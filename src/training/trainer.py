"""Three-phase progressive training for DeepArtNet.

Training strategy
-----------------
Phase 1  Freeze BiLSTM + Attention, train heads + backbone projection.
         LR = 1e-3, 20 epochs, batch = 64.
Phase 2  Freeze backbone blocks 0-2, fine-tune LSTM + attention + heads.
         LR = 5e-4, 30 epochs, batch = 32.
Phase 3  Unfreeze all; end-to-end fine-tuning.
         LR = 1e-5, 20 epochs, batch = 32.

Features
--------
- Mixed-precision training (``torch.cuda.amp``)
- Gradient clipping (max_norm = 1.0)
- Best-checkpoint saving when mean val accuracy across 3 tasks improves
- TensorBoard logging: total loss, per-task loss, per-task accuracy,
  attention entropy
- ``--resume`` support via ``load_from_checkpoint``
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models.deepartnet import DeepArtNet
from src.training.losses import MultiTaskLoss
from src.training.scheduler import build_scheduler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase configuration
# ---------------------------------------------------------------------------

_PHASE_DEFAULTS: Dict[int, Dict[str, Any]] = {
    1: {"epochs": 20, "lr": 1e-3,  "batch_size": 64, "freeze_lstm": True,  "freeze_backbone_blocks": []},
    2: {"epochs": 30, "lr": 5e-4,  "batch_size": 32, "freeze_lstm": False, "freeze_backbone_blocks": [0, 1, 2]},
    3: {"epochs": 20, "lr": 1e-5,  "batch_size": 32, "freeze_lstm": False, "freeze_backbone_blocks": []},
}


class Trainer:
    """Manages the three-phase training loop for DeepArtNet.

    Args:
        model: The :class:`~src.models.deepartnet.DeepArtNet` instance to train.
        config: Flat config dict (typically loaded from ``train_config.yaml``).
            Keys used: ``training.phase{1,2,3}``, ``training.weight_decay``,
            ``training.grad_clip``, ``training.label_smoothing``,
            ``training.focal_gamma``, ``training.mixed_precision``,
            ``training.scheduler``, ``output.checkpoint_dir``,
            ``output.log_dir``.
        device: Target device string or :class:`torch.device` (e.g. ``"cuda"``).

    Example::

        trainer = Trainer(model, config, device="cuda")
        trainer.train_all_phases(train_loader, val_loader)
    """

    def __init__(
        self,
        model: DeepArtNet,
        config: Dict[str, Any],
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)

        training_cfg = config.get("training", {})
        output_cfg = config.get("output", {})

        self.weight_decay: float = float(training_cfg.get("weight_decay", 1e-4))
        self.grad_clip: float = float(training_cfg.get("grad_clip", 1.0))
        self.mixed_precision: bool = bool(training_cfg.get("mixed_precision", True))

        self.checkpoint_dir = pathlib.Path(output_cfg.get("checkpoint_dir", "outputs/checkpoints"))
        self.log_dir = pathlib.Path(output_cfg.get("log_dir", "outputs/logs"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = MultiTaskLoss(
            gamma=float(training_cfg.get("focal_gamma", 2.0)),
            label_smoothing=float(training_cfg.get("label_smoothing", 0.1)),
        ).to(self.device)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.scaler = GradScaler(enabled=self.mixed_precision and self.device.type == "cuda")

        self._best_mean_acc: float = 0.0
        self._global_step: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_all_phases(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_path: Optional[str | pathlib.Path] = None,
    ) -> None:
        """Run all three training phases sequentially.

        Args:
            train_loader: DataLoader for the training split.
            val_loader: DataLoader for the validation split.
            resume_path: Optional path to a checkpoint to resume from.
                If provided the model weights are loaded before phase 1.
        """
        if resume_path is not None:
            self._resume(pathlib.Path(resume_path))

        for phase in (1, 2, 3):
            logger.info("=" * 60)
            logger.info("Starting Phase %d", phase)
            logger.info("=" * 60)
            self.train_phase(phase, train_loader, val_loader)

        self.writer.close()

    def train_phase(
        self,
        phase: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """Run a single training phase.

        Args:
            phase: Phase number (1, 2, or 3).
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
        """
        phase_cfg = self._get_phase_cfg(phase)
        epochs: int = phase_cfg["epochs"]
        lr: float = phase_cfg["lr"]

        self._apply_phase_freezing(phase_cfg)

        optimizer = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, self.model.parameters()))
            + list(self.criterion.parameters()),
            lr=lr,
            weight_decay=self.weight_decay,
        )

        scheduler_cfg = self.config.get("training", {}).get(
            "scheduler", {"type": "cosine_annealing", "T_max": epochs, "eta_min": 1e-7}
        )
        scheduler = build_scheduler(optimizer, {**scheduler_cfg, "T_max": epochs})

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer)
            val_metrics = self._val_epoch(val_loader)
            scheduler.step()

            mean_acc = sum(
                val_metrics[t]["top1"] for t in ("style", "genre", "artist")
            ) / 3.0

            self._log_epoch(phase, epoch, train_loss, val_metrics, mean_acc, optimizer)

            if mean_acc > self._best_mean_acc:
                self._best_mean_acc = mean_acc
                ckpt_path = self.checkpoint_dir / f"phase{phase}_best.pth"
                self.model.save_checkpoint(
                    ckpt_path,
                    metadata={
                        "phase": phase,
                        "epoch": epoch,
                        "mean_val_acc": mean_acc,
                        **{f"val_acc_{t}": val_metrics[t]["top1"] for t in ("style", "genre", "artist")},
                    },
                )
                logger.info(
                    "Phase %d epoch %d — new best mean acc %.4f — saved %s",
                    phase, epoch, mean_acc, ckpt_path,
                )

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def _freeze_module(self, module: nn.Module) -> None:
        """Disable gradient computation for all parameters in ``module``."""
        for param in module.parameters():
            param.requires_grad = False

    def _unfreeze_module(self, module: nn.Module) -> None:
        """Enable gradient computation for all parameters in ``module``."""
        for param in module.parameters():
            param.requires_grad = True

    def _apply_phase_freezing(self, phase_cfg: Dict[str, Any]) -> None:
        """Apply the freeze/unfreeze schedule for a given phase config."""
        # Start by unfreezing everything
        self._unfreeze_module(self.model)

        if phase_cfg.get("freeze_lstm", False):
            self._freeze_module(self.model.encoder)
            self._freeze_module(self.model.attention)

        freeze_blocks = phase_cfg.get("freeze_backbone_blocks", [])
        if freeze_blocks:
            self.model.backbone.freeze_blocks(freeze_blocks)

    # ------------------------------------------------------------------
    # Training / validation loops
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Run one full pass over the training set.

        Returns:
            Mean total loss over all batches.
        """
        self.model.train()
        self.criterion.train()
        total_loss = 0.0
        num_batches = 0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = {k: v.to(self.device, non_blocking=True) for k, v in labels.items()}

            optimizer.zero_grad()

            with autocast(enabled=self.mixed_precision and self.device.type == "cuda"):
                outputs = self.model(images)
                loss, _ = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.criterion.parameters()),
                max_norm=self.grad_clip,
            )
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1
            self._global_step += 1

        return total_loss / max(num_batches, 1)

    def _val_epoch(
        self,
        loader: DataLoader,
    ) -> Dict[str, Dict[str, float]]:
        """Run one full pass over the validation set.

        Returns:
            Nested dict ``{task: {"top1": float, "top5": float}}``.
        """
        self.model.eval()
        tasks = ("style", "genre", "artist")
        correct_top1: Dict[str, int] = {t: 0 for t in tasks}
        correct_top5: Dict[str, int] = {t: 0 for t in tasks}
        valid_total: Dict[str, int] = {t: 0 for t in tasks}

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = {k: v.to(self.device, non_blocking=True) for k, v in labels.items()}
                outputs = self.model(images)

                for task in tasks:
                    logits = outputs[task]        # (B, C)
                    targets = labels[task]        # (B,)
                    valid_mask = targets != -1
                    if valid_mask.sum() == 0:
                        continue

                    vlogits = logits[valid_mask]
                    vtargets = targets[valid_mask]
                    n = vtargets.size(0)
                    valid_total[task] += n

                    # Top-1
                    pred1 = vlogits.argmax(dim=1)
                    correct_top1[task] += (pred1 == vtargets).sum().item()

                    # Top-5 (handle fewer than 5 classes)
                    k = min(5, vlogits.size(1))
                    top5_preds = vlogits.topk(k, dim=1).indices
                    correct_top5[task] += (
                        top5_preds == vtargets.unsqueeze(1)
                    ).any(dim=1).sum().item()

        metrics: Dict[str, Dict[str, float]] = {}
        for task in tasks:
            n = max(valid_total[task], 1)
            metrics[task] = {
                "top1": correct_top1[task] / n,
                "top5": correct_top5[task] / n,
            }
        return metrics

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_epoch(
        self,
        phase: int,
        epoch: int,
        train_loss: float,
        val_metrics: Dict[str, Dict[str, float]],
        mean_acc: float,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Write scalars to TensorBoard and log to console."""
        tag_prefix = f"phase{phase}"
        step = self._global_step

        self.writer.add_scalar(f"{tag_prefix}/train_loss", train_loss, step)
        self.writer.add_scalar(f"{tag_prefix}/val_mean_acc_top1", mean_acc, step)
        self.writer.add_scalar(f"{tag_prefix}/lr", optimizer.param_groups[0]["lr"], step)

        for task, m in val_metrics.items():
            self.writer.add_scalar(f"{tag_prefix}/val_{task}_top1", m["top1"], step)
            self.writer.add_scalar(f"{tag_prefix}/val_{task}_top5", m["top5"], step)

        task_weights = self.criterion.get_task_weights()
        for task, w in task_weights.items():
            self.writer.add_scalar(f"{tag_prefix}/task_weight_{task}", w, step)

        logger.info(
            "Phase %d | Epoch %3d | loss %.4f | "
            "style %.3f | genre %.3f | artist %.3f | mean %.3f",
            phase, epoch, train_loss,
            val_metrics["style"]["top1"],
            val_metrics["genre"]["top1"],
            val_metrics["artist"]["top1"],
            mean_acc,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_phase_cfg(self, phase: int) -> Dict[str, Any]:
        """Merge default phase config with any overrides from ``self.config``."""
        defaults = _PHASE_DEFAULTS[phase].copy()
        training_cfg = self.config.get("training", {})
        user_phase = training_cfg.get(f"phase{phase}", {})
        defaults.update(user_phase)
        return defaults

    def _resume(self, path: pathlib.Path) -> None:
        """Load model weights from a checkpoint file.

        Args:
            path: Path to a ``.pth`` checkpoint saved by
                :meth:`~src.models.deepartnet.DeepArtNet.save_checkpoint`.
        """
        import torch as _torch
        payload = _torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state_dict"])
        meta = payload.get("metadata", {})
        logger.info("Resumed from %s (metadata: %s)", path, meta)
