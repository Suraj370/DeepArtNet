"""Loss functions for multi-task, class-imbalanced art classification.

Provides:
  - ``FocalLoss``     — cross-entropy reweighted by (1-p_t)^gamma to down-weight
                        easy examples; supports label smoothing and ignore_index.
  - ``MultiTaskLoss`` — Kendall et al. (2018) homoscedastic uncertainty weighting
                        across style, genre, and artist tasks; each task weight is
                        a learned parameter.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss with label smoothing and ignore-index masking.

    Computes cross-entropy with label smoothing, then reweights each sample
    by ``(1 - p_t) ** gamma`` where ``p_t`` is the model's estimated
    probability for the correct class.  Samples whose label equals
    ``ignore_index`` are masked out before any computation.

    Args:
        gamma: Focusing parameter.  Higher values down-weight well-classified
            examples more aggressively (default 2.0).
        alpha: Unused scalar kept for API compatibility; focal weighting
            already handles class balance (default 0.25).
        label_smoothing: Smoothing factor forwarded to
            ``F.cross_entropy`` (default 0.1).
        ignore_index: Label value to exclude from the loss (default -1).

    Example::

        criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
        loss = criterion(logits, labels)   # logits (B, C), labels (B,)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        label_smoothing: float = 0.1,
        ignore_index: int = -1,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute focal loss over a batch.

        Args:
            logits: Raw class scores of shape ``(B, num_classes)``.
            labels: Integer class labels of shape ``(B,)``.  Entries equal to
                ``ignore_index`` are excluded from the loss.

        Returns:
            Scalar mean focal loss over valid samples.  Returns
            ``torch.tensor(0.0)`` when all labels are ``ignore_index``.
        """
        # Mask out ignore_index samples
        valid_mask = labels != self.ignore_index
        if valid_mask.sum() == 0:
            return logits.sum() * 0.0  # zero with grad

        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]

        # Standard cross-entropy with label smoothing (per-sample)
        ce = F.cross_entropy(
            valid_logits,
            valid_labels,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        # p_t = probability assigned to the correct class (no smoothing for focal weight)
        with torch.no_grad():
            probs = F.softmax(valid_logits, dim=-1)
            p_t = probs.gather(1, valid_labels.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - p_t) ** self.gamma
        loss = (focal_weight * ce).mean()
        return loss


class MultiTaskLoss(nn.Module):
    """Homoscedastic uncertainty-weighted multi-task loss (Kendall et al., 2018).

    Learns one ``log_sigma2`` parameter per task.  The total loss is::

        L = Σ_i [ exp(-log_sigma2_i) * focal_i + log_sigma2_i ]

    This is equivalent to maximising the log-likelihood under a Gaussian
    observation model and allows the network to balance task contributions
    automatically without manual tuning.

    Args:
        tasks: Ordered list of task names matching the keys in the logits
            and labels dicts (default ``["style", "genre", "artist"]``).
        gamma: Focal loss focusing parameter (default 2.0).
        label_smoothing: Label smoothing for the underlying focal losses
            (default 0.1).
        ignore_index: Label value treated as missing / masked (default -1).

    Example::

        criterion = MultiTaskLoss()
        total, per_task = criterion(logits_dict, labels_dict)
    """

    def __init__(
        self,
        tasks: List[str] | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        ignore_index: int = -1,
    ) -> None:
        super().__init__()
        if tasks is None:
            tasks = ["style", "genre", "artist"]
        self.tasks = tasks

        # One learnable log(σ²) per task; initialised to 0 → σ²=1
        self.log_sigma2 = nn.Parameter(torch.zeros(len(tasks)))

        self._focal = FocalLoss(
            gamma=gamma,
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
        )

    def forward(
        self,
        logits_dict: Dict[str, torch.Tensor],
        labels_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total and per-task losses.

        Args:
            logits_dict: Dict mapping task name → logit tensor ``(B, C)``.
            labels_dict: Dict mapping task name → label tensor ``(B,)``.

        Returns:
            A 2-tuple ``(total_loss, per_task_losses)`` where:

            - **total_loss** — scalar weighted sum across all tasks.
            - **per_task_losses** — dict mapping task name to its unweighted
              focal loss scalar (useful for TensorBoard logging).
        """
        total_loss = torch.tensor(0.0, device=self.log_sigma2.device)
        per_task: Dict[str, torch.Tensor] = {}

        for i, task in enumerate(self.tasks):
            focal = self._focal(logits_dict[task], labels_dict[task])
            per_task[task] = focal

            # Kendall weighting: exp(-log_sigma2) * L + log_sigma2
            precision = torch.exp(-self.log_sigma2[i])
            total_loss = total_loss + precision * focal + self.log_sigma2[i]

        return total_loss, per_task

    def get_task_weights(self) -> Dict[str, float]:
        """Return the current effective weight per task for logging.

        The effective weight is ``exp(-log_sigma2_i)`` — higher means the
        model is more confident about that task's predictions.

        Returns:
            Dict mapping task name to its current precision weight (float).
        """
        with torch.no_grad():
            precisions = torch.exp(-self.log_sigma2)
        return {task: float(precisions[i].item()) for i, task in enumerate(self.tasks)}
