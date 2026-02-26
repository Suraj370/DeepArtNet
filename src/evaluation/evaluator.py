"""Evaluation loop and reporting for DeepArtNet.

Provides :class:`Evaluator` which runs the full validation set through the
model, aggregates metrics, prints a formatted report, and optionally saves
per-attribute confusion matrix figures.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_all_metrics
from src.models.deepartnet import DeepArtNet

logger = logging.getLogger(__name__)

_NUM_CLASSES = {"style": 27, "genre": 10, "artist": 23}


class Evaluator:
    """Runs evaluation and generates reports for all three attributes.

    Example::

        evaluator = Evaluator()
        metrics = evaluator.evaluate(model, val_loader, device="cuda")
        evaluator.print_report(metrics)
        evaluator.save_confusion_matrices(metrics, save_dir="outputs/visualizations")
    """

    def evaluate(
        self,
        model: DeepArtNet,
        val_loader: DataLoader,
        device: str | torch.device = "cpu",
    ) -> Dict[str, Dict[str, object]]:
        """Run a full evaluation pass over the validation loader.

        Args:
            model: Trained :class:`~src.models.deepartnet.DeepArtNet` instance.
            val_loader: DataLoader over the validation split.
            device: Device to run inference on.

        Returns:
            Nested metrics dict as returned by
            :func:`~src.evaluation.metrics.compute_all_metrics`.
        """
        device = torch.device(device)
        model = model.to(device)
        model.eval()

        all_logits: Dict[str, list] = {"style": [], "genre": [], "artist": []}
        all_labels: Dict[str, list] = {"style": [], "genre": [], "artist": []}

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                outputs = model(images)

                for attr in ("style", "genre", "artist"):
                    all_logits[attr].append(outputs[attr].cpu())
                    all_labels[attr].append(labels[attr].cpu())

        logits_dict = {attr: torch.cat(v, dim=0) for attr, v in all_logits.items()}
        labels_dict = {attr: torch.cat(v, dim=0) for attr, v in all_labels.items()}

        return compute_all_metrics(
            logits_dict, labels_dict, num_classes=_NUM_CLASSES
        )

    def print_report(self, metrics: Dict[str, Dict[str, object]]) -> None:
        """Print a formatted accuracy table to the logger.

        Args:
            metrics: Output of :meth:`evaluate`.
        """
        header = f"{'Attribute':<12} {'Top-1 Acc':>10} {'Top-5 Acc':>10}"
        separator = "-" * len(header)
        logger.info(separator)
        logger.info(header)
        logger.info(separator)

        for attr, m in metrics.items():
            logger.info(
                "%-12s %10.4f %10.4f",
                attr,
                float(m["top1"]),  # type: ignore[arg-type]
                float(m["top5"]),  # type: ignore[arg-type]
            )
        logger.info(separator)

        mean_top1 = sum(float(m["top1"]) for m in metrics.values()) / len(metrics)  # type: ignore[arg-type]
        logger.info("%-12s %10.4f", "Mean", mean_top1)
        logger.info(separator)

    def save_confusion_matrices(
        self,
        metrics: Dict[str, Dict[str, object]],
        save_dir: str | pathlib.Path,
        class_names: Optional[Dict[str, list]] = None,
    ) -> None:
        """Save per-attribute confusion matrix plots as PNG files.

        Args:
            metrics: Output of :meth:`evaluate`.
            save_dir: Directory where PNG files will be written.
            class_names: Optional dict mapping attribute name to list of class
                name strings.  When provided, axis ticks are labelled.
        """
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for attr, m in metrics.items():
            cm: np.ndarray = m["confusion_matrix"]  # type: ignore[assignment]
            nc = cm.shape[0]

            fig, ax = plt.subplots(figsize=(max(8, nc * 0.4), max(6, nc * 0.4)))
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            fig.colorbar(im, ax=ax)

            if class_names and attr in class_names:
                names = class_names[attr]
                ax.set_xticks(range(nc))
                ax.set_yticks(range(nc))
                ax.set_xticklabels(names, rotation=90, fontsize=6)
                ax.set_yticklabels(names, fontsize=6)

            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Confusion Matrix — {attr.capitalize()}")

            out_path = save_dir / f"confusion_matrix_{attr}.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            logger.info("Saved confusion matrix: %s", out_path)
