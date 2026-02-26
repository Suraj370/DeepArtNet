"""Evaluation metrics for multi-attribute art classification.

Provides functions to compute top-k accuracy, per-class accuracy, and
confusion matrices, all with support for ``ignore_index`` masking so that
samples with missing labels are excluded from statistics.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch


def compute_topk_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: Tuple[int, ...] = (1, 5),
    ignore_index: int = -1,
) -> Dict[str, float]:
    """Compute top-k accuracy for one or more values of k.

    Args:
        logits: Raw class scores of shape ``(N, num_classes)``.
        labels: Ground-truth integer labels of shape ``(N,)``.
        k: Tuple of k values to evaluate (default ``(1, 5)``).
        ignore_index: Label value to exclude from evaluation (default -1).

    Returns:
        Dict mapping ``"top1"``, ``"top5"``, … to accuracy floats in [0, 1].
        Returns 0.0 for each k when all labels are masked.
    """
    valid_mask = labels != ignore_index
    if valid_mask.sum() == 0:
        return {f"top{ki}": 0.0 for ki in k}

    vlogits = logits[valid_mask]   # (M, C)
    vtargets = labels[valid_mask]  # (M,)
    n = vtargets.size(0)

    max_k = max(k)
    actual_k = min(max_k, vlogits.size(1))
    topk_preds = vlogits.topk(actual_k, dim=1).indices  # (M, actual_k)

    results: Dict[str, float] = {}
    for ki in k:
        ki_clamped = min(ki, actual_k)
        correct = (topk_preds[:, :ki_clamped] == vtargets.unsqueeze(1)).any(dim=1).sum().item()
        results[f"top{ki}"] = correct / n
    return results


def compute_confusion_matrix(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> np.ndarray:
    """Compute a confusion matrix from logits and labels.

    Args:
        logits: Raw class scores of shape ``(N, num_classes)``.
        labels: Ground-truth integer labels of shape ``(N,)``.
        num_classes: Total number of classes (sets matrix dimensions).
        ignore_index: Label value to exclude (default -1).

    Returns:
        Confusion matrix as ``np.ndarray`` of shape
        ``(num_classes, num_classes)`` with dtype ``int64``.
        Entry ``[i, j]`` is the count of samples with true label ``i``
        predicted as class ``j``.
    """
    valid_mask = labels != ignore_index
    if valid_mask.sum() == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)

    preds = logits[valid_mask].argmax(dim=1).cpu().numpy()
    targets = labels[valid_mask].cpu().numpy()

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (targets, preds), 1)
    return cm


def per_class_accuracy(confusion_matrix: np.ndarray) -> np.ndarray:
    """Compute per-class accuracy from a confusion matrix.

    Args:
        confusion_matrix: Square matrix of shape ``(C, C)`` as returned by
            :func:`compute_confusion_matrix`.

    Returns:
        Array of shape ``(C,)`` where entry ``i`` is the recall (accuracy)
        for class ``i``.  Classes with zero true samples get accuracy 0.0.
    """
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid division by zero
    return (confusion_matrix.diagonal() / row_sums.squeeze()).astype(float)


def compute_all_metrics(
    outputs_dict: Dict[str, torch.Tensor],
    labels_dict: Dict[str, torch.Tensor],
    num_classes: Dict[str, int],
    ignore_index: int = -1,
) -> Dict[str, Dict[str, object]]:
    """Compute top-k accuracy and confusion matrix for every attribute.

    Args:
        outputs_dict: Dict mapping attribute name to logit tensor ``(N, C)``.
        labels_dict: Dict mapping attribute name to label tensor ``(N,)``.
        num_classes: Dict mapping attribute name to number of classes.
        ignore_index: Label value to ignore (default -1).

    Returns:
        Nested dict::

            {
                "style": {
                    "top1": float,
                    "top5": float,
                    "confusion_matrix": np.ndarray (27, 27),
                    "per_class_acc": np.ndarray (27,),
                },
                "genre": { ... },
                "artist": { ... },
            }
    """
    results: Dict[str, Dict[str, object]] = {}
    for attr in outputs_dict:
        logits = outputs_dict[attr]
        labels = labels_dict[attr]
        nc = num_classes[attr]

        topk = compute_topk_accuracy(logits, labels, k=(1, 5), ignore_index=ignore_index)
        cm = compute_confusion_matrix(logits, labels, num_classes=nc, ignore_index=ignore_index)
        pca = per_class_accuracy(cm)

        results[attr] = {
            "top1": topk["top1"],
            "top5": topk["top5"],
            "confusion_matrix": cm,
            "per_class_acc": pca,
        }
    return results
