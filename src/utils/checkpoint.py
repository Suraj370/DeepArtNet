"""Checkpoint management utilities for DeepArtNet.

Thin wrappers around :meth:`~src.models.deepartnet.DeepArtNet.save_checkpoint`
and :meth:`~src.models.deepartnet.DeepArtNet.load_from_checkpoint` that add
top-k rotation (keep only the best k checkpoints in a directory) and a
convenience function for listing saved checkpoints.
"""

from __future__ import annotations

import heapq
import logging
import pathlib
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: Any,
    path: str | pathlib.Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a model checkpoint, delegating to the model's own method.

    Args:
        model: A :class:`~src.models.deepartnet.DeepArtNet` instance.
        path: Destination ``.pth`` file path.
        metadata: Optional extra metadata to embed (epoch, accuracy, etc.).
    """
    model.save_checkpoint(path, metadata=metadata)
    logger.info("Checkpoint saved: %s", path)


def load_checkpoint(
    model_cls: Any,
    path: str | pathlib.Path,
    map_location: Optional[str | torch.device] = None,
) -> Any:
    """Load a checkpoint and return the reconstructed model.

    Args:
        model_cls: The model class (e.g. ``DeepArtNet``) that implements
            ``load_from_checkpoint``.
        path: Path to a ``.pth`` file.
        map_location: Passed to ``torch.load`` (e.g. ``"cpu"``).

    Returns:
        A model instance with weights loaded, in ``eval()`` mode.
    """
    model = model_cls.load_from_checkpoint(path, map_location=map_location)
    logger.info("Checkpoint loaded: %s", path)
    return model


def rotate_checkpoints(
    checkpoint_dir: str | pathlib.Path,
    keep_top_k: int = 3,
    metric_key: str = "mean_val_acc",
) -> None:
    """Delete the lowest-scoring checkpoints, keeping only the best ``k``.

    Reads the ``metadata`` field embedded in each ``.pth`` file and ranks
    them by ``metadata[metric_key]`` (higher is better).  Files that cannot
    be read or are missing the key are treated as score 0.

    Args:
        checkpoint_dir: Directory to scan for ``.pth`` files.
        keep_top_k: Number of top checkpoints to retain (default 3).
        metric_key: Key inside ``metadata`` used for ranking (default
            ``"mean_val_acc"``).
    """
    ckpt_dir = pathlib.Path(checkpoint_dir)
    pth_files = list(ckpt_dir.glob("*.pth"))

    if len(pth_files) <= keep_top_k:
        return  # nothing to delete

    scored: List[tuple[float, pathlib.Path]] = []
    for pth in pth_files:
        try:
            payload = torch.load(pth, map_location="cpu", weights_only=False)
            score = float(payload.get("metadata", {}).get(metric_key, 0.0))
        except Exception:
            score = 0.0
        scored.append((score, pth))

    # Keep the top-k; delete the rest
    top_k = heapq.nlargest(keep_top_k, scored, key=lambda x: x[0])
    top_k_paths = {p for _, p in top_k}

    for score, pth in scored:
        if pth not in top_k_paths:
            pth.unlink()
            logger.info("Rotated out checkpoint (score=%.4f): %s", score, pth)


def list_checkpoints(checkpoint_dir: str | pathlib.Path) -> List[pathlib.Path]:
    """Return all ``.pth`` files in ``checkpoint_dir``, sorted by mtime (newest first).

    Args:
        checkpoint_dir: Directory to scan.

    Returns:
        List of :class:`pathlib.Path` objects.
    """
    ckpt_dir = pathlib.Path(checkpoint_dir)
    return sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
