"""Learning-rate scheduler utilities for DeepArtNet training.

Wraps ``torch.optim.lr_scheduler.CosineAnnealingLR`` and exposes a single
factory function so the trainer can build the scheduler from the YAML config
without importing PyTorch scheduler classes directly.
"""

from __future__ import annotations

from typing import Any, Dict

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler


def build_scheduler(
    optimizer: Optimizer,
    scheduler_cfg: Dict[str, Any],
) -> LRScheduler:
    """Construct a learning-rate scheduler from a config dict.

    Currently supports ``type: cosine_annealing`` only.  Additional
    scheduler types can be added here without changing the Trainer.

    Args:
        optimizer: The optimizer whose LR the scheduler will adjust.
        scheduler_cfg: Sub-dict from the training config, e.g.::

                {"type": "cosine_annealing", "T_max": 20, "eta_min": 1e-7}

    Returns:
        A configured :class:`~torch.optim.lr_scheduler.LRScheduler` instance.

    Raises:
        ValueError: If ``scheduler_cfg["type"]`` is not recognised.
    """
    scheduler_type: str = scheduler_cfg.get("type", "cosine_annealing")

    if scheduler_type == "cosine_annealing":
        return CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_cfg.get("T_max", 20)),
            eta_min=float(scheduler_cfg.get("eta_min", 1e-7)),
        )

    raise ValueError(
        f"Unknown scheduler type {scheduler_type!r}. "
        "Supported types: 'cosine_annealing'."
    )
