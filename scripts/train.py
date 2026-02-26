"""CLI training script for DeepArtNet.

Supports single-phase and all-phases training, checkpoint resuming, and
per-GPU multi-device selection.

Usage examples::

    # Train all phases with default config
    python scripts/train.py --config configs/train_config.yaml

    # Train only phase 2, resuming from a previous checkpoint
    python scripts/train.py --phase 2 --resume outputs/checkpoints/phase1_best.pth

    # Override data and output directories
    python scripts/train.py --data_dir data/wikiart --output_dir runs/exp1
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import random
import sys

import numpy as np
import torch
import yaml

# Ensure project root is on the path when run as a script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data.dataloader import build_dataloaders
from src.data.dataset import load_class_names
from src.models.deepartnet import DeepArtNet
from src.training.trainer import Trainer
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(config_path: pathlib.Path) -> dict:
    base_cfg_path = config_path.parent / "base_config.yaml"
    config: dict = {}
    if base_cfg_path.exists():
        with base_cfg_path.open() as f:
            config.update(yaml.safe_load(f) or {})
    with config_path.open() as f:
        config.update(yaml.safe_load(f) or {})
    return config


def _merge_cli(config: dict, args: argparse.Namespace) -> dict:
    """Override config values with explicit CLI arguments."""
    data_cfg = config.setdefault("data", {})
    output_cfg = config.setdefault("output", {})

    if args.data_dir:
        data_cfg["data_dir"] = args.data_dir
    if args.image_dir:
        data_cfg["image_dir"] = args.image_dir
    if args.output_dir:
        output_cfg["checkpoint_dir"] = str(pathlib.Path(args.output_dir) / "checkpoints")
        output_cfg["log_dir"] = str(pathlib.Path(args.output_dir) / "logs")
    if args.seed is not None:
        config["seed"] = args.seed

    return config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepArtNet")
    parser.add_argument("--config", default="configs/train_config.yaml",
                        help="Path to train_config.yaml")
    parser.add_argument("--data_dir", default=None,
                        help="Root data directory (overrides config)")
    parser.add_argument("--image_dir", default=None,
                        help="Image directory (overrides config)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for checkpoints + logs")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=None,
                        help="Run a single phase (default: all 3)")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--gpus", default="0",
                        help="Comma-separated GPU indices, or 'cpu'")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = pathlib.Path(args.config)
    config = _load_config(config_path)
    config = _merge_cli(config, args)

    setup_logging(
        log_file=pathlib.Path(config.get("output", {}).get("log_dir", "outputs/logs")) / "train.log"
    )

    seed = int(config.get("seed", 42))
    _set_seed(seed)
    logger.info("Random seed: %d", seed)

    # Device
    if args.gpus.lower() == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        gpu_idx = int(args.gpus.split(",")[0])
        device = torch.device(f"cuda:{gpu_idx}")
    logger.info("Using device: %s", device)

    # Class names
    data_cfg = config.get("data", {})
    style_names = load_class_names(data_cfg["style_class_file"])
    genre_names = load_class_names(data_cfg["genre_class_file"])
    artist_names = load_class_names(data_cfg["artist_class_file"])

    # DataLoaders
    training_cfg = config.get("training", {})
    phase1_batch = training_cfg.get("phase1", {}).get("batch_size", 64)
    train_loader, val_loader = build_dataloaders(
        image_dir=data_cfg["image_dir"],
        style_train_csv=data_cfg["style_train_csv"],
        style_val_csv=data_cfg["style_val_csv"],
        genre_train_csv=data_cfg["genre_train_csv"],
        genre_val_csv=data_cfg["genre_val_csv"],
        artist_train_csv=data_cfg["artist_train_csv"],
        artist_val_csv=data_cfg["artist_val_csv"],
        batch_size=phase1_batch,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )

    # Model
    model = DeepArtNet(
        style_class_names=style_names,
        genre_class_names=genre_names,
        artist_class_names=artist_names,
        pretrained_backbone=bool(config.get("model", {}).get("pretrained", True)),
    )
    logger.info("Model initialised: %d trainable params",
                sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Trainer
    trainer = Trainer(model=model, config=config, device=device)

    if args.phase is not None:
        logger.info("Running single phase: %d", args.phase)
        if args.resume:
            trainer._resume(pathlib.Path(args.resume))
        trainer.train_phase(args.phase, train_loader, val_loader)
    else:
        trainer.train_all_phases(train_loader, val_loader, resume_path=args.resume)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
