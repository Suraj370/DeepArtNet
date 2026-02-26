"""CLI evaluation script for DeepArtNet.

Runs the full validation set through a trained model and prints per-attribute
top-1 / top-5 accuracy, then optionally saves confusion matrix plots.

Usage::

    python scripts/evaluate.py \\
        --checkpoint outputs/checkpoints/phase3_best.pth \\
        --data_dir data/wikiart \\
        --output_dir outputs/eval
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data.dataloader import build_dataloaders
from src.data.dataset import load_class_names
from src.evaluation.evaluator import Evaluator
from src.models.deepartnet import DeepArtNet
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DeepArtNet checkpoint")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pth checkpoint")
    parser.add_argument("--data_dir", default="data/wikiart",
                        help="Root WikiArt data directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", default="outputs/eval",
                        help="Directory to save confusion matrix plots")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=output_dir / "evaluate.log")

    data_dir = pathlib.Path(args.data_dir)

    # Load model from checkpoint (class names are embedded in checkpoint)
    model = DeepArtNet.load_from_checkpoint(
        args.checkpoint, map_location=args.device
    )
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    # Build val loader only
    _, val_loader = build_dataloaders(
        image_dir=str(data_dir / "images"),
        style_train_csv=str(data_dir / "style_train.csv"),
        style_val_csv=str(data_dir / "style_val.csv"),
        genre_train_csv=str(data_dir / "genre_train.csv"),
        genre_val_csv=str(data_dir / "genre_val.csv"),
        artist_train_csv=str(data_dir / "artist_train.csv"),
        artist_val_csv=str(data_dir / "artist_val.csv"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=False,
    )

    evaluator = Evaluator()
    metrics = evaluator.evaluate(model, val_loader, device=args.device)
    evaluator.print_report(metrics)

    class_names = {
        "style": model.style_class_names,
        "genre": model.genre_class_names,
        "artist": model.artist_class_names,
    }
    evaluator.save_confusion_matrices(metrics, save_dir=output_dir, class_names=class_names)
    logger.info("Evaluation complete. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
