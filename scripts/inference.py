"""CLI inference script for DeepArtNet.

Loads a trained checkpoint, runs prediction on a single image, prints a
formatted results table, and optionally saves an attention heatmap overlay.

Usage::

    python scripts/inference.py \\
        --checkpoint outputs/checkpoints/phase3_best.pth \\
        --image path/to/painting.jpg

    # With attention visualisation
    python scripts/inference.py \\
        --checkpoint outputs/checkpoints/phase3_best.pth \\
        --image path/to/painting.jpg \\
        --visualize_attention \\
        --output_dir outputs/visualizations
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.models.deepartnet import DeepArtNet
from src.utils.logging_utils import setup_logging
from src.utils.visualization import visualize_attention

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained DeepArtNet model")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pth checkpoint")
    parser.add_argument("--image", required=True,
                        help="Path to input image")
    parser.add_argument("--visualize_attention", action="store_true",
                        help="Save attention heatmap overlay")
    parser.add_argument("--output_dir", default="outputs/visualizations",
                        help="Directory to save the attention heatmap (if requested)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _print_table(predictions: dict) -> None:
    """Print predictions as a formatted table."""
    col_w = (12, 30, 12)
    header = f"{'Attribute':<{col_w[0]}} {'Predicted Class':<{col_w[1]}} {'Confidence':>{col_w[2]}}"
    sep = "-" * sum(col_w)
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    for attr, result in predictions.items():
        logger.info(
            "%-*s %-*s %*s",
            col_w[0], attr,
            col_w[1], result["label"],
            col_w[2], f"{result['confidence']:.4f}",
        )
    logger.info(sep)


def main() -> None:
    args = parse_args()
    setup_logging()

    image_path = pathlib.Path(args.image)
    if not image_path.exists():
        logger.error("Image not found: %s", image_path)
        sys.exit(1)

    model = DeepArtNet.load_from_checkpoint(args.checkpoint, map_location=args.device)
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    pil_image = Image.open(image_path)
    predictions = model.predict(pil_image)

    logger.info("Predictions for: %s", image_path.name)
    _print_table(predictions)

    if args.visualize_attention:
        output_dir = pathlib.Path(args.output_dir)
        save_path = output_dir / f"{image_path.stem}_attention.png"
        visualize_attention(model, pil_image, save_path=save_path)
        logger.info("Attention heatmap saved: %s", save_path)


if __name__ == "__main__":
    main()
