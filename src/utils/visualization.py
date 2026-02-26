"""Attention heatmap visualisation for DeepArtNet.

Provides ``visualize_attention`` which overlays the model's spatial attention
weights (B=1, 49) as a jet-colourmap heatmap on the original input image.
The 7×7 weight grid is upsampled to 224×224 via bilinear interpolation.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


def visualize_attention(
    model: "src.models.deepartnet.DeepArtNet",  # type: ignore[name-defined]
    pil_image: Image.Image,
    save_path: Optional[str | pathlib.Path] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay spatial attention weights as a heatmap on the original image.

    Runs a single forward pass (batch size = 1) through ``model`` to obtain
    the ``attention_weights`` tensor of shape ``(1, 49)``.  The weights are
    reshaped to ``(7, 7)``, upsampled to ``(224, 224)`` via bilinear
    interpolation, and blended with the original image using a jet colourmap.

    Args:
        model: A :class:`~src.models.deepartnet.DeepArtNet` instance.
        pil_image: Input PIL image (any mode; converted to RGB internally).
        save_path: If provided the overlay is saved as a PNG at this path.
            Parent directories are created automatically.
        alpha: Blending factor for the heatmap overlay (0 = original image
            only, 1 = heatmap only).  Default 0.5.

    Returns:
        The blended overlay as a ``np.ndarray`` of shape ``(224, 224, 3)``
        with dtype ``uint8``.
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    from torchvision import transforms as T
    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    device = next(model.parameters()).device
    tensor = val_transform(pil_image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(tensor)

    # attention_weights: (1, 49) → (1, 1, 7, 7)
    weights = output["attention_weights"]  # (1, 49)
    weights_2d = weights.reshape(1, 1, 7, 7)

    # Upsample to (224, 224)
    heatmap = F.interpolate(weights_2d, size=(224, 224), mode="bilinear", align_corners=False)
    heatmap = heatmap.squeeze().cpu().numpy()  # (224, 224)

    # Normalise to [0, 1]
    heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
    if heatmap_max > heatmap_min:
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

    # Resize original image for overlay
    base_img = pil_image.resize((224, 224), Image.BILINEAR)
    base_arr = np.asarray(base_img, dtype=np.float32) / 255.0  # (224, 224, 3)

    # Apply jet colourmap to heatmap
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heatmap)[:, :, :3]  # (224, 224, 3), float32

    # Blend
    overlay = (1 - alpha) * base_arr + alpha * heat_rgb
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(save_path)
        logger.info("Attention heatmap saved: %s", save_path)

    return overlay
