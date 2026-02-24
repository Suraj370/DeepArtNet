"""DeepArtNet full model assembly.

Composes all five pipeline stages into a single ``nn.Module``:

  Step 1  CNNBackbone       (B, 3, 224, 224) → (B, 512, 7, 7)
  Step 2  SpatialSequencer  (B, 512, 7, 7)   → (B, 49, 512)
  Step 3  BiLSTMEncoder     (B, 49, 512)     → (B, 49, 512)
  Step 4  AdditiveAttention (B, 49, 512)     → context (B, 512), weights (B, 49)
  Step 5  MultiTaskHeads    (B, 512)         → {style (B,27), genre (B,10), artist (B,23)}

Also provides:
  - ``predict(pil_image)``            → human-readable label + confidence per attribute
  - ``save_checkpoint(path, meta)``   → serialise weights + class names + metadata
  - ``load_from_checkpoint(path)``    → classmethod to reconstruct and return model
"""

from __future__ import annotations

import pathlib
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models.attention import AdditiveAttention
from src.models.bilstm_encoder import BiLSTMEncoder
from src.models.classification_heads import MultiTaskHeads
from src.models.cnn_backbone import CNNBackbone


# ---------------------------------------------------------------------------
# Step 2 helper — no learnable parameters
# ---------------------------------------------------------------------------

class SpatialSequencer(nn.Module):
    """Flatten spatial grid into a token sequence for the RNN.

    Reshapes the CNN feature map from ``(B, C, H, W)`` to ``(B, H*W, C)``
    so each spatial location becomes one token in a sequence.

    For the default backbone output ``(B, 512, 7, 7)`` this yields
    ``(B, 49, 512)`` — 49 spatial tokens of dimension 512.

    No learnable parameters; pure reshape + transpose.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape feature map to sequence.

        Args:
            x: Feature map of shape ``(B, C, H, W)``.

        Returns:
            Sequence tensor of shape ``(B, H*W, C)``.
        """
        B, C, H, W = x.shape
        # (B, C, H, W) → (B, C, H*W) → (B, H*W, C)
        return x.flatten(start_dim=2).transpose(1, 2)


# ---------------------------------------------------------------------------
# Preprocessing used by predict() — mirrors get_val_transforms()
# ---------------------------------------------------------------------------

_VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class DeepArtNet(nn.Module):
    """Hybrid CNN-RNN model for multi-attribute fine art classification.

    Composes :class:`CNNBackbone`, :class:`SpatialSequencer`,
    :class:`BiLSTMEncoder`, :class:`AdditiveAttention`, and
    :class:`MultiTaskHeads` into a single end-to-end module.

    Args:
        style_class_names: Ordered list of 27 style class names.
        genre_class_names: Ordered list of 10 genre class names.
        artist_class_names: Ordered list of 23 artist class names.
        pretrained_backbone: Whether to initialise EfficientNet-B4 with
            ImageNet weights (default ``True``).
        freeze_backbone_blocks: Backbone stage indices to freeze at init.

    Example::

        model = DeepArtNet(style_names, genre_names, artist_names)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        # out["style"]            : (2, 27)
        # out["genre"]            : (2, 10)
        # out["artist"]           : (2, 23)
        # out["attention_weights"]: (2, 49)
    """

    def __init__(
        self,
        style_class_names: list[str],
        genre_class_names: list[str],
        artist_class_names: list[str],
        pretrained_backbone: bool = True,
        freeze_backbone_blocks: Optional[list[int]] = None,
    ) -> None:
        super().__init__()

        self.style_class_names = style_class_names
        self.genre_class_names = genre_class_names
        self.artist_class_names = artist_class_names

        # Pipeline stages
        self.backbone = CNNBackbone(
            pretrained=pretrained_backbone,
            freeze_blocks=freeze_backbone_blocks or [],
        )
        self.sequencer = SpatialSequencer()
        self.encoder = BiLSTMEncoder(input_size=512, hidden_size=256, num_layers=2, dropout=0.3)
        self.attention = AdditiveAttention(dim=512)
        self.heads = MultiTaskHeads(
            in_features=512,
            hidden_size=256,
            num_style_classes=len(style_class_names),
            num_genre_classes=len(genre_class_names),
            num_artist_classes=len(artist_class_names),
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the full CNN-RNN pipeline.

        Args:
            x: Batch of images, shape ``(B, 3, 224, 224)``.

        Returns:
            Dictionary with keys:

            - ``"style"``             — logits ``(B, 27)``
            - ``"genre"``             — logits ``(B, 10)``
            - ``"artist"``            — logits ``(B, 23)``
            - ``"attention_weights"`` — softmax weights ``(B, 49)``
        """
        features = self.backbone(x)           # (B, 512, 7, 7)
        tokens = self.sequencer(features)     # (B, 49, 512)
        encoded = self.encoder(tokens)        # (B, 49, 512)
        context, weights = self.attention(encoded)  # (B, 512), (B, 49)
        logits = self.heads(context)          # {"style", "genre", "artist"}

        return {
            "style": logits["style"],
            "genre": logits["genre"],
            "artist": logits["artist"],
            "attention_weights": weights,
        }

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------

    def predict(self, pil_image: Image.Image) -> dict[str, dict[str, Any]]:
        """Classify a single PIL image and return human-readable results.

        Applies the same validation preprocessing as the training pipeline
        (Resize 256 → CenterCrop 224 → ImageNet normalisation), runs a
        forward pass in ``torch.no_grad()`` mode, and converts raw logits
        to probabilities.

        Args:
            pil_image: A ``PIL.Image.Image`` in RGB mode.  Non-RGB images
                are converted automatically.

        Returns:
            Nested dict keyed by attribute name, each containing:

            - ``"label"``      — predicted class name string
            - ``"confidence"`` — probability of the predicted class (0–1)

            Example::

                {
                    "style":  {"label": "Impressionism", "confidence": 0.87},
                    "genre":  {"label": "landscape",     "confidence": 0.91},
                    "artist": {"label": "Claude_Monet",  "confidence": 0.73},
                }
        """
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        device = next(self.parameters()).device
        tensor = _VAL_TRANSFORM(pil_image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

        self.eval()
        with torch.no_grad():
            output = self.forward(tensor)

        results: dict[str, dict[str, Any]] = {}
        for attr, class_names in (
            ("style", self.style_class_names),
            ("genre", self.genre_class_names),
            ("artist", self.artist_class_names),
        ):
            probs = F.softmax(output[attr][0], dim=0)  # (N,)
            idx = int(probs.argmax().item())
            results[attr] = {
                "label": class_names[idx],
                "confidence": round(float(probs[idx].item()), 4),
            }

        return results

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: str | pathlib.Path,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Serialise model weights, class names, and optional metadata.

        Args:
            path: Destination ``.pth`` file path.  Parent directories are
                created automatically.
            metadata: Arbitrary dict of extra information to embed in the
                checkpoint (e.g. epoch, validation accuracy, config).
                Defaults to an empty dict.

        Example::

            model.save_checkpoint(
                "outputs/checkpoints/best.pth",
                metadata={"epoch": 42, "val_acc_style": 0.76},
            )
        """
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "style_class_names": self.style_class_names,
            "genre_class_names": self.genre_class_names,
            "artist_class_names": self.artist_class_names,
            "metadata": metadata or {},
        }
        torch.save(payload, path)

    @classmethod
    def load_from_checkpoint(
        cls,
        path: str | pathlib.Path,
        map_location: Optional[str | torch.device] = None,
    ) -> "DeepArtNet":
        """Reconstruct a :class:`DeepArtNet` instance from a saved checkpoint.

        Args:
            path: Path to a ``.pth`` file created by :meth:`save_checkpoint`.
            map_location: Passed directly to :func:`torch.load`; use
                ``"cpu"`` to load a GPU checkpoint on a CPU-only machine.

        Returns:
            A :class:`DeepArtNet` instance with weights loaded and set to
            ``eval()`` mode.

        Example::

            model = DeepArtNet.load_from_checkpoint(
                "outputs/checkpoints/best.pth",
                map_location="cpu",
            )
        """
        path = pathlib.Path(path)
        payload: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)

        model = cls(
            style_class_names=payload["style_class_names"],
            genre_class_names=payload["genre_class_names"],
            artist_class_names=payload["artist_class_names"],
            pretrained_backbone=False,  # weights come from checkpoint
        )
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        return model
