"""DeepArtNet full model assembly.

Builds the hybrid CNN-RNN pipeline stage by stage:
  Step 1  CNNBackbone      (B, 3, 224, 224) → (B, 512, 7, 7)
  Step 2  SpatialSequencer (B, 512, 7, 7)   → (B, 49, 512)
"""

from __future__ import annotations

import torch
import torch.nn as nn


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
