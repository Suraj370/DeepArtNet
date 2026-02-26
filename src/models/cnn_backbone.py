"""CNN Backbone module for DeepArtNet.

Wraps a pretrained EfficientNet-B4 (via timm) as a feature extractor, then projects
the last feature-stage output from 1792 → 512 channels via a 1×1 conv + BN + ReLU.
Supports per-stage freezing so the trainer can progressively unfreeze the backbone.

Output shape contract:
    Input : (B, 3, 224, 224)
    Output: (B, 512, 7, 7)
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import timm


class CNNBackbone(nn.Module):
    """EfficientNet-B4 feature extractor with a 1×1 projection head.

    Uses ``timm.create_model`` in ``features_only=True`` mode to expose all
    five feature stages of EfficientNet-B4.  Only the last stage's output
    (channels=1792, spatial=7×7 for a 224×224 input) is used; it is then
    projected to 512 channels with ``Conv2d(1792, 512, kernel_size=1,
    bias=False) → BatchNorm2d(512) → ReLU``.

    Args:
        pretrained: Whether to load ImageNet-pretrained weights.
        freeze_blocks: Indices of EfficientNet feature blocks (0–4) to freeze.
            Gradients are disabled for all parameters within those blocks.
            Pass an empty list (default) to leave all blocks trainable.

    Example::

        backbone = CNNBackbone(pretrained=True, freeze_blocks=[0, 1, 2])
        x = torch.randn(2, 3, 224, 224)
        out = backbone(x)          # (2, 512, 7, 7)
    """

    # Channel counts per feature stage for EfficientNet-B4 (features_only=True)
    # Note: 1792 is the classifier head — not exposed by features_only.
    # The actual last feature stage (index 4) outputs 448 channels.
    _STAGE_CHANNELS: List[int] = [24, 32, 56, 160, 448]
    _LAST_STAGE_IDX: int = 4
    _LAST_STAGE_CHANNELS: int = 448
    _PROJECTION_CHANNELS: int = 512

    def __init__(
        self,
        pretrained: bool = True,
        freeze_blocks: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        if freeze_blocks is None:
            freeze_blocks = []

        # Load backbone in features-only mode; returns list of stage outputs
        self.backbone: nn.Module = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            features_only=True,
        )

        # 1×1 projection: 1792 → 512
        self.projection = nn.Sequential(
            nn.Conv2d(
                self._LAST_STAGE_CHANNELS,
                self._PROJECTION_CHANNELS,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(self._PROJECTION_CHANNELS),
            nn.ReLU(inplace=True),
        )

        # Apply initial freezing
        self._freeze_blocks(freeze_blocks)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and project features from the last EfficientNet-B4 stage.

        Args:
            x: Input image tensor of shape ``(B, 3, H, W)``.  For the spatial
                contract of ``(B, 512, 7, 7)`` to hold, ``H == W == 224``.

        Returns:
            Projected feature map of shape ``(B, 512, 7, 7)``.
        """
        # backbone returns a list of 5 stage feature maps; take the last one
        stage_features: List[torch.Tensor] = self.backbone(x)
        last_features: torch.Tensor = stage_features[self._LAST_STAGE_IDX]  # (B, 448, 7, 7)
        projected: torch.Tensor = self.projection(last_features)             # (B, 512, 7, 7)
        return projected

    def freeze_blocks(self, block_indices: List[int]) -> None:
        """Freeze the given feature-stage blocks (disable gradient computation).

        Args:
            block_indices: List of stage indices (0–4) to freeze.  Calling
                with an empty list is a no-op.
        """
        self._freeze_blocks(block_indices)

    def unfreeze_blocks(self, block_indices: List[int]) -> None:
        """Unfreeze the given feature-stage blocks (re-enable gradients).

        Args:
            block_indices: List of stage indices (0–4) to unfreeze.  Calling
                with an empty list is a no-op.
        """
        feature_info = self.backbone.feature_info  # metadata list, one entry per stage
        for idx in block_indices:
            if idx < 0 or idx >= len(feature_info):
                raise ValueError(
                    f"Block index {idx} is out of range. "
                    f"EfficientNet-B4 has {len(feature_info)} feature stages (0–{len(feature_info)-1})."
                )
            stage_name: str = feature_info[idx]["module"]
            stage_module: nn.Module = self._get_submodule_by_name(stage_name)
            for param in stage_module.parameters():
                param.requires_grad = True

    def freeze_all_backbone(self) -> None:
        """Freeze every parameter in the EfficientNet-B4 backbone (not the projection head)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_all_backbone(self) -> None:
        """Unfreeze every parameter in the EfficientNet-B4 backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Return all parameters that require gradient updates.

        Returns:
            Flat list of ``nn.Parameter`` objects with ``requires_grad=True``.
        """
        return [p for p in self.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _freeze_blocks(self, block_indices: List[int]) -> None:
        """Internal implementation for freezing backbone stages.

        Args:
            block_indices: Stage indices to freeze.
        """
        if not block_indices:
            return

        feature_info = self.backbone.feature_info  # timm FeatureInfo list
        for idx in block_indices:
            if idx < 0 or idx >= len(feature_info):
                raise ValueError(
                    f"Block index {idx} is out of range. "
                    f"EfficientNet-B4 has {len(feature_info)} feature stages (0–{len(feature_info)-1})."
                )
            stage_name: str = feature_info[idx]["module"]
            stage_module: nn.Module = self._get_submodule_by_name(stage_name)
            for param in stage_module.parameters():
                param.requires_grad = False

    def _get_submodule_by_name(self, dotted_name: str) -> nn.Module:
        """Traverse the backbone's module tree using a dot-separated name.

        Args:
            dotted_name: Dot-separated attribute path, e.g. ``"blocks.2"``.

        Returns:
            The ``nn.Module`` at that path within ``self.backbone``.

        Raises:
            AttributeError: If the path does not exist.
        """
        module: nn.Module = self.backbone
        for part in dotted_name.split("."):
            module = getattr(module, part)
        return module
