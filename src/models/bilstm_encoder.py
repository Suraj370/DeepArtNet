"""Bidirectional LSTM encoder for DeepArtNet.

Consumes the token sequence produced by SpatialSequencer and enriches each
spatial token with global context by running a 2-layer BiLSTM over the 49
spatial positions.  A LayerNorm is applied to the concatenated forward +
backward hidden states before the output is passed to the attention module.

Output shape contract:
    Input : (B, 49, 512)   — sequence of spatial tokens from SpatialSequencer
    Output: (B, 49, 512)   — 256 forward + 256 backward hidden states, normalised
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class BiLSTMEncoder(nn.Module):
    """2-layer Bidirectional LSTM with post-LayerNorm.

    Each of the 49 spatial tokens (dim 512) is processed by a 2-layer
    BiLSTM with ``hidden_size=256``.  The concatenated forward and backward
    states (256 + 256 = 512) are layer-normalised and returned as the
    encoded sequence.

    Args:
        input_size: Feature dimension of each input token.  Must match the
            projection channels of :class:`~src.models.cnn_backbone.CNNBackbone`
            (default 512).
        hidden_size: Hidden units *per direction*.  The effective output
            dimension is ``2 * hidden_size`` (default 256 → 512 out).
        num_layers: Number of stacked LSTM layers (default 2).
        dropout: Dropout probability applied between LSTM layers.  Ignored
            when ``num_layers == 1`` (PyTorch behaviour).  Default 0.3.

    Example::

        encoder = BiLSTMEncoder()
        tokens = torch.randn(4, 49, 512)   # (B, seq_len, input_size)
        out = encoder(tokens)              # (4, 49, 512)
    """

    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # output_size = 2 * hidden_size (forward + backward)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Encode a spatial token sequence with a bidirectional LSTM.

        Args:
            x: Token sequence of shape ``(B, seq_len, input_size)``.
                Typically ``(B, 49, 512)`` from :class:`SpatialSequencer`.
            hx: Optional initial hidden/cell state tuple ``(h_0, c_0)``.
                When ``None`` (default) PyTorch initialises both to zeros.

        Returns:
            Normalised output tensor of shape ``(B, seq_len, 2 * hidden_size)``,
            i.e. ``(B, 49, 512)`` under the default configuration.
        """
        # lstm_out: (B, seq_len, 2 * hidden_size)
        lstm_out, _ = self.lstm(x, hx)
        return self.layer_norm(lstm_out)
