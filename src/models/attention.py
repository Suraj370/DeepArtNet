"""Additive (Bahdanau-style) attention module for DeepArtNet.

Computes a soft attention distribution over the 49 spatial tokens produced
by BiLSTMEncoder, then returns a single context vector as the weighted sum
of those tokens.  The attention weights are also returned so that callers
can overlay them as a heatmap on the original image.

Output shape contract:
    Input : (B, 49, 512)   — encoded token sequence from BiLSTMEncoder
    Output: context  (B, 512)   — attended summary vector
            weights  (B, 49)    — softmax distribution over spatial tokens
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention with a global learnable query.

    The mechanism computes a scalar alignment score for each of the
    ``seq_len`` (=49) tokens, then normalises them with softmax to produce
    attention weights.  A context vector is formed as the weighted sum of
    the input tokens.

    Scoring function (per token ``i``)::

        key_i   = W_k · h_i          # Linear(d, d, bias=False)
        score_i = v · tanh(q + key_i) # Linear(d, 1, bias=False)
        weight  = softmax(score)
        context = Σ weight_i · h_i

    where ``q`` is a global learnable query vector of shape ``(d,)``.

    Args:
        dim: Dimensionality of the token vectors and the attention space.
            Must match the output size of :class:`~src.models.bilstm_encoder.BiLSTMEncoder`
            (default 512).

    Example::

        attn = AdditiveAttention(dim=512)
        tokens = torch.randn(4, 49, 512)   # (B, seq_len, dim)
        context, weights = attn(tokens)
        # context : (4, 512)
        # weights : (4, 49)  — sums to 1.0 along dim=-1
    """

    def __init__(self, dim: int = 512) -> None:
        super().__init__()

        # Global learnable query vector (shared across the batch)
        self.query = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.query, mean=0.0, std=0.01)

        # Key projection: projects each token into attention space
        self.W_k = nn.Linear(dim, dim, bias=False)

        # Scoring layer: collapses the dim-dimensional vector to a scalar
        self.v = nn.Linear(dim, 1, bias=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute context vector and attention weights.

        Args:
            x: Encoded token sequence of shape ``(B, seq_len, dim)``.
                Typically ``(B, 49, 512)`` from :class:`BiLSTMEncoder`.

        Returns:
            A 2-tuple ``(context, weights)`` where:

            - **context** — ``(B, dim)`` weighted sum of input tokens.
            - **weights** — ``(B, seq_len)`` softmax attention distribution.
              Sums to ``1.0`` along ``dim=-1``.  Useful for heatmap
              visualisation over the 7×7 spatial grid.
        """
        # keys  : (B, seq_len, dim)
        keys = self.W_k(x)

        # query broadcast: (dim,) → (1, 1, dim)
        query = self.query.unsqueeze(0).unsqueeze(0)

        # scores: (B, seq_len, 1) → (B, seq_len)
        scores = self.v(torch.tanh(query + keys)).squeeze(-1)

        # weights: (B, seq_len)
        weights = F.softmax(scores, dim=-1)

        # context: (B, dim)  via batched matrix-vector multiply
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)

        return context, weights
