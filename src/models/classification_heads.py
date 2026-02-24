"""Multi-task classification heads for DeepArtNet.

Each attribute (style, genre, artist) gets its own two-layer MLP head that
maps the 512-dim context vector from AdditiveAttention to class logits.
MultiTaskHeads composes all three heads and returns a dict of logits so the
loss function and evaluator can address each task by name.

Output shape contract:
    Input : (B, 512)   — context vector from AdditiveAttention
    Output: {
        "style":  (B, 27),
        "genre":  (B, 10),
        "artist": (B, 23),
    }
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Two-layer MLP classification head for a single attribute.

    Architecture::

        Linear(in_features, hidden_size) → ReLU → Dropout(dropout) → Linear(hidden_size, num_classes)

    Args:
        in_features: Dimensionality of the input context vector (default 512).
        hidden_size: Width of the intermediate layer (default 256).
        num_classes: Number of output classes.
        dropout: Dropout probability applied after ReLU (default 0.4).

    Example::

        head = ClassificationHead(in_features=512, hidden_size=256, num_classes=27)
        context = torch.randn(4, 512)
        logits = head(context)   # (4, 27)
    """

    def __init__(
        self,
        in_features: int = 512,
        hidden_size: int = 256,
        num_classes: int = 1,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits from a context vector.

        Args:
            x: Context tensor of shape ``(B, in_features)``.

        Returns:
            Logit tensor of shape ``(B, num_classes)``.
        """
        return self.net(x)


class MultiTaskHeads(nn.Module):
    """Three classification heads — style, genre, artist — in one module.

    Wraps individual :class:`ClassificationHead` instances and dispatches
    the shared context vector to each, returning a dict of raw logits.

    Args:
        in_features: Dimensionality of the shared input context (default 512).
        hidden_size: Hidden layer width for every head (default 256).
        num_style_classes: Output classes for the style head (default 27).
        num_genre_classes: Output classes for the genre head (default 10).
        num_artist_classes: Output classes for the artist head (default 23).
        dropout: Dropout probability for every head (default 0.4).

    Example::

        heads = MultiTaskHeads()
        context = torch.randn(4, 512)
        logits = heads(context)
        # logits["style"]  : (4, 27)
        # logits["genre"]  : (4, 10)
        # logits["artist"] : (4, 23)
    """

    def __init__(
        self,
        in_features: int = 512,
        hidden_size: int = 256,
        num_style_classes: int = 27,
        num_genre_classes: int = 10,
        num_artist_classes: int = 23,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        self.style = ClassificationHead(in_features, hidden_size, num_style_classes, dropout)
        self.genre = ClassificationHead(in_features, hidden_size, num_genre_classes, dropout)
        self.artist = ClassificationHead(in_features, hidden_size, num_artist_classes, dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute logits for all three attributes.

        Args:
            x: Shared context tensor of shape ``(B, in_features)``.

        Returns:
            Dictionary with keys ``"style"``, ``"genre"``, and ``"artist"``,
            each mapping to a raw logit tensor of shape ``(B, num_classes)``.
        """
        return {
            "style": self.style(x),
            "genre": self.genre(x),
            "artist": self.artist(x),
        }
