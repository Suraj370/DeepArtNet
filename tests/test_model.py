"""Unit tests for the full DeepArtNet forward pass and inference API.

All tests use random tensors / synthetic PIL images so no real image files
or downloaded weights are required.  The EfficientNet backbone is initialised
with ``pretrained=False`` to keep tests fast and offline-friendly.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from src.models.deepartnet import DeepArtNet

# Minimal class name lists matching the expected class counts
STYLE_NAMES = [f"style_{i}" for i in range(27)]
GENRE_NAMES = [f"genre_{i}" for i in range(10)]
ARTIST_NAMES = [f"artist_{i}" for i in range(23)]


@pytest.fixture(scope="module")
def model() -> DeepArtNet:
    """Shared model instance (no pretrained weights — fast to construct)."""
    return DeepArtNet(
        style_class_names=STYLE_NAMES,
        genre_class_names=GENRE_NAMES,
        artist_class_names=ARTIST_NAMES,
        pretrained_backbone=False,
    ).eval()


# ---------------------------------------------------------------------------
# Forward pass output shapes
# ---------------------------------------------------------------------------

class TestForwardShapes:
    def test_style_logits_shape(self, model):
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out["style"].shape == (2, 27)

    def test_genre_logits_shape(self, model):
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out["genre"].shape == (2, 10)

    def test_artist_logits_shape(self, model):
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out["artist"].shape == (2, 23)

    def test_attention_weights_shape(self, model):
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out["attention_weights"].shape == (2, 49)

    def test_output_keys(self, model):
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert set(out.keys()) == {"style", "genre", "artist", "attention_weights"}


# ---------------------------------------------------------------------------
# Attention weights sum to 1
# ---------------------------------------------------------------------------

class TestAttentionWeights:
    def test_weights_sum_to_one(self, model):
        x = torch.randn(4, 3, 224, 224)
        out = model(x)
        sums = out["attention_weights"].sum(dim=-1)  # (B,)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_weights_non_negative(self, model):
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert (out["attention_weights"] >= 0).all()


# ---------------------------------------------------------------------------
# predict() API
# ---------------------------------------------------------------------------

class TestPredict:
    @pytest.fixture
    def pil_image(self) -> Image.Image:
        return Image.new("RGB", (256, 256), color=(128, 64, 32))

    def test_predict_returns_dict(self, model, pil_image):
        result = model.predict(pil_image)
        assert isinstance(result, dict)

    def test_predict_keys(self, model, pil_image):
        result = model.predict(pil_image)
        assert set(result.keys()) == {"style", "genre", "artist"}

    def test_predict_inner_keys(self, model, pil_image):
        result = model.predict(pil_image)
        for attr_result in result.values():
            assert "label" in attr_result
            assert "confidence" in attr_result

    def test_predict_label_is_string(self, model, pil_image):
        result = model.predict(pil_image)
        for attr_result in result.values():
            assert isinstance(attr_result["label"], str)

    def test_predict_confidence_in_range(self, model, pil_image):
        result = model.predict(pil_image)
        for attr_result in result.values():
            assert 0.0 <= attr_result["confidence"] <= 1.0

    def test_predict_accepts_non_rgb(self, model):
        """Grayscale image should be converted and not raise."""
        grey = Image.new("L", (224, 224), color=128)
        result = model.predict(grey)
        assert set(result.keys()) == {"style", "genre", "artist"}
