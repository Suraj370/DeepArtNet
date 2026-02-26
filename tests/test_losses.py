"""Unit tests for FocalLoss and MultiTaskLoss."""

from __future__ import annotations

import torch
import pytest

from src.training.losses import FocalLoss, MultiTaskLoss


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------

class TestFocalLoss:
    @pytest.fixture
    def criterion(self) -> FocalLoss:
        return FocalLoss(gamma=2.0, label_smoothing=0.1, ignore_index=-1)

    def test_returns_scalar(self, criterion):
        logits = torch.randn(8, 27)
        labels = torch.randint(0, 27, (8,))
        loss = criterion(logits, labels)
        assert loss.shape == ()

    def test_loss_is_positive(self, criterion):
        logits = torch.randn(8, 27)
        labels = torch.randint(0, 27, (8,))
        loss = criterion(logits, labels)
        assert loss.item() > 0.0

    def test_ignore_index_masks_samples(self, criterion):
        """Loss should be lower when half the labels are masked out."""
        logits = torch.randn(8, 27)
        full_labels = torch.randint(0, 27, (8,))
        masked_labels = full_labels.clone()
        masked_labels[:4] = -1  # mask first half

        loss_full = criterion(logits, full_labels)
        loss_masked = criterion(logits, masked_labels)
        # They should differ (masked has fewer samples)
        assert not torch.isclose(loss_full, loss_masked)

    def test_all_ignored_returns_zero(self, criterion):
        logits = torch.randn(4, 27)
        labels = torch.full((4,), -1, dtype=torch.long)
        loss = criterion(logits, labels)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_gradient_flows(self, criterion):
        logits = torch.randn(4, 27, requires_grad=True)
        labels = torch.randint(0, 27, (4,))
        loss = criterion(logits, labels)
        loss.backward()
        assert logits.grad is not None


# ---------------------------------------------------------------------------
# MultiTaskLoss
# ---------------------------------------------------------------------------

class TestMultiTaskLoss:
    @pytest.fixture
    def criterion(self) -> MultiTaskLoss:
        return MultiTaskLoss(tasks=["style", "genre", "artist"])

    def _make_batch(self, batch_size: int = 4):
        logits = {
            "style":  torch.randn(batch_size, 27),
            "genre":  torch.randn(batch_size, 10),
            "artist": torch.randn(batch_size, 23),
        }
        labels = {
            "style":  torch.randint(0, 27, (batch_size,)),
            "genre":  torch.randint(0, 10, (batch_size,)),
            "artist": torch.randint(0, 23, (batch_size,)),
        }
        return logits, labels

    def test_returns_scalar_total(self, criterion):
        logits, labels = self._make_batch()
        total, _ = criterion(logits, labels)
        assert total.shape == ()

    def test_returns_per_task_dict(self, criterion):
        logits, labels = self._make_batch()
        _, per_task = criterion(logits, labels)
        assert set(per_task.keys()) == {"style", "genre", "artist"}

    def test_per_task_losses_are_scalars(self, criterion):
        logits, labels = self._make_batch()
        _, per_task = criterion(logits, labels)
        for loss in per_task.values():
            assert loss.shape == ()

    def test_total_loss_is_positive(self, criterion):
        logits, labels = self._make_batch()
        total, _ = criterion(logits, labels)
        assert total.item() > 0.0

    def test_log_sigma2_requires_grad(self, criterion):
        assert criterion.log_sigma2.requires_grad

    def test_log_sigma2_count(self, criterion):
        assert criterion.log_sigma2.shape == (3,)

    def test_gradient_flows_through_total(self, criterion):
        logits = {
            "style":  torch.randn(4, 27, requires_grad=True),
            "genre":  torch.randn(4, 10, requires_grad=True),
            "artist": torch.randn(4, 23, requires_grad=True),
        }
        labels = {
            "style":  torch.randint(0, 27, (4,)),
            "genre":  torch.randint(0, 10, (4,)),
            "artist": torch.randint(0, 23, (4,)),
        }
        total, _ = criterion(logits, labels)
        total.backward()
        for name, logit in logits.items():
            assert logit.grad is not None, f"No gradient for {name}"

    def test_get_task_weights_returns_dict(self, criterion):
        weights = criterion.get_task_weights()
        assert set(weights.keys()) == {"style", "genre", "artist"}
        for w in weights.values():
            assert isinstance(w, float)
            assert w > 0.0
