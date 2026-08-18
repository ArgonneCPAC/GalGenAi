"""Tests for the training diagnostic plots."""

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

from galgenai.training.base_trainer import BaseTrainer  # noqa: E402
from galgenai.utils.plotting import (  # noqa: E402
    images_to_display,
    plot_real_vs_generated,
)


def test_images_to_display_shape_and_range():
    images = torch.randn(4, 5, 8, 8)
    display, lo, hi = images_to_display(images)
    assert display.shape == (4, 8, 8, 3)
    assert lo.shape == (4,) and hi.shape == (4,)
    assert display.min() >= 0.0 and display.max() <= 1.0


def test_images_to_display_grayscale_for_few_channels():
    display, _, _ = images_to_display(torch.rand(2, 1, 8, 8))
    assert display.shape == (2, 8, 8, 3)
    np.testing.assert_allclose(display[..., 0], display[..., 2])


def test_images_to_display_shared_scale_keeps_faint_faint():
    real = torch.rand(3, 5, 8, 8)
    faint = real * 0.1
    _, lo, hi = images_to_display(real)
    shared, _, _ = images_to_display(faint, lo=lo, hi=hi)
    own, _, _ = images_to_display(faint)
    # Self-scaling brightens the faint images back up; borrowing the
    # real black/white points must not.
    assert shared.mean() < own.mean()


@pytest.mark.parametrize("n", [16, 7])
def test_plot_real_vs_generated_writes_png(tmp_path, n):
    out = plot_real_vs_generated(
        torch.rand(n, 5, 16, 16),
        torch.rand(n, 5, 16, 16),
        tmp_path / "grid.png",
        conditioning=torch.rand(n, 5),
        condition_labels=["g", "r", "i", "z", "y"],
        condition_denorm_fn=lambda c: c * 5.0 + 18.0,
        title="epoch 1",
    )
    assert out.exists() and out.stat().st_size > 0


def test_plot_real_vs_generated_single_channel(tmp_path):
    out = plot_real_vs_generated(
        torch.rand(4, 1, 16, 16),
        torch.rand(4, 1, 16, 16),
        tmp_path / "grid_1band.png",
        ncol=2,
    )
    assert out.exists()


class _HistoryOnlyTrainer(BaseTrainer):
    """Bypasses BaseTrainer.__init__ to test save_loss_plot alone."""

    def __init__(self, output_dir, loss_history):
        self.output_dir = output_dir
        self.loss_history = loss_history

    def _train_step(self, batch):  # pragma: no cover - unused
        raise NotImplementedError

    def _setup_optimizer(self):  # pragma: no cover - unused
        raise NotImplementedError

    def train(self):  # pragma: no cover - unused
        raise NotImplementedError


def test_save_loss_plot_with_lr(tmp_path):
    history = [
        {
            "step": i,
            "epoch": i,
            "loss": 1.0 / (i + 1),
            "val_loss": 1.2 / (i + 1),
            "lr": 1e-4 * (i + 1),
        }
        for i in range(5)
    ]
    trainer = _HistoryOnlyTrainer(tmp_path, history)
    out = trainer.save_loss_plot()
    assert out is not None and out.exists()


def test_save_loss_plot_without_lr(tmp_path):
    history = [
        {"step": i, "epoch": i, "loss": 1.0 / (i + 1)} for i in range(3)
    ]
    out = _HistoryOnlyTrainer(tmp_path, history).save_loss_plot()
    assert out is not None and out.exists()


def test_save_loss_plot_no_losses_returns_none(tmp_path):
    history = [{"step": 0, "epoch": 0, "lr": 1e-4}]
    assert _HistoryOnlyTrainer(tmp_path, history).save_loss_plot() is None
