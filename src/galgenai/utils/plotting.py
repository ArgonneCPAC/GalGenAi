"""Plotting helpers for training-time diagnostics."""

import math
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from matplotlib import patheffects as path_effects


def images_to_display(
    images: torch.Tensor,
    rgb_bands: Sequence[int] = (2, 1, 0),
    lo: Optional[np.ndarray] = None,
    hi: Optional[np.ndarray] = None,
    percentiles: Tuple[float, float] = (0.5, 99.5),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a batch of images to display arrays in ``[0, 1]``.

    Parameters:
    -----------
    images: Tensor of shape ``(N, C, H, W)``. Values are expected to be
        already stretched (e.g. the arcsinh + min-max normalized images
        the models are trained on), so no extra stretch is applied here.
    rgb_bands: Band indices mapped to (R, G, B) when ``C >= 3``. The
        default ``(2, 1, 0)`` maps i, r, g -> R, G, B for grizy data.
        With fewer than three channels the first band is replicated into
        a grayscale RGB image.
    lo, hi: Per-image black/white points of shape ``(N,)``. When not
        given they are the ``percentiles`` of each image, so every tile
        is scaled on its own. Passing them in applies a stretch borrowed
        from elsewhere, which is how the real and generated panels of a
        comparison grid are put on a common scale: a generated galaxy
        that is systematically too faint then renders too faint instead
        of being silently brightened to fill the range.

    Returns:
    --------
    ``(display, lo, hi)`` with shapes ``(N, H, W, 3)``, ``(N,)``,
    ``(N,)``.
    """
    arr = images.detach().float().cpu().numpy()
    n, c = arr.shape[0], arr.shape[1]

    if c >= 3:
        rgb = arr[:, list(rgb_bands)]
    else:
        rgb = np.repeat(arr[:, :1], 3, axis=1)

    if lo is None or hi is None:
        pct = np.stack(
            [np.percentile(rgb[i], list(percentiles)) for i in range(n)]
        )
        lo, hi = pct[:, 0], pct[:, 1]
    lo = np.asarray(lo, dtype=np.float64).reshape(n)
    hi = np.asarray(hi, dtype=np.float64).reshape(n)

    span = np.maximum(hi - lo, 1e-8).reshape(n, 1, 1, 1)
    out = np.clip((rgb - lo.reshape(n, 1, 1, 1)) / span, 0.0, 1.0)
    return np.transpose(out, (0, 2, 3, 1)), lo, hi


def _condition_text(values: np.ndarray) -> str:
    """Format one conditioning vector as a compact tile annotation.

    Values only, one per line, in the order of the conditioning vector:
    at tile size there is no room for the column names, which are named
    once in the figure footer instead.
    """
    return "\n".join(f"{v:.2f}" for v in values)


def _draw_panel(
    fig,
    gridspec,
    display: np.ndarray,
    label: str,
    annotations: Optional[List[str]],
):
    """Fill one gridspec with a grid of image tiles."""
    nrow, ncol = gridspec.get_geometry()
    for idx in range(nrow * ncol):
        ax = fig.add_subplot(gridspec[idx // ncol, idx % ncol])
        ax.set_xticks([])
        ax.set_yticks([])
        if idx >= display.shape[0]:
            ax.set_axis_off()
            continue
        ax.imshow(display[idx], origin="lower", interpolation="nearest")
        if annotations is not None:
            text = ax.text(
                0.04,
                0.96,
                annotations[idx],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6,
                color="white",
                linespacing=1.15,
            )
            text.set_path_effects(
                [path_effects.withStroke(linewidth=1.4, foreground="black")]
            )
    bbox = gridspec.get_grid_positions(fig)
    fig.text(
        0.5 * (bbox[2][0] + bbox[3][-1]),
        bbox[1][0] + 0.012,
        label,
        ha="center",
        va="bottom",
        fontsize=11,
    )


def plot_real_vs_generated(
    real: torch.Tensor,
    generated: torch.Tensor,
    output_path: str | Path,
    conditioning: Optional[torch.Tensor] = None,
    condition_labels: Optional[Sequence[str]] = None,
    condition_denorm_fn: Optional[Callable] = None,
    ncol: int = 4,
    rgb_bands: Sequence[int] = (2, 1, 0),
    title: Optional[str] = None,
) -> Path:
    """Save a real-vs-generated comparison grid.

    Tile ``i`` is the same object on both sides: the generated panel is
    conditioned on the conditioning vector of the real galaxy in the
    matching slot, so the two panels are read pairwise. Both panels
    share the real tile's black/white points (see
    ``images_to_display``), so a generated galaxy whose flux scale is
    off renders too faint or too bright instead of each tile
    self-normalizing and hiding it.

    Parameters:
    -----------
    real: Real images, shape ``(N, C, H, W)``.
    generated: Generated images conditioned on the same vectors, same
        shape as ``real``.
    output_path: Where to write the PNG.
    conditioning: Conditioning vectors of shape ``(N, D)``. When given,
        each tile of the real panel is annotated with its values, one
        per line in vector order.
    condition_labels: Names of the conditioning columns, listed in the
        figure footer to identify the annotated values.
    condition_denorm_fn: Optional map from normalized conditioning back
        to physical units, applied before annotating.
    ncol: Number of columns per panel.
    rgb_bands: Band indices mapped to (R, G, B); see
        ``images_to_display``.
    title: Optional figure title.
    """
    import matplotlib.pyplot as plt

    n = min(real.shape[0], generated.shape[0])
    real, generated = real[:n], generated[:n]

    real_display, lo, hi = images_to_display(real, rgb_bands=rgb_bands)
    gen_display, _, _ = images_to_display(
        generated, rgb_bands=rgb_bands, lo=lo, hi=hi
    )

    annotations = None
    if conditioning is not None:
        cond = conditioning[:n].detach().cpu()
        if condition_denorm_fn is not None:
            cond = condition_denorm_fn(cond)
        cond = cond.float().numpy()
        annotations = [_condition_text(cond[i]) for i in range(n)]

    nrow = math.ceil(n / ncol)
    footer = (
        ", ".join(condition_labels)
        if annotations is not None and condition_labels is not None
        else None
    )

    tile = 1.6
    header = 0.45 + (0.3 if title else 0.0)
    fig_h = nrow * tile + header + (0.3 if footer else 0.1)
    fig = plt.figure(figsize=(2 * ncol * tile + 0.4, fig_h), dpi=150)

    bottom = (0.3 if footer else 0.05) / fig_h
    top = 1.0 - header / fig_h
    gs_kwargs = dict(top=top, bottom=bottom, wspace=0.04, hspace=0.04)
    gap = 0.2 / (2 * ncol * tile + 0.4)
    gs_left = fig.add_gridspec(
        nrow, ncol, left=0.005, right=0.5 - gap, **gs_kwargs
    )
    gs_right = fig.add_gridspec(
        nrow, ncol, left=0.5 + gap, right=0.995, **gs_kwargs
    )
    _draw_panel(fig, gs_left, real_display, "real", annotations)
    _draw_panel(fig, gs_right, gen_display, "generated", None)

    if title:
        fig.suptitle(title, y=1.0 - 0.1 / fig_h, va="top")
    if footer:
        fig.text(
            0.5,
            0.06 / fig_h,
            f"annotations: {footer}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="0.3",
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
