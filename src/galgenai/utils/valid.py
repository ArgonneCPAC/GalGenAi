"""Validation utilities: measure galaxy properties from images.

These helpers turn a batch of real (test) images and a corresponding
batch of generated (predicted) images into per-galaxy properties --
aperture photometry (flux, AB magnitude) and morphology (axis
ratio, size) -- so generative-model fidelity can be assessed downstream
(plotting, metrics) without this module knowing about either.

Photometry follows the approach validated in the CFM notebooks:
- morphology is measured by connected-component segmentation on the
  band-summed *normalized* reference (well-behaved noise; the arcsinh
  de-normalization exponentially amplifies faint noise and would wreck
  the image moments);
- the resulting elliptical aperture is integrated on the de-normalized
  physical-flux bands.

The estimator is applied identically to test and predicted images, so
comparisons isolate model fidelity from estimator bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from photutils.aperture import EllipticalAperture
from photutils.segmentation import SourceCatalog, detect_sources
from surveycodex import get_survey

from .units import counts_to_magnitude

# Default config values (overridable via kwargs).
DEFAULT_N_SIGMA = 3.0  # aperture size in units of the moment sigma
DEFAULT_THRESH_SIGMA = 1.5  # detection threshold above background, in sigma
DEFAULT_NPIX = 5  # min connected pixels for a detection
DEFAULT_SURVEY = "HSC"  # surveycodex survey for the photometric zeropoint
DEFAULT_BANDS = ("g", "r", "i", "z", "y")  # filter per channel, in order


@dataclass
class GalaxyProperties:
    """Per-galaxy properties for a batch of images.

    Attributes (N galaxies, C bands):
        flux:        (N, C) aperture flux per band (electron counts).
        mag:         (N, C) AB magnitude (survey-zeropoint calibrated).
        axis_ratio:  (N,)   b/a (semiminor/semimajor sigma), in (0, 1].
        size:        (N,)   semimajor sigma in px (Gaussian-equivalent).
        orientation: (N,)   semimajor-axis position angle in degrees,
                     CCW from the +x axis, in (-90, 90].

    Undetected sources have NaN in every field.
    """

    flux: np.ndarray
    mag: np.ndarray
    axis_ratio: np.ndarray
    size: np.ndarray
    orientation: np.ndarray

    @classmethod
    def concatenate(
        cls, items: Sequence["GalaxyProperties"]
    ) -> "GalaxyProperties":
        """Concatenate per-batch properties along the galaxy axis."""
        return cls(
            flux=np.concatenate([p.flux for p in items]),
            mag=np.concatenate([p.mag for p in items]),
            axis_ratio=np.concatenate([p.axis_ratio for p in items]),
            size=np.concatenate([p.size for p in items]),
            orientation=np.concatenate([p.orientation for p in items]),
        )


def _resolve_survey(survey):
    """Accept a surveycodex Survey, a survey-name string, or None."""
    if survey is None or isinstance(survey, str):
        return get_survey(survey_name=survey or DEFAULT_SURVEY)
    return survey


def fluxes_to_mags(flux, survey=None, bands=DEFAULT_BANDS):
    """Convert (N, C) aperture flux (counts) to (N, C) AB magnitude.

    Uses :func:`galgenai.utils.counts_to_magnitude`, which applies the
    survey/filter zeropoint (the analytical inverse of the
    ``mag2counts`` used to simulate the images), so magnitudes are on
    the catalog scale rather than zeropoint-less instrumental values.
    NaN where flux <= 0.

    ``survey`` may be a surveycodex Survey, a survey-name string, or
    None (defaults to ``DEFAULT_SURVEY``); ``bands`` is the filter name
    per channel, in order.
    """
    survey = _resolve_survey(survey)
    mag = np.full_like(flux, np.nan)
    for k, band in enumerate(bands):
        col = flux[:, k]
        good = col > 0
        mag[good, k] = counts_to_magnitude(col[good], survey, band)
    return mag


def measure_source(
    image: torch.Tensor,
    denorm_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    n_sigma: float = DEFAULT_N_SIGMA,
    thresh_sigma: float = DEFAULT_THRESH_SIGMA,
    npix: int = DEFAULT_NPIX,
) -> tuple[np.ndarray, float, float, float]:
    """Aperture photometry + morphology for one (C, H, W) stamp.

    Parameters:
    -----------
    image: (C, H, W) tensor of normalized image.
    denorm_fn: Function to denormalize the image from normalized to
        physical flux units. Should accept a torch.Tensor and return
        a torch.Tensor.

    Returns ``(flux, axis_ratio, size, orientation)``:
        flux: (C,) aperture flux per band,
        axis_ratio: b/a (semiminor/semimajor sigma), in (0, 1],
        size: semimajor sigma in px (Gaussian-equivalent),
        orientation: semimajor-axis position angle in degrees, CCW
            from the +x axis, in (-90, 90].

    Shape/centroid come from the source at the stamp centre (segmented
    on the band-summed normalized reference); that single elliptical
    aperture is integrated in every de-normalized band. Every returned
    value is NaN if nothing is detected.
    """
    norm = image.detach().cpu().numpy()  # (C, H, W) normalized
    phys = denorm_fn(image.detach().cpu()).numpy()
    ref = norm.sum(axis=0)  # (H, W) high-S/N reference
    h, w = ref.shape

    _, med, std = sigma_clipped_stats(ref)
    sub = ref - med
    segm = detect_sources(sub, threshold=thresh_sigma * std, n_pixels=npix)
    if segm is None:
        return np.full(phys.shape[0], np.nan), np.nan, np.nan, np.nan

    cat = SourceCatalog(sub, segm)
    lbl = segm.data[h // 2, w // 2]  # prefer the source at the stamp centre
    if lbl == 0:  # centre is background: fall back to the largest source
        idx = int(np.argmax(cat.area.value))
    else:
        idx = int(np.where(cat.labels == lbl)[0][0])
    src = cat[idx]

    smaj = src.semimajor_axis.value  # px (Gaussian-equiv), not * n_sigma
    smin = src.semiminor_axis.value
    q = smin / smaj  # axis ratio b/a, scale-free
    orient = src.orientation.to(u.deg).value  # PA in deg, in (-90, 90]

    x0, y0 = float(src.x_centroid), float(src.y_centroid)
    a = n_sigma * smaj
    b = n_sigma * smin
    theta = src.orientation.to(u.rad).value
    ap = EllipticalAperture((x0, y0), a, b, theta=theta)

    out = np.empty(phys.shape[0])
    for k, band in enumerate(phys):
        _, med_b, _ = sigma_clipped_stats(band)
        out[k] = ap.do_photometry(band - med_b)[0][0]  # bkg-subtracted sum
    return out, q, smaj, orient


def measure_batch(
    images: torch.Tensor,
    denorm_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    survey=None,
    bands=DEFAULT_BANDS,
    n_sigma: float = DEFAULT_N_SIGMA,
    thresh_sigma: float = DEFAULT_THRESH_SIGMA,
    npix: int = DEFAULT_NPIX,
) -> GalaxyProperties:
    """Measure :class:`GalaxyProperties` for a batch of stamps.

    Parameters:
    -----------
    images: (N, C, H, W) tensor (e.g. real test images or model samples)
        in normalized units.
    denorm_fn: Function to denormalize the images from normalized to
        physical flux units.
    survey: surveycodex Survey or survey name for magnitude calibration.
    bands: Filter names per channel.

    ``survey`` and ``bands`` set the zeropoint used to turn flux into AB
    magnitudes (see :func:`fluxes_to_mags`).
    """
    cfg = dict(n_sigma=n_sigma, thresh_sigma=thresh_sigma, npix=npix)
    fluxes, ratios, sizes, orients = [], [], [], []
    for img in images:
        flux, q, size, orient = measure_source(img, denorm_fn, **cfg)
        fluxes.append(flux)
        ratios.append(q)
        sizes.append(size)
        orients.append(orient)
    flux = np.array(fluxes)  # (N, C)
    return GalaxyProperties(
        flux=flux,
        mag=fluxes_to_mags(flux, survey=survey, bands=bands),
        axis_ratio=np.array(ratios),
        size=np.array(sizes),
        orientation=np.array(orients),
    )


def measure_galaxy_properties(
    test_images: torch.Tensor,
    pred_images: torch.Tensor,
    denorm_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    survey=None,
    bands=DEFAULT_BANDS,
    n_sigma: float = DEFAULT_N_SIGMA,
    thresh_sigma: float = DEFAULT_THRESH_SIGMA,
    npix: int = DEFAULT_NPIX,
) -> tuple[GalaxyProperties, GalaxyProperties]:
    """Measure properties for a test batch and its predictions.

    Parameters:
    -----------
    test_images: (N, C, H, W) tensor of test/real images in normalized
        units.
    pred_images: (N, C, H, W) tensor of predicted/generated images in
        normalized units.
    denorm_fn: Function to denormalize the images from normalized to
        physical flux units.
    survey: surveycodex Survey or survey name for magnitude calibration.
    bands: Filter names per channel.

    Returns ``(test_props, pred_props)``, each a
    :class:`GalaxyProperties`. The same estimator and config are applied
    to both so the two are directly comparable.
    """
    survey = _resolve_survey(survey)  # resolve once for both batches
    cfg = dict(
        survey=survey,
        bands=bands,
        n_sigma=n_sigma,
        thresh_sigma=thresh_sigma,
        npix=npix,
    )
    return (
        measure_batch(test_images, denorm_fn, **cfg),
        measure_batch(pred_images, denorm_fn, **cfg),
    )
