"""Shared training utilities."""

from typing import Optional, Tuple

import torch
import torch.nn as nn


def extract_batch_data(
    batch, device: torch.device
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Extract data from batch, handling both tuple and tensor formats.

    Args:
        batch: Either a tensor or tuple of (flux, ivar, mask).
        device: Device to move tensors to.

    Returns:
        Tuple of (data, ivar, mask). ivar and mask may be None.
    """
    if isinstance(batch, (tuple, list)):
        data = batch[0].to(device, non_blocking=True)
        ivar = (
            batch[1].to(device, non_blocking=True) if len(batch) > 1 else None
        )
        mask = (
            batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
        )
    else:
        data = batch.to(device, non_blocking=True)
        ivar = None
        mask = None
    return data, ivar, mask


def vae_loss(
    reconstruction: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    reconstruction_loss_fn: str = "mse",
    beta: float = 1.0,
    ivar: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate VAE loss = reconstruction_loss + beta * KL_divergence.

    Args:
        reconstruction: Reconstructed images.
        x: Original images.
        mu: Mean of latent distribution.
        logvar: Log variance of latent distribution.
        reconstruction_loss_fn: Type of reconstruction loss
            ('mse' or 'masked_weighted_mse').
        beta: Weight for KL divergence term (beta-VAE).
        ivar: Inverse variance weights for each pixel (optional).
        mask: Boolean mask indicating valid pixels (optional).

    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_divergence).
    """
    if reconstruction_loss_fn == "mse":
        recon_loss = nn.functional.mse_loss(reconstruction, x, reduction="sum")
    elif reconstruction_loss_fn == "masked_weighted_mse":
        if ivar is None or mask is None:
            raise ValueError(
                "masked_weighted_mse requires both ivar and mask arguments"
            )
        squared_error = (reconstruction - x).pow(2)
        weighted_error = squared_error * ivar * mask.float()
        num_valid_pixels = mask.float().sum()
        recon_loss = weighted_error.sum() / num_valid_pixels.clamp(min=1.0)
        total_pixels = torch.tensor(
            reconstruction.numel(),
            dtype=torch.float32,
            device=reconstruction.device,
        )
        recon_loss = recon_loss * total_pixels
    else:
        raise ValueError(
            f"Unknown reconstruction loss: {reconstruction_loss_fn}"
        )

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + beta * kl_div
    return total_loss, recon_loss, kl_div
