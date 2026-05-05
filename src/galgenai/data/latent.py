"""Latent representation dataset for CNF training."""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from galgenai.models import VAEEncoder


class LatentDataset(Dataset):
    """Dataset for latent representations that samples from the VAE posterior.

    Loads the posterior mean (mu) and log-variance (logvar) from a .pt cache file
    into memory and samples from N(mu, sigma^2) when accessed. This preserves the
    stochastic nature of the latent representation during CNF training.

    Parameters:
    -----------
    cache_path: Path to .pt cache file containing mu, logvar, and conditions tensors.

    """

    def __init__(self, cache_path: str | Path):
        self.cache_path = Path(cache_path)
        # Load cached latents into memory
        data = torch.load(self.cache_path, map_location="cpu", weights_only=True)
        self.mu = data["mu"]
        self.logvar = data["logvar"]
        self.conditions = data["conditions"]

        self.latent_dim = self.mu.shape[1]
        self.condition_dim = self.conditions.shape[1]

    def __len__(self):
        return len(self.mu)

    def __getitem__(self, idx):
        """Sample from the latent posterior N(mu, sigma^2)."""
        mu = self.mu[idx]
        logvar = self.logvar[idx]
        cond = self.conditions[idx]

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, cond


@torch.no_grad()
def precompute_latents(
    encoder: VAEEncoder,
    loader: DataLoader,
    device: torch.device,
    cache_path: str | Path,
    return_dataloader: bool = False,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
) -> LatentDataset | DataLoader:
    """Encode all images in ``loader`` and cache to .pt file.

    The LatentDataset samples from N(mu, sigma^2) each time it's accessed,
    preserving stochasticity.

    Always computes latents and writes to cache, overwriting any existing file.

    Parameters:
    -----------
    encoder: Frozen VAE encoder in eval mode.
    loader: DataLoader returning batches. Supported formats:
        - (flux, condition): 2-tuple from return_aux_data=False with conditioning
        - (flux, ivar, mask, condition): 4-tuple from return_aux_data=True with conditioning
    device: Device to run the encoder on.
    cache_path: Path to .pt cache file. Will be overwritten if it exists.
    return_dataloader: If True, return a DataLoader instead of LatentDataset.
    batch_size: Batch size for the returned DataLoader. If None, uses the batch size
        from the input loader. Only used when return_dataloader=True.
    shuffle: Whether to shuffle the DataLoader. Only used when return_dataloader=True.
        Default True.

    Returns:
    --------
    If return_dataloader=False: LatentDataset 
        where mu, logvar, and conditions each having shape (N, latent_dim).
    If return_dataloader=True: DataLoader 
        wrapping the LatentDataset.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print("Computing latents from encoder...")
    encoder.eval()

    all_mu: list[torch.Tensor] = []
    all_logvar: list[torch.Tensor] = []
    all_cond: list[torch.Tensor] = []

    for batch in loader:
        # Handle both 2-tuple and 4-tuple formats
        # flux is always first, condition is always last
        images = batch[0].to(device, non_blocking=True)
        cond = batch[-1]

        mu, logvar = encoder(images)
        # Keep on device; copy to CPU once after the loop. Avoids a
        # host-device sync on every batch. Latents are tiny
        # (latent_dim ~ 32), so accumulating on GPU is cheap.
        all_mu.append(mu)
        all_logvar.append(logvar)
        all_cond.append(cond)

    # Concatenate all batches and move to CPU for caching
    mu = torch.cat(all_mu, dim=0).cpu()
    logvar = torch.cat(all_logvar, dim=0).cpu()
    conditions = torch.cat(all_cond, dim=0)
    if conditions.is_cuda:
        conditions = conditions.cpu()

    print(f"Computed {len(mu)} latents")

    # Save to cache
    print(f"Saving to cache: {cache_path}")
    torch.save(
        {
            "mu": mu,
            "logvar": logvar,
            "conditions": conditions,
        },
        cache_path,
    )
    print(f"Cache saved successfully")

    # Create dataset from cache
    dataset = LatentDataset(cache_path=cache_path)

    if return_dataloader:
        if batch_size is None:
            batch_size = loader.batch_size
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
        )
    else:
        return dataset
