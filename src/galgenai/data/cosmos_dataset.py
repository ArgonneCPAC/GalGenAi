from typing import Callable, Optional

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, random_split

from galgenai.data.hsc import HSCDataset, custom_collate_fn

# Import load_fits_dataset from the simulation package
# This function loads FITS datasets produced by galgenai-sims
from galgenai_sims import load_fits_dataset  # noqa: F401


def make_loaders(
    dataset_raw: Dataset,
    nx: int,
    batch_size: int,
    num_workers: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_seed: int = 42,
    image_norm_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    return_aux_data: bool = True,
    return_noiseless_flux: bool = False,
    condition_cols: Optional[list] = None,
    conditional_norm_fn: Optional[
        Callable[[torch.Tensor], torch.Tensor]
    ] = None,
    shuffle: bool = True,
    invert_mask: bool = False,
    augment_train: bool = False,
):
    """Build train/val/test DataLoaders from a raw dataset.

    Takes a raw dataset and splits it into train/val/test,
    wraps each split in HSCDataset, and creates dataloaders.
    Similar to get_dataset_and_loaders in hsc.py.

    Supports multiple data modes:
    1. VAE training: return_aux_data=True, no conditioning,
       shuffle=True. Returns (flux, ivar, mask) tuples for
       training.
    2. Latent precomputation: return_aux_data=False, with
       conditioning. Returns (flux, condition) tuples for
       encoding.
    3. CNF training on raw images: return_aux_data=False,
       with conditioning, shuffle=True. Returns (flux,
       condition) tuples for direct CNF training (rare).

    Parameters:
    -----------
    dataset_raw: Raw HuggingFace Dataset to be split
    nx: Side length of center-cropped output patch
    batch_size: Batch size for DataLoaders
    num_workers: Number of DataLoader worker processes
    train_ratio: Fraction of data for training split.
        Default 0.8.
    val_ratio: Fraction of data for validation split.
        Default 0.1.
    random_seed: Random seed for reproducible splits.
        Default 42.
    image_norm_fn: Optional image normalization function.
        Create externally using get_image_norm_fn()
        [see normalization.py] and pass here.
    return_aux_data: If True, return (flux, ivar, mask).
        If False with conditioning, return (flux,
        condition). Default True.
    return_noiseless_flux: If True, return noiseless flux.
        Dataset must have been loaded with load_noiseless=True.
        Default False.
    condition_cols: Optional list of column names for
        conditioning variables. If provided, enables
        conditioning mode.
    conditional_norm_fn: Optional function to normalize
        conditioning variables. Create externally using
        get_conditional_norm_fn() and pass here.
        Required if condition_cols is provided.
        [see normalization.py]
    shuffle: Whether to shuffle training data. Default
        True.
    invert_mask: If True, flip the per-pixel mask
        emitting it. Set this when the source survey writes
        ``1 = bad pixel`` rather than the ``1 = valid pixel`` convention
        the trainers assume. Default False.
    augment_train: If True, apply random rotations and
        flips to training data only. Validation and test
        data are never augmented. Default False.

    Returns:
    --------
    (train_loader, val_loader, test_loader)
        where test_loader is None if train_ratio + val_ratio == 1.0
    """
    # Validate conditioning parameters
    if condition_cols is not None and conditional_norm_fn is None:
        raise ValueError(
            "conditional_norm_fn must be provided when using condition_cols. "
            "Use get_conditional_norm_fn() to create it."
        )

    # Determine if pin_memory should be used (only supported on CUDA)
    use_pin_memory = torch.cuda.is_available()

    # Split raw dataset first
    # This is to apply different augmentation to train vs val/test
    test_ratio = 1.0 - train_ratio - val_ratio
    train_raw, val_raw, test_raw = random_split(
        dataset_raw,
        [train_ratio, val_ratio, test_ratio],
        generator=torch.Generator().manual_seed(random_seed),
    )

    # Create HSCDataset instances with different augmentation settings
    datasets = []
    for raw_ds, augment in [
        (train_raw, augment_train),
        (val_raw, False),
        (test_raw, False),
    ]:
        ds = HSCDataset(
            raw_ds,
            nx=nx,
            image_norm_fn=image_norm_fn,
            return_aux_data=return_aux_data,
            return_noiseless_flux=return_noiseless_flux,
            condition_cols=condition_cols or [],
            conditional_norm_fn=conditional_norm_fn,
            invert_mask=invert_mask,
            augment=augment,
        )
        datasets.append(ds)

    train_ds, val_ds, test_ds = datasets

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=True
        if num_workers > 0
        else False,  # Reuse worker processes across epochs
        prefetch_factor=num_workers * 4
        if num_workers > 0
        else None,  # Prefetch batches
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,  # Validation is never shuffled
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=num_workers * 4 if num_workers > 0 else None,
        collate_fn=custom_collate_fn,
    )

    # Create test loader if test set exists
    if test_ratio > 0:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,  # Test is never shuffled
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=num_workers * 4 if num_workers > 0 else None,
            collate_fn=custom_collate_fn,
        )
    else:
        test_loader = None

    return train_loader, val_loader, test_loader
