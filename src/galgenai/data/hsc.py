import re
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from datasets import Dataset, DatasetInfo, concatenate_datasets

from .augmentation import random_rotation_and_flip


def custom_collate_fn(batch):
    """
    Custom collate function that handles None values in batches.

    HSCDataset returns 5-tuples: (flux, ivar, mask, noiseless, cond)
    where some values may be None. This function batches non-None values
    and keeps None values as None.

    Args:
        batch: List of 5-tuples from HSCDataset

    Returns:
        5-tuple of batched tensors or None
    """
    # Transpose batch (list of tuples -> tuple of lists)
    transposed = list(zip(*batch, strict=False))

    batched = []
    for samples in transposed:
        # Check if all samples are None
        if all(s is None for s in samples):
            batched.append(None)
        else:
            # Stack non-None tensors using default collate
            batched.append(torch.utils.data.default_collate(samples))

    return tuple(batched)


def load_hsc_mmu_dataset(
    data_dir,
    split: str = "train",
    format: Optional[str] = "torch",
    condition_cols: Optional[list] = None,
    filter_invalid_conditions: bool = True,
) -> Dataset:
    """Load an HSC MultiModalUniverse dataset saved by ``save_to_disk``.

    This is the HSC-MMU counterpart to
    ``cosmos_dataset.load_fits_dataset``: it returns a raw HuggingFace
    Dataset whose ``image`` column is the nested dict (``flux``,
    ``ivar``, ``mask``, ``band``, ...) that ``HSCDataset`` consumes,
    plus every catalog column. Split it with
    ``cosmos_dataset.make_loaders``.

    Unlike ``datasets.load_from_disk``, this loads whichever Arrow
    shards are actually present rather than failing on the first missing
    one. Partial copies of the survey are common (the full dataset is
    ~94 GB), so a truncated copy stays usable and transparently grows to
    the full dataset once the remaining shards are synced.

    Parameters:
    -----------
    data_dir : str or Path
        Dataset root. Either a ``DatasetDict`` layout (containing a
        ``<split>/`` subdirectory) or a bare ``Dataset`` directory
        holding the ``.arrow`` shards directly. Both are handled.
    split : str
        Split subdirectory to look for. Default "train".
    format : str or None
        Output format for arrays, as in ``Dataset.with_format``.
        Options: "torch" (default), "numpy", "tensorflow", or None
        (Python lists).
    condition_cols : list of str or None
        Conditioning column names. Only used to drop rows with invalid
        conditioning; pass None to skip that check entirely.
    filter_invalid_conditions : bool
        If True (default) and ``condition_cols`` is given, drop rows
        where any conditioning value is non-finite (NaN or inf). HSC MMU
        uses NaN rather than a sentinel for missing catalog entries.

    Returns:
    --------
    datasets.Dataset
        HuggingFace Dataset with PyTorch tensors
        (default format="torch").
    """
    data_dir = Path(data_dir).expanduser()
    split_dir = data_dir / split
    if not split_dir.is_dir():
        # Bare Dataset layout: shards live directly in data_dir.
        split_dir = data_dir

    shards = sorted(split_dir.glob("data-*.arrow"))
    if not shards:
        raise FileNotFoundError(
            f"No Arrow shards (data-*.arrow) found in {split_dir}. "
            "Expected a HuggingFace dataset saved with save_to_disk()."
        )

    # Shard names encode the expected total: data-00000-of-00091.arrow
    n_expected = None
    match = re.search(r"-of-(\d+)\.arrow$", shards[0].name)
    if match:
        n_expected = int(match.group(1))

    print(f"Loading HSC MMU dataset from: {split_dir}")
    if n_expected is not None and len(shards) < n_expected:
        print(
            f"  WARNING: only {len(shards)}/{n_expected} Arrow shards are "
            f"present. Loading the available subset; copy the missing "
            f"shards to train on the full dataset."
        )

    info = DatasetInfo.from_directory(str(split_dir))
    parts = [Dataset.from_file(str(p), info=info) for p in shards]
    dataset = parts[0] if len(parts) == 1 else concatenate_datasets(parts)

    n_loaded = len(dataset)
    msg = f"  Loaded {n_loaded:,} galaxies from {len(shards)} shard(s)"
    if info.splits is not None and split in info.splits:
        # Note: this is the count recorded at build time, which may be
        # stale for a partial copy. Reported for context only.
        msg += f" (dataset_info reports {info.splits[split].num_examples:,})"
    print(msg)

    if condition_cols and filter_invalid_conditions:
        # Project to the scalar conditioning columns before scanning, so
        # we never materialise the image arrays, then use select() to
        # keep the filtering lazy (an indices mapping, not a rewrite).
        cond_tbl = dataset.select_columns(list(condition_cols))
        cond_tbl = cond_tbl.with_format("numpy")[:]
        values = np.stack(
            [
                np.asarray(cond_tbl[c], dtype=np.float64)
                for c in condition_cols
            ],
            axis=1,
        )
        valid = np.isfinite(values).all(axis=1)
        n_invalid = int((~valid).sum())
        if n_invalid > 0:
            dataset = dataset.select(np.flatnonzero(valid).tolist())
            print(
                f"  Filtered {n_invalid} galaxies with non-finite "
                f"conditioning values. Remaining: {len(dataset):,}"
            )

    if format is not None:
        dataset = dataset.with_format(format)

    return dataset


class HSCDataset(torch.utils.data.Dataset):
    """Unified dataset for galaxy images.

    For HSC/COSMOS with optional conditioning.

    Always returns 5-tuple:
    (flux, ivar, mask, noiseless_flux, condition)
    where non-requested values are None:
    - return_aux_data=True: ivar and mask are tensors
    - return_aux_data=False: ivar and mask are None
    - return_noiseless_flux=True: noiseless_flux is tensor
    - return_noiseless_flux=False: noiseless_flux is None
    - condition_cols specified: condition is tensor
    - condition_cols not specified: condition is None

    Mask convention: ``1 = valid, 0 = invalid``.

    Mask convention:
    downstream losses (VAE/CFM weighted MSE) treat the
    emitted mask as ``1 = valid, 0 = invalid``.
    Set ``invert_mask=True`` when the source survey writes
    the opposite convention (``1 = bad pixel flag``).

    Args:
        hf_dataset: HuggingFace Dataset with 'image' column
        nx: Side length of center-cropped patch
        image_norm_fn: Optional normalization
        return_aux_data: Return auxiliary data
        return_noiseless_flux: Return noiseless flux if available
        condition_cols: Optional column names
        conditional_norm_fn: Optional function
        invert_mask: If True, flip mask
        augment: If True, apply random rotations and flips
    """

    def __init__(
        self,
        hf_dataset,
        nx: int,
        image_norm_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_aux_data: bool = True,
        return_noiseless_flux: bool = False,
        condition_cols: Optional[list] = None,
        conditional_norm_fn: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None,
        invert_mask: bool = False,
        augment: bool = False,
    ):
        self.dataset = hf_dataset
        self.nx = nx
        self.image_norm_fn = image_norm_fn
        self.return_aux_data = return_aux_data
        self.return_noiseless_flux = return_noiseless_flux
        self.condition_cols = condition_cols or []
        self.conditional_norm_fn = conditional_norm_fn
        self.invert_mask = invert_mask
        self.augment = augment

        # crop
        self.og_nx2 = self.dataset[0]["image"]["flux"].shape[1] // 2
        self.og_ny2 = self.dataset[0]["image"]["flux"].shape[2] // 2
        self.nx2 = nx // 2

        # bands
        self.bands = self.dataset[0]["image"]["band"]
        self.n_bands = self.dataset[0]["image"]["flux"].shape[0]

        # Check if noiseless data is available when requested
        if self.return_noiseless_flux:
            if "noiseless" not in self.dataset[0]["image"]:
                raise ValueError(
                    "return_noiseless_flux=True "
                    "but 'noiseless' field not found in dataset."
                )

    def __len__(self):
        return len(self.dataset)

    def normalize(self, img):
        if self.image_norm_fn is not None:
            return self.image_norm_fn(img)
        return img

    def crop(self, img):
        return img[
            :,
            self.og_nx2 - self.nx2 : self.og_nx2 + self.nx2,
            self.og_ny2 - self.nx2 : self.og_ny2 + self.nx2,
        ]

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image_data = sample["image"]

        # Extract and crop flux
        flux = self.crop(image_data["flux"])

        # Extract and crop noiseless flux if requested
        if self.return_noiseless_flux:
            noiseless_flux = self.crop(image_data["noiseless"])
        else:
            noiseless_flux = None

        # Process auxiliary data if requested
        if self.return_aux_data:
            # Extract and crop inverse variance
            ivar = self.crop(image_data["ivar"])

            # Cast unconditionally: HSC MMU stores the mask as bool, and
            # bool tensors support neither ``1 - mask`` nor the
            # arithmetic the weighted-MSE losses do downstream.
            mask = torch.as_tensor(image_data["mask"], dtype=torch.float32)
            mask = self.crop(mask)
            if self.invert_mask:
                mask = 1 - mask

            # Apply augmentation to flux, ivar, mask, and [noiseless]
            if self.augment:
                if self.return_noiseless_flux:
                    flux, ivar, mask, noiseless_flux = (
                        random_rotation_and_flip(
                            flux, ivar, mask, noiseless_flux
                        )
                    )
                else:
                    flux, ivar, mask = random_rotation_and_flip(
                        flux, ivar, mask
                    )
        else:
            # Apply augmentation to flux and optionally noiseless flux
            if self.augment:
                if self.return_noiseless_flux:
                    flux, noiseless_flux = random_rotation_and_flip(
                        flux, noiseless_flux
                    )
                else:
                    (flux,) = random_rotation_and_flip(flux)

        # Normalize flux after augmentation
        flux_normalized = self.normalize(flux)

        if self.return_noiseless_flux:
            noiseless_flux_normalized = self.normalize(noiseless_flux)
        else:
            noiseless_flux_normalized = None

        # Set ivar and mask to None if not requested
        if not self.return_aux_data:
            ivar = None
            mask = None

        # Get conditioning if requested
        cond = None
        if self.condition_cols:
            cond = torch.tensor(
                [float(sample[c]) for c in self.condition_cols],
                dtype=torch.float32,
            )
            # Normalize conditioning if function provided
            if self.conditional_norm_fn is not None:
                cond = self.conditional_norm_fn(cond)

        # Always return 5-tuple:
        # (flux, ivar, mask, noiseless_flux, condition)
        # Non-requested values are None
        return (flux_normalized, ivar, mask, noiseless_flux_normalized, cond)


def get_dataset_and_loaders(
    dataset_raw: Dataset,
    nx: int = 64,
    image_norm_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    split: float = 0.8,
    batch_size: int = 128,
    num_workers: int = 8,
    invert_mask: bool = False,
    augment: bool = False,
) -> Tuple[HSCDataset, DataLoader, DataLoader]:
    dataset_raw = dataset_raw.select_columns(["image"]).with_format("torch")

    n_gals = len(dataset_raw)

    dataset = HSCDataset(
        dataset_raw,
        nx=nx,
        image_norm_fn=image_norm_fn,
        invert_mask=invert_mask,
        augment=augment,
    )
    n_bands, n_x, n_y = dataset[0][0].shape  # First element of tuple is flux
    print(f"Images dimension: {n_bands}*{n_x}*{n_y} ({n_gals} galaxies)")

    dataset_train, dataset_test = random_split(dataset, [split, 1 - split])

    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    return dataset, train_loader, test_loader
