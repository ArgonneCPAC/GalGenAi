"""In-memory copy of a HuggingFace galaxy dataset.

Row access on an Arrow-backed HuggingFace Dataset costs several ms per
sample (nested lists are decoded into tensors row by row), which makes
training data-bound for small images. ``InMemoryDataset`` decodes the
needed columns once, in bulk, into contiguous tensors, and then serves
rows with the same structure as the HuggingFace Dataset in torch
format, so it can be passed to ``HSCDataset`` unchanged.
"""

from typing import Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from datasets import Dataset


def _hf_torch_dtype(dtype: torch.dtype) -> torch.dtype:
    """Dtype HuggingFace's torch formatter would return for ``dtype``.

    The formatter casts integers to int64 and floats to float32;
    matching it keeps batches identical to the Arrow-backed path.
    """
    if dtype.is_floating_point:
        return torch.float32
    if dtype != torch.bool:
        return torch.int64
    return dtype


def _nested_list_to_tensor(column: pa.Array, n_rows: int) -> torch.Tensor:
    """Convert a fixed-shape nested-list Arrow column to a tensor.

    The column must hold arrays of identical shape in every row (e.g.
    ``(bands, H, W)`` images); the result has shape
    ``(n_rows, *shape)``.
    """
    if isinstance(column, pa.ExtensionArray):  # e.g. Array3D features
        column = column.storage

    first = column[0].as_py()
    shape = np.shape(first)
    values = column
    while (
        pa.types.is_list(values.type)
        or pa.types.is_large_list(values.type)
        or pa.types.is_fixed_size_list(values.type)
    ):
        values = values.flatten()

    expected = n_rows * int(np.prod(shape))
    if len(values) != expected:
        raise ValueError(
            f"Column does not have a fixed shape {shape} across rows "
            f"({len(values)} values, expected {expected})"
        )
    array = values.to_numpy(zero_copy_only=False).reshape(n_rows, *shape)
    return torch.from_numpy(np.ascontiguousarray(array))


class InMemoryDataset(torch.utils.data.Dataset):
    """Drop-in, in-memory replacement for a HuggingFace galaxy dataset.

    Rows are returned as ``{"image": {field: tensor, ...,
    "band": [...]}, col: scalar_tensor, ...}``, matching what
    ``HSCDataset`` reads from a torch-formatted HuggingFace Dataset.
    Only the requested image fields and scalar columns are loaded.
    Images are stored in their native dtype (e.g. an int32 mask stays
    int32) and cast per row on access, to keep memory down.

    Args:
        hf_dataset: HuggingFace Dataset with an ``image`` struct column.
        image_fields: Fields of ``image`` to load (fixed-shape arrays).
        columns: Top-level scalar columns to load (e.g. conditioning
            columns).
    """

    def __init__(
        self,
        hf_dataset: Dataset,
        image_fields: Sequence[str] = ("flux", "ivar", "mask"),
        columns: Optional[Sequence[str]] = None,
    ):
        columns = list(columns or [])
        # Arrow-formatted slice applies any indices mapping left by
        # select()/filter(), so rows match hf_dataset[i]
        table = hf_dataset.select_columns(["image", *columns]).with_format(
            "arrow"
        )[:]
        n_rows = table.num_rows

        image = table.column("image").combine_chunks()
        self.images = {
            field: _nested_list_to_tensor(
                pc.struct_field(image, field), n_rows
            )
            for field in image_fields
        }
        self.bands = pc.struct_field(image, "band").to_pylist()

        self.columns = {
            col: torch.from_numpy(table.column(col).to_numpy())
            for col in columns
        }
        self.out_dtypes = {
            name: _hf_torch_dtype(t.dtype)
            for name, t in [*self.images.items(), *self.columns.items()]
        }
        self.n_rows = n_rows

    def __len__(self):
        return self.n_rows

    def __getitem__(self, idx):
        dtypes = self.out_dtypes
        image = {
            field: t[idx].to(dtypes[field]) for field, t in self.images.items()
        }
        image["band"] = self.bands[idx]
        row = {"image": image}
        row.update(
            {col: t[idx].to(dtypes[col]) for col, t in self.columns.items()}
        )
        return row

    def nbytes(self) -> int:
        """Memory held by the loaded image and column tensors."""
        tensors = [*self.images.values(), *self.columns.values()]
        return sum(t.element_size() * t.nelement() for t in tensors)
