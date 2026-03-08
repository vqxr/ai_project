from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class MemmapDataset:
    path: Path
    dtype: np.dtype
    length: int

    @staticmethod
    def open(path: Path, meta_path: Path | None = None) -> MemmapDataset:
        dtype = None
        if meta_path is not None and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            dt = meta.get("dtype")
            if dt == "uint16":
                dtype = np.uint16
            elif dt == "uint32":
                dtype = np.uint32

        if dtype is None:
            dtype = np.uint16

        size = path.stat().st_size
        bytes_per = np.dtype(dtype).itemsize
        length = size // bytes_per
        return MemmapDataset(path=path, dtype=dtype, length=int(length))

    def memmap(self):
        return np.memmap(self.path, mode="r", dtype=self.dtype, shape=(self.length,))


def get_batch(mm: np.memmap, block_size: int, batch_size: int, device: torch.device):
    # Sample random contiguous sequences.
    max_start = mm.shape[0] - block_size - 1
    ix = np.random.randint(0, max_start, size=(batch_size,))
    x = np.stack([mm[i : i + block_size] for i in ix])
    y = np.stack([mm[i + 1 : i + 1 + block_size] for i in ix])
    x = torch.from_numpy(x.astype(np.int64, copy=False)).to(device)
    y = torch.from_numpy(y.astype(np.int64, copy=False)).to(device)
    return x, y
