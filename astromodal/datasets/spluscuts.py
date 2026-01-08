"""
Straight S-PLUS cutouts dataset (no masking).

Loads multi-band cutouts from a Polars DataFrame and applies per-band,
per-object robust arcsinh normalization.

Returns:
- x_norm: (C, H, W) float32
- m_valid: (C, H, W) float32 (1=valid pixel, 0=invalid)
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional

import polars as pl
import torch
from torch.utils.data import Dataset


def to_image_torch(flat, img_size: Optional[int] = None) -> torch.Tensor:
    """
    Convert flattened array-like into a square image tensor.

    If img_size is provided, reshapes to (img_size, img_size).
    Otherwise infers a square side from length.
    """
    arr = torch.as_tensor(flat, dtype=torch.float32)

    if arr.ndim == 2:
        return arr

    if arr.ndim != 1:
        raise ValueError(f"Unexpected ndim={arr.ndim} for image data")

    n = arr.numel()
    if img_size is None:
        side = int(math.isqrt(n))
        if side * side != n:
            raise ValueError(f"Cannot reshape length {n} into a square image")
        return arr.view(side, side)

    if img_size * img_size != n:
        raise ValueError(
            f"Expected {img_size}x{img_size}={img_size*img_size} pixels, got {n}"
        )
    return arr.view(img_size, img_size)


def arcsinh_norm_per_band(
    x: torch.Tensor,
    m_valid: torch.Tensor,
    eps: float = 1e-6,
    min_valid: int = 10,
    mad_mult: float = 3.0,
) -> torch.Tensor:
    """
    Robust per-band arcsinh normalization using valid pixels only.

    For each band c:
      med = median(vals)
      mad = median(|vals - med|)
      scale = mad_mult * mad + eps
      x_norm = asinh((x - med) / scale)

    If too few valid pixels, falls back to using all pixels in the band.
    """
    C, H, W = x.shape
    x_out = torch.empty_like(x)

    for c in range(C):
        vals = x[c][m_valid[c] > 0.5]
        if vals.numel() < min_valid:
            vals = x[c].reshape(-1)

        if vals.numel() == 0:
            x_out[c] = x[c]
            continue

        med = vals.median()
        mad = (vals - med).abs().median()
        scale = mad_mult * mad + eps

        x_out[c] = torch.asinh((x[c] - med) / scale)

    return x_out


class SplusCutoutsDataset(Dataset):
    """
    Straight S-PLUS cutouts dataset (no object masking).

    Parameters
    ----------
    df : pl.DataFrame
        Must contain columns: splus_cut_{band} for each band.
    bands : list[str], optional
        If None, uses default 12 S-PLUS bands.
    img_size : int
        Expected cutout size (default 96).
    return_valid_mask : bool
        If True returns (x_norm, m_valid). Otherwise returns only x_norm.
    """

    DEFAULT_BANDS = [
        "F378", "F395", "F410", "F430",
        "F515", "F660", "F861",
        "U", "G", "R", "I", "Z",
    ]

    def __init__(
        self,
        df: pl.DataFrame,
        bands: Optional[List[str]] = None,
        img_size: int = 96,
        return_valid_mask: bool = True,
        eps: float = 1e-6,
        min_valid: int = 10,
        mad_mult: float = 3.0,
    ):
        self.df = df
        self.bands = bands if bands is not None else self.DEFAULT_BANDS
        self.img_size = img_size
        self.return_valid_mask = return_valid_mask

        self.eps = eps
        self.min_valid = min_valid
        self.mad_mult = mad_mult

        self._len = df.height

        # Optional sanity check (fast): ensure columns exist
        missing = [f"splus_cut_{b}" for b in self.bands if f"splus_cut_{b}" not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        imgs = []
        masks = []

        for band in self.bands:
            col = f"splus_cut_{band}"
            flat = self.df[col][idx]

            img = to_image_torch(flat, img_size=self.img_size)  # (H, W)

            # validity: finite AND non-zero (you can change if you want zeros to be valid)
            valid = torch.isfinite(img) & (img != 0.0)

            # clean NaN/Inf -> 0 (keeps shape stable)
            img = img.clone()
            img[~torch.isfinite(img)] = 0.0

            imgs.append(img)
            masks.append(valid.float())

        x = torch.stack(imgs, dim=0)          # (C, H, W)
        m_valid = torch.stack(masks, dim=0)   # (C, H, W)

        x_norm = arcsinh_norm_per_band(
            x,
            m_valid,
            eps=self.eps,
            min_valid=self.min_valid,
            mad_mult=self.mad_mult,
        )

        if self.return_valid_mask:
            return x_norm, m_valid
        return x_norm