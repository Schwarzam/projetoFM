"""
Image dataset for S-PLUS cutouts with preprocessing.

This module provides a PyTorch Dataset for loading S-PLUS multi-band image cutouts
with AION-style preprocessing including elliptical masking and per-band normalization.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


# Cache for coordinate grids
_GRID_CACHE = {}


def get_grid(H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get cached coordinate grid for elliptical masking.

    Parameters
    ----------
    H : int
        Image height
    W : int
        Image width
    device : torch.device
        Device to create tensors on

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (yy, xx) coordinate grids, each of shape (H, W)
    """
    key = (H, W, device.type)
    if key not in _GRID_CACHE:
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        _GRID_CACHE[key] = (yy.float(), xx.float())
    return _GRID_CACHE[key]


def to_image_torch(flat) -> torch.Tensor:
    """
    Convert flattened array-like into a square image tensor.

    Parameters
    ----------
    flat : array-like
        Flattened pixel data (should be a perfect square length)

    Returns
    -------
    torch.Tensor
        Square image tensor of shape (H, W)

    Raises
    ------
    ValueError
        If the flattened array cannot be reshaped into a square
    """
    arr = torch.tensor(flat, dtype=torch.float32)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 1:
        n = arr.numel()
        side = int(math.isqrt(n))
        if side * side != n:
            raise ValueError(f"Cannot reshape length {n} into a square image")
        return arr.view(side, side)

    raise ValueError(f"Unexpected ndim={arr.ndim} for image data")


def elliptical_mask(
    H: int,
    W: int,
    x0: float,
    y0: float,
    a: float,
    b: float,
    theta: float,
    device: str = "cpu",
    expand_factor: float = 4.0,
) -> torch.Tensor:
    """
    Create binary elliptical mask.

    Parameters
    ----------
    H : int
        Image height
    W : int
        Image width
    x0 : float
        Ellipse center x-coordinate
    y0 : float
        Ellipse center y-coordinate
    a : float
        Semi-major axis length
    b : float
        Semi-minor axis length
    theta : float
        Rotation angle in radians
    device : str, default="cpu"
        Device to create tensor on
    expand_factor : float, default=4.0
        Factor to expand ellipse size (larger = more pixels included)

    Returns
    -------
    torch.Tensor
        Binary mask of shape (H, W), with 1.0 inside expanded ellipse, 0.0 outside
    """
    device = torch.device(device)
    yy, xx = get_grid(H, W, device=device)

    a_scaled = a * expand_factor
    b_scaled = b * expand_factor

    X = xx - x0
    Y = yy - y0

    ct = torch.cos(theta)
    st = torch.sin(theta)

    Xp = X * ct + Y * st
    Yp = -X * st + Y * ct

    mask = (Xp / a_scaled) ** 2 + (Yp / b_scaled) ** 2 <= 1.0
    return mask.float()


class SplusDataset(Dataset):
    """
    Dataset for loading S-PLUS image cutouts with preprocessing.

    This dataset implements AION-style preprocessing:
    - Multi-band image construction (C, H, W)
    - Valid pixel mask per band (NaN/zero handling)
    - Elliptical object mask based on detection parameters
    - Per-band robust arcsinh normalization using local background

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame containing S-PLUS cutout data and detection parameters.
        Required columns:
        - splus_cut_{band} for each band: flattened image cutouts
        - a_pixel_det: semi-major axis in pixels
        - b_pixel_det: semi-minor axis in pixels
        - theta_det: position angle in degrees
    bands : List[str], default=None
        List of S-PLUS band names to use. If None, uses default 12-band list.
    img_size : int, default=96
        Expected square image size

    Attributes
    ----------
    df : pl.DataFrame
        Input DataFrame
    bands : List[str]
        Band names
    img_size : int
        Image size

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.read_parquet("datacube_HYDRA-0011.parquet")
    >>> dataset = SplusDataset(df)
    >>> len(dataset)
    12345
    >>> x_norm, m_obj, m_valid = dataset[0]
    >>> x_norm.shape
    torch.Size([12, 96, 96])
    >>> m_obj.shape
    torch.Size([12, 96, 96])

    Notes
    -----
    The normalization is done per-band, per-object:
    1. Estimate background median and MAD from valid pixels
    2. Compute scale = MAD * 3.0 + eps
    3. Apply arcsinh((x - median) / scale)

    This provides robust normalization that handles varying background levels
    and flux ranges across bands and objects.
    """

    DEFAULT_BANDS = [
        "F378", "F395", "F410", "F430",
        "F515", "F660", "F861",
        "U", "G", "R", "I", "Z",
    ]

    def __init__(
        self,
        df: pl.DataFrame,
        bands: List[str] = None,
        img_size: int = 96,
    ):
        self.df = df
        self.bands = bands if bands is not None else self.DEFAULT_BANDS
        self.img_size = img_size
        self._len = df.height

    def __len__(self) -> int:
        """Return number of objects in dataset."""
        return self._len

    @staticmethod
    def arcsinh_norm_per_band(
        x: torch.Tensor,
        mask_valid: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Apply per-band robust arcsinh normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (C, H, W)
        mask_valid : torch.Tensor
            Valid pixel mask of shape (C, H, W), with 1.0 for valid pixels
        eps : float, default=1e-6
            Small constant for numerical stability

        Returns
        -------
        torch.Tensor
            Normalized image tensor of shape (C, H, W)

        Notes
        -----
        For each band:
        1. Extract valid pixel values
        2. Compute robust background estimate (median)
        3. Compute robust scale estimate (MAD * 3.0)
        4. Apply arcsinh transformation: arcsinh((x - median) / scale)
        """
        C, H, W = x.shape
        x_out = torch.zeros_like(x)

        for c in range(C):
            m = mask_valid[c] > 0.5
            vals = x[c][m]

            # Fallback if too few valid pixels
            if vals.numel() < 10:
                vals = x[c].reshape(-1)

            if vals.numel() == 0:
                x_out[c] = x[c]
                continue

            # Robust background and scale estimation
            med = vals.median()
            mad = (vals - med).abs().median()
            scale = mad * 3.0 + eps

            # Apply arcsinh normalization
            x_out[c] = torch.asinh((x[c] - med) / scale)

        return x_out

    def build_single(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build preprocessed image for a single object.

        Parameters
        ----------
        idx : int
            Object index in DataFrame

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - x_norm: Normalized image (C, H, W)
            - m_obj: Object mask (C, H, W) - ellipse intersection with valid pixels
            - m_valid: Basic validity mask (C, H, W) - finite, non-zero pixels
        """
        imgs = []
        masks_pix_binary = []

        # Build per-band images and basic validity masks
        for band in self.bands:
            col_name = f"splus_cut_{band}"
            flat = self.df[col_name][idx]
            img = to_image_torch(flat)  # (H, W)

            # Mark valid pixels (finite and non-zero)
            valid = torch.isfinite(img) & (img != 0.0)
            img_clean = img.clone()
            img_clean[~torch.isfinite(img_clean)] = 0.0

            imgs.append(img_clean)
            masks_pix_binary.append(valid.float())

        x = torch.stack(imgs, dim=0)  # (C, H, W)
        m_pix_basic = torch.stack(masks_pix_binary, dim=0)  # (C, H, W)

        C, H, W = x.shape
        device_t = torch.device("cpu")
        x = x.to(device_t)
        m_pix_basic = m_pix_basic.to(device_t)

        # Build elliptical object mask
        x0 = torch.tensor(self.img_size // 2, dtype=torch.float32, device=device_t)
        y0 = torch.tensor(self.img_size // 2, dtype=torch.float32, device=device_t)

        a = torch.tensor(float(self.df["a_pixel_det"][idx]), device=device_t)
        b = torch.tensor(float(self.df["b_pixel_det"][idx]), device=device_t)
        th = torch.tensor(float(self.df["theta_det"][idx]), device=device_t)
        theta = th * math.pi / 180.0

        obj_mask = elliptical_mask(H, W, x0, y0, a, b, theta, device=device_t)
        obj_mask_full = obj_mask.unsqueeze(0).expand(C, H, W)

        # Refined pixel mask: finite pixels inside ellipse
        m_pix = m_pix_basic * obj_mask_full

        # AION-like normalization (per band, per object)
        x_norm = self.arcsinh_norm_per_band(x, m_pix_basic)

        return x_norm, m_pix, m_pix_basic

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get preprocessed image for a single object.

        Parameters
        ----------
        idx : int
            Object index

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - x_norm: Normalized image (C, H, W)
            - m_obj: Object mask (C, H, W)
            - m_valid: Validity mask (C, H, W)
        """
        return self.build_single(idx)
