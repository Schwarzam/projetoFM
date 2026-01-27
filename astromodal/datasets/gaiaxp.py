from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from astromodal.scalers.scaler1d import StandardScaler1D
from pathlib import Path

def error_to_weights(err: np.ndarray, mask: np.ndarray, clip_p99: float = 0.99) -> np.ndarray:
    """
    Convert flux_error -> weights ~ 1/sigma^2, with clipping + normalization for stability.
    """
    w = np.zeros_like(err, dtype=np.float32)
    ok = mask & np.isfinite(err) & (err > 0)
    if not np.any(ok):
        return w

    w_ok = 1.0 / (err[ok] ** 2 + 1e-12)

    # clip extreme weights (equivalently clip tiny errors)
    hi = np.quantile(w_ok, clip_p99)
    if np.isfinite(hi) and hi > 0:
        w_ok = np.clip(w_ok, 0.0, hi)

    # normalize average weight ~ 1
    w_ok = w_ok / (np.mean(w_ok) + 1e-12)

    w[ok] = w_ok.astype(np.float32)
    return w


# -----------------------------
# Helpers to safely convert list-like cells
# -----------------------------
def _as_1d_float(x: Any) -> np.ndarray:
    if x is None:
        return np.empty(0, dtype=np.float64)
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 0:
        if np.isfinite(a.item()):
            return a.reshape(1)
        return np.empty(0, dtype=np.float64)
    return np.ravel(a)


# -----------------------------
# Dataset
# -----------------------------
class GaiaXPCalibratedDataset(Dataset):
    """
    Expects a Polars DF where each row has:
      - flux_col: list/array of calibrated GaiaXP flux samples
      - err_col : list/array of flux_error samples (same length)
    Returns per item:
      x    : float32 [L, 1]   (normalized flux)
      mask : bool    [L]      (valid)
      w    : float32 [L]      (weights from error; 0 for invalid)
      stats: FluxStats (optional if return_stats=True)
    """
    def __init__(
        self,
        df,
        scaler_path = None,
        flux_col: str = "xp_flux",
        err_col: str = "xp_flux_error",
        return_stats: bool = False,
    ):
        self.df = df
        self.flux_col = flux_col
        self.err_col = err_col
        self.return_stats = bool(return_stats)
        
        self.scaler = StandardScaler1D(mean=0.0, std=1.0)
        if scaler_path and Path(scaler_path).exists():
            self.scaler = StandardScaler1D.load(scaler_path)

    def __len__(self) -> int:
        return self.df.height

    def __getitem__(self, idx: int):
        row = self.df.row(idx, named=True)

        f = _as_1d_float(row.get(self.flux_col))
        e = _as_1d_float(row.get(self.err_col))

        L = min(f.size, e.size)

        f = f[:L].astype(np.float64, copy=False)
        e = e[:L].astype(np.float64, copy=False)

        mask = np.isfinite(f) & np.isfinite(e) & (e > 0)
        
        f_norm = self.scaler.transform_x(f)

        x = f_norm[:, None]  # [L,1]
        w = error_to_weights(e, mask)  # [L]

        return torch.from_numpy(x), torch.from_numpy(mask.astype(np.bool_)), torch.from_numpy(w)


# -----------------------------
# Collate: pad to max length
# -----------------------------
def gaiaxp_collate_pad_flux_only(batch, pad_value_x: float = 0.0):
    """
    batch items: (x[L,1], mask[L], w[L]) or (x, mask, w, stats)
    returns:
      x_pad    : [B, Lmax, 1] float32
      mask_pad : [B, Lmax]    bool
      w_pad    : [B, Lmax]    float32
      lengths  : [B]          int64
      stats    : optional list[FluxStats]
    """
    has_stats = len(batch[0]) == 4

    if has_stats:
        xs, ms, ws, stats = zip(*batch)
    else:
        xs, ms, ws = zip(*batch)
        stats = None

    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    Lmax = int(lengths.max().item())
    B = len(xs)

    x_pad = torch.full((B, Lmax, 1), float(pad_value_x), dtype=torch.float32)
    mask_pad = torch.zeros((B, Lmax), dtype=torch.bool)
    w_pad = torch.zeros((B, Lmax), dtype=torch.float32)

    for i, (x, m, w) in enumerate(zip(xs, ms, ws)):
        L = x.shape[0]
        x_pad[i, :L] = x
        mask_pad[i, :L] = m
        w_pad[i, :L] = w

    if has_stats:
        return x_pad, mask_pad, w_pad, lengths, stats
    return x_pad, mask_pad, w_pad, lengths


# -----------------------------
# Usage: build DataLoader
# -----------------------------
def make_gaiaxp_dataloader(
    df,
    flux_col: str,
    err_col: str,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    return_stats: bool = False,
):
    ds = GaiaXPCalibratedDataset(
        df,
        flux_col=flux_col,
        err_col=err_col,
        return_stats=return_stats,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=gaiaxp_collate_pad_flux_only,
    )
    return ds, dl