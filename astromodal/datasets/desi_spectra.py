from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import torch.nn.functional as F
from dataclasses import dataclass

from typing import Optional, Any

@dataclass
class NormStats:
    mu: float
    sigma: float

def robust_flux_stats(flux: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> NormStats:
    """
    Robust per-spectrum stats using median + MAD.
    flux: [L]
    mask: [L] bool (True valid pixel)
    """
    v = flux[mask] if mask is not None else flux
    v = v[np.isfinite(v)]
    if v.size == 0:
        return NormStats(mu=0.0, sigma=1.0)

    mu = float(np.median(v))
    mad = float(np.median(np.abs(v - mu)))
    sigma = 1.4826 * mad  # MAD -> ~sigma for normal
    if not np.isfinite(sigma) or sigma < eps:
        # fallback to std
        sigma2 = float(np.std(v))
        sigma = sigma2 if (np.isfinite(sigma2) and sigma2 >= eps) else 1.0

    return NormStats(mu=mu, sigma=sigma)

def normalize_flux(flux: np.ndarray, stats: NormStats) -> np.ndarray:
    return (flux - stats.mu) / (stats.sigma + 1e-12)

def denormalize_flux(flux_norm: np.ndarray, stats: NormStats) -> np.ndarray:
    return flux_norm * stats.sigma + stats.mu

def ivar_to_weights_for_normalized_flux(
    ivar: np.ndarray,
    stats: NormStats,
    mask: np.ndarray,
    *,
    use_sqrt: bool = True,
    w_clip_quantile: float = 0.995,
    eps: float = 1e-12
) -> np.ndarray:
    """
    If f' = (f - mu)/sigma, then ivar' = ivar * sigma^2.

    We then optionally use sqrt(ivar') to reduce domination by very high ivar,
    and clip weights at a high quantile among valid pixels.
    """
    if ivar.size == 0:
        return np.zeros_like(ivar, dtype=np.float32)

    iv = np.asarray(ivar, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)

    # scale to match normalized flux units
    iv_scaled = iv * (stats.sigma ** 2)

    # valid weights: finite and >0 and valid pixel
    w = np.zeros_like(iv_scaled, dtype=np.float64)
    good = m & np.isfinite(iv_scaled) & (iv_scaled > 0)
    if np.any(good):
        w_good = iv_scaled[good]

        if use_sqrt:
            w_good = np.sqrt(w_good)

        # clip extreme weights (optional but recommended)
        if 0.0 < w_clip_quantile < 1.0:
            cap = np.quantile(w_good, w_clip_quantile)
            if np.isfinite(cap) and cap > eps:
                w_good = np.minimum(w_good, cap)

        w[good] = w_good

    return w.astype(np.float32)


# -------------------------
# Stitching (your code, unchanged)
# -------------------------

def _as_1d_float(x) -> np.ndarray:
    if x is None:
        return np.empty(0, dtype=np.float64)
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 0:
        if np.isfinite(a.item()):
            return a.reshape(1)
        return np.empty(0, dtype=np.float64)
    return np.ravel(a)

def _trim_to_min_len(w, f, iv):
    n = min(w.size, f.size, iv.size)
    if n == 0:
        return (np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64))
    return (w[:n], f[:n], iv[:n])

def stitch_desi_arms(
    wave_b, flux_b, ivar_b,
    wave_r, flux_r, ivar_r,
    wave_z, flux_z, ivar_z,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns: wave, flux, ivar, mask
      mask True means valid pixel (ivar > 0 and finite) after stitching/combining.
    """

    wb = _as_1d_float(wave_b); fb = _as_1d_float(flux_b); ivb = _as_1d_float(ivar_b)
    wb, fb, ivb = _trim_to_min_len(wb, fb, ivb)

    wr = _as_1d_float(wave_r); fr = _as_1d_float(flux_r); ivr = _as_1d_float(ivar_r)
    wr, fr, ivr = _trim_to_min_len(wr, fr, ivr)

    wz = _as_1d_float(wave_z); fz = _as_1d_float(flux_z); ivz = _as_1d_float(ivar_z)
    wz, fz, ivz = _trim_to_min_len(wz, fz, ivz)

    w = np.concatenate([wb, wr, wz]) if (wb.size + wr.size + wz.size) > 0 else np.empty(0, dtype=np.float64)
    f = np.concatenate([fb, fr, fz]) if (fb.size + fr.size + fz.size) > 0 else np.empty(0, dtype=np.float64)
    iv = np.concatenate([ivb, ivr, ivz]) if (ivb.size + ivr.size + ivz.size) > 0 else np.empty(0, dtype=np.float64)

    if w.size == 0:
        return w, f, iv, np.empty(0, dtype=bool)

    m = np.isfinite(w) & np.isfinite(f) & np.isfinite(iv) & (iv >= 0)
    w, f, iv = w[m], f[m], iv[m]
    if w.size == 0:
        return w, f, iv, np.empty(0, dtype=bool)

    order = np.argsort(w)
    w, f, iv = w[order], f[order], iv[order]

    w_unique, inv, counts = np.unique(w, return_inverse=True, return_counts=True)

    if np.any(counts > 1):
        num = np.zeros_like(w_unique, dtype=np.float64)
        den = np.zeros_like(w_unique, dtype=np.float64)
        np.add.at(num, inv, iv * f)
        np.add.at(den, inv, iv)

        flux_out = np.zeros_like(w_unique, dtype=np.float64)
        mask_out = den > 0
        flux_out[mask_out] = num[mask_out] / den[mask_out]

        bad = ~mask_out
        if np.any(bad):
            sum_f = np.zeros_like(w_unique, dtype=np.float64)
            np.add.at(sum_f, inv, f)
            flux_out[bad] = sum_f[bad] / counts[bad]

        ivar_out = den
        return w_unique, flux_out, ivar_out, mask_out

    mask = iv > 0
    return w, f, iv, mask


# -------------------------
# Dataset: returns flux only + mask + weights (from ivar)
# -------------------------

class DesiStitchedFluxOnlyDataset(torch.utils.data.Dataset):
    """
    Returns:
      x    : float32 [L, 1]  (normalized flux)
      mask : bool    [L]     (valid pixel)
      w    : float32 [L]     (loss weights; derived from ivar; zero for invalid)
      stats: float32 [2]     (mu, sigma) optional, useful for denorm/debug
    """

    def __init__(
        self,
        df,
        *,
        return_stats: bool = True,
        use_sqrt_weights: bool = True,
        w_clip_quantile: float = 0.995,
    ):
        self.df = df
        self.return_stats = return_stats
        self.use_sqrt_weights = use_sqrt_weights
        self.w_clip_quantile = w_clip_quantile

    def __len__(self):
        return self.df.height

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)

        wv, f, iv, m = stitch_desi_arms(
            row.get("desi_wave_b"), row.get("desi_flux_b"), row.get("desi_ivar_b"),
            row.get("desi_wave_r"), row.get("desi_flux_r"), row.get("desi_ivar_r"),
            row.get("desi_wave_z"), row.get("desi_flux_z"), row.get("desi_ivar_z"),
        )

        # If empty: return dummy length-1, fully invalid
        if f.size == 0:
            x = np.zeros((1, 1), dtype=np.float32)
            mask = np.zeros((1,), dtype=np.bool_)
            w = np.zeros((1,), dtype=np.float32)
            if self.return_stats:
                stats = np.array([0.0, 1.0], dtype=np.float32)
                return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(w), torch.from_numpy(stats)
            return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(w)

        mask = m.astype(bool)

        # robust per-spectrum normalization on VALID pixels
        stats = robust_flux_stats(f, mask)
        f_norm = normalize_flux(f, stats).astype(np.float32)

        # weights from ivar scaled into normalized units
        w_loss = ivar_to_weights_for_normalized_flux(
            iv,
            stats,
            mask,
            use_sqrt=self.use_sqrt_weights,
            w_clip_quantile=self.w_clip_quantile,
        )  # [L] float32, already zeroed where invalid

        x = f_norm[:, None]  # [L,1]
        if self.return_stats:
            stats_arr = np.array([stats.mu, stats.sigma], dtype=np.float32)
            return (torch.from_numpy(x),
                    torch.from_numpy(mask),
                    torch.from_numpy(w_loss),
                    torch.from_numpy(stats_arr))
        return (torch.from_numpy(x),
                torch.from_numpy(mask),
                torch.from_numpy(w_loss))


def desi_collate_pad_flux_only(batch, pad_x: float = 0.0):
    """
    batch entries:
      either (x[L,1], mask[L], w[L]) or (x[L,1], mask[L], w[L], stats[2])

    returns:
      x_pad    : [B, Lmax, 1] float32
      mask_pad : [B, Lmax]    bool
      w_pad    : [B, Lmax]    float32
      lengths  : [B]          int64
      stats    : [B, 2]       float32 (optional)
    """
    has_stats = (len(batch[0]) == 4)

    if has_stats:
        xs, ms, ws, st = zip(*batch)
        stats = torch.stack(st, dim=0).to(torch.float32)  # [B,2]
    else:
        xs, ms, ws = zip(*batch)
        stats = None

    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    Lmax = int(lengths.max().item())
    B = len(xs)

    x_pad = torch.full((B, Lmax, 1), float(pad_x), dtype=torch.float32)
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

class LatentsPolarsDataset(Dataset):
    """
    df columns:
      - id_col: object id
      - latent_col: list-like latent stored as:
          (a) flat:  [latent_dim*T]
          (b) nested: [latent_dim][T]
          (c) nested: [T][latent_dim]

    returns:
      x    : float32 [T, C]   (sequence length T, channels C=latent_dim)
      id   : python object (string/int)
    """
    def __init__(
        self,
        df,
        *,
        id_col: str = "id",
        latent_col: str = "latent",
        latent_dim: Optional[int] = 16,  # needed for flat case
        nested_layout: str = "auto",      # "auto" | "C_T" | "T_C"
    ):
        self.df = df
        self.id_col = id_col
        self.latent_col = latent_col
        self.latent_dim = latent_dim
        self.nested_layout = nested_layout

    def __len__(self) -> int:
        return self.df.height

    def _to_TC(self, latent: Any) -> np.ndarray:
        """
        Convert latent to ndarray [T, C].
        """
        arr = np.asarray(latent, dtype=np.float32)

        # --- flat case: [C*T]
        if arr.ndim == 1:
            if self.latent_dim is None:
                raise ValueError("latent_dim must be provided when latent is stored flat.")
            C = int(self.latent_dim)
            if arr.size % C != 0:
                raise ValueError(f"Flat latent length {arr.size} not divisible by latent_dim={C}.")
            T = arr.size // C
            return arr.reshape(C, T).T  # [T,C]

        # --- nested case
        if arr.ndim == 2:
            a0, a1 = arr.shape

            if self.nested_layout == "C_T":
                return arr.T  # [T,C]
            if self.nested_layout == "T_C":
                return arr    # [T,C]

            # auto: decide by latent_dim if available, else fallback heuristics
            if self.latent_dim is not None:
                C = int(self.latent_dim)
                if a0 == C:
                    return arr.T
                if a1 == C:
                    return arr

            # heuristic: treat smaller dimension as C if it looks like channels
            # (common: C=8/16/32/64)
            if a0 in (8, 16, 32, 64):
                return arr.T
            return arr

        raise ValueError(f"Unsupported latent array ndim={arr.ndim}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        row = self.df.row(idx, named=True)
        obj_id = row[self.id_col]
        latent = row[self.latent_col]

        x_tc = self._to_TC(latent)  # [T,C]

        if x_tc.size == 0:
            # keep tokenizer safe
            x_tc = np.zeros((1, int(self.latent_dim or 1)), dtype=np.float32)

        return torch.from_numpy(x_tc), obj_id
    
    
import math
import torch

def collate_latents_pad_to_patch_multiple(batch, patch_size: int, pad_value: float = 0.0):
    """
    batch: list of (x[T,C], id)
    returns:
      x_pad   : [B, Lpad, C] float32
      mask    : [B, Lpad]    bool   (True for real tokens)
      lengths : [B]          int64  (original T)
      ids     : list         (object ids)
    """
    xs, ids = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    Tmax = int(lengths.max().item())
    C = int(xs[0].shape[1])

    Lpad = int(math.ceil(Tmax / patch_size) * patch_size)
    B = len(xs)

    x_pad = torch.full((B, Lpad, C), float(pad_value), dtype=torch.float32)
    mask = torch.zeros((B, Lpad), dtype=torch.bool)

    for i, x in enumerate(xs):
        T = x.shape[0]
        x_pad[i, :T] = x
        mask[i, :T] = True

    return x_pad, mask, lengths, list(ids)

