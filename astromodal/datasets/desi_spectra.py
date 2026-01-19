from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _as_1d_float(x) -> np.ndarray:
    """
    Convert x to a 1D float64 array safely.
    - None -> empty
    - scalar -> length-1 if finite, else empty
    - array/list -> ravel to 1D
    """
    if x is None:
        return np.empty(0, dtype=np.float64)

    a = np.asarray(x, dtype=np.float64)

    if a.ndim == 0:
        # scalar
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
    Stitch DESI b/r/z arms into a single spectrum.

    Returns: wave, flux, ivar, mask
      mask True means valid pixel (ivar > 0 and finite).
    """

    wb = _as_1d_float(wave_b)
    fb = _as_1d_float(flux_b)
    ivb = _as_1d_float(ivar_b)
    wb, fb, ivb = _trim_to_min_len(wb, fb, ivb)

    wr = _as_1d_float(wave_r)
    fr = _as_1d_float(flux_r)
    ivr = _as_1d_float(ivar_r)
    wr, fr, ivr = _trim_to_min_len(wr, fr, ivr)

    wz = _as_1d_float(wave_z)
    fz = _as_1d_float(flux_z)
    ivz = _as_1d_float(ivar_z)
    wz, fz, ivz = _trim_to_min_len(wz, fz, ivz)

    # concatenate (arms can be empty)
    w = np.concatenate([wb, wr, wz]) if (wb.size + wr.size + wz.size) > 0 else np.empty(0, dtype=np.float64)
    f = np.concatenate([fb, fr, fz]) if (fb.size + fr.size + fz.size) > 0 else np.empty(0, dtype=np.float64)
    iv = np.concatenate([ivb, ivr, ivz]) if (ivb.size + ivr.size + ivz.size) > 0 else np.empty(0, dtype=np.float64)

    if w.size == 0:
        return w, f, iv, np.empty(0, dtype=bool)

    # finite + non-negative ivar
    m = np.isfinite(w) & np.isfinite(f) & np.isfinite(iv) & (iv >= 0)
    w, f, iv = w[m], f[m], iv[m]

    if w.size == 0:
        return w, f, iv, np.empty(0, dtype=bool)

    # sort by wavelength
    order = np.argsort(w)
    w, f, iv = w[order], f[order], iv[order]

    # combine duplicate wavelengths (arm overlaps)
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


class DesiStitchedSpectraDataset(Dataset):
    """
    Expects df.row(i, named=True) with list/array columns:
      desi_wave_b, desi_flux_b, desi_ivar_b, ...
    Returns:
      x    : float32 [L, 2] (flux, ivar)
      mask : bool    [L]
    """
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return self.df.height

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)

        w, f, iv, m = stitch_desi_arms(
            row.get("desi_wave_b"), row.get("desi_flux_b"), row.get("desi_ivar_b"),
            row.get("desi_wave_r"), row.get("desi_flux_r"), row.get("desi_ivar_r"),
            row.get("desi_wave_z"), row.get("desi_flux_z"), row.get("desi_ivar_z"),
        )

        # If empty, return a dummy length-1 sample (mask False) so model/conv won't crash.
        if f.size == 0:
            x = np.zeros((1, 2), dtype=np.float32)
            mask = np.zeros((1,), dtype=np.bool_)
            return torch.from_numpy(x), torch.from_numpy(mask)

        x = np.stack([f.astype(np.float32), iv.astype(np.float32)], axis=1)  # [L,2]
        return torch.from_numpy(x), torch.from_numpy(m.astype(np.bool_))


def desi_collate_pad(batch, pad_value_x=0.0, pad_value_mask=False):
    """
    batch: list of (x[L,2], mask[L])
    returns:
      x_pad    : [B, Lmax, 2] float32
      mask_pad : [B, Lmax]    bool
      lengths  : [B]          int64
    """
    xs, ms = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    Lmax = int(lengths.max().item())
    B = len(xs)

    x_pad = torch.full((B, Lmax, 2), float(pad_value_x), dtype=torch.float32)
    mask_pad = torch.full((B, Lmax), bool(pad_value_mask), dtype=torch.bool)

    for i, (x, m) in enumerate(zip(xs, ms)):
        L = x.shape[0]
        x_pad[i, :L] = x
        mask_pad[i, :L] = m

    return x_pad, mask_pad, lengths