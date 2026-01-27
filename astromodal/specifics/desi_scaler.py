from astromodal.datasets.desi_spectra import stitch_desi_arms
from astromodal.scalers.scaler1d import StandardScaler1D

from typing import Optional
import numpy as np

def fit_standard_scaler_1d_from_desi_df(
    df,
    *,
    max_rows: Optional[int] = 20000,
    seed: int = 0,
    transform: str = "asinh",      # "none" or "asinh"
    asinh_scale: float = 1.0,      # tune if you want (e.g. 1, 10, median abs flux, etc.)
    clip_quantile: float = 0.999,  # clip extremes before mean/std
) -> StandardScaler1D:
    """
    Fit a 1D StandardScaler on DESI stitched flux values pooled across objects.
    Uses your stitch_desi_arms to get flux + mask; then aggregates values.

    Note: this reads rows and stitches, so it's heavier than reading a simple float column.
    Use max_rows to subsample.
    """
    import random

    rng = random.Random(seed)
    n = df.height
    idxs = list(range(n))
    if max_rows is not None and n > max_rows:
        idxs = rng.sample(idxs, k=max_rows)

    vals = []

    for idx in idxs:
        row = df.row(idx, named=True)
        _, f, _, m = stitch_desi_arms(
            row.get("desi_wave_b"), row.get("desi_flux_b"), row.get("desi_ivar_b"),
            row.get("desi_wave_r"), row.get("desi_flux_r"), row.get("desi_ivar_r"),
            row.get("desi_wave_z"), row.get("desi_flux_z"), row.get("desi_ivar_z"),
        )
        if f.size == 0:
            continue
        mask = m.astype(bool)
        v = f[mask]
        v = v[np.isfinite(v)]
        if v.size:
            vals.append(v.astype(np.float64, copy=False))

    if not vals:
        return StandardScaler1D(mean=0.0, std=1.0, transform=transform, asinh_scale=asinh_scale)

    v = np.concatenate(vals)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return StandardScaler1D(mean=0.0, std=1.0, transform=transform, asinh_scale=asinh_scale)

    # apply pre-transform
    if transform == "asinh":
        v = np.arcsinh(v / (asinh_scale if asinh_scale > 0 else 1.0))

    # optional clipping
    if 0.0 < clip_quantile < 1.0 and v.size > 10:
        lo = np.quantile(v, 1.0 - clip_quantile)
        hi = np.quantile(v, clip_quantile)
        v = np.clip(v, lo, hi)

    mean = float(np.mean(v))
    std = float(np.std(v))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0

    return StandardScaler1D(mean=mean, std=std, transform=transform, asinh_scale=asinh_scale)