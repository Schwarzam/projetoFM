from typing import Optional
import numpy as np

from astromodal.scalers.standardvec import StandardScalerVec

def fit_standard_scaler_vec_from_gaiaxp(
    df,
    *,
    col: str = "bp_coefficients",
    max_rows: Optional[int] = 20000,
    seed: int = 0,
    transform: str = "none",        # "none" | "asinh"
    asinh_scale: float = 1.0,       # used if transform == "asinh"
    clip_quantile: float = 0.999,   # clip per-dimension before mean/std
    min_std: float = 1e-6,          # floor for std
) -> "StandardScalerVec":
    """
    Fit a per-coefficient StandardScalerVec for Gaia XP coefficient vectors.

    - Samples up to max_rows rows from df (polars) deterministically by seed
    - Reads list column 'col' with shape [D]
    - Optionally applies asinh pre-transform per element
    - Optionally clips extremes *per coefficient index* (robust)
    - Returns StandardScalerVec(mean[D], std[D], transform, asinh_scale)

    Notes:
      - This assumes each row has a coefficient vector of same length D.
      - If some rows have missing/short vectors, they are skipped.
    """
    import random

    rng = random.Random(seed)
    n = df.height
    idxs = list(range(n))

    if max_rows is not None and n > max_rows:
        idxs = rng.sample(idxs, k=max_rows)

    rows = []
    D = None

    for idx in idxs:
        row = df.row(idx, named=True)
        coeff = row.get(col)

        if coeff is None:
            continue

        x = np.asarray(coeff, dtype=np.float64)
        if x.ndim != 1 or x.size == 0:
            continue

        if D is None:
            D = int(x.size)
        if int(x.size) != D:
            # skip inconsistent vectors
            continue

        # finite mask per row
        if not np.all(np.isfinite(x)):
            # keep if at least some finite; replace non-finite with nan
            x = x.copy()
            x[~np.isfinite(x)] = np.nan

        rows.append(x)

    if not rows or D is None:
        mean = np.zeros(55, dtype=np.float64)  # fallback guess
        std  = np.ones(55, dtype=np.float64)
        return StandardScalerVec(mean=mean, std=std, transform=transform, asinh_scale=asinh_scale)

    X = np.stack(rows, axis=0)  # [N,D]

    # drop rows that are completely NaN
    good_row = np.isfinite(X).any(axis=1)
    X = X[good_row]
    if X.shape[0] == 0:
        mean = np.zeros(D, dtype=np.float64)
        std  = np.ones(D, dtype=np.float64)
        return StandardScalerVec(mean=mean, std=std, transform=transform, asinh_scale=asinh_scale)

    # ---- pre-transform ----
    if transform == "asinh":
        s0 = float(asinh_scale) if asinh_scale > 0 else 1.0
        X = np.arcsinh(X / s0)

    # ---- optional robust clipping per dimension ----
    if 0.0 < clip_quantile < 1.0 and X.shape[0] > 10:
        lo_q = 1.0 - clip_quantile
        hi_q = clip_quantile

        lo = np.nanquantile(X, lo_q, axis=0)  # [D]
        hi = np.nanquantile(X, hi_q, axis=0)  # [D]

        # broadcast clip
        X = np.clip(X, lo[None, :], hi[None, :])

    mean = np.nanmean(X, axis=0)  # [D]
    std  = np.nanstd(X, axis=0)   # [D]

    # sanitize
    std = np.where(np.isfinite(std) & (std >= min_std), std, 1.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)

    return StandardScalerVec(mean=mean.astype(np.float64),
                             std=std.astype(np.float64),
                             transform=transform,
                             asinh_scale=float(asinh_scale))