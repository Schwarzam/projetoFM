#!/usr/bin/env python3
from __future__ import annotations

# ---- avoid OpenBLAS thread explosion on big nodes ----
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import contextlib
import tempfile
import glob

import numpy as np
import polars as pl
from sklearn.cluster import MiniBatchKMeans
import joblib
from tqdm.auto import tqdm

# =========================
# CONFIG
# =========================

INPUT_DIR = "/home/astrodados4/downloads/hypercube"
FILE_PATTERN = "datacube_*.parquet"

TOKENIZER_CONFIG = "spectrum_tokenizer_config.npz"
CODEBOOK_SIZE = 2048

MAX_FEATURES_PER_GROUP = 2_000_000
MAX_ROWS_PER_FILE = 5000
MAX_POINTS_PER_ROW = 2000

DO_ZTF_AUTODETECT = False
ZTF_MAG_PREFIX = "ztf_mag_"
ZTF_MAGERR_PREFIX = "ztf_magerr_"
ZTF_TIME_PREFIX = "ztf_hmjd_"
ZTF_ERR_EPS = 1e-3

# ---- GaiaXPy strategy ----
GAIAXPY_MODE = "convert"   # "convert" or "calibrate"
GAIAXPY_SAMPLING = None    # e.g. np.linspace(330, 1050, 600)
GAIAXPY_TRUNCATION = False
GAIAXPY_WITH_CORRELATION = False

# GaiaXP feature details
GAIAXP_ERR_EPS = 1e-6          # precision floor
GAIAXP_CLIP_NORM = 12.0        # clip normalized flux to reduce extreme tails

BASE_GROUPS = [
    {"name": "desi_b", "kind": "flux_ivar", "flux_col": "desi_flux_b", "ivar_col": "desi_ivar_b"},
    {"name": "desi_r", "kind": "flux_ivar", "flux_col": "desi_flux_r", "ivar_col": "desi_ivar_r"},
    {"name": "desi_z", "kind": "flux_ivar", "flux_col": "desi_flux_z", "ivar_col": "desi_ivar_z"},

    # ✅ Better: separate codebooks (BP vs RP)
    {"name": "gaiaxp_bp", "kind": "gaiaxpy_xp", "xp_band": "BP"},
    {"name": "gaiaxp_rp", "kind": "gaiaxpy_xp", "xp_band": "RP"},
]

# =========================
# HELPERS
# =========================

def list_parquet_files() -> List[Path]:
    p = Path(INPUT_DIR)
    files = sorted(p.glob(FILE_PATTERN))
    if not files:
        raise RuntimeError(f"No parquet files found in {INPUT_DIR} matching {FILE_PATTERN}")
    return files

def ensure_dir_for_file(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def read_schema_fast(path: Path) -> List[str]:
    df0 = pl.read_parquet(path, n_rows=0)
    return df0.columns

def union_columns(files: List[Path]) -> List[str]:
    cols = set()
    for f in tqdm(files, desc="[schema] union columns", unit="file"):
        try:
            cols.update(read_schema_fast(f))
        except Exception as e:
            tqdm.write(f"[warn] Could not read schema from {f.name}: {e}")
    return sorted(cols)

def autodetect_ztf_groups(all_cols: List[str]) -> List[Dict[str, Any]]:
    mags, magerrs, times = {}, {}, {}

    for c in all_cols:
        if c.startswith(ZTF_MAG_PREFIX):
            band = c[len(ZTF_MAG_PREFIX):]
            mags[band] = c
        elif c.startswith(ZTF_MAGERR_PREFIX):
            band = c[len(ZTF_MAGERR_PREFIX):]
            magerrs[band] = c
        elif c.startswith(ZTF_TIME_PREFIX):
            band = c[len(ZTF_TIME_PREFIX):]
            times[band] = c

    bands = sorted(set(mags.keys()) & set(magerrs.keys()))
    groups: List[Dict[str, Any]] = []
    for b in bands:
        groups.append({
            "name": f"ztf_{b}",
            "kind": "ztf_mag_err",
            "mag_col": mags[b],
            "err_col": magerrs[b],
            "time_col": times.get(b, None),
        })

    if groups:
        tqdm.write("\nDetected ZTF groups:")
        for g in groups:
            tinfo = f", time_col={g['time_col']}" if g.get("time_col") else ""
            tqdm.write(f"  - {g['name']}: mag_col={g['mag_col']}, err_col={g['err_col']}{tinfo}")
    else:
        tqdm.write("\nNo ZTF (mag+magerr) groups detected in union schema.")

    return groups

def _sample_points(feats: np.ndarray) -> np.ndarray:
    if feats.ndim == 1:
        feats = feats.reshape(-1, 1)
    L = feats.shape[0]
    if L > MAX_POINTS_PER_ROW:
        idx = np.random.choice(L, size=MAX_POINTS_PER_ROW, replace=False)
        return feats[idx]
    return feats

def _finite_rows_mask(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return np.isfinite(x).all(axis=1)

def _cell_to_1d_float(cell) -> np.ndarray:
    if cell is None:
        return np.zeros((0,), dtype=np.float32)

    if isinstance(cell, pl.Series):
        cell = cell.to_list()

    if isinstance(cell, (list, tuple, np.ndarray)):
        arr = np.asarray(cell, dtype=np.float32).reshape(-1)
        return arr[np.isfinite(arr)]

    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return np.zeros((0,), dtype=np.float32)

        # may be '"[1,2,3]"'
        try:
            decoded = json.loads(s)
            if isinstance(decoded, str):
                s = decoded.strip()
            else:
                arr = np.asarray(decoded, dtype=np.float32).reshape(-1)
                return arr[np.isfinite(arr)]
        except Exception:
            pass

        try:
            vals = json.loads(s)
            arr = np.asarray(vals, dtype=np.float32).reshape(-1)
            return arr[np.isfinite(arr)]
        except Exception:
            return np.zeros((0,), dtype=np.float32)

    try:
        v = float(cell)
        if np.isfinite(v):
            return np.asarray([v], dtype=np.float32)
    except Exception:
        pass

    return np.zeros((0,), dtype=np.float32)

# =========================
# VALID ROW SAMPLING
# =========================

def _mask_list_nonempty(col: str) -> pl.Expr:
    e = pl.col(col)
    return e.is_not_null() & (e.list.len() > 0)

def _mask_str_nonempty(col: str) -> pl.Expr:
    e = pl.col(col)
    s = e.str.strip_chars()
    return e.is_not_null() & (s.str.len_chars() > 2) & (s != "[]") & s.str.contains(r"\[")

def _nonempty_cell_mask(df: pl.DataFrame, col: str) -> pl.Expr:
    try:
        df.select(_mask_list_nonempty(col).head(1))
        return _mask_list_nonempty(col)
    except Exception:
        try:
            df.select(_mask_str_nonempty(col).head(1))
            return _mask_str_nonempty(col)
        except Exception:
            return pl.col(col).is_not_null()

def _valid_row_indices_for_group(df: pl.DataFrame, group: Dict[str, Any]) -> np.ndarray:
    kind = group["kind"]

    if kind == "flux_ivar":
        fcol = group["flux_col"]
        wcol = group["ivar_col"]
        f = pl.col(fcol)
        w = pl.col(wcol)

        mask = _nonempty_cell_mask(df, fcol) & _nonempty_cell_mask(df, wcol)
        try:
            df.select((f.list.len() == w.list.len()).head(1))
            mask = mask & (f.list.len() == w.list.len())
        except Exception:
            pass

    elif kind == "ztf_mag_err":
        mcol = group["mag_col"]
        ecol = group["err_col"]
        m = pl.col(mcol)
        e = pl.col(ecol)

        mask = _nonempty_cell_mask(df, mcol) & _nonempty_cell_mask(df, ecol)
        try:
            df.select((m.list.len() == e.list.len()).head(1))
            mask = mask & (m.list.len() == e.list.len())
        except Exception:
            pass

    elif kind == "gaiaxpy_xp":
        # resolved per-file; mask built after loading columns
        mask = pl.all_horizontal([pl.lit(True)])

    else:
        raise ValueError(f"Unknown group kind: {kind}")

    idx = df.select(pl.arg_where(mask)).to_series().to_numpy()
    return idx.astype(np.int64)

def _sample_valid_row_indices(valid_idx: np.ndarray) -> np.ndarray:
    n = int(valid_idx.size)
    if n == 0:
        return valid_idx
    if n > MAX_ROWS_PER_FILE:
        pick = np.random.choice(n, size=MAX_ROWS_PER_FILE, replace=False)
        return valid_idx[pick]
    return valid_idx

# =========================
# GaiaXPy temp-file integration
# =========================

def _resolve_gaiaxp_continuous_columns(cols_in_file: List[str]) -> Optional[Dict[str, str]]:
    """
    Accept both:
      - bp_coefficients / rp_coefficients ...
      - gaiaxp_bp_coefficients / gaiaxp_rp_coefficients ...
    We'll synthesize errors/std/cov if missing.
    """
    cols = set(cols_in_file)
    prefixes = ["gaiaxp_", ""]

    required = [
        "source_id",
        "bp_coefficients",
        "rp_coefficients",
        "bp_coefficient_correlations",
        "rp_coefficient_correlations",
        "bp_n_parameters",
        "rp_n_parameters",
    ]
    optional = [
        "bp_coefficient_errors",
        "rp_coefficient_errors",
        "bp_standard_deviation",
        "rp_standard_deviation",
        "bp_covariance_matrix",
        "rp_covariance_matrix",
    ]

    out: Dict[str, str] = {}
    for name in required + optional:
        for pref in prefixes:
            cand = f"{pref}{name}"
            if cand in cols:
                out[name] = cand
                break

    for name in required:
        if name not in out:
            return None

    return out

def _polars_to_pandas_gaiaxpy_input(df: pl.DataFrame, colmap: Dict[str, str]):
    """
    Build a pandas DataFrame with EXACT canonical column names GaiaXPy expects,
    and ensure values are plain Python lists / floats (robust for CSV roundtrip).
    """
    import pandas as pd
    import numpy as np

    def as_list1d(x):
        return _cell_to_1d_float(x).astype(float, copy=False).tolist()

    def as_int(x, default=0):
        try:
            return int(x)
        except Exception:
            return default

    def rms_scalar_from_vec(x) -> float:
        arr = _cell_to_1d_float(x)
        if arr.size == 0:
            return np.nan
        return float(np.sqrt(np.mean(arr * arr)))

    def scalar_or_rms(x) -> float:
        if x is None:
            return np.nan
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        return rms_scalar_from_vec(x)

    def corrvec_to_corrmat(corr_vec: np.ndarray, n: int) -> np.ndarray:
        corr_vec = np.asarray(corr_vec, dtype=np.float32).reshape(-1)
        need = n * (n - 1) // 2
        if corr_vec.size != need:
            return np.eye(n, dtype=np.float32)
        C = np.eye(n, dtype=np.float32)
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                C[i, j] = corr_vec[k]
                C[j, i] = corr_vec[k]
                k += 1
        return np.clip(C, -1.0, 1.0)

    def cov_from_corr_and_err(corr_vec_cell, err_list: List[float]) -> List[List[float]]:
        err = np.asarray(err_list, dtype=np.float32).reshape(-1)
        n = int(err.size)
        if n == 0:
            return []
        corr_vec = _cell_to_1d_float(corr_vec_cell).astype(np.float32, copy=False)
        Corr = corrvec_to_corrmat(corr_vec, n)
        Cov = Corr * (err[:, None] * err[None, :])
        return Cov.astype(float).tolist()

    src = df[colmap["source_id"]].cast(pl.Int64).to_list()

    bp_coeff_raw = df[colmap["bp_coefficients"]].to_list()
    rp_coeff_raw = df[colmap["rp_coefficients"]].to_list()
    bp_corr_raw  = df[colmap["bp_coefficient_correlations"]].to_list()
    rp_corr_raw  = df[colmap["rp_coefficient_correlations"]].to_list()
    bp_npar_raw  = df[colmap["bp_n_parameters"]].to_list()
    rp_npar_raw  = df[colmap["rp_n_parameters"]].to_list()

    bp_coeff = [as_list1d(x) for x in bp_coeff_raw]
    rp_coeff = [as_list1d(x) for x in rp_coeff_raw]
    bp_corr1 = [as_list1d(x) for x in bp_corr_raw]
    rp_corr1 = [as_list1d(x) for x in rp_corr_raw]

    bp_err = None
    rp_err = None
    if "bp_coefficient_errors" in colmap and colmap["bp_coefficient_errors"] in df.columns:
        bp_err = [as_list1d(x) for x in df[colmap["bp_coefficient_errors"]].to_list()]
    if "rp_coefficient_errors" in colmap and colmap["rp_coefficient_errors"] in df.columns:
        rp_err = [as_list1d(x) for x in df[colmap["rp_coefficient_errors"]].to_list()]

    bp_std = None
    rp_std = None
    if "bp_standard_deviation" in colmap and colmap["bp_standard_deviation"] in df.columns:
        bp_std = [scalar_or_rms(x) for x in df[colmap["bp_standard_deviation"]].to_list()]
    if "rp_standard_deviation" in colmap and colmap["rp_standard_deviation"] in df.columns:
        rp_std = [scalar_or_rms(x) for x in df[colmap["rp_standard_deviation"]].to_list()]

    if bp_std is None:
        bp_std = [rms_scalar_from_vec(e) for e in bp_err] if bp_err is not None else [np.nan] * len(bp_coeff)
    if rp_std is None:
        rp_std = [rms_scalar_from_vec(e) for e in rp_err] if rp_err is not None else [np.nan] * len(rp_coeff)

    if bp_err is None:
        bp_err = []
        for L, s in zip([len(x) for x in bp_coeff], bp_std):
            bp_err.append([] if L == 0 else [float(s) if np.isfinite(s) else np.nan] * L)

    if rp_err is None:
        rp_err = []
        for L, s in zip([len(x) for x in rp_coeff], rp_std):
            rp_err.append([] if L == 0 else [float(s) if np.isfinite(s) else np.nan] * L)

    bp_cov = [cov_from_corr_and_err(c, e) for c, e in zip(bp_corr_raw, bp_err)]
    rp_cov = [cov_from_corr_and_err(c, e) for c, e in zip(rp_corr_raw, rp_err)]

    return pd.DataFrame(
        {
            "source_id": [int(x) for x in src],

            "bp_coefficients": bp_coeff,
            "rp_coefficients": rp_coeff,

            "bp_coefficient_errors": bp_err,
            "rp_coefficient_errors": rp_err,

            "bp_coefficient_correlations": bp_corr1,
            "rp_coefficient_correlations": rp_corr1,

            "bp_n_parameters": [as_int(x, 0) for x in bp_npar_raw],
            "rp_n_parameters": [as_int(x, 0) for x in rp_npar_raw],

            "bp_standard_deviation": [float(x) if np.isfinite(x) else np.nan for x in bp_std],
            "rp_standard_deviation": [float(x) if np.isfinite(x) else np.nan for x in rp_std],

            "bp_covariance_matrix": bp_cov,
            "rp_covariance_matrix": rp_cov,
        }
    )

@contextlib.contextmanager
def _suppress_all_output():
    devnull = open(os.devnull, "w")
    old_out, old_err = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
    finally:
        try:
            os.dup2(old_out, 1)
            os.dup2(old_err, 2)
        finally:
            os.close(old_out)
            os.close(old_err)
            devnull.close()

import ast

def _cell_to_list_float(cell) -> np.ndarray:
    """
    GaiaXPy long output puts 'flux'/'flux_error' as python lists OR strings like '[1,2,3]'.
    Convert to 1D float32 finite array.
    """
    if cell is None:
        return np.zeros((0,), dtype=np.float32)

    if isinstance(cell, np.ndarray):
        arr = cell.astype(np.float32, copy=False).reshape(-1)
        return arr[np.isfinite(arr)]

    if isinstance(cell, (list, tuple)):
        arr = np.asarray(cell, dtype=np.float32).reshape(-1)
        return arr[np.isfinite(arr)]

    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return np.zeros((0,), dtype=np.float32)

        try:
            vals = ast.literal_eval(s)
            arr = np.asarray(vals, dtype=np.float32).reshape(-1)
            return arr[np.isfinite(arr)]
        except Exception:
            try:
                vals = json.loads(s)
                arr = np.asarray(vals, dtype=np.float32).reshape(-1)
                return arr[np.isfinite(arr)]
            except Exception:
                return np.zeros((0,), dtype=np.float32)

    try:
        v = float(cell)
        if np.isfinite(v):
            return np.asarray([v], dtype=np.float32)
    except Exception:
        pass

    return np.zeros((0,), dtype=np.float32)

def _robust_center_scale(x: np.ndarray) -> np.ndarray:
    """
    Robust per-spectrum normalization: (x - median) / (MAD + eps).
    """
    if x.size == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = mad * 1.4826
    if not np.isfinite(scale) or scale <= 0:
        # fallback to std
        scale = float(np.std(x)) if np.std(x) > 0 else 1.0
    y = (x - med) / (scale + 1e-8)
    if GAIAXP_CLIP_NORM is not None and GAIAXP_CLIP_NORM > 0:
        y = np.clip(y, -GAIAXP_CLIP_NORM, GAIAXP_CLIP_NORM)
    return y.astype(np.float32, copy=False)

def _gaiaxpy_out_to_feature_points_2d(out_df, remaining_budget: int, xp_band: Optional[str]) -> np.ndarray:
    """
    GaiaXP -> (n_points, 2) float32 features: [flux_norm, precision]
    - Supports GaiaXPy LONG format: ['source_id','xp','flux','flux_error']
    - Filters by xp_band ('BP'/'RP') if provided.
    - Per-spectrum robust normalization of flux.
    """
    if remaining_budget <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if out_df is None:
        return np.zeros((0, 2), dtype=np.float32)

    # We only support LONG reliably for BP/RP splitting
    cols = set(getattr(out_df, "columns", []))
    if not (("flux" in cols) and ("flux_error" in cols)):
        return np.zeros((0, 2), dtype=np.float32)

    # Filter band
    if xp_band is not None and ("xp" in cols):
        try:
            sub = out_df[out_df["xp"] == xp_band]
        except Exception:
            sub = out_df
    else:
        sub = out_df

    if sub is None or len(sub) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    chunks = []
    flux_col = sub["flux"].values
    err_col  = sub["flux_error"].values

    # Build (flux_norm, precision) per-row, then concatenate
    for fcell, ecell in zip(flux_col, err_col):
        flux = _cell_to_list_float(fcell)
        ferr = _cell_to_list_float(ecell)

        if flux.size == 0 or ferr.size == 0:
            continue
        if flux.size != ferr.size:
            L = min(flux.size, ferr.size)
            if L <= 0:
                continue
            flux = flux[:L]
            ferr = ferr[:L]

        flux_n = _robust_center_scale(flux)

        ferr = np.clip(ferr, a_min=GAIAXP_ERR_EPS, a_max=None)
        prec = 1.0 / ferr

        feats = np.stack([flux_n, prec.astype(np.float32, copy=False)], axis=-1)  # (L,2)
        feats = feats[_finite_rows_mask(feats)]
        if feats.size == 0:
            continue

        # per-row sampling so giant spectra don't dominate
        feats = _sample_points(feats)
        if feats.size == 0:
            continue

        chunks.append(feats.astype(np.float32, copy=False))
        if sum(x.shape[0] for x in chunks) >= remaining_budget:
            break

    if not chunks:
        return np.zeros((0, 2), dtype=np.float32)

    X = np.concatenate(chunks, axis=0)
    if X.shape[0] > remaining_budget:
        idx = np.random.choice(X.shape[0], size=remaining_budget, replace=False)
        X = X[idx]

    X = X.astype(np.float32, copy=False)
    X = X[_finite_rows_mask(X)]
    return X

def _gaiaxpy_run_via_tempfile(df_pd):
    """
    Robust GaiaXPy runner returning the RAW out_df.
    """
    import pandas as pd
    import os
    import glob
    from contextlib import contextmanager

    @contextmanager
    def _chdir(path: str):
        old = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(old)

    with tempfile.TemporaryDirectory(prefix="gaiaxpy_tmp_") as td:
        in_csv = os.path.join(td, "xp_input.csv")
        df_pd.to_csv(in_csv, index=False)

        mode = GAIAXPY_MODE.lower().strip()
        if mode not in ("convert", "calibrate"):
            raise ValueError(f"GAIAXPY_MODE must be 'convert' or 'calibrate' (got {GAIAXPY_MODE!r})")

        # 1) preferred: in-memory return
        try:
            with _suppress_all_output():
                if mode == "convert":
                    from gaiaxpy import convert
                    if GAIAXPY_SAMPLING is None:
                        out_df, _ = convert(in_csv, truncation=GAIAXPY_TRUNCATION, save_file=False)
                    else:
                        out_df, _ = convert(
                            in_csv,
                            sampling=GAIAXPY_SAMPLING,
                            truncation=GAIAXPY_TRUNCATION,
                            save_file=False
                        )
                else:
                    from gaiaxpy import calibrate
                    if GAIAXPY_SAMPLING is None:
                        out_df, _ = calibrate(
                            in_csv,
                            truncation=GAIAXPY_TRUNCATION,
                            save_file=False,
                            with_correlation=GAIAXPY_WITH_CORRELATION
                        )
                    else:
                        out_df, _ = calibrate(
                            in_csv,
                            sampling=GAIAXPY_SAMPLING,
                            truncation=GAIAXPY_TRUNCATION,
                            save_file=False,
                            with_correlation=GAIAXPY_WITH_CORRELATION
                        )
            return out_df
        except Exception:
            pass

        # 2) fallback: force file output
        out_dir = os.path.join(td, "out")
        os.makedirs(out_dir, exist_ok=True)

        with _chdir(out_dir):
            with _suppress_all_output():
                if mode == "convert":
                    from gaiaxpy import convert
                    if GAIAXPY_SAMPLING is None:
                        convert(in_csv, truncation=GAIAXPY_TRUNCATION, save_file=True)
                    else:
                        convert(in_csv, sampling=GAIAXPY_SAMPLING, truncation=GAIAXPY_TRUNCATION, save_file=True)
                else:
                    from gaiaxpy import calibrate
                    if GAIAXPY_SAMPLING is None:
                        calibrate(in_csv, truncation=GAIAXPY_TRUNCATION, save_file=True, with_correlation=GAIAXPY_WITH_CORRELATION)
                    else:
                        calibrate(in_csv, sampling=GAIAXPY_SAMPLING, truncation=GAIAXPY_TRUNCATION, save_file=True, with_correlation=GAIAXPY_WITH_CORRELATION)

        candidates = sorted(
            glob.glob(os.path.join(td, "**", "*.csv"), recursive=True),
            key=os.path.getmtime
        )
        candidates = [c for c in candidates if os.path.basename(c) != "xp_input.csv"]
        if not candidates:
            return None

        newest = candidates[-1]
        out_df = pd.read_csv(newest)
        return out_df

# =========================
# FEATURE COLLECTION
# =========================

def collect_features_for_group(group: Dict[str, Any], files: List[Path]) -> np.ndarray:
    name = group["name"]
    kind = group["kind"]

    tqdm.write(f"\n[group {name}] Collecting features (kind={kind})...")

    all_feats: List[np.ndarray] = []
    total = 0

    files_ok = 0
    skipped_missing_cols = 0
    skipped_read_error = 0
    skipped_no_valid_rows = 0
    skipped_gx_error = 0

    if kind == "flux_ivar":
        cols_to_read = [group["flux_col"], group["ivar_col"]]
    elif kind == "ztf_mag_err":
        cols_to_read = [group["mag_col"], group["err_col"]]
    elif kind == "gaiaxpy_xp":
        cols_to_read = []
    else:
        raise ValueError(f"Unknown group kind: {kind}")

    tqdm.write(f"  [group {name}] Reading columns: {cols_to_read if cols_to_read else '(resolved per file)'}")

    pbar_total = tqdm(
        total=MAX_FEATURES_PER_GROUP,
        desc=f"[group {name}] collected",
        unit="feat",
        leave=True,
        dynamic_ncols=True,
    )
    pbar_files = tqdm(
        files,
        desc=f"[group {name}] files",
        unit="file",
        leave=False,
        dynamic_ncols=True,
    )

    def _set_file_postfix():
        pbar_files.set_postfix(
            {
                "ok": files_ok,
                "miss": skipped_missing_cols,
                "readerr": skipped_read_error,
                "novalid": skipped_no_valid_rows,
                "gxerr": skipped_gx_error,
            }
        )

    try:
        for path in pbar_files:
            if total >= MAX_FEATURES_PER_GROUP:
                break

            try:
                cols_in_file = read_schema_fast(path)
                cols_set = set(cols_in_file)
            except Exception:
                cols_in_file = []
                cols_set = set()

            colmap = None
            if kind == "gaiaxpy_xp":
                colmap = _resolve_gaiaxp_continuous_columns(cols_in_file)
                if colmap is None:
                    skipped_missing_cols += 1
                    _set_file_postfix()
                    continue
                cols_to_read_now = [colmap[k] for k in colmap.keys()]
            else:
                if not all(c in cols_set for c in cols_to_read):
                    skipped_missing_cols += 1
                    _set_file_postfix()
                    continue
                cols_to_read_now = cols_to_read

            try:
                df = pl.read_parquet(path, columns=cols_to_read_now)
            except Exception:
                skipped_read_error += 1
                _set_file_postfix()
                continue

            if df.height == 0:
                skipped_no_valid_rows += 1
                _set_file_postfix()
                continue

            # subsample rows
            if kind != "gaiaxpy_xp":
                valid_idx = _valid_row_indices_for_group(df, group)
                if valid_idx.size == 0:
                    skipped_no_valid_rows += 1
                    _set_file_postfix()
                    continue
                row_indices = _sample_valid_row_indices(valid_idx)
                df = df[row_indices]
            else:
                assert colmap is not None
                mask = (
                    _nonempty_cell_mask(df, colmap["bp_coefficients"])
                    & _nonempty_cell_mask(df, colmap["rp_coefficients"])
                    & _nonempty_cell_mask(df, colmap["bp_coefficient_correlations"])
                    & _nonempty_cell_mask(df, colmap["rp_coefficient_correlations"])
                    & pl.col(colmap["bp_n_parameters"]).is_not_null()
                    & pl.col(colmap["rp_n_parameters"]).is_not_null()
                )
                valid_idx = df.select(pl.arg_where(mask)).to_series().to_numpy().astype(np.int64)
                if valid_idx.size == 0:
                    skipped_no_valid_rows += 1
                    _set_file_postfix()
                    continue
                row_indices = _sample_valid_row_indices(valid_idx)
                df = df[row_indices]

            files_ok += 1

            # extract features
            if kind == "flux_ivar":
                flux_series = df[group["flux_col"]]
                ivar_series = df[group["ivar_col"]]

                for ridx in range(df.height):
                    if total >= MAX_FEATURES_PER_GROUP:
                        break
                    flux = _cell_to_1d_float(flux_series[ridx])
                    ivar = _cell_to_1d_float(ivar_series[ridx])
                    if flux.size == 0 or ivar.size == 0 or ivar.size != flux.size:
                        continue
                    ivar = np.clip(ivar, a_min=0.0, a_max=None)
                    prec = np.sqrt(ivar)
                    feats = np.stack([flux, prec], axis=-1)
                    feats = _sample_points(feats)
                    feats = feats[_finite_rows_mask(feats)]
                    if feats.size == 0:
                        continue
                    all_feats.append(feats.astype(np.float32))
                    n_new = int(feats.shape[0])
                    total += n_new
                    pbar_total.update(n_new)

            elif kind == "ztf_mag_err":
                mag_series = df[group["mag_col"]]
                err_series = df[group["err_col"]]
                for ridx in range(df.height):
                    if total >= MAX_FEATURES_PER_GROUP:
                        break
                    mag = _cell_to_1d_float(mag_series[ridx])
                    err = _cell_to_1d_float(err_series[ridx])
                    if mag.size == 0 or err.size == 0 or err.size != mag.size:
                        continue
                    err = np.clip(err, a_min=ZTF_ERR_EPS, a_max=None)
                    prec = 1.0 / err
                    feats = np.stack([mag, prec], axis=-1)
                    feats = _sample_points(feats)
                    feats = feats[_finite_rows_mask(feats)]
                    if feats.size == 0:
                        continue
                    all_feats.append(feats.astype(np.float32))
                    n_new = int(feats.shape[0])
                    total += n_new
                    pbar_total.update(n_new)

            elif kind == "gaiaxpy_xp":
                try:
                    assert colmap is not None
                    df_pd = _polars_to_pandas_gaiaxpy_input(df, colmap)

                    out_df = _gaiaxpy_run_via_tempfile(df_pd)
                    if out_df is None:
                        skipped_no_valid_rows += 1
                        _set_file_postfix()
                        continue

                except Exception as e:
                    skipped_gx_error += 1
                    if skipped_gx_error <= 3:
                        tqdm.write(f"[gaiaxpy warn] {type(e).__name__}: {e}")
                    _set_file_postfix()
                    continue

                remaining = MAX_FEATURES_PER_GROUP - total
                if remaining <= 0:
                    break

                xp_band = group.get("xp_band", None)

                # ✅ Key change: GaiaXP features are now 2D: [flux_norm, precision]
                feats = _gaiaxpy_out_to_feature_points_2d(out_df, remaining_budget=remaining, xp_band=xp_band)

                if feats.shape[0] == 0:
                    skipped_no_valid_rows += 1
                    _set_file_postfix()
                    continue

                all_feats.append(feats)
                n_new = int(feats.shape[0])
                total += n_new
                pbar_total.update(n_new)

            pbar_total.set_postfix({"files_ok": files_ok, "total": total})
            _set_file_postfix()

    finally:
        pbar_files.close()
        pbar_total.close()

    if skipped_missing_cols:
        tqdm.write(f"[group {name}] skipped {skipped_missing_cols} files (missing required columns).")
    if skipped_read_error:
        tqdm.write(f"[group {name}] skipped {skipped_read_error} files (read errors).")
    if skipped_no_valid_rows:
        tqdm.write(f"[group {name}] skipped {skipped_no_valid_rows} files (no valid rows for this group).")
    if skipped_gx_error:
        tqdm.write(f"[group {name}] GaiaXPy errors in {skipped_gx_error} files/batches.")

    if not all_feats:
        tqdm.write(f"  [group {name}] No features collected.")
        D = 2 if kind in ("flux_ivar", "ztf_mag_err", "gaiaxpy_xp") else 1
        return np.zeros((0, D), dtype=np.float32)

    X = np.concatenate(all_feats, axis=0)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    tqdm.write(f"  [group {name}] Collected {X.shape[0]} feature vectors, dim={X.shape[1]}")
    return X.astype(np.float32)

# =========================
# TRAIN / REBUILD CONFIG
# =========================

def train_spectrum_tokenizers():
    files = list_parquet_files()
    tqdm.write(f"Found {len(files)} parquet files.")

    all_cols = union_columns(files)

    if DO_ZTF_AUTODETECT:
        ztf_groups = autodetect_ztf_groups(all_cols)
    else:
        ztf_groups = []

    groups = list(BASE_GROUPS) + ztf_groups
    tqdm.write(f"\nTotal groups to consider: {len(groups)}")

    group_names: List[str] = []
    group_means: Dict[str, np.ndarray] = {}
    group_stds: Dict[str, np.ndarray] = {}
    group_codebooks: Dict[str, np.ndarray] = {}
    group_feature_dim: Dict[str, int] = {}
    group_columns: Dict[str, Dict[str, Any]] = {}

    for group in tqdm(groups, desc="[groups]", unit="group"):
        name = group["name"]
        out_joblib = Path(f"spectrum_kmeans_{name}.joblib")

        feats = collect_features_for_group(group, files)
        if feats.shape[0] < max(1000, CODEBOOK_SIZE):
            tqdm.write(f"[group {name}] Not enough data even for mean/std (got {feats.shape[0]}). Skipping group.")
            continue

        mean = feats.mean(axis=0)
        std = feats.std(axis=0) + 1e-8

        if out_joblib.exists():
            tqdm.write(f"[group {name}] {out_joblib.name} exists -> loading model (no retrain).")
            kmeans = joblib.load(out_joblib)
        else:
            if feats.shape[0] < CODEBOOK_SIZE * 10:
                tqdm.write(f"[group {name}] Not enough data for robust KMeans (got {feats.shape[0]}). Skipping training.")
                continue

            feats_std = (feats - mean) / std

            tqdm.write(f"[group {name}] Training MiniBatchKMeans with {CODEBOOK_SIZE} clusters...")
            kmeans = MiniBatchKMeans(
                n_clusters=CODEBOOK_SIZE,
                batch_size=4096,
                max_iter=50,
                verbose=0,
                n_init=1,   # (optional: set to 5 for more stable codebooks)
            )
            kmeans.fit(feats_std)

            joblib.dump(kmeans, out_joblib)
            tqdm.write(f"[group {name}] Done. Saved KMeans model to {out_joblib}")

        centers = getattr(kmeans, "cluster_centers_", None)
        if centers is None:
            tqdm.write(f"[group {name}] Loaded model has no cluster_centers_. Skipping.")
            continue

        group_names.append(name)
        group_means[name] = mean.astype(np.float32)
        group_stds[name] = std.astype(np.float32)
        group_codebooks[name] = np.asarray(centers, dtype=np.float32)
        group_feature_dim[name] = int(feats.shape[1])
        group_columns[name] = group

    if not group_names:
        raise RuntimeError("No groups could be written to the npz (no data / missing models / all skipped).")

    ensure_dir_for_file(TOKENIZER_CONFIG)
    np.savez_compressed(
        TOKENIZER_CONFIG,
        group_names=np.array(group_names, dtype=object),
        group_means=group_means,
        group_stds=group_stds,
        group_codebooks=group_codebooks,
        group_feature_dim=group_feature_dim,
        group_columns=group_columns,
        CODEBOOK_SIZE=int(CODEBOOK_SIZE),
        MAX_POINTS_PER_ROW=int(MAX_POINTS_PER_ROW),
        MAX_ROWS_PER_FILE=int(MAX_ROWS_PER_FILE),
        GAIAXPY_MODE=str(GAIAXPY_MODE),
        GAIAXPY_TRUNCATION=bool(GAIAXPY_TRUNCATION),
        GAIAXPY_WITH_CORRELATION=bool(GAIAXPY_WITH_CORRELATION),
    )

    tqdm.write(f"\nSaved spectrum tokenizer config to {TOKENIZER_CONFIG}")
    tqdm.write(f"Groups saved: {group_names}")

if __name__ == "__main__":
    train_spectrum_tokenizers()