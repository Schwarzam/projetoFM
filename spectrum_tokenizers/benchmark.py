#!/usr/bin/env python3
from __future__ import annotations

import os

# ---- HARD LIMIT BLAS THREADS (prevents OpenBLAS segfault) ----
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import contextlib
import tempfile
import glob
import ast

import numpy as np
import polars as pl
import pandas as pd
import joblib
from tqdm.auto import tqdm

# =========================
# CONFIG
# =========================

INPUT_DIR = "/home/astrodados4/downloads/hypercube"
FILE_PATTERN = "datacube_*.parquet"

TOKENIZER_CONFIG = "spectrum_tokenizer_config.npz"  # produced by training script
KMEANS_GLOB = "spectrum_kmeans_*.joblib"

# Benchmark sampling (keep smaller than training to run fast)
MAX_FILES = 999999
MAX_ROWS_PER_FILE = 2000
MAX_POINTS_PER_ROW = 2000
MAX_FEATURES_PER_GROUP = 500_000  # per group, for benchmark

# For ZTF precision floor
ZTF_ERR_EPS = 1e-3

# Output
OUT_JSON = "spectrum_codebook_benchmark.json"

# ---- GaiaXPy bench mode ----
GAIAXPY_MODE = "convert"       # "convert" or "calibrate"
GAIAXPY_SAMPLING = None        # e.g. np.linspace(330, 1050, 600)
GAIAXPY_TRUNCATION = False
GAIAXPY_WITH_CORRELATION = False  # only used in calibrate

# ---- GaiaXP feature design (MUST MATCH TRAINING) ----
GAIAXP_ERR_EPS = 1e-6
GAIAXP_CLIP_NORM = 12.0

# =========================
# UTIL
# =========================

def list_parquet_files() -> List[Path]:
    p = Path(INPUT_DIR)
    files = sorted(p.glob(FILE_PATTERN))
    if not files:
        raise RuntimeError(f"No parquet files found in {INPUT_DIR} matching {FILE_PATTERN}")
    return files[:MAX_FILES]

def read_schema_fast(path: Path) -> List[str]:
    df0 = pl.read_parquet(path, n_rows=0)
    return df0.columns

def _sample_points(feats: np.ndarray, max_points: int) -> np.ndarray:
    if feats.ndim == 1:
        feats = feats.reshape(-1, 1)
    L = feats.shape[0]
    if L > max_points:
        idx = np.random.choice(L, size=max_points, replace=False)
        return feats[idx]
    return feats

def _finite_rows_mask(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return np.isfinite(x).all(axis=1)

def _cell_to_1d_float(cell) -> np.ndarray:
    """
    Convert a polars/pandas cell into a 1D float32 numpy array.
    Handles:
      - list/ndarray
      - JSON list stored as string
      - JSON list stored as *quoted* string: "\"[1,2,3]\""
      - python-literal lists ("[1,2,3]")
      - polars Series inner values
    """
    if cell is None:
        return np.zeros((0,), dtype=np.float32)

    if isinstance(cell, pl.Series):
        cell = cell.to_list()

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

        # A) JSON decode (maybe returns list OR string)
        try:
            decoded = json.loads(s)
            if isinstance(decoded, str):
                s = decoded.strip()
            else:
                arr = np.asarray(decoded, dtype=np.float32).reshape(-1)
                return arr[np.isfinite(arr)]
        except Exception:
            pass

        # B) JSON list
        try:
            vals = json.loads(s)
            arr = np.asarray(vals, dtype=np.float32).reshape(-1)
            return arr[np.isfinite(arr)]
        except Exception:
            pass

        # C) python literal list
        try:
            vals = ast.literal_eval(s)
            arr = np.asarray(vals, dtype=np.float32).reshape(-1)
            return arr[np.isfinite(arr)]
        except Exception:
            return np.zeros((0,), dtype=np.float32)

    return np.zeros((0,), dtype=np.float32)

# =========================
# VALID ROW SAMPLING (LIST OR STR)
# =========================

def _mask_list_nonempty(col: str) -> pl.Expr:
    e = pl.col(col)
    return e.is_not_null() & (e.list.len() > 0)

def _mask_str_nonempty(col: str) -> pl.Expr:
    e = pl.col(col).cast(pl.Utf8)
    s = e.str.strip_chars()
    return (
        e.is_not_null()
        & (s.str.len_chars() > 2)
        & (s != "[]")
        & s.str.contains(r"\[")
    )

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

def _valid_row_indices_for_group(df: pl.DataFrame, group_spec: Dict[str, Any]) -> np.ndarray:
    kind = group_spec["kind"]

    if kind == "flux_ivar":
        fcol = group_spec["flux_col"]
        wcol = group_spec["ivar_col"]
        f = pl.col(fcol)
        w = pl.col(wcol)

        mask = _nonempty_cell_mask(df, fcol) & _nonempty_cell_mask(df, wcol)
        try:
            df.select((f.list.len() == w.list.len()).head(1))
            mask = mask & (f.list.len() == w.list.len())
        except Exception:
            pass

    elif kind == "value_only":
        fcol = group_spec["flux_col"]
        mask = _nonempty_cell_mask(df, fcol)

    elif kind == "ztf_mag_err":
        mcol = group_spec["mag_col"]
        ecol = group_spec["err_col"]
        m = pl.col(mcol)
        e = pl.col(ecol)

        mask = _nonempty_cell_mask(df, mcol) & _nonempty_cell_mask(df, ecol)
        try:
            df.select((m.list.len() == e.list.len()).head(1))
            mask = mask & (m.list.len() == e.list.len())
        except Exception:
            pass

    elif kind == "gaiaxpy_xp":
        mask = pl.all_horizontal([pl.lit(True)])

    else:
        raise ValueError(f"Unknown kind={kind}")

    idx = df.select(pl.arg_where(mask)).to_series().to_numpy()
    return idx.astype(np.int64)

def _sample_valid_row_indices(valid_idx: np.ndarray, max_rows: int) -> np.ndarray:
    n = int(valid_idx.size)
    if n == 0:
        return valid_idx
    if n > max_rows:
        pick = np.random.choice(n, size=max_rows, replace=False)
        return valid_idx[pick]
    return valid_idx

# =========================
# LOAD TOKENIZER METADATA
# =========================

def load_tokenizer_config(path: str) -> Dict[str, Any]:
    npz = np.load(path, allow_pickle=True)
    group_names = list(npz["group_names"].tolist())
    group_means: Dict[str, np.ndarray] = npz["group_means"].item()
    group_stds: Dict[str, np.ndarray] = npz["group_stds"].item()
    group_columns: Dict[str, Dict[str, Any]] = npz["group_columns"].item()
    return {
        "group_names": group_names,
        "group_means": group_means,
        "group_stds": group_stds,
        "group_columns": group_columns,
    }

def load_kmeans_models() -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    for p in sorted(Path(".").glob(KMEANS_GLOB)):
        name = p.name.replace("spectrum_kmeans_", "").replace(".joblib", "")
        models[name] = joblib.load(p)
    if not models:
        raise RuntimeError(f"No models found matching {KMEANS_GLOB} in current directory.")
    return models

# =========================
# GaiaXPy helpers
# =========================

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

def _resolve_gaiaxp_continuous_columns(cols_in_file: List[str]) -> Optional[Dict[str, str]]:
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

def _polars_to_pandas_gaiaxpy_input(df: pl.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    """
    Canonical GaiaXPy input columns (lists/floats) — robust for CSV roundtrip.
    Always synthesize covariance matrices.
    """
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

def _gaiaxpy_run_via_tempfile_raw(df_pd: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Returns RAW GaiaXPy dataframe (ideally LONG format).
    """
    @contextlib.contextmanager
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

        # 1) try in-memory return
        try:
            with _suppress_all_output():
                if mode == "convert":
                    from gaiaxpy import convert
                    if GAIAXPY_SAMPLING is None:
                        out_df, _ = convert(in_csv, truncation=GAIAXPY_TRUNCATION, save_file=False)
                    else:
                        out_df, _ = convert(in_csv, sampling=GAIAXPY_SAMPLING, truncation=GAIAXPY_TRUNCATION, save_file=False)
                else:
                    from gaiaxpy import calibrate
                    if GAIAXPY_SAMPLING is None:
                        out_df, _ = calibrate(in_csv, truncation=GAIAXPY_TRUNCATION, save_file=False, with_correlation=GAIAXPY_WITH_CORRELATION)
                    else:
                        out_df, _ = calibrate(in_csv, sampling=GAIAXPY_SAMPLING, truncation=GAIAXPY_TRUNCATION, save_file=False, with_correlation=GAIAXPY_WITH_CORRELATION)
            if out_df is not None and len(out_df) > 0:
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

        candidates = sorted(glob.glob(os.path.join(td, "**", "*.csv"), recursive=True), key=os.path.getmtime)
        candidates = [c for c in candidates if os.path.basename(c) != "xp_input.csv"]
        if not candidates:
            return None

        newest = candidates[-1]
        out_df = pd.read_csv(newest)
        return out_df

def _robust_center_scale(x: np.ndarray) -> np.ndarray:
    """
    Robust per-spectrum normalization: (x - median) / (MAD*1.4826 + eps)
    then clip.
    """
    if x.size == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = mad * 1.4826
    if not np.isfinite(scale) or scale <= 0:
        sd = float(np.std(x))
        scale = sd if sd > 0 else 1.0
    y = (x - med) / (scale + 1e-8)
    y = y.astype(np.float32, copy=False)
    if GAIAXP_CLIP_NORM is not None and GAIAXP_CLIP_NORM > 0:
        y = np.clip(y, -GAIAXP_CLIP_NORM, GAIAXP_CLIP_NORM)
    return y

def _gaiaxpy_out_to_feature_points_2d(out_df: pd.DataFrame, remaining_budget: int, xp_band: Optional[str]) -> np.ndarray:
    """
    GaiaXP -> (n_points, 2) features: [flux_norm, precision]
    Requires LONG output with columns: xp, flux, flux_error for BP/RP splitting.
    """
    if remaining_budget <= 0 or out_df is None or len(out_df) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    cols = set(out_df.columns)
    if not (("flux" in cols) and ("flux_error" in cols) and ("xp" in cols)):
        # cannot split BP/RP robustly without xp
        return np.zeros((0, 2), dtype=np.float32)

    if xp_band is not None:
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

    collected = 0
    for fcell, ecell in zip(flux_col, err_col):
        if collected >= remaining_budget:
            break

        flux = _cell_to_1d_float(fcell)
        ferr = _cell_to_1d_float(ecell)

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
        prec = (1.0 / ferr).astype(np.float32, copy=False)

        feats = np.stack([flux_n, prec], axis=-1)  # (L,2)
        feats = feats[_finite_rows_mask(feats)]
        if feats.size == 0:
            continue

        feats = _sample_points(feats, MAX_POINTS_PER_ROW)
        feats = feats[_finite_rows_mask(feats)]
        if feats.size == 0:
            continue

        chunks.append(feats.astype(np.float32, copy=False))
        collected += int(feats.shape[0])

    if not chunks:
        return np.zeros((0, 2), dtype=np.float32)

    X = np.concatenate(chunks, axis=0)
    if X.shape[0] > remaining_budget:
        idx = np.random.choice(X.shape[0], size=remaining_budget, replace=False)
        X = X[idx]

    X = X.astype(np.float32, copy=False)
    X = X[_finite_rows_mask(X)]
    return X

# =========================
# FEATURE COLLECTION (BENCH)
# =========================

def collect_features_for_group_bench(group_spec: Dict[str, Any], files: List[Path]) -> np.ndarray:
    """
    Feature semantics:
      - flux_ivar:   feats=[flux, sqrt(ivar)]
      - value_only:  feats=[value]
      - ztf_mag_err: feats=[mag, 1/magerr]
      - gaiaxpy_xp:  feats=[flux_norm, precision]  (dim=2) with BP/RP split using xp_band
    """
    name = group_spec["name"]
    kind = group_spec["kind"]

    if kind == "flux_ivar":
        cols_to_read = [group_spec["flux_col"], group_spec["ivar_col"]]
    elif kind == "value_only":
        cols_to_read = [group_spec["flux_col"]]
    elif kind == "ztf_mag_err":
        cols_to_read = [group_spec["mag_col"], group_spec["err_col"]]
    elif kind == "gaiaxpy_xp":
        cols_to_read = []  # resolved per file
    else:
        raise ValueError(f"Unknown kind={kind}")

    all_feats: List[np.ndarray] = []
    total = 0

    skipped_missing_cols = 0
    skipped_read_error = 0
    skipped_no_valid_rows = 0
    skipped_gx_error = 0
    files_ok = 0

    pbar_total = tqdm(total=MAX_FEATURES_PER_GROUP, desc=f"[bench:{name}] collected", unit="feat", leave=False)
    pbar_files = tqdm(files, desc=f"[bench:{name}] files", unit="file", leave=False)

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
                    pbar_files.set_postfix_str(
                        f"ok={files_ok} miss={skipped_missing_cols} readerr={skipped_read_error} novalid={skipped_no_valid_rows} gxerr={skipped_gx_error}"
                    )
                    continue
                cols_to_read_now = [colmap[k] for k in colmap.keys()]
            else:
                if not all(c in cols_set for c in cols_to_read):
                    skipped_missing_cols += 1
                    pbar_files.set_postfix_str(
                        f"ok={files_ok} miss={skipped_missing_cols} readerr={skipped_read_error} novalid={skipped_no_valid_rows} gxerr={skipped_gx_error}"
                    )
                    continue
                cols_to_read_now = cols_to_read

            try:
                df = pl.read_parquet(path, columns=cols_to_read_now)
            except Exception:
                skipped_read_error += 1
                pbar_files.set_postfix_str(
                    f"ok={files_ok} miss={skipped_missing_cols} readerr={skipped_read_error} novalid={skipped_no_valid_rows} gxerr={skipped_gx_error}"
                )
                continue

            if df.height == 0:
                skipped_no_valid_rows += 1
                pbar_files.set_postfix_str(
                    f"ok={files_ok} miss={skipped_missing_cols} readerr={skipped_read_error} novalid={skipped_no_valid_rows} gxerr={skipped_gx_error}"
                )
                continue

            # valid rows + subsample
            if kind != "gaiaxpy_xp":
                valid_idx = _valid_row_indices_for_group(df, group_spec)
                if valid_idx.size == 0:
                    skipped_no_valid_rows += 1
                    pbar_files.set_postfix_str(
                        f"ok={files_ok} miss={skipped_missing_cols} readerr={skipped_read_error} novalid={skipped_no_valid_rows} gxerr={skipped_gx_error}"
                    )
                    continue
                row_indices = _sample_valid_row_indices(valid_idx, MAX_ROWS_PER_FILE)
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
                    pbar_files.set_postfix_str(
                        f"ok={files_ok} miss={skipped_missing_cols} readerr={skipped_read_error} novalid={skipped_no_valid_rows} gxerr={skipped_gx_error}"
                    )
                    continue
                row_indices = _sample_valid_row_indices(valid_idx, MAX_ROWS_PER_FILE)
                df = df[row_indices]  # keep only sampled rows for GaiaXPy

            files_ok += 1

            # ---- extract features
            if kind == "flux_ivar":
                flux_series = df[group_spec["flux_col"]]
                ivar_series = df[group_spec["ivar_col"]]

                for ridx in row_indices:
                    if total >= MAX_FEATURES_PER_GROUP:
                        break
                    ridx = int(ridx)
                    flux = _cell_to_1d_float(flux_series[ridx])
                    ivar = _cell_to_1d_float(ivar_series[ridx])
                    if flux.size == 0 or ivar.size == 0 or ivar.size != flux.size:
                        continue

                    ivar = np.clip(ivar, a_min=0.0, a_max=None)
                    prec = np.sqrt(ivar)
                    feats = np.stack([flux, prec], axis=-1)

                    feats = _sample_points(feats, MAX_POINTS_PER_ROW)
                    feats = feats[_finite_rows_mask(feats)]
                    if feats.size == 0:
                        continue

                    all_feats.append(feats.astype(np.float32))
                    n_new = int(feats.shape[0])
                    total += n_new
                    pbar_total.update(n_new)

            elif kind == "value_only":
                series = df[group_spec["flux_col"]]
                for ridx in row_indices:
                    if total >= MAX_FEATURES_PER_GROUP:
                        break
                    ridx = int(ridx)
                    arr = _cell_to_1d_float(series[ridx])
                    if arr.size == 0:
                        continue

                    feats = arr.reshape(-1, 1)
                    feats = _sample_points(feats, MAX_POINTS_PER_ROW)
                    feats = feats[_finite_rows_mask(feats)]
                    if feats.size == 0:
                        continue

                    all_feats.append(feats.astype(np.float32))
                    n_new = int(feats.shape[0])
                    total += n_new
                    pbar_total.update(n_new)

            elif kind == "ztf_mag_err":
                mag_series = df[group_spec["mag_col"]]
                err_series = df[group_spec["err_col"]]

                for ridx in row_indices:
                    if total >= MAX_FEATURES_PER_GROUP:
                        break
                    ridx = int(ridx)
                    mag = _cell_to_1d_float(mag_series[ridx])
                    err = _cell_to_1d_float(err_series[ridx])
                    if mag.size == 0 or err.size == 0 or err.size != mag.size:
                        continue

                    err = np.clip(err, a_min=ZTF_ERR_EPS, a_max=None)
                    prec = 1.0 / err
                    feats = np.stack([mag, prec], axis=-1)

                    feats = _sample_points(feats, MAX_POINTS_PER_ROW)
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
                    out_df = _gaiaxpy_run_via_tempfile_raw(df_pd)  # RAW GaiaXPy df
                except Exception as e:
                    skipped_gx_error += 1
                    if skipped_gx_error <= 3:
                        tqdm.write(f"[gaiaxpy warn] {type(e).__name__}: {e}")
                    pbar_files.set_postfix_str(
                        f"ok={files_ok} miss={skipped_missing_cols} readerr={skipped_read_error} novalid={skipped_no_valid_rows} gxerr={skipped_gx_error}"
                    )
                    continue

                if out_df is None or len(out_df) == 0:
                    skipped_no_valid_rows += 1
                    pbar_files.set_postfix_str(
                        f"ok={files_ok} miss={skipped_missing_cols} readerr={skipped_read_error} novalid={skipped_no_valid_rows} gxerr={skipped_gx_error}"
                    )
                    continue

                remaining = MAX_FEATURES_PER_GROUP - total
                if remaining <= 0:
                    break

                xp_band = group_spec.get("xp_band", None)
                feats = _gaiaxpy_out_to_feature_points_2d(out_df, remaining_budget=remaining, xp_band=xp_band)

                if feats.shape[0] == 0:
                    skipped_no_valid_rows += 1
                    pbar_files.set_postfix_str(
                        f"ok={files_ok} miss={skipped_missing_cols} readerr={skipped_read_error} novalid={skipped_no_valid_rows} gxerr={skipped_gx_error}"
                    )
                    continue

                all_feats.append(feats)
                n_new = int(feats.shape[0])
                total += n_new
                pbar_total.update(n_new)

            pbar_files.set_postfix_str(
                f"ok={files_ok} miss={skipped_missing_cols} readerr={skipped_read_error} novalid={skipped_no_valid_rows} gxerr={skipped_gx_error}"
            )

    finally:
        pbar_files.close()
        pbar_total.close()

    if not all_feats:
        D = 2 if kind in ("flux_ivar", "ztf_mag_err", "gaiaxpy_xp") else 1
        return np.zeros((0, D), dtype=np.float32)

    X = np.concatenate(all_feats, axis=0)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X.astype(np.float32)

# =========================
# METRICS
# =========================

def benchmark_group(name, kmeans, X_raw, mean, std):
    if X_raw.shape[0] == 0:
        return {"n_points": 0, "status": "no_data"}

    mean = np.asarray(mean, dtype=np.float32).reshape(1, -1)
    std  = np.asarray(std, dtype=np.float32).reshape(1, -1)
    X_raw = np.asarray(X_raw, dtype=np.float32)

    if X_raw.ndim == 1:
        X_raw = X_raw.reshape(-1, 1)

    X = (X_raw - mean) / (std + 1e-8)

    labels = kmeans.predict(X)
    centers = np.asarray(kmeans.cluster_centers_, dtype=np.float32)

    assigned = centers[labels]
    diff = X - assigned
    sq = np.sum(diff * diff, axis=1)
    dist = np.sqrt(np.maximum(sq, 0.0))

    n = int(X.shape[0])
    inertia_per_point = float(np.mean(sq))
    rmse = float(np.sqrt(inertia_per_point))
    mae_dist = float(np.mean(dist))
    p95_dist = float(np.quantile(dist, 0.95))

    K = int(centers.shape[0])
    counts = np.bincount(labels, minlength=K).astype(np.float64)
    used = int(np.sum(counts > 0))
    p = counts / max(counts.sum(), 1.0)
    p_nz = p[p > 0]
    entropy = float(-np.sum(p_nz * np.log(p_nz)))
    perplexity = float(np.exp(entropy))
    top10_mass = float(np.sort(p)[-10:].sum()) if K >= 10 else float(np.sort(p).sum())

    return {
        "status": "ok",
        "n_points": n,
        "inertia_per_point": inertia_per_point,
        "rmse": rmse,
        "mae_dist": mae_dist,
        "p95_dist": p95_dist,
        "code_usage": used,
        "code_usage_frac": float(used / max(K, 1)),
        "perplexity": perplexity,
        "perplexity_frac": float(perplexity / max(K, 1)),
        "top10_mass": top10_mass,
        "dist_min": float(dist.min()),
        "dist_median": float(np.median(dist)),
        "dist_max": float(dist.max()),
    }

# =========================
# MAIN
# =========================

def main():
    files = list_parquet_files()
    print(f"Found {len(files)} parquet files.")

    cfg = load_tokenizer_config(TOKENIZER_CONFIG)
    group_names_cfg: List[str] = cfg["group_names"]
    group_means: Dict[str, np.ndarray] = cfg["group_means"]
    group_stds: Dict[str, np.ndarray] = cfg["group_stds"]
    group_columns: Dict[str, Dict[str, Any]] = cfg["group_columns"]

    models = load_kmeans_models()
    print(f"Loaded {len(models)} KMeans models from {KMEANS_GLOB}")

    group_names = [g for g in group_names_cfg if g in models]
    missing_models = [g for g in group_names_cfg if g not in models]
    if missing_models:
        print("[warn] Groups in config but missing joblib:", missing_models)

    results: Dict[str, Any] = {
        "config": {
            "INPUT_DIR": INPUT_DIR,
            "FILE_PATTERN": FILE_PATTERN,
            "TOKENIZER_CONFIG": TOKENIZER_CONFIG,
            "MAX_FILES": MAX_FILES,
            "MAX_ROWS_PER_FILE": MAX_ROWS_PER_FILE,
            "MAX_POINTS_PER_ROW": MAX_POINTS_PER_ROW,
            "MAX_FEATURES_PER_GROUP": MAX_FEATURES_PER_GROUP,
            "GAIAXPY_MODE": GAIAXPY_MODE,
            "GAIAXPY_TRUNCATION": GAIAXPY_TRUNCATION,
            "GAIAXPY_WITH_CORRELATION": GAIAXPY_WITH_CORRELATION,
        },
        "groups": {},
    }

    for name in tqdm(group_names, desc="[bench] groups", unit="group"):
        kmeans = models[name]
        spec = dict(group_columns[name])
        spec["name"] = name

        X_raw = collect_features_for_group_bench(spec, files)

        metrics = benchmark_group(
            name=name,
            kmeans=kmeans,
            X_raw=X_raw,
            mean=group_means[name].astype(np.float32),
            std=group_stds[name].astype(np.float32),
        )

        results["groups"][name] = {
            "kind": spec.get("kind"),
            "columns": {k: v for k, v in spec.items() if k != "name"},
            "metrics": metrics,
        }

        print(
            f"[{name}] n={metrics.get('n_points', 0)} "
            f"rmse={metrics.get('rmse', None)} "
            f"usage={metrics.get('code_usage', None)}/{kmeans.n_clusters} "
            f"perp={metrics.get('perplexity', None)}"
        )

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark report to {OUT_JSON}")

if __name__ == "__main__":
    main()