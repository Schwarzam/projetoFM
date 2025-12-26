#!/usr/bin/env python3
from __future__ import annotations

import os
# ---- HARD LIMIT BLAS THREADS (prevents OpenBLAS segfault) ----
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("GOTO_NUM_THREADS", "1")

from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import json
import contextlib
import tempfile
import glob

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# =========================
# CONFIG
# =========================

INPUT_DIR = "/home/astrodados4/downloads/hypercube"
FILE_PATTERN = "datacube_*.parquet"

TOKENIZER_CONFIG = "spectrum_tokenizer_config.npz"
OUTPUT_DIR = "spectrum_tokens"

PREFERRED_ID_COLS = ["id"]
TOKEN_DTYPE = np.uint16  # fits up to 65535 codes
SAVE_ROW_INDEX_IF_NO_ID = True

ZTF_ERR_EPS = 1e-3
PAD_TRUNCATE_LENGTH_MISMATCH = True

# If True, only encode rows where the group values exist (List OR GaiaXP str).
# If False, encode all rows and fill missing rows with zeros.
ENCODE_ONLY_VALID_ROWS = True

# ---- GaiaXPy encoding (must match training/benchmark logic) ----
GAIAXPY_MODE = "convert"       # "convert" or "calibrate"
GAIAXPY_SAMPLING = None        # e.g. np.linspace(330, 1050, 600)
GAIAXPY_TRUNCATION = False
GAIAXPY_WITH_CORRELATION = False

# Cap per-row output for GaiaXP (recommended)
MAX_POINTS_PER_ROW_GAIAXP = 512  # each flux bin becomes a token; cap keeps files smaller

N_WORKERS = 12  # tune

# =========================
# HELPERS
# =========================

def list_parquet_files() -> List[Path]:
    p = Path(INPUT_DIR)
    files = sorted(p.glob(FILE_PATTERN))
    if not files:
        raise RuntimeError(f"No parquet files found in {INPUT_DIR} matching {FILE_PATTERN}")
    return files

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def read_schema_fast(path: Path) -> List[str]:
    df0 = pl.read_parquet(path, n_rows=0)
    return df0.columns

def choose_id_column(columns: List[str]) -> Optional[str]:
    for c in PREFERRED_ID_COLS:
        if c in columns:
            return c
    return None

def _to_str(x: Any) -> str:
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode()
    return str(x)

def _normalize_dict_keys(d: Dict[Any, Any]) -> Dict[str, Any]:
    return {_to_str(k): v for k, v in d.items()}

def load_tokenizer_config(config_path: str):
    data = np.load(config_path, allow_pickle=True)

    raw_names = data["group_names"]
    group_names = [_to_str(n) for n in raw_names]

    group_means = _normalize_dict_keys(data["group_means"].item())
    group_stds = _normalize_dict_keys(data["group_stds"].item())
    group_codebooks = _normalize_dict_keys(data["group_codebooks"].item())
    codebook_size = int(data["CODEBOOK_SIZE"])

    group_columns: Dict[str, Dict[str, Any]] = {}
    if "group_columns" in data.files:
        group_columns = _normalize_dict_keys(data["group_columns"].item())
        norm2 = {}
        for gname, spec in group_columns.items():
            if isinstance(spec, dict):
                spec2 = {}
                for k, v in spec.items():
                    kk = _to_str(k)
                    if isinstance(v, (bytes, np.bytes_)):
                        vv = v.decode()
                    else:
                        vv = v
                    spec2[kk] = vv
                norm2[gname] = spec2
            else:
                norm2[gname] = spec
        group_columns = norm2

    for k, v in list(group_means.items()):
        group_means[k] = np.asarray(v, dtype=np.float32)
    for k, v in list(group_stds.items()):
        group_stds[k] = np.asarray(v, dtype=np.float32)
    for k, v in list(group_codebooks.items()):
        group_codebooks[k] = np.asarray(v, dtype=np.float32)

    return group_names, group_means, group_stds, group_codebooks, group_columns, codebook_size

def _cell_to_1d_float(cell) -> np.ndarray:
    """
    Supports:
      - list/tuple/ndarray
      - GaiaXP-style Utf8 where cell is a quoted JSON string: "\"[... ]\""
      - Utf8 that is directly JSON list: "[... ]"
    """
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

        # First decode JSON string wrapper if present (returns list OR inner string)
        try:
            decoded = json.loads(s)
            if isinstance(decoded, str):
                s = decoded.strip()
            else:
                arr = np.asarray(decoded, dtype=np.float32).reshape(-1)
                return arr[np.isfinite(arr)]
        except Exception:
            pass

        # Then parse actual list
        try:
            vals = json.loads(s)
            arr = np.asarray(vals, dtype=np.float32).reshape(-1)
            return arr[np.isfinite(arr)]
        except Exception:
            return np.zeros((0,), dtype=np.float32)

    return np.zeros((0,), dtype=np.float32)

def _pad_or_truncate_1d(x: np.ndarray, L: int) -> np.ndarray:
    if x.size == L:
        return x
    if x.size > L:
        return x[:L]
    out = np.zeros(L, dtype=x.dtype)
    out[: x.size] = x
    return out

def _mask_list_nonempty(col: str) -> pl.Expr:
    e = pl.col(col)
    return e.is_not_null() & (e.list.len() > 0)

def _mask_str_nonempty(col: str) -> pl.Expr:
    e = pl.col(col)
    s = e.cast(pl.Utf8).str.strip_chars()
    return e.is_not_null() & (s.str.len_chars() > 2) & (s != "[]") & s.str.contains(r"\[")

def _nonempty_cell_mask(df: pl.DataFrame, col: str) -> pl.Expr:
    """
    Prefer list semantics (DESI). If list ops fail, fall back to Utf8 list-in-string semantics (GaiaXP).
    """
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

        # If both are lists, enforce equal length (skip constraint if not list)
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
        # stricter mask using resolved colmap is applied later
        mask = pl.all_horizontal([pl.lit(True)])

    else:
        raise ValueError(f"Unknown group kind: {kind}")

    idx = df.select(pl.arg_where(mask)).to_series().to_numpy()
    return idx.astype(np.int64)

# =========================
# GaiaXPy integration (BP/RP split)
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
    Always synthesize covariance matrices (GaiaXPy can be picky).
    """
    import pandas as pd

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

    # optional errors
    bp_err = None
    rp_err = None
    if "bp_coefficient_errors" in colmap and colmap["bp_coefficient_errors"] in df.columns:
        bp_err = [as_list1d(x) for x in df[colmap["bp_coefficient_errors"]].to_list()]
    if "rp_coefficient_errors" in colmap and colmap["rp_coefficient_errors"] in df.columns:
        rp_err = [as_list1d(x) for x in df[colmap["rp_coefficient_errors"]].to_list()]

    # optional stddev
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

    # covariance matrices (force existence; GaiaXPy is picky)
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

def _gaiaxpy_run_via_tempfile(df_pd):
    """
    Returns RAW GaiaXPy out_df (pandas DataFrame).
    We accept both LONG (source_id,xp,flux,flux_error) and WIDE numeric formats.
    """
    import pandas as pd

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

        # 1) preferred: in-memory return
        try:
            with _suppress_all_output():
                if mode == "convert":
                    from gaiaxpy import convert
                    if GAIAXPY_SAMPLING is None:
                        out_df, _ = convert(in_csv, truncation=GAIAXPY_TRUNCATION, save_file=False)
                    else:
                        out_df, _ = convert(in_csv, sampling=GAIAXPY_SAMPLING,
                                            truncation=GAIAXPY_TRUNCATION, save_file=False)
                else:
                    from gaiaxpy import calibrate
                    if GAIAXPY_SAMPLING is None:
                        out_df, _ = calibrate(in_csv, truncation=GAIAXPY_TRUNCATION,
                                              save_file=False, with_correlation=GAIAXPY_WITH_CORRELATION)
                    else:
                        out_df, _ = calibrate(in_csv, sampling=GAIAXPY_SAMPLING,
                                              truncation=GAIAXPY_TRUNCATION, save_file=False,
                                              with_correlation=GAIAXPY_WITH_CORRELATION)
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
                        convert(in_csv, sampling=GAIAXPY_SAMPLING,
                                truncation=GAIAXPY_TRUNCATION, save_file=True)
                else:
                    from gaiaxpy import calibrate
                    if GAIAXPY_SAMPLING is None:
                        calibrate(in_csv, truncation=GAIAXPY_TRUNCATION,
                                  save_file=True, with_correlation=GAIAXPY_WITH_CORRELATION)
                    else:
                        calibrate(in_csv, sampling=GAIAXPY_SAMPLING,
                                  truncation=GAIAXPY_TRUNCATION, save_file=True,
                                  with_correlation=GAIAXPY_WITH_CORRELATION)

        candidates = sorted(
            glob.glob(os.path.join(td, "**", "*.csv"), recursive=True),
            key=os.path.getmtime
        )
        candidates = [c for c in candidates if os.path.basename(c) != "xp_input.csv"]
        if not candidates:
            return None

        newest = candidates[-1]
        return pd.read_csv(newest)

def _gaiaxp_extract_flux_by_source_and_band(out_df, xp_band: str) -> Dict[int, np.ndarray]:
    """
    Returns dict: source_id -> 1D float32 flux array (finite-only).
    Works with GaiaXPy LONG format:
      columns: source_id, xp, flux, flux_error
    """
    if out_df is None or len(out_df) == 0:
        return {}

    # LONG format
    if "flux" in out_df.columns and "source_id" in out_df.columns:
        band = str(xp_band).strip().lower()
        if "xp" in out_df.columns:
            xpcol = out_df["xp"].astype(str).str.lower()
            m = xpcol == band
            sub = out_df.loc[m]
        else:
            sub = out_df

        # group by source_id and concatenate
        mapping: Dict[int, List[np.ndarray]] = {}
        for sid, flux_cell in zip(sub["source_id"].values, sub["flux"].values):
            sid = int(sid)
            arr = _cell_to_1d_float(flux_cell)
            if arr.size == 0:
                continue
            mapping.setdefault(sid, []).append(arr)

        out: Dict[int, np.ndarray] = {}
        for sid, chunks in mapping.items():
            flat = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
            flat = flat[np.isfinite(flat)]
            if flat.size:
                out[sid] = flat
        return out

    # WIDE numeric fallback (rare): treat each row as a source_id spectrum
    if "source_id" in out_df.columns:
        num_cols = []
        for c in out_df.columns:
            if c == "source_id":
                continue
            dt = getattr(out_df[c], "dtype", None)
            kind = getattr(dt, "kind", None)
            if kind in ("f", "i"):
                num_cols.append(c)
        if not num_cols:
            return {}
        X = out_df[num_cols].to_numpy(dtype=np.float32, copy=False)
        out: Dict[int, np.ndarray] = {}
        for sid, row in zip(out_df["source_id"].values, X):
            arr = np.asarray(row, dtype=np.float32).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                out[int(sid)] = arr
        return out

    return {}

def _tokenize_scalar_vector(values: np.ndarray, mean: np.ndarray, std: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    values: (L,) -> tokens (L,)
    mean/std: (1,) or (1,1)
    centers: (K,1)
    """
    v = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    mean = np.asarray(mean, dtype=np.float32).reshape(1, -1)
    std  = np.asarray(std, dtype=np.float32).reshape(1, -1)
    std = np.where(std == 0, 1e-8, std)

    x = (v - mean) / std  # (L,1)
    centers = np.asarray(centers, dtype=np.float32)  # (K,1)
    c2 = np.sum(centers * centers, axis=1, keepdims=True).T  # (1,K)

    x2 = np.sum(x * x, axis=1, keepdims=True)   # (L,1)
    xc = x @ centers.T                          # (L,K)
    dist2 = x2 + c2 - 2.0 * xc                  # (L,K)
    return np.argmin(dist2, axis=1).astype(TOKEN_DTYPE)

# =========================
# TOKENIZATION (DESI/ZTF + GaiaXP)
# =========================

def _process_one_file(path: Path) -> Tuple[str, bool, str]:
    """
    Returns (filename, ok, message)
    """
    try:
        out_path = Path(OUTPUT_DIR) / (path.stem + "_spectrum_tokens.npz")
        if out_path.exists():
            return (Path(path).name, True, f"already exists: {out_path.name}")

        # (everything you currently do inside `for path in files:` goes here)
        # IMPORTANT: load tokenizer config inside worker OR pass only the config path.
        group_names, group_means, group_stds, group_codebooks, group_columns, codebook_size = load_tokenizer_config(TOKENIZER_CONFIG)

        p = Path(path)

        cols_in_file = read_schema_fast(path)
        cols_set = set(cols_in_file)
        id_col = choose_id_column(cols_in_file)

        gaia_colmap = _resolve_gaiaxp_continuous_columns(cols_in_file)

        available_group_specs: List[Dict[str, Any]] = []
        for gname in group_names:
            gname = _to_str(gname)
            spec = group_columns.get(gname)
            if not isinstance(spec, dict):
                continue

            kind = spec.get("kind")
            spec = dict(spec)
            spec["name"] = gname

            if kind == "flux_ivar":
                need = [spec["flux_col"], spec["ivar_col"]]
                if all(c in cols_set for c in need):
                    available_group_specs.append(spec)

            elif kind == "value_only":
                need = [spec["flux_col"]]
                if all(c in cols_set for c in need):
                    available_group_specs.append(spec)

            elif kind == "ztf_mag_err":
                need = [spec["mag_col"], spec["err_col"]]
                if all(c in cols_set for c in need):
                    available_group_specs.append(spec)

            elif kind == "gaiaxpy_xp":
                if gaia_colmap is None:
                    continue
                if "xp_band" not in spec:
                    continue
                spec["_gaia_colmap"] = gaia_colmap
                available_group_specs.append(spec)

        if not available_group_specs:
            return (p.name, False, "no groups present")

        cols_to_read: List[str] = []
        if id_col is not None:
            cols_to_read.append(id_col)

        for spec in available_group_specs:
            kind = spec["kind"]
            if kind == "flux_ivar":
                cols_to_read.extend([spec["flux_col"], spec["ivar_col"]])
            elif kind == "value_only":
                cols_to_read.append(spec["flux_col"])
            elif kind == "ztf_mag_err":
                cols_to_read.extend([spec["mag_col"], spec["err_col"]])
            elif kind == "gaiaxpy_xp":
                assert gaia_colmap is not None
                cols_to_read.extend([gaia_colmap[k] for k in gaia_colmap.keys()])

        cols_to_read = list(dict.fromkeys(cols_to_read))
        df = pl.read_parquet(path, columns=cols_to_read)

        n_rows = df.height

        out_data: Dict[str, Any] = {}
        if id_col is not None and id_col in df.columns:
            out_data["ids"] = df[id_col].to_numpy()
            out_data["id_col"] = np.array([id_col], dtype=object)
        else:
            out_data["ids"] = np.array([], dtype=np.int64)
            out_data["id_col"] = np.array([""], dtype=object)
            if SAVE_ROW_INDEX_IF_NO_ID:
                out_data["row_index"] = np.arange(n_rows, dtype=np.int64)

        for spec in available_group_specs:
            name = spec["name"]
            payload = compute_tokens_for_group_in_file(
                df=df,
                group_spec=spec,
                mean=group_means[name],
                std=group_stds[name],
                centers=group_codebooks[name],
            )
            if "tokens" in payload:
                out_data[f"tokens_{name}"] = payload["tokens"]
            if "tokens_flat" in payload and "tokens_indptr" in payload:
                out_data[f"tokens_{name}_flat"] = payload["tokens_flat"]
                out_data[f"tokens_{name}_indptr"] = payload["tokens_indptr"]

        ensure_dir(OUTPUT_DIR)
        out_path = Path(OUTPUT_DIR) / (p.stem + "_spectrum_tokens.npz")
        np.savez_compressed(out_path, **out_data)

        return (p.name, True, f"saved {out_path.name}")

    except Exception as e:
        return (Path(path).name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

def compute_tokens_for_group_in_file(
    df: pl.DataFrame,
    group_spec: Dict[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
    centers: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Returns a dict of arrays to store for this group.

    - flux_ivar/value_only/ztf_mag_err: returns {"tokens": (n_rows, L)}
    - gaiaxpy_xp: returns ragged:
        {
          "tokens_flat": (N_total,),
          "tokens_indptr": (n_rows+1,),
        }
      where row r tokens are tokens_flat[indptr[r]:indptr[r+1]].

    If ENCODE_ONLY_VALID_ROWS=True:
      - for rectangular groups: invalid rows are all-zeros
      - for gaiaxpy_xp: invalid rows have empty spans
    """
    name = group_spec.get("name", "")
    kind = group_spec["kind"]
    n_rows = df.height
    if n_rows == 0:
        return {"tokens": np.zeros((0, 0), dtype=TOKEN_DTYPE)}

    # ---------- DESI / ZTF ----------
    if kind in ("flux_ivar", "value_only", "ztf_mag_err"):
        std = np.asarray(std, dtype=np.float32)
        std = np.where(std == 0, 1e-8, std)
        mean = np.asarray(mean, dtype=np.float32)
        centers = np.asarray(centers, dtype=np.float32)
        c2 = np.sum(centers * centers, axis=1, keepdims=True).T  # (1,K)

        valid_idx = None
        if ENCODE_ONLY_VALID_ROWS:
            valid_idx = _valid_row_indices_for_group(df, group_spec)
            if valid_idx.size == 0:
                return {"tokens": np.zeros((n_rows, 0), dtype=TOKEN_DTYPE)}

        def _infer_L_from_indices(getter, indices: np.ndarray) -> Optional[int]:
            for i in indices:
                arr = getter(int(i))
                if arr.size > 0:
                    return int(arr.size)
            return None

        if kind == "flux_ivar":
            flux_col = group_spec["flux_col"]
            ivar_col = group_spec["ivar_col"]
            flux_series = df[flux_col]
            ivar_series = df[ivar_col]

            def get_flux(i: int) -> np.ndarray:
                return _cell_to_1d_float(flux_series[i])

            def get_ivar(i: int) -> np.ndarray:
                return _cell_to_1d_float(ivar_series[i])

            idx_for_L = valid_idx if valid_idx is not None else np.arange(n_rows, dtype=np.int64)
            L = _infer_L_from_indices(get_flux, idx_for_L)
            if L is None:
                return {"tokens": np.zeros((n_rows, 0), dtype=TOKEN_DTYPE)}

            tokens = np.zeros((n_rows, L), dtype=TOKEN_DTYPE)
            loop_indices = valid_idx if valid_idx is not None else np.arange(n_rows, dtype=np.int64)

            for i in tqdm(loop_indices, desc=f"  [tokens {name}] rows", leave=False):
                i = int(i)
                flux = get_flux(i)
                ivar = get_ivar(i)
                if flux.size == 0 or ivar.size == 0:
                    continue

                if flux.size != L:
                    if PAD_TRUNCATE_LENGTH_MISMATCH:
                        flux = _pad_or_truncate_1d(flux, L)
                    else:
                        continue
                if ivar.size != L:
                    if PAD_TRUNCATE_LENGTH_MISMATCH:
                        ivar = _pad_or_truncate_1d(ivar, L)
                    else:
                        continue

                ivar = np.clip(ivar, a_min=0.0, a_max=None)
                prec = np.sqrt(ivar)
                feats = np.stack([flux, prec], axis=-1)          # (L,2)
                feats_std = (feats - mean) / std                 # (L,2)

                x = feats_std.astype(np.float32, copy=False)
                x2 = np.sum(x * x, axis=1, keepdims=True)        # (L,1)
                xc = x @ centers.T                               # (L,K)
                dist2 = x2 + c2 - 2.0 * xc                       # (L,K)
                tokens[i, :] = np.argmin(dist2, axis=1).astype(TOKEN_DTYPE)

            return {"tokens": tokens}

        if kind == "value_only":
            flux_col = group_spec["flux_col"]
            series = df[flux_col]

            def get_arr(i: int) -> np.ndarray:
                return _cell_to_1d_float(series[i])

            idx_for_L = valid_idx if valid_idx is not None else np.arange(n_rows, dtype=np.int64)
            L = _infer_L_from_indices(get_arr, idx_for_L)
            if L is None:
                return {"tokens": np.zeros((n_rows, 0), dtype=TOKEN_DTYPE)}

            tokens = np.zeros((n_rows, L), dtype=TOKEN_DTYPE)
            loop_indices = valid_idx if valid_idx is not None else np.arange(n_rows, dtype=np.int64)

            for i in tqdm(loop_indices, desc=f"  [tokens {name}] rows", leave=False):
                i = int(i)
                arr = get_arr(i)
                if arr.size == 0:
                    continue
                if arr.size != L:
                    if PAD_TRUNCATE_LENGTH_MISMATCH:
                        arr = _pad_or_truncate_1d(arr, L)
                    else:
                        continue

                feats = arr.reshape(-1, 1)
                feats_std = (feats - mean) / std

                x = feats_std.astype(np.float32, copy=False)
                x2 = np.sum(x * x, axis=1, keepdims=True)
                xc = x @ centers.T
                dist2 = x2 + c2 - 2.0 * xc
                tokens[i, :] = np.argmin(dist2, axis=1).astype(TOKEN_DTYPE)

            return {"tokens": tokens}

        if kind == "ztf_mag_err":
            mag_col = group_spec["mag_col"]
            err_col = group_spec["err_col"]
            mag_series = df[mag_col]
            err_series = df[err_col]

            def get_mag(i: int) -> np.ndarray:
                return _cell_to_1d_float(mag_series[i])

            def get_err(i: int) -> np.ndarray:
                return _cell_to_1d_float(err_series[i])

            idx_for_L = valid_idx if valid_idx is not None else np.arange(n_rows, dtype=np.int64)
            L = _infer_L_from_indices(get_mag, idx_for_L)
            if L is None:
                return {"tokens": np.zeros((n_rows, 0), dtype=TOKEN_DTYPE)}

            tokens = np.zeros((n_rows, L), dtype=TOKEN_DTYPE)
            loop_indices = valid_idx if valid_idx is not None else np.arange(n_rows, dtype=np.int64)

            for i in tqdm(loop_indices, desc=f"  [tokens {name}] rows", leave=False):
                i = int(i)
                mag = get_mag(i)
                err = get_err(i)
                if mag.size == 0 or err.size == 0:
                    continue

                if mag.size != L:
                    if PAD_TRUNCATE_LENGTH_MISMATCH:
                        mag = _pad_or_truncate_1d(mag, L)
                    else:
                        continue
                if err.size != L:
                    if PAD_TRUNCATE_LENGTH_MISMATCH:
                        err = _pad_or_truncate_1d(err, L)
                    else:
                        continue

                err = np.clip(err, a_min=ZTF_ERR_EPS, a_max=None)
                prec = 1.0 / err
                feats = np.stack([mag, prec], axis=-1)
                feats_std = (feats - mean) / std

                x = feats_std.astype(np.float32, copy=False)
                x2 = np.sum(x * x, axis=1, keepdims=True)
                xc = x @ centers.T
                dist2 = x2 + c2 - 2.0 * xc
                tokens[i, :] = np.argmin(dist2, axis=1).astype(TOKEN_DTYPE)

            return {"tokens": tokens}

    # ---------- GaiaXP (BP/RP split) ----------
    if kind == "gaiaxpy_xp":
        xp_band = group_spec.get("xp_band", None)
        if xp_band is None:
            raise ValueError(f"[{name}] gaiaxpy_xp requires spec['xp_band'] = 'BP' or 'RP'")

        # We'll produce ragged tokens for all rows.
        indptr = np.zeros(n_rows + 1, dtype=np.int64)
        flat_tokens: List[np.ndarray] = []

        # stricter mask using resolved GaiaXP columns
        colmap = group_spec.get("_gaia_colmap", None)
        if colmap is None:
            raise RuntimeError(f"[{name}] internal error: missing _gaia_colmap in group_spec")

        mask = (
            _nonempty_cell_mask(df, colmap["bp_coefficients"])
            & _nonempty_cell_mask(df, colmap["rp_coefficients"])
            & _nonempty_cell_mask(df, colmap["bp_coefficient_correlations"])
            & _nonempty_cell_mask(df, colmap["rp_coefficient_correlations"])
            & pl.col(colmap["bp_n_parameters"]).is_not_null()
            & pl.col(colmap["rp_n_parameters"]).is_not_null()
        )
        valid_idx = df.select(pl.arg_where(mask)).to_series().to_numpy().astype(np.int64)

        if ENCODE_ONLY_VALID_ROWS and valid_idx.size == 0:
            return {"tokens_flat": np.zeros((0,), dtype=TOKEN_DTYPE), "tokens_indptr": indptr}

        # Convert valid subset to GaiaXPy input, run once, then map flux per source_id
        # (This is MUCH faster than per-row GaiaXPy calls.)
        import pandas as pd  # lazy import

        df_valid = df[valid_idx] if ENCODE_ONLY_VALID_ROWS else df
        df_pd = _polars_to_pandas_gaiaxpy_input(df_valid, colmap)

        try:
            out_df = _gaiaxpy_run_via_tempfile(df_pd)
        except Exception as e:
            tqdm.write(f"[gaiaxpy] {name} failed: {type(e).__name__}: {e}")
            return {"tokens_flat": np.zeros((0,), dtype=TOKEN_DTYPE), "tokens_indptr": indptr}

        if out_df is None:
            return {"tokens_flat": np.zeros((0,), dtype=TOKEN_DTYPE), "tokens_indptr": indptr}

        flux_map = _gaiaxp_extract_flux_by_source_and_band(out_df, xp_band=str(xp_band).strip())

        # For writing tokens, we need the original row order.
        # We only have source_id in df_pd, but we can index df_valid rows 1:1.
        valid_source_ids = df_valid[colmap["source_id"]].cast(pl.Int64).to_numpy()

        # build a row->flux mapping for valid rows
        row_flux: Dict[int, np.ndarray] = {}
        for local_i, sid in enumerate(valid_source_ids):
            sid_int = int(sid)
            arr = flux_map.get(sid_int, None)
            if arr is None or arr.size == 0:
                continue
            if MAX_POINTS_PER_ROW_GAIAXP is not None and arr.size > int(MAX_POINTS_PER_ROW_GAIAXP):
                # keep a consistent "cover the spectrum" sampling
                pick = np.linspace(0, arr.size - 1, int(MAX_POINTS_PER_ROW_GAIAXP)).astype(np.int64)
                arr = arr[pick]
            row_flux[int(valid_idx[local_i]) if ENCODE_ONLY_VALID_ROWS else int(local_i)] = arr

        # tokenize per row (ragged)
        total = 0
        for r in range(n_rows):
            indptr[r] = total
            arr = row_flux.get(r, None)
            if arr is None or arr.size == 0:
                continue
            toks = _tokenize_scalar_vector(arr, mean=mean, std=std, centers=centers)
            flat_tokens.append(toks)
            total += int(toks.size)
        indptr[n_rows] = total

        if total == 0:
            return {"tokens_flat": np.zeros((0,), dtype=TOKEN_DTYPE), "tokens_indptr": indptr}

        flat = np.concatenate(flat_tokens, axis=0).astype(TOKEN_DTYPE, copy=False)
        return {"tokens_flat": flat, "tokens_indptr": indptr}

    raise ValueError(f"Unknown group kind: {kind}")

# =========================
# MAIN ENCODING
# =========================

def encode_spectrum_tokens():
    files_all = list_parquet_files()
    ensure_dir(OUTPUT_DIR)

    files = []
    skipped = 0
    for f in files_all:
        out_path = Path(OUTPUT_DIR) / (f.stem + "_spectrum_tokens.npz")
        if out_path.exists():
            skipped += 1
            continue
        files.append(f)

    print(f"Found {len(files_all)} parquet files.")
    print(f"Skipping {skipped} already-encoded outputs in {OUTPUT_DIR}.")
    print(f"To process: {len(files)}")
    print(f"Parallel workers: {N_WORKERS}")

    if not files:
        print("Nothing to do.")
        return

    ok = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(_process_one_file, f) for f in files]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="files"):
            fname, success, msg = fut.result()
            if success:
                ok += 1
            else:
                print(f"[warn] {fname}: {msg}")

    print(f"Done. OK: {ok}/{len(files)}")


if __name__ == "__main__":
    encode_spectrum_tokens()