#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import polars as pl


# =========================
# CONFIG
# =========================

INPUT_DIR = "/home/astrodados4/downloads/hypercube"
FILE_PATTERN = "datacube_*.parquet"

OUTPUT_CONFIG = "scalar_tokenizer_config.npz"

N_BINS = 1024

MAX_VALUES_PER_COL = 2_000_000
MAX_VALUES_PER_FILE_PER_COL = 100_000

RNG_SEED = 0

# --- row filter: keep only objects within this r-band range ---
MAG_FILTER_COL = "mag_pstotal_r"
MAG_MIN = 14.0
MAG_MAX = 22.0
APPLY_ROW_FILTER = True  # <--- set False if you ever want to disable this

# --- value-level filter for S-PLUS magnitude errors ---
APPLY_MAGERR_FILTER = True
MAGERR_MAX = 2.0  # keep only 0 < err < 2 for err_mag_pstotal_*


# -------------------------
# YOU CHOOSE SCALAR COLUMNS
# -------------------------

splus_bands = [
    "u", "i", "r", "g", "z",
    "j0378", "j0395", "j0410", "j0430",
    "j0515", "j0660", "j0861",
]

SCALAR_COLUMNS: List[str] = [
    "ellipticity_det",
    "elongation_det",
    "a_pixel_det",
    "b_pixel_det",
    "theta_det",
    "fwhm_n_det",
    *[f"mag_pstotal_{b}" for b in splus_bands],
    *[f"err_mag_pstotal_{b}" for b in splus_bands],
    "gaia_parallax",
    "gaia_parallax_error",
    "gaia_pmra",
    "gaia_pmdec",
    "gaia_pmra_error",
    "gaia_pmdec_error",
    "gaia_phot_bp_mean_flux",
    "gaia_phot_rp_mean_flux",
    "gaia_phot_g_mean_flux",
    "gaia_phot_bp_mean_flux_error",
    "gaia_phot_rp_mean_flux_error",
    "gaia_phot_g_mean_flux_error",
    "gaia_teff_gspphot",
    "gaia_logg_gspphot",
    "gaia_mh_gspphot",
    "specz_z",
    "specz_e_z",
    "vista_yapermag6",
    "vista_yapermag6err",
    "vista_japermag6",
    "vista_japermag6err",
    "vista_hapermag6",
    "vista_hapermag6err",
    "vista_ksapermag6",
    "vista_ksapermag6err",
]

ALLOW_BOOL = False

EXCLUDE_PREFIXES = (
    "splus_cut_",
    "desi_wave_",
    "desi_flux_",
    "desi_ivar_",
    "ztf_mag_",
    "ztf_magerr_",
    "ztf_hmjd_",
)
EXCLUDE_SUBSTRINGS = (
    "gaiaxp_bp_coefficients",
    "gaiaxp_rp_coefficients",
)

MIN_VALUES_TO_FIT = 1000
EDGE_EPS = 1e-8


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


def is_excluded(col: str) -> bool:
    for pref in EXCLUDE_PREFIXES:
        if col.startswith(pref):
            return True
    for sub in EXCLUDE_SUBSTRINGS:
        if sub in col:
            return True
    return False


def is_splus_mag(col: str) -> bool:
    # apply value-level cut (14..22) only to these columns
    return col.startswith("mag_pstotal_")


def is_splus_magerr(col: str) -> bool:
    return col.startswith("err_mag_pstotal_")


def dtype_is_allowed_scalar(dt: pl.DataType) -> bool:
    # Polars dtype checks WITHOUT polars.datatypes.is_numeric
    if dt.is_numeric():
        return True
    if ALLOW_BOOL and dt == pl.Boolean:
        return True
    return False


def read_schema_fast(path: Path) -> Dict[str, pl.DataType]:
    df0 = pl.read_parquet(path, n_rows=0)
    return {c: df0.schema[c] for c in df0.columns}


def find_first_file_with_columns(files: List[Path], cols: List[str]) -> Optional[Path]:
    need = set(cols)
    for f in files:
        try:
            sch = read_schema_fast(f)
        except Exception:
            continue
        if need.issubset(set(sch.keys())):
            return f
    return None


def check_columns_exist_somewhere(files: List[Path], requested: List[str]) -> Tuple[List[str], List[str]]:
    """
    Returns (present_somewhere, missing_everywhere) without building union schema.
    """
    missing = set(requested)
    present = set()

    for f in files:
        if not missing:
            break
        try:
            sch = read_schema_fast(f)
        except Exception:
            continue
        have = set(sch.keys())
        hit = missing.intersection(have)
        if hit:
            present.update(hit)
            missing.difference_update(hit)

    present_list = [c for c in requested if c in present]
    missing_list = [c for c in requested if c in missing]
    return present_list, missing_list


def make_strictly_increasing_edges(edges: np.ndarray, eps: float = EDGE_EPS) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.float64)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps
    return edges


def collect_values_for_column(
    col: str,
    files: List[Path],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Collect values for ONE column across many files, with:
      - optional row filtering by MAG_FILTER_COL in (MAG_MIN, MAG_MAX)
      - per-file subsampling
      - global cap
      - extra value-level cut for S-PLUS magnitude columns (mag_pstotal_*)
      - extra value-level cut for S-PLUS magnitude error columns (err_mag_pstotal_*): 0 < err < MAGERR_MAX
    """
    chunks: List[np.ndarray] = []
    total = 0

    for path in files:
        if total >= MAX_VALUES_PER_COL:
            break

        # read column plus filter column (so filter applies at the row-level)
        cols_to_read = [col]
        if APPLY_ROW_FILTER and MAG_FILTER_COL not in cols_to_read:
            cols_to_read.append(MAG_FILTER_COL)

        try:
            df = pl.read_parquet(path, columns=cols_to_read)
        except Exception:
            continue

        # apply row filter (mag_pstotal_r)
        if APPLY_ROW_FILTER:
            if MAG_FILTER_COL not in df.columns:
                continue

            # keep only finite r and within range
            df = df.filter(
                pl.col(MAG_FILTER_COL).is_finite()
                & (pl.col(MAG_FILTER_COL) > MAG_MIN)
                & (pl.col(MAG_FILTER_COL) < MAG_MAX)
            )
            if df.height == 0:
                continue

        if col not in df.columns:
            continue

        s = df[col]
        dt = s.dtype

        # dtype gate (some columns may drift)
        if not dtype_is_allowed_scalar(dt):
            continue

        # Convert to numpy float64 + finite filtering for floats
        try:
            if dt == pl.Boolean:
                arr = s.cast(pl.UInt8).to_numpy().astype(np.float64, copy=False)
            elif dt.is_float():
                arr = s.to_numpy()
                arr = arr[np.isfinite(arr)]
                arr = arr.astype(np.float64, copy=False)
            elif dt.is_integer():
                arr = s.to_numpy().astype(np.float64, copy=False)
            else:
                continue
        except Exception:
            continue

        if arr.size == 0:
            continue

        # extra value-level cut for S-PLUS mags (keeps bins sane)
        if is_splus_mag(col):
            arr = arr[(arr > MAG_MIN) & (arr < MAG_MAX)]
            if arr.size == 0:
                continue

        # extra value-level cut for S-PLUS mag errors (fix giant outliers)
        if APPLY_MAGERR_FILTER and is_splus_magerr(col):
            arr = arr[(arr > 0.0) & (arr < MAGERR_MAX)]
            if arr.size == 0:
                continue

        # subsample per file
        if arr.size > MAX_VALUES_PER_FILE_PER_COL:
            idx = rng.choice(arr.size, size=MAX_VALUES_PER_FILE_PER_COL, replace=False)
            arr = arr[idx]

        # respect global cap
        remaining = int(MAX_VALUES_PER_COL - total)
        if remaining <= 0:
            break
        if arr.size > remaining:
            idx = rng.choice(arr.size, size=remaining, replace=False)
            arr = arr[idx]

        chunks.append(arr)
        total += arr.size

    if not chunks:
        return np.array([], dtype=np.float64)

    return np.concatenate(chunks, axis=0)


# =========================
# MAIN
# =========================

def fit_scalar_tokenizer():
    rng = np.random.default_rng(RNG_SEED)

    files = list_parquet_files()
    print(f"Found {len(files)} parquet files in {INPUT_DIR} matching {FILE_PATTERN}")

    # quick sanity: ensure row filter column exists somewhere (if enabled)
    if APPLY_ROW_FILTER:
        probe = find_first_file_with_columns(files, [MAG_FILTER_COL])
        if probe is None:
            raise RuntimeError(
                f"Row filter enabled, but '{MAG_FILTER_COL}' was not found in any file.\n"
                "Set APPLY_ROW_FILTER=False or change MAG_FILTER_COL."
            )
        print(f"Row filter enabled: {MAG_MIN} < {MAG_FILTER_COL} < {MAG_MAX}")

    if APPLY_MAGERR_FILTER:
        print(f"Mag-error filter enabled: 0 < err_mag_pstotal_* < {MAGERR_MAX}")

    # filter/exclude requested columns
    requested = [c for c in SCALAR_COLUMNS if (not is_excluded(c))]
    present, missing = check_columns_exist_somewhere(files, requested)

    if missing:
        print("\n[warn] Requested but not found in ANY file:")
        for c in missing:
            print("  -", c)

    scalar_cols = present
    print(f"\nSelected for tokenization (exists somewhere & not excluded): {len(scalar_cols)}")

    if not scalar_cols:
        raise RuntimeError("No scalar columns selected for tokenization after existence check.")

    # Precompute quantile positions once
    quantiles = np.linspace(0.0, 1.0, N_BINS + 1)

    bin_edges: Dict[str, np.ndarray] = {}
    col_means: Dict[str, float] = {}
    col_stds: Dict[str, float] = {}
    col_dtypes: Dict[str, str] = {}

    for col in scalar_cols:
        print(f"\nFitting tokenizer for column: {col}")
        vals = collect_values_for_column(col, files, rng=rng)
        n = int(vals.size)
        print(f"  -> collected {n} values (after caps/subsampling/filter)")

        if n < MIN_VALUES_TO_FIT:
            print(f"  -> too few values (<{MIN_VALUES_TO_FIT}); skipping this column.")
            continue

        vals_sorted = np.sort(vals)

        mean = float(np.mean(vals_sorted))
        std = float(np.std(vals_sorted) + 1e-8)
        col_means[col] = mean
        col_stds[col] = std

        edges = np.quantile(vals_sorted, quantiles)
        edges = make_strictly_increasing_edges(edges, eps=EDGE_EPS)

        bin_edges[col] = edges.astype(np.float64, copy=False)

        # record dtype from the *first file where it exists*
        dt_str = "unknown"
        for f in files:
            try:
                sch = read_schema_fast(f)
            except Exception:
                continue
            if col in sch:
                dt_str = str(sch[col])
                break
        col_dtypes[col] = dt_str

        print(f"  -> bin edges: {edges.shape}, min={edges[0]:.6g}, max={edges[-1]:.6g}")

    if not bin_edges:
        raise RuntimeError("No columns produced bin edges (all had too few values?).")

    ensure_dir_for_file(OUTPUT_CONFIG)
    np.savez_compressed(
        OUTPUT_CONFIG,
        bin_edges=bin_edges,
        col_means=col_means,
        col_stds=col_stds,
        col_dtypes=col_dtypes,
        N_BINS=int(N_BINS),
        input_dir=str(INPUT_DIR),
        file_pattern=str(FILE_PATTERN),
        rng_seed=int(RNG_SEED),
        max_values_per_col=int(MAX_VALUES_PER_COL),
        max_values_per_file_per_col=int(MAX_VALUES_PER_FILE_PER_COL),
        allow_bool=bool(ALLOW_BOOL),
        min_values_to_fit=int(MIN_VALUES_TO_FIT),
        edge_eps=float(EDGE_EPS),
        apply_row_filter=bool(APPLY_ROW_FILTER),
        mag_filter_col=str(MAG_FILTER_COL),
        mag_min=float(MAG_MIN),
        mag_max=float(MAG_MAX),
        apply_magerr_filter=bool(APPLY_MAGERR_FILTER),
        magerr_max=float(MAGERR_MAX),
    )

    print(f"\nSaved scalar tokenizer config to {OUTPUT_CONFIG}")
    print(f"Columns saved: {len(bin_edges)}")


if __name__ == "__main__":
    fit_scalar_tokenizer()