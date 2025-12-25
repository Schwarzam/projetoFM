#!/usr/bin/env python3
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import polars as pl

# =========================
# CONFIG
# =========================

# Folder where your hypercube parquet tables live
INPUT_DIR = "/home/astrodados4/downloads/hypercube"

# File pattern for hypercube tables
FILE_PATTERN = "datacube_*.parquet"

# Output config with bin edges
OUTPUT_CONFIG = "scalar_tokenizer_config.npz"

# How many bins per scalar (AION-style)
N_BINS = 1024

# To avoid storing billions of values in RAM, subsample per column
MAX_VALUES_PER_COL = 2_000_000           # global cap across all files
MAX_VALUES_PER_FILE_PER_COL = 100_000    # per-file cap

# -------------------------
# YOU CHOOSE SCALAR COLUMNS
# -------------------------
# Put the scalar columns you want here. They may appear only in some files.
# If empty, the script will auto-discover numeric scalar columns from the UNION schema of all files.
SCALAR_COLUMNS: List[str] = [
    # Examples:
    # "ra", "dec", "r_med_photogeo",
    # "mag_psf_u", "mag_psf_g", "mag_psf_r",
]

# If True, boolean columns are allowed (they'll be tokenized as 0/1)
ALLOW_BOOL = False

# Exclusion rules (apply in BOTH manual and auto mode)
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def is_excluded(col: str) -> bool:
    for pref in EXCLUDE_PREFIXES:
        if col.startswith(pref):
            return True
    for sub in EXCLUDE_SUBSTRINGS:
        if sub in col:
            return True
    return False


def read_schema_fast(path: Path) -> Dict[str, pl.DataType]:
    """
    Read schema without loading data. n_rows=0 is typically cheap.
    """
    df0 = pl.read_parquet(path, n_rows=0)
    return dict(zip(df0.columns, [df0[c].dtype for c in df0.columns]))


def union_schemas(files: List[Path]) -> Dict[str, pl.DataType]:
    """
    Build a union schema: col -> dtype (first-seen dtype).
    If same col appears with different dtypes across files, we keep first and warn.
    """
    union: Dict[str, pl.DataType] = {}
    for f in files:
        sch = read_schema_fast(f)
        for c, dt in sch.items():
            if c not in union:
                union[c] = dt
            else:
                # dtype drift warning
                if union[c] != dt:
                    # keep first dtype; warn once per file occurrence
                    print(f"[warn] dtype drift for '{c}': seen {union[c]} then {dt} in {f.name} (keeping {union[c]})")
    return union


def dtype_is_allowed_scalar(dt: pl.DataType) -> bool:
    """
    Decide if a dtype is a scalar numeric we want to tokenize.
    """
    if dt.is_numeric():
        return True
    if ALLOW_BOOL and dt == pl.Boolean:
        return True
    return False


def autodiscover_scalar_columns(schema: Dict[str, pl.DataType]) -> List[str]:
    cols = []
    for c, dt in schema.items():
        if is_excluded(c):
            continue
        if dtype_is_allowed_scalar(dt):
            cols.append(c)
    return sorted(cols)


def filter_requested_columns(
    requested: List[str],
    schema: Dict[str, pl.DataType],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (selected_cols, missing_cols, excluded_cols)

    - missing: not present in union schema
    - excluded: present but excluded by name rules
    - selected: present, not excluded, allowed dtype
    """
    selected = []
    missing = []
    excluded = []

    for c in requested:
        if c not in schema:
            missing.append(c)
            continue
        if is_excluded(c):
            excluded.append(c)
            continue
        dt = schema[c]
        if not dtype_is_allowed_scalar(dt):
            excluded.append(f"{c} (dtype={dt})")
            continue
        selected.append(c)

    return selected, missing, excluded


# =========================
# MAIN TRAINING LOGIC
# =========================

def collect_values_for_column(col: str, files: List[Path]) -> np.ndarray:
    """
    Stream over all files and collect numeric values for one column,
    with subsampling to control memory.

    Important: if the column doesn't exist in a file, it is skipped.
    """
    all_vals = []
    total_collected = 0

    for path in files:
        if total_collected >= MAX_VALUES_PER_COL:
            break

        # Read only if column exists in that file (cheap schema check)
        try:
            sch = read_schema_fast(path)
        except Exception as e:
            print(f"  [warn] Could not read schema from {path}: {e}")
            continue

        if col not in sch:
            continue

        # Now read column data
        try:
            df = pl.read_parquet(path, columns=[col])
        except Exception as e:
            print(f"  [warn] Could not read column '{col}' from {path}: {e}")
            continue

        s = df[col]

        # bool handling (optional)
        if s.dtype == pl.Boolean:
            arr = s.cast(pl.UInt8).to_numpy().astype(np.float64)
        elif s.dtype.is_float():
            arr = s.to_numpy()
            arr = arr[np.isfinite(arr)]
        elif s.dtype.is_integer():
            arr = s.to_numpy().astype(np.float64)
        else:
            # Shouldn't happen if dtype filtering was done, but safe-guard
            continue

        if arr.size == 0:
            continue

        # subsample from this file
        if arr.size > MAX_VALUES_PER_FILE_PER_COL:
            idx = np.random.choice(arr.size, size=MAX_VALUES_PER_FILE_PER_COL, replace=False)
            arr = arr[idx]

        all_vals.append(arr)
        total_collected += arr.size

    if not all_vals:
        return np.array([], dtype=np.float64)

    return np.concatenate(all_vals, axis=0)


def fit_scalar_tokenizer():
    files = list_parquet_files()
    print("Found parquet files:")
    for f in files:
        print("  -", f)

    print("\nBuilding UNION schema (so we can select columns that are not in the first file)...")
    schema = union_schemas(files)
    print(f"  -> union schema columns: {len(schema)}")

    # Decide which scalar columns to tokenize
    if SCALAR_COLUMNS:
        print("\nUsing user-provided SCALAR_COLUMNS list...")
        selected_cols, missing_cols, excluded_cols = filter_requested_columns(SCALAR_COLUMNS, schema)

        print("\nRequested scalar columns:")
        for c in SCALAR_COLUMNS:
            print("  -", c)

        if selected_cols:
            print("\nSelected for tokenization:")
            for c in selected_cols:
                print("  -", c)

        if missing_cols:
            print("\n[warn] Requested but not found in ANY file (union schema):")
            for c in missing_cols:
                print("  -", c)

        if excluded_cols:
            print("\n[warn] Requested but excluded (name rule or dtype):")
            for c in excluded_cols:
                print("  -", c)

        scalar_cols = selected_cols
    else:
        print("\nSCALAR_COLUMNS is empty -> auto-discover numeric scalar columns from union schema...")
        scalar_cols = autodiscover_scalar_columns(schema)
        print(f"\nAuto-discovered scalar columns: {len(scalar_cols)}")
        for c in scalar_cols:
            print("  -", c)

    if not scalar_cols:
        raise RuntimeError("No scalar columns selected for tokenization. Check SCALAR_COLUMNS and exclusions.")

    bin_edges: Dict[str, np.ndarray] = {}
    col_means: Dict[str, float] = {}
    col_stds: Dict[str, float] = {}

    for col in scalar_cols:
        print(f"\nFitting tokenizer for column: {col}")
        vals = collect_values_for_column(col, files)
        n = vals.size
        print(f"  -> collected {n} values (after subsampling/caps)")

        if n < 1000:
            print("  -> too few values; skipping this column.")
            continue

        vals_sorted = np.sort(vals)

        mean = float(np.mean(vals_sorted))
        std = float(np.std(vals_sorted) + 1e-8)
        col_means[col] = mean
        col_stds[col] = std

        # equal-mass binning (quantile bins)
        quantiles = np.linspace(0.0, 1.0, N_BINS + 1)
        edges = np.quantile(vals_sorted, quantiles)

        # ensure strictly increasing (avoid duplicate quantiles)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-8

        bin_edges[col] = edges
        print(f"  -> bin edges: {edges.shape}, min={edges[0]:.6g}, max={edges[-1]:.6g}")

    if not bin_edges:
        raise RuntimeError("No columns produced bin edges (all had too few values?).")

    ensure_dir(OUTPUT_CONFIG)
    np.savez_compressed(
        OUTPUT_CONFIG,
        bin_edges=bin_edges,
        col_means=col_means,
        col_stds=col_stds,
        N_BINS=int(N_BINS),
        input_dir=str(INPUT_DIR),
        file_pattern=str(FILE_PATTERN),
    )
    print(f"\nSaved scalar tokenizer config to {OUTPUT_CONFIG}")
    print(f"Columns saved: {len(bin_edges)}")


if __name__ == "__main__":
    fit_scalar_tokenizer()