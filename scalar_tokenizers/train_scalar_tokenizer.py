#!/usr/bin/env python
import math
from pathlib import Path
from typing import List, Dict

import numpy as np
import polars as pl
from tqdm.auto import tqdm

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


# =========================
# HELPERS
# =========================

def list_parquet_files() -> List[Path]:
    p = Path(INPUT_DIR)
    files = sorted(p.glob(FILE_PATTERN))
    if not files:
        raise RuntimeError(f"No parquet files found in {INPUT_DIR} matching {FILE_PATTERN}")
    return files


def infer_scalar_columns(df: pl.DataFrame) -> List[str]:
    """
    From a sample DataFrame, choose scalar columns that are 'reasonable':
      - numeric dtypes (Int, Float)
      - NOT the huge image/spectrum arrays
    We exclude splus_cut_*, desi_wave_*, desi_flux_*, desi_ivar_*,
    Gaia XP coefficient arrays, ZTF lightcurve arrays, etc.
    You can refine this list later.
    """
    cols = []
    for c in df.columns:
        dt = df[c].dtype

        # skip obvious big arrays / non-scalar
        if c.startswith("splus_cut_"):
            continue
        if c.startswith("desi_wave_") or c.startswith("desi_flux_") or c.startswith("desi_ivar_"):
            continue
        if "gaiaxp_bp_coefficients" in c or "gaiaxp_rp_coefficients" in c:
            continue
        if c.startswith("ztf_mag_") or c.startswith("ztf_magerr_") or c.startswith("ztf_hmjd_"):
            continue

        # keep scalar-like numeric columns
        if dt.is_numeric():
            cols.append(c)

    return cols


def ensure_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# =========================
# MAIN TRAINING LOGIC
# =========================

def collect_values_for_column(col: str, files: List[Path]) -> np.ndarray:
    """
    Stream over all files and collect numeric values for one column,
    with subsampling to control memory.
    """
    all_vals = []
    total_collected = 0

    for path in files:
        if total_collected >= MAX_VALUES_PER_COL:
            break

        try:
            df = pl.read_parquet(path, columns=[col])
        except Exception as e:
            print(f"  [warn] Could not read column '{col}' from {path}: {e}")
            continue

        if col not in df.columns:
            continue

        s = df[col]
        # filter non-finite for floats
        if s.dtype.is_float():
            arr = s.to_numpy()
            mask = np.isfinite(arr)
            arr = arr[mask]
        else:
            # integers, keep all but cast to float64
            arr = s.to_numpy().astype(np.float64)

        if arr.size == 0:
            continue

        # subsample from this file
        if arr.size > MAX_VALUES_PER_FILE_PER_COL:
            idx = np.random.choice(arr.size, size=MAX_VALUES_PER_FILE_PER_COL, replace=False)
            arr = arr[idx]

        all_vals.append(arr)
        total_collected += arr.size

        if total_collected >= MAX_VALUES_PER_COL:
            break

    if not all_vals:
        return np.array([], dtype=np.float64)

    vals = np.concatenate(all_vals, axis=0)
    return vals


def fit_scalar_tokenizer():
    files = list_parquet_files()
    print("Found parquet files:")
    for f in files:
        print("  -", f)

    print("\nReading sample file to infer scalar columns...")
    sample_df = pl.read_parquet(files[0])
    scalar_cols = infer_scalar_columns(sample_df)

    print("\nScalar columns selected for tokenization:")
    for c in scalar_cols:
        print("  -", c)

    bin_edges: Dict[str, np.ndarray] = {}
    col_means: Dict[str, float] = {}
    col_stds: Dict[str, float] = {}

    for col in scalar_cols:
        print(f"\nFitting tokenizer for column: {col}")
        vals = collect_values_for_column(col, files)
        n = vals.size
        print(f"  -> collected {n} values")

        if n < 1000:
            print("  -> too few values; skipping this column.")
            continue

        vals_sorted = np.sort(vals)

        mean = float(np.mean(vals_sorted))
        std = float(np.std(vals_sorted) + 1e-8)
        col_means[col] = mean
        col_stds[col] = std

        # equal-mass binning
        quantiles = np.linspace(0.0, 1.0, N_BINS + 1)
        edges = np.quantile(vals_sorted, quantiles)

        # ensure strictly increasing
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-8

        bin_edges[col] = edges
        print(f"  -> bin edges shape: {edges.shape}, min={edges[0]:.4g}, max={edges[-1]:.4g}")

    ensure_dir(OUTPUT_CONFIG)
    np.savez_compressed(
        OUTPUT_CONFIG,
        bin_edges=bin_edges,
        col_means=col_means,
        col_stds=col_stds,
        N_BINS=N_BINS,
    )
    print(f"\nSaved scalar tokenizer config to {OUTPUT_CONFIG}")


if __name__ == "__main__":
    fit_scalar_tokenizer()