#!/usr/bin/env python
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import polars as pl
from tqdm.auto import tqdm

# =========================
# CONFIG
# =========================

INPUT_DIR = "/home/astrodados4/downloads/hypercube"
FILE_PATTERN = "datacube_*.parquet"

TOKENIZER_CONFIG = "scalar_tokenizer_config.npz"
OUTPUT_DIR = "scalar_tokens"

# Which column to use as ID in priority order
PREFERRED_ID_COLS = ["gaia_source_id", "id"]

TOKEN_DTYPE = np.uint16   # fits N_BINS=1024


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


def load_tokenizer_config(config_path: str):
    data = np.load(config_path, allow_pickle=True)
    bin_edges: Dict[str, np.ndarray] = data["bin_edges"].item()
    col_means: Dict[str, float] = data["col_means"].item()
    col_stds: Dict[str, float] = data["col_stds"].item()
    n_bins = int(data["N_BINS"])
    return bin_edges, col_means, col_stds, n_bins


def choose_id_column(df: pl.DataFrame) -> Optional[str]:
    for c in PREFERRED_ID_COLS:
        if c in df.columns:
            return c
    return None


def infer_scalar_columns_from_config(
    df: pl.DataFrame,
    bin_edges: Dict[str, np.ndarray],
) -> List[str]:
    """
    Use the intersection of:
      - columns present in the DataFrame
      - columns that have bin_edges in the config
    """
    cols = []
    for c in df.columns:
        if c in bin_edges:
            cols.append(c)
    return cols


def encode_column_values_to_tokens(
    vals: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """
    vals: (N,) float values
    edges: (N_BINS+1,) bin edges (strictly increasing)
    returns tokens: (N,) integers in [0, N_BINS-1]
    """
    idx = np.searchsorted(edges, vals, side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    return idx.astype(TOKEN_DTYPE)


# =========================
# MAIN ENCODING
# =========================

def encode_all_files():
    files = list_parquet_files()
    print("Found parquet files:")
    for f in files:
        print("  -", f)

    bin_edges, col_means, col_stds, n_bins = load_tokenizer_config(TOKENIZER_CONFIG)
    print(f"\nLoaded tokenizer config with {len(bin_edges)} scalar columns and N_BINS={n_bins}")

    ensure_dir(OUTPUT_DIR)

    for path in files:
        p = Path(path)
        print(f"\nEncoding scalar tokens for file: {p.name}")

        # small sample to decide which scalar columns are present
        df_sample = pl.read_parquet(path, n_rows=5)
        scalar_cols = infer_scalar_columns_from_config(df_sample, bin_edges)

        if not scalar_cols:
            print("  -> no scalar columns from config present in this file, skipping.")
            continue

        id_col = choose_id_column(df_sample)
        if id_col is None:
            print("  -> no suitable ID column found, proceeding without IDs.")
            cols_to_read = scalar_cols
        else:
            cols_to_read = scalar_cols + [id_col]

        df = pl.read_parquet(path, columns=cols_to_read)
        n_rows = df.height
        print(f"  -> rows: {n_rows}")
        print(f"  -> scalar columns used: {len(scalar_cols)}")

        scalar_tokens = np.empty((n_rows, len(scalar_cols)), dtype=TOKEN_DTYPE)

        for j, col in enumerate(scalar_cols):
            s = df[col]
            arr = s.to_numpy().astype(np.float64)

            mask_finite = np.isfinite(arr)
            default_val = float(col_means[col]) if col in col_means else 0.0
            arr[~mask_finite] = default_val

            edges = bin_edges[col]
            tokens_col = encode_column_values_to_tokens(arr, edges)
            scalar_tokens[:, j] = tokens_col

        ids = None
        if id_col is not None and id_col in df.columns:
            ids = df[id_col].to_numpy()

        out_name = p.stem + "_scalars_tokens.npz"
        out_path = Path(OUTPUT_DIR) / out_name

        np.savez_compressed(
            out_path,
            scalar_tokens=scalar_tokens,
            scalar_cols=np.array(scalar_cols, dtype=object),
            ids=ids if ids is not None else np.array([], dtype=np.int64),
            id_col=id_col if id_col is not None else "",
        )

        print(
            f"  -> saved {out_path} "
            f"(scalar_tokens shape={scalar_tokens.shape}, n_cols={len(scalar_cols)})"
        )


if __name__ == "__main__":
    encode_all_files()