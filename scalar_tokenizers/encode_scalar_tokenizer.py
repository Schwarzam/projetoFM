#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import numpy as np
import polars as pl

# =========================
# CONFIG
# =========================

INPUT_DIR = "/home/astrodados4/downloads/hypercube"
FILE_PATTERN = "datacube_*.parquet"

TOKENIZER_CONFIG = "scalar_tokenizer_config.npz"
OUTPUT_DIR = "scalar_tokens"

# Which column to use as ID in priority order
PREFERRED_ID_COLS = ["gaia_source_id", "id"]

# Token range: N_BINS=1024 fits in uint16
TOKEN_DTYPE = np.uint16

# If no ID column exists, optionally store row indices so you can align later
SAVE_ROW_INDEX_IF_NO_ID = True


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
    """Read parquet schema (column names) without loading any rows."""
    df0 = pl.read_parquet(path, n_rows=0)
    return df0.columns


def _to_str_key(k: Any) -> str:
    if isinstance(k, (bytes, np.bytes_)):
        return k.decode()
    return str(k)


def normalize_dict_keys(d: Dict[Any, Any]) -> Dict[str, Any]:
    return {_to_str_key(k): v for k, v in d.items()}


def load_tokenizer_config(config_path: str):
    data = np.load(config_path, allow_pickle=True)

    bin_edges_raw: Dict[Any, np.ndarray] = data["bin_edges"].item()
    col_means_raw: Dict[Any, float] = data["col_means"].item()
    col_stds_raw: Dict[Any, float] = data["col_stds"].item()
    n_bins = int(data["N_BINS"])

    bin_edges = normalize_dict_keys(bin_edges_raw)
    col_means = normalize_dict_keys(col_means_raw)
    col_stds = normalize_dict_keys(col_stds_raw)

    # Ensure edges are numpy arrays float64
    for k, v in list(bin_edges.items()):
        bin_edges[k] = np.asarray(v, dtype=np.float64)

    return bin_edges, col_means, col_stds, n_bins


def choose_id_column(columns: List[str]) -> Optional[str]:
    for c in PREFERRED_ID_COLS:
        if c in columns:
            return c
    return None


def pick_scalar_cols_for_file(
    file_columns: List[str],
    bin_edges: Dict[str, np.ndarray],
) -> List[str]:
    """
    Pick scalar columns that:
      - exist in this file schema
      - exist in tokenizer config (bin_edges)
    Keeps a stable order: config order first.
    """
    file_set = set(file_columns)

    # stable order based on config insertion order
    scalar_cols = [c for c in bin_edges.keys() if c in file_set]
    return scalar_cols


def encode_column_values_to_tokens(vals: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    vals: (N,) float64
    edges: (N_BINS+1,) strictly increasing
    returns tokens: (N,) integers in [0, N_BINS-1]
    """
    idx = np.searchsorted(edges, vals, side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    return idx.astype(TOKEN_DTYPE)


def series_to_float64_array(s: pl.Series) -> np.ndarray:
    """
    Convert numeric (int/float/bool) Polars Series into float64 numpy array.
    """
    if s.dtype == pl.Boolean:
        return s.cast(pl.UInt8).to_numpy().astype(np.float64)
    if s.dtype.is_float():
        return s.to_numpy().astype(np.float64)
    if s.dtype.is_integer():
        return s.to_numpy().astype(np.float64)

    # Fallback: try to cast in polars then to numpy
    return s.cast(pl.Float64).to_numpy()


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

        # schema-only to decide which columns exist
        try:
            file_cols = read_schema_fast(path)
        except Exception as e:
            print(f"  [warn] Could not read schema for {p.name}: {e}")
            continue

        scalar_cols = pick_scalar_cols_for_file(file_cols, bin_edges)
        if not scalar_cols:
            print("  -> no scalar columns from config present in this file, skipping.")
            continue

        id_col = choose_id_column(file_cols)

        # columns to read (unique, preserve order)
        cols_to_read: List[str] = list(scalar_cols)
        if id_col is not None:
            cols_to_read.append(id_col)
        cols_to_read = list(dict.fromkeys(cols_to_read))

        # read only needed columns
        try:
            df = pl.read_parquet(path, columns=cols_to_read)
        except Exception as e:
            print(f"  [warn] Could not read columns from {p.name}: {e}")
            continue

        n_rows = df.height
        print(f"  -> rows: {n_rows}")
        print(f"  -> scalar columns used: {len(scalar_cols)}")
        print(f"  -> id_col: {id_col if id_col is not None else '(none)'}")

        scalar_tokens = np.empty((n_rows, len(scalar_cols)), dtype=TOKEN_DTYPE)

        for j, col in enumerate(scalar_cols):
            s = df[col]
            arr = series_to_float64_array(s)

            # fill non-finite values with mean (or 0.0)
            default_val = float(col_means.get(col, 0.0))
            if arr.size:
                bad = ~np.isfinite(arr)
                if np.any(bad):
                    arr[bad] = default_val

            edges = bin_edges[col]
            scalar_tokens[:, j] = encode_column_values_to_tokens(arr, edges)

        # ids (optional)
        out_ids: np.ndarray
        out_id_col: str
        if id_col is not None and id_col in df.columns:
            out_ids = df[id_col].to_numpy()
            out_id_col = id_col
        else:
            out_ids = np.array([], dtype=np.int64)
            out_id_col = ""

        out_data: Dict[str, Any] = {
            "scalar_tokens": scalar_tokens,
            "scalar_cols": np.array(scalar_cols, dtype=object),
            "ids": out_ids,
            "id_col": np.array([out_id_col], dtype=object),
            "N_BINS": np.array([n_bins], dtype=np.int64),
        }

        if out_id_col == "" and SAVE_ROW_INDEX_IF_NO_ID:
            out_data["row_index"] = np.arange(n_rows, dtype=np.int64)

        out_name = p.stem + "_scalar_tokens.npz"
        out_path = Path(OUTPUT_DIR) / out_name
        np.savez_compressed(out_path, **out_data)

        print(
            f"  -> saved {out_path} "
            f"(scalar_tokens shape={scalar_tokens.shape})"
        )


if __name__ == "__main__":
    encode_all_files()