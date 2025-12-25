#!/usr/bin/env python
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import polars as pl
from tqdm.auto import tqdm
from sklearn.cluster import MiniBatchKMeans
import joblib

# =========================
# CONFIG
# =========================

INPUT_DIR = "/home/astrodados4/downloads/hypercube"
FILE_PATTERN = "datacube_*.parquet"

TOKENIZER_CONFIG = "spectrum_tokenizer_config.npz"
OUTPUT_DIR = "spectrum_tokens"

# Same groups as before; must match training
SPECTRAL_GROUPS = [
    {"name": "desi_b",  "flux_col": "desi_flux_b",  "ivar_col": "desi_ivar_b"},
    {"name": "desi_r",  "flux_col": "desi_flux_r",  "ivar_col": "desi_ivar_r"},
    {"name": "desi_z",  "flux_col": "desi_flux_z",  "ivar_col": "desi_ivar_z"},
    {"name": "gaiaxp_bp", "flux_col": "gaiaxp_bp_coefficients", "ivar_col": None},
    {"name": "gaiaxp_rp", "flux_col": "gaiaxp_rp_coefficients", "ivar_col": None},
]

# Which ID column to prefer
PREFERRED_ID_COLS = ["gaia_source_id", "id"]

TOKEN_DTYPE = np.uint16  # fits up to 65535 codes


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
    group_names = data["group_names"]
    group_means: Dict[str, np.ndarray] = data["group_means"].item()
    group_stds: Dict[str, np.ndarray] = data["group_stds"].item()
    group_codebooks: Dict[str, np.ndarray] = data["group_codebooks"].item()
    codebook_size = int(data["CODEBOOK_SIZE"])

    # restore MiniBatchKMeans objects from centers, for convenience we will do manual nearest-center
    # but we can also build a tiny wrapper; here we'll just keep centers.
    return group_names, group_means, group_stds, group_codebooks, codebook_size


def choose_id_column(df: pl.DataFrame) -> Optional[str]:
    for c in PREFERRED_ID_COLS:
        if c in df.columns:
            return c
    return None


def compute_tokens_for_group_in_file(
    df: pl.DataFrame,
    group: Dict[str, str],
    mean: np.ndarray,
    std: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """
    df: DataFrame with columns flux_col (+ ivar_col if needed)
    Returns tokens array of shape (N_rows, L) with dtype TOKEN_DTYPE.
    Assumes all rows for this group share same length L.
    """
    name = group["name"]
    flux_col = group["flux_col"]
    ivar_col = group["ivar_col"]

    flux_series = df[flux_col]
    ivar_series = df[ivar_col] if ivar_col is not None else None

    n_rows = df.height
    if n_rows == 0:
        return np.zeros((0, 0), dtype=TOKEN_DTYPE)

    # determine length L from first non-empty row
    L = None
    for i in range(n_rows):
        arr = np.asarray(flux_series[i])
        if arr.size > 0:
            L = arr.size
            break
    if L is None:
        return np.zeros((n_rows, 0), dtype=TOKEN_DTYPE)

    tokens = np.empty((n_rows, L), dtype=TOKEN_DTYPE)

    # We'll do manual nearest-center search using Euclidean distance
    # centers: (K, D), D=1 or 2
    centers = centers.astype(np.float32)
    for i in tqdm(range(n_rows), desc=f"  [tokens {name}] rows", leave=False):
        flux_arr = np.asarray(flux_series[i], dtype=np.float32)
        if flux_arr.size == 0:
            tokens[i, :] = 0
            continue

        if flux_arr.size != L:
            # length mismatch; simple handling: pad or truncate to L
            if flux_arr.size > L:
                flux_arr = flux_arr[:L]
            else:
                flux_tmp = np.zeros(L, dtype=np.float32)
                flux_tmp[:flux_arr.size] = flux_arr
                flux_arr = flux_tmp

        if ivar_col is not None:
            ivar_arr = np.asarray(ivar_series[i], dtype=np.float32)
            if ivar_arr.size != flux_arr.size:
                if ivar_arr.size > L:
                    ivar_arr = ivar_arr[:L]
                else:
                    ivar_tmp = np.zeros(L, dtype=np.float32)
                    ivar_tmp[:ivar_arr.size] = ivar_arr
                    ivar_arr = ivar_tmp
            ivar_arr = np.clip(ivar_arr, a_min=0.0, a_max=None)
            prec_arr = np.sqrt(ivar_arr)
            feats = np.stack([flux_arr, prec_arr], axis=-1)  # (L,2)
        else:
            feats = flux_arr.reshape(-1, 1)  # (L,1)

        # standardize
        feats_std = (feats - mean) / std

        # nearest center: compute distances
        # feats_std: (L, D), centers: (K, D) -> distances: (L, K)
        # to save memory, do in chunks if needed; here we assume L*K is manageable.
        # dist^2 = sum((x-c)^2) = x^2 + c^2 - 2 x·c
        # simpler: brute-force with broadcasting
        diffs = feats_std[:, None, :] - centers[None, :, :]  # (L,K,D)
        dist2 = np.sum(diffs * diffs, axis=-1)  # (L,K)
        codes = np.argmin(dist2, axis=1).astype(TOKEN_DTYPE)  # (L,)

        tokens[i, :] = codes

    return tokens


# =========================
# MAIN ENCODING
# =========================

def encode_spectrum_tokens():
    files = list_parquet_files()
    print("Found parquet files:")
    for f in files:
        print("  -", f)

    group_names, group_means, group_stds, group_codebooks, codebook_size = load_tokenizer_config(TOKENIZER_CONFIG)
    print(f"\nLoaded spectrum tokenizer config for groups: {list(group_names)} (K={codebook_size})")

    ensure_dir(OUTPUT_DIR)

    # Build lookup from name -> group spec
    group_by_name = {g["name"]: g for g in SPECTRAL_GROUPS}

    for path in files:
        p = Path(path)
        print(f"\nProcessing file: {p.name}")

        # read a small sample to decide which groups are present + pick ID column
        df_sample = pl.read_parquet(path, n_rows=5)
        id_col = choose_id_column(df_sample)
        cols_to_read = [id_col] if id_col is not None else []

        available_groups = []
        for gname in group_names:
            g = group_by_name.get(gname)
            if g is None:
                continue
            flux_col = g["flux_col"]
            if flux_col in df_sample.columns:
                available_groups.append(g)

        if not available_groups:
            print("  -> no spectrum groups present in this file according to config, skipping.")
            continue

        # read full DataFrame with ID + all spectral columns for the available groups
        extra_cols = []
        for g in available_groups:
            extra_cols.append(g["flux_col"])
            if g["ivar_col"] is not None:
                extra_cols.append(g["ivar_col"])

        cols_to_read = list(dict.fromkeys(cols_to_read + extra_cols))  # unique preserve order
        df = pl.read_parquet(path, columns=cols_to_read)
        n_rows = df.height
        print(f"  -> rows: {n_rows}")
        print(f"  -> spectrum groups used: {[g['name'] for g in available_groups]}")

        # Prepare output dict
        out_data = {}

        # IDs
        if id_col is not None and id_col in df.columns:
            out_data["ids"] = df[id_col].to_numpy()
            out_data["id_col"] = id_col
        else:
            out_data["ids"] = np.array([], dtype=np.int64)
            out_data["id_col"] = ""

        # encode each spectrum group
        for g in available_groups:
            name = g["name"]
            mean = group_means[name]
            std = group_stds[name]
            centers = group_codebooks[name]

            tokens = compute_tokens_for_group_in_file(df, g, mean, std, centers)
            out_data[f"tokens_{name}"] = tokens

        out_name = p.stem + "_spectrum_tokens.npz"
        out_path = Path(OUTPUT_DIR) / out_name
        np.savez_compressed(out_path, **out_data)

        shapes = {k: v.shape for k, v in out_data.items() if k.startswith("tokens_")}
        print(f"  -> saved {out_path} with token arrays: {shapes}")


if __name__ == "__main__":
    encode_spectrum_tokens()