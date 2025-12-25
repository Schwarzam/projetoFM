#!/usr/bin/env python
import random
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import polars as pl
from tqdm.auto import tqdm
from sklearn.cluster import MiniBatchKMeans
import joblib  # pip install scikit-learn joblib

# =========================
# CONFIG
# =========================

INPUT_DIR = "/home/astrodados4/downloads/hypercube"
FILE_PATTERN = "datacube_*.parquet"

# Output tokenizer config (will contain one codebook per spectrum type)
TOKENIZER_CONFIG = "spectrum_tokenizer_config.npz"

# Codebook size per spectrum type
CODEBOOK_SIZE = 2048

# Sampling / RAM control
MAX_FEATURES_PER_GROUP = 2_000_000   # max feature vectors per spectrum group
MAX_ROWS_PER_FILE = 5000             # max rows per file to sample from
MAX_PIXELS_PER_ROW = 2000            # max pixels per row to sample from

# Spectral array groups: name + columns
# - If ivar_col is not None: use feature=[flux, sqrt(ivar)]
# - If ivar_col is None: use feature=[value]
SPECTRAL_GROUPS = [
    {"name": "desi_b",  "flux_col": "desi_flux_b",  "ivar_col": "desi_ivar_b"},
    {"name": "desi_r",  "flux_col": "desi_flux_r",  "ivar_col": "desi_ivar_r"},
    {"name": "desi_z",  "flux_col": "desi_flux_z",  "ivar_col": "desi_ivar_z"},
    {"name": "gaiaxp_bp", "flux_col": "gaiaxp_bp_coefficients", "ivar_col": None},
    {"name": "gaiaxp_rp", "flux_col": "gaiaxp_rp_coefficients", "ivar_col": None},
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


def ensure_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def collect_features_for_group(
    group: Dict[str, str],
    files: List[Path],
) -> np.ndarray:
    """
    Collect spectral feature vectors for one group across all files.
    For DESI: [flux, sqrt(ivar)] per pixel.
    For Gaia XP: [coeff] per coefficient.
    Sampling is used to control memory.
    """
    name = group["name"]
    flux_col = group["flux_col"]
    ivar_col = group["ivar_col"]

    print(f"\n[group {name}] Collecting features...")
    all_feats = []
    total = 0

    for path in files:
        if total >= MAX_FEATURES_PER_GROUP:
            break

        cols_to_read = [flux_col]
        if ivar_col is not None:
            cols_to_read.append(ivar_col)

        try:
            df = pl.read_parquet(path, columns=cols_to_read)
        except Exception as e:
            print(f"  [warn] Could not read {cols_to_read} from {path.name}: {e}")
            continue

        if flux_col not in df.columns:
            # this file just doesn't have this spectrum type
            continue

        n_rows = df.height
        if n_rows == 0:
            continue

        # pick a subset of rows
        if n_rows > MAX_ROWS_PER_FILE:
            row_indices = np.random.choice(n_rows, size=MAX_ROWS_PER_FILE, replace=False)
        else:
            row_indices = np.arange(n_rows)

        flux_series = df[flux_col]
        ivar_series = df[ivar_col] if ivar_col is not None else None

        for ridx in row_indices:
            if total >= MAX_FEATURES_PER_GROUP:
                break

            flux_arr = np.asarray(flux_series[ridx], dtype=np.float32)
            if flux_arr.size == 0:
                continue

            if ivar_col is not None:
                ivar_arr = np.asarray(ivar_series[ridx], dtype=np.float32)
                if ivar_arr.size != flux_arr.size:
                    # length mismatch; skip this row
                    continue
                # sqrt(ivar) as "precision" proxy; clip negatives
                ivar_arr = np.clip(ivar_arr, a_min=0.0, a_max=None)
                prec_arr = np.sqrt(ivar_arr)
                feats = np.stack([flux_arr, prec_arr], axis=-1)  # (L, 2)
            else:
                feats = flux_arr.reshape(-1, 1)  # (L,1)

            L = feats.shape[0]
            if L > MAX_PIXELS_PER_ROW:
                idx = np.random.choice(L, size=MAX_PIXELS_PER_ROW, replace=False)
                feats = feats[idx]

            all_feats.append(feats)
            total += feats.shape[0]

    if not all_feats:
        print(f"  [group {name}] No features collected.")
        return np.zeros((0, 1), dtype=np.float32)

    features = np.concatenate(all_feats, axis=0)
    print(f"  [group {name}] Collected {features.shape[0]} feature vectors, dim={features.shape[1]}")
    return features


# =========================
# MAIN TRAINING
# =========================

def train_spectrum_tokenizers():
    files = list_parquet_files()
    print("Found parquet files:")
    for f in files:
        print("  -", f)

    group_names = []
    group_means = {}
    group_stds = {}
    group_codebooks = {}

    for group in SPECTRAL_GROUPS:
        name = group["name"]
        feats = collect_features_for_group(group, files)
        if feats.shape[0] < CODEBOOK_SIZE * 10:
            print(f"[group {name}] Not enough data for robust KMeans; skipping.")
            continue

        # compute mean/std & standardize
        mean = feats.mean(axis=0)
        std = feats.std(axis=0) + 1e-8
        feats_std = (feats - mean) / std

        print(f"[group {name}] Fitting MiniBatchKMeans with {CODEBOOK_SIZE} clusters...")
        kmeans = MiniBatchKMeans(
            n_clusters=CODEBOOK_SIZE,
            batch_size=4096,
            max_iter=50,
            verbose=0,
            n_init=1,
        )
        kmeans.fit(feats_std)

        group_names.append(name)
        group_means[name] = mean
        group_stds[name] = std
        group_codebooks[name] = kmeans.cluster_centers_

        # optional: save KMeans model separately
        joblib.dump(kmeans, f"spectrum_kmeans_{name}.joblib")
        print(f"[group {name}] Done. Saved KMeans model to spectrum_kmeans_{name}.joblib")

    if not group_names:
        raise RuntimeError("No spectrum groups had enough data to fit a tokenizer.")

    ensure_dir(TOKENIZER_CONFIG)
    np.savez_compressed(
        TOKENIZER_CONFIG,
        group_names=np.array(group_names, dtype=object),
        group_means=group_means,
        group_stds=group_stds,
        group_codebooks=group_codebooks,
        CODEBOOK_SIZE=CODEBOOK_SIZE,
    )
    print(f"\nSaved spectrum tokenizer config to {TOKENIZER_CONFIG}")


if __name__ == "__main__":
    train_spectrum_tokenizers()