#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

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

# Output tokenizer config (one codebook per group)
TOKENIZER_CONFIG = "spectrum_tokenizer_config.npz"

# Codebook size per group
CODEBOOK_SIZE = 2048

# Sampling / RAM control
MAX_FEATURES_PER_GROUP = 2_000_000   # max feature vectors per group
MAX_ROWS_PER_FILE = 5000             # max rows per file to sample from
MAX_POINTS_PER_ROW = 2000            # max points per row (pixels/coeffs/obs) sampled

# ZTF option A config
ZTF_MAG_PREFIX = "ztf_mag_"
ZTF_MAGERR_PREFIX = "ztf_magerr_"
ZTF_TIME_PREFIX = "ztf_hmjd_"         # not used in clustering, but detected for sanity
ZTF_ERR_EPS = 1e-3                    # floor for magerr to avoid inf precision

# Fixed groups (DESI + GaiaXP)
BASE_GROUPS = [
    {"name": "desi_b",    "kind": "flux_ivar", "flux_col": "desi_flux_b",            "ivar_col": "desi_ivar_b"},
    {"name": "desi_r",    "kind": "flux_ivar", "flux_col": "desi_flux_r",            "ivar_col": "desi_ivar_r"},
    {"name": "desi_z",    "kind": "flux_ivar", "flux_col": "desi_flux_z",            "ivar_col": "desi_ivar_z"},
    {"name": "gaiaxp_bp", "kind": "value_only","flux_col": "gaiaxp_bp_coefficients", "ivar_col": None},
    {"name": "gaiaxp_rp", "kind": "value_only","flux_col": "gaiaxp_rp_coefficients", "ivar_col": None},
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


def read_schema_fast(path: Path) -> List[str]:
    """Read schema without loading data."""
    df0 = pl.read_parquet(path, n_rows=0)
    return df0.columns


def union_columns(files: List[Path]) -> List[str]:
    """Union of all column names across files."""
    cols = set()
    for f in files:
        try:
            cols.update(read_schema_fast(f))
        except Exception as e:
            print(f"[warn] Could not read schema from {f.name}: {e}")
    return sorted(cols)


def autodetect_ztf_groups(all_cols: List[str]) -> List[Dict[str, Any]]:
    """
    Detect ZTF bands by matching ztf_mag_<band> with ztf_magerr_<band>.
    Creates groups with kind='ztf_mag_err' (Option A): features=[mag, 1/magerr]
    """
    mags = {}
    magerrs = {}
    times = {}

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
            "time_col": times.get(b, None),  # not used in training
        })

    if groups:
        print("\nDetected ZTF groups:")
        for g in groups:
            tinfo = f", time_col={g['time_col']}" if g["time_col"] is not None else ""
            print(f"  - {g['name']}: mag_col={g['mag_col']}, err_col={g['err_col']}{tinfo}")
    else:
        print("\nNo ZTF (mag+magerr) groups detected in union schema.")

    return groups


def _sample_row_indices(n_rows: int) -> np.ndarray:
    if n_rows <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n_rows > MAX_ROWS_PER_FILE:
        return np.random.choice(n_rows, size=MAX_ROWS_PER_FILE, replace=False)
    return np.arange(n_rows)


def _sample_points(feats: np.ndarray) -> np.ndarray:
    """feats: (L, D) -> subsample rows if L is large."""
    L = feats.shape[0]
    if L > MAX_POINTS_PER_ROW:
        idx = np.random.choice(L, size=MAX_POINTS_PER_ROW, replace=False)
        return feats[idx]
    return feats


def _finite_rows_mask(x: np.ndarray) -> np.ndarray:
    """Mask rows where all columns are finite."""
    return np.isfinite(x).all(axis=1)


def collect_features_for_group(group: Dict[str, Any], files: List[Path]) -> np.ndarray:
    """
    Collect feature vectors for one group across all files, with sampling.
    Group kinds:
      - flux_ivar: feats=[flux, sqrt(ivar)]  (DESI)
      - value_only: feats=[value]           (GaiaXP)
      - ztf_mag_err: feats=[mag, 1/magerr]  (ZTF Option A)
    """
    name = group["name"]
    kind = group["kind"]

    print(f"\n[group {name}] Collecting features (kind={kind})...")
    all_feats: List[np.ndarray] = []
    total = 0

    # Determine which columns to read
    if kind == "flux_ivar":
        cols_to_read = [group["flux_col"], group["ivar_col"]]
    elif kind == "value_only":
        cols_to_read = [group["flux_col"]]
    elif kind == "ztf_mag_err":
        cols_to_read = [group["mag_col"], group["err_col"]]
    else:
        raise ValueError(f"Unknown group kind: {kind}")

    for path in files:
        if total >= MAX_FEATURES_PER_GROUP:
            break

        # quick schema check: skip files without needed columns
        try:
            cols_in_file = set(read_schema_fast(path))
        except Exception:
            cols_in_file = set()
        if not all(c in cols_in_file for c in cols_to_read):
            continue

        try:
            df = pl.read_parquet(path, columns=cols_to_read)
        except Exception as e:
            print(f"  [warn] Could not read {cols_to_read} from {path.name}: {e}")
            continue

        n_rows = df.height
        if n_rows == 0:
            continue

        row_indices = _sample_row_indices(n_rows)

        if kind == "flux_ivar":
            flux_series = df[group["flux_col"]]
            ivar_series = df[group["ivar_col"]]

            for ridx in row_indices:
                if total >= MAX_FEATURES_PER_GROUP:
                    break

                flux = np.asarray(flux_series[ridx], dtype=np.float32)
                if flux.size == 0:
                    continue

                ivar = np.asarray(ivar_series[ridx], dtype=np.float32)
                if ivar.size != flux.size:
                    continue

                ivar = np.clip(ivar, a_min=0.0, a_max=None)
                prec = np.sqrt(ivar)  # (L,)
                feats = np.stack([flux, prec], axis=-1)  # (L,2)

                feats = _sample_points(feats)
                m = _finite_rows_mask(feats)
                feats = feats[m]
                if feats.size == 0:
                    continue

                all_feats.append(feats)
                total += feats.shape[0]

        elif kind == "value_only":
            series = df[group["flux_col"]]
            for ridx in row_indices:
                if total >= MAX_FEATURES_PER_GROUP:
                    break

                arr = np.asarray(series[ridx], dtype=np.float32)
                if arr.size == 0:
                    continue

                feats = arr.reshape(-1, 1)  # (L,1)
                feats = _sample_points(feats)

                m = _finite_rows_mask(feats)
                feats = feats[m]
                if feats.size == 0:
                    continue

                all_feats.append(feats)
                total += feats.shape[0]

        elif kind == "ztf_mag_err":
            mag_series = df[group["mag_col"]]
            err_series = df[group["err_col"]]

            for ridx in row_indices:
                if total >= MAX_FEATURES_PER_GROUP:
                    break

                mag = np.asarray(mag_series[ridx], dtype=np.float32)
                err = np.asarray(err_series[ridx], dtype=np.float32)

                if mag.size == 0 or err.size == 0:
                    continue
                if err.size != mag.size:
                    continue

                # precision proxy = 1/magerr
                err = np.clip(err, a_min=ZTF_ERR_EPS, a_max=None)
                prec = 1.0 / err

                feats = np.stack([mag, prec], axis=-1)  # (L,2)
                feats = _sample_points(feats)

                m = _finite_rows_mask(feats)
                feats = feats[m]
                if feats.size == 0:
                    continue

                all_feats.append(feats)
                total += feats.shape[0]

    if not all_feats:
        print(f"  [group {name}] No features collected.")
        # Return correct dimensionality for downstream
        D = 2 if kind in ("flux_ivar", "ztf_mag_err") else 1
        return np.zeros((0, D), dtype=np.float32)

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

    # Detect ZTF groups (Option A) from union schema
    all_cols = union_columns(files)
    ztf_groups = autodetect_ztf_groups(all_cols)

    # Final group list
    groups = list(BASE_GROUPS) + ztf_groups
    print(f"\nTotal groups to consider: {len(groups)}")

    group_names: List[str] = []
    group_means: Dict[str, np.ndarray] = {}
    group_stds: Dict[str, np.ndarray] = {}
    group_codebooks: Dict[str, np.ndarray] = {}
    group_feature_dim: Dict[str, int] = {}
    group_columns: Dict[str, Dict[str, Any]] = {}

    for group in groups:
        name = group["name"]
        feats = collect_features_for_group(group, files)

        if feats.shape[0] < CODEBOOK_SIZE * 10:
            print(f"[group {name}] Not enough data for robust KMeans (got {feats.shape[0]}). Skipping.")
            continue

        # mean/std & standardize
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
        group_means[name] = mean.astype(np.float32)
        group_stds[name] = std.astype(np.float32)
        group_codebooks[name] = kmeans.cluster_centers_.astype(np.float32)
        group_feature_dim[name] = int(feats.shape[1])
        group_columns[name] = group  # save spec used

        # optional: save model separately
        joblib.dump(kmeans, f"spectrum_kmeans_{name}.joblib")
        print(f"[group {name}] Done. Saved KMeans model to spectrum_kmeans_{name}.joblib")

    if not group_names:
        raise RuntimeError("No groups had enough data to fit a tokenizer.")

    ensure_dir(TOKENIZER_CONFIG)
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
    )
    print(f"\nSaved spectrum tokenizer config to {TOKENIZER_CONFIG}")
    print(f"Groups saved: {group_names}")


if __name__ == "__main__":
    train_spectrum_tokenizers()