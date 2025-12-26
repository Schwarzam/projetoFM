#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple
import json

import numpy as np
import polars as pl
from sklearn.cluster import MiniBatchKMeans
import joblib
from tqdm.auto import tqdm

# =========================
# CONFIG
# =========================

INPUT_DIR = "/home/astrodados4/downloads/hypercube"
FILE_PATTERN = "datacube_*.parquet"

TOKENIZER_CONFIG = "spectrum_tokenizer_config.npz"
CODEBOOK_SIZE = 2048

MAX_FEATURES_PER_GROUP = 2_000_000
MAX_ROWS_PER_FILE = 5000
MAX_POINTS_PER_ROW = 2000

DO_ZTF_AUTODETECT = False
ZTF_MAG_PREFIX = "ztf_mag_"
ZTF_MAGERR_PREFIX = "ztf_magerr_"
ZTF_TIME_PREFIX = "ztf_hmjd_"
ZTF_ERR_EPS = 1e-3

BASE_GROUPS = [
    {"name": "desi_b",    "kind": "flux_ivar",  "flux_col": "desi_flux_b",            "ivar_col": "desi_ivar_b"},
    {"name": "desi_r",    "kind": "flux_ivar",  "flux_col": "desi_flux_r",            "ivar_col": "desi_ivar_r"},
    {"name": "desi_z",    "kind": "flux_ivar",  "flux_col": "desi_flux_z",            "ivar_col": "desi_ivar_z"},
    {"name": "gaiaxp_bp", "kind": "value_only", "flux_col": "gaiaxp_bp_coefficients", "ivar_col": None},
    {"name": "gaiaxp_rp", "kind": "value_only", "flux_col": "gaiaxp_rp_coefficients", "ivar_col": None},
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

def ensure_dir_for_file(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def read_schema_fast(path: Path) -> List[str]:
    df0 = pl.read_parquet(path, n_rows=0)
    return df0.columns

def union_columns(files: List[Path]) -> List[str]:
    cols = set()
    for f in tqdm(files, desc="[schema] union columns", unit="file"):
        try:
            cols.update(read_schema_fast(f))
        except Exception as e:
            print(f"[warn] Could not read schema from {f.name}: {e}")
    
    print(f"Union schema has {len(cols)} columns.")

    for c in sorted(cols):
        print(f"{c} ", end="")
        
    return sorted(cols)

def autodetect_ztf_groups(all_cols: List[str]) -> List[Dict[str, Any]]:
    mags, magerrs, times = {}, {}, {}

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
            "time_col": times.get(b, None),
        })

    if groups:
        print("\nDetected ZTF groups:")
        for g in groups:
            tinfo = f", time_col={g['time_col']}" if g.get("time_col") else ""
            print(f"  - {g['name']}: mag_col={g['mag_col']}, err_col={g['err_col']}{tinfo}")
    else:
        print("\nNo ZTF (mag+magerr) groups detected in union schema.")

    return groups

def _sample_row_indices(n_rows: int) -> np.ndarray:
    if n_rows <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n_rows > MAX_ROWS_PER_FILE:
        return np.random.choice(n_rows, size=MAX_ROWS_PER_FILE, replace=False).astype(np.int64)
    return np.arange(n_rows, dtype=np.int64)

def _sample_points(feats: np.ndarray) -> np.ndarray:
    if feats.ndim == 1:
        feats = feats.reshape(-1, 1)
    L = feats.shape[0]
    if L > MAX_POINTS_PER_ROW:
        idx = np.random.choice(L, size=MAX_POINTS_PER_ROW, replace=False)
        return feats[idx]
    return feats

def _finite_rows_mask(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return np.isfinite(x).all(axis=1)

def _cell_to_1d_float(cell) -> np.ndarray:
    if cell is None:
        return np.zeros((0,), dtype=np.float32)

    if isinstance(cell, (list, tuple, np.ndarray)):
        arr = np.asarray(cell, dtype=np.float32).reshape(-1)
        return arr[np.isfinite(arr)]

    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return np.zeros((0,), dtype=np.float32)
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            s = s[1:-1]
        try:
            vals = json.loads(s)
            arr = np.asarray(vals, dtype=np.float32).reshape(-1)
            return arr[np.isfinite(arr)]
        except Exception:
            return np.zeros((0,), dtype=np.float32)

    return np.zeros((0,), dtype=np.float32)

def collect_features_for_group(group: Dict[str, Any], files: List[Path]) -> np.ndarray:
    name = group["name"]
    kind = group["kind"]

    print(f"\n[group {name}] Collecting features (kind={kind})...")
    all_feats: List[np.ndarray] = []
    total = 0

    if kind == "flux_ivar":
        cols_to_read = [group["flux_col"], group["ivar_col"]]
    elif kind == "value_only":
        cols_to_read = [group["flux_col"]]
    elif kind == "ztf_mag_err":
        cols_to_read = [group["mag_col"], group["err_col"]]
    else:
        raise ValueError(f"Unknown group kind: {kind}")

    for path in tqdm(files, desc=f"[group {name}] files", unit="file"):
        if total >= MAX_FEATURES_PER_GROUP:
            break

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
            for ridx in tqdm(row_indices, desc=f"[group {name}] rows ({path.name})", unit="row", leave=False):
                if total >= MAX_FEATURES_PER_GROUP:
                    break
                ridx = int(ridx)

                flux = _cell_to_1d_float(flux_series[ridx])
                if flux.size == 0:
                    continue
                ivar = _cell_to_1d_float(ivar_series[ridx])
                if ivar.size != flux.size:
                    continue

                ivar = np.clip(ivar, a_min=0.0, a_max=None)
                prec = np.sqrt(ivar)
                feats = np.stack([flux, prec], axis=-1)

                feats = _sample_points(feats)
                feats = feats[_finite_rows_mask(feats)]
                if feats.size == 0:
                    continue

                all_feats.append(feats.astype(np.float32))
                total += feats.shape[0]

        elif kind == "value_only":
            series = df[group["flux_col"]]
            for ridx in tqdm(row_indices, desc=f"[group {name}] rows ({path.name})", unit="row", leave=False):
                if total >= MAX_FEATURES_PER_GROUP:
                    break
                ridx = int(ridx)

                arr = _cell_to_1d_float(series[ridx])
                if arr.size == 0:
                    continue

                feats = arr.reshape(-1, 1)
                feats = _sample_points(feats)
                feats = feats[_finite_rows_mask(feats)]
                if feats.size == 0:
                    continue

                all_feats.append(feats.astype(np.float32))
                total += feats.shape[0]

        elif kind == "ztf_mag_err":
            mag_series = df[group["mag_col"]]
            err_series = df[group["err_col"]]
            for ridx in tqdm(row_indices, desc=f"[group {name}] rows ({path.name})", unit="row", leave=False):
                if total >= MAX_FEATURES_PER_GROUP:
                    break
                ridx = int(ridx)

                mag = _cell_to_1d_float(mag_series[ridx])
                err = _cell_to_1d_float(err_series[ridx])
                if mag.size == 0 or err.size == 0:
                    continue
                if err.size != mag.size:
                    continue

                err = np.clip(err, a_min=ZTF_ERR_EPS, a_max=None)
                prec = 1.0 / err
                feats = np.stack([mag, prec], axis=-1)

                feats = _sample_points(feats)
                feats = feats[_finite_rows_mask(feats)]
                if feats.size == 0:
                    continue

                all_feats.append(feats.astype(np.float32))
                total += feats.shape[0]

    if not all_feats:
        print(f"  [group {name}] No features collected.")
        D = 2 if kind in ("flux_ivar", "ztf_mag_err") else 1
        return np.zeros((0, D), dtype=np.float32)

    X = np.concatenate(all_feats, axis=0)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    print(f"  [group {name}] Collected {X.shape[0]} feature vectors, dim={X.shape[1]}")
    return X.astype(np.float32)

# =========================
# TRAIN / REBUILD CONFIG
# =========================

def train_spectrum_tokenizers():
    files = list_parquet_files()
    print(f"Found {len(files)} parquet files.")

    all_cols = union_columns(files)

    if DO_ZTF_AUTODETECT:
        ztf_groups = autodetect_ztf_groups(all_cols)
    else:
        ztf_groups = []

    groups = list(BASE_GROUPS) + ztf_groups
    print(f"\nTotal groups to consider: {len(groups)}")

    group_names: List[str] = []
    group_means: Dict[str, np.ndarray] = {}
    group_stds: Dict[str, np.ndarray] = {}
    group_codebooks: Dict[str, np.ndarray] = {}
    group_feature_dim: Dict[str, int] = {}
    group_columns: Dict[str, Dict[str, Any]] = {}

    for group in tqdm(groups, desc="[groups]", unit="group"):
        name = group["name"]
        out_joblib = Path(f"spectrum_kmeans_{name}.joblib")

        # Always collect a sample to compute mean/std for config
        feats = collect_features_for_group(group, files)
        if feats.shape[0] < max(1000, CODEBOOK_SIZE):  # small but nonzero threshold for stats
            print(f"[group {name}] Not enough data even for mean/std (got {feats.shape[0]}). Skipping group.")
            continue

        mean = feats.mean(axis=0)
        std = feats.std(axis=0) + 1e-8

        # If model exists, LOAD it; otherwise train it
        if out_joblib.exists():
            print(f"[group {name}] {out_joblib.name} exists -> loading model (no retrain).")
            kmeans = joblib.load(out_joblib)
        else:
            if feats.shape[0] < CODEBOOK_SIZE * 10:
                print(f"[group {name}] Not enough data for robust KMeans (got {feats.shape[0]}). Skipping training.")
                continue

            feats_std = (feats - mean) / std

            print(f"[group {name}] Training MiniBatchKMeans with {CODEBOOK_SIZE} clusters...")
            kmeans = MiniBatchKMeans(
                n_clusters=CODEBOOK_SIZE,
                batch_size=4096,
                max_iter=50,
                verbose=0,
                n_init=1,
            )
            kmeans.fit(feats_std)

            joblib.dump(kmeans, out_joblib)
            print(f"[group {name}] Done. Saved KMeans model to {out_joblib}")

        # Save group info into config (works for both trained and loaded)
        centers = getattr(kmeans, "cluster_centers_", None)
        if centers is None:
            print(f"[group {name}] Loaded model has no cluster_centers_. Skipping.")
            continue

        group_names.append(name)
        group_means[name] = mean.astype(np.float32)
        group_stds[name] = std.astype(np.float32)
        group_codebooks[name] = np.asarray(centers, dtype=np.float32)
        group_feature_dim[name] = int(feats.shape[1])
        group_columns[name] = group

    if not group_names:
        raise RuntimeError("No groups could be written to the npz (no data / missing models / all skipped).")

    ensure_dir_for_file(TOKENIZER_CONFIG)
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