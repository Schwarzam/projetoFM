#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json

import numpy as np
import polars as pl
from tqdm.auto import tqdm

# =========================
# CONFIG
# =========================

INPUT_DIR = "/home/astrodados4/downloads/hypercube"
FILE_PATTERN = "datacube_*.parquet"

TOKENIZER_CONFIG = "scalar_tokenizer_config.npz"
OUT_JSON = "scalar_tokenizer_benchmark.json"

# Fast benchmarking caps
MAX_FILES = 50                 # increase for better stats
MAX_ROWS_PER_FILE = 200_000    # rows sampled per file
MAX_VALUES_PER_COL = 1_000_000 # cap total values per column across all files

# If None, benchmark all columns in config
COLUMNS: Optional[List[str]] = None

# Reproducible sampling
RNG_SEED = 0

# =========================
# IO
# =========================

def list_parquet_files() -> List[Path]:
    p = Path(INPUT_DIR)
    files = sorted(p.glob(FILE_PATTERN))
    if not files:
        raise RuntimeError(f"No parquet files in {INPUT_DIR} matching {FILE_PATTERN}")
    return files[:MAX_FILES]

def load_scalar_tokenizer_config(path: str) -> Dict[str, Any]:
    npz = np.load(path, allow_pickle=True)
    bin_edges: Dict[str, np.ndarray] = npz["bin_edges"].item()
    N_BINS = int(npz["N_BINS"])
    return {"bin_edges": bin_edges, "N_BINS": N_BINS, "meta": {k: npz[k].item() if k in npz.files else None for k in []}}

# =========================
# TOKENIZATION
# =========================

def encode_scalar_tokens(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    values: float array (finite)
    edges: shape (N_BINS+1,), strictly increasing
    returns token in [0, N_BINS-1]
    """
    # bin index such that edges[i] <= x < edges[i+1]
    # searchsorted(..., side="right") - 1 gives i
    t = np.searchsorted(edges, values, side="right") - 1
    # clamp to [0, N_BINS-1]
    t = np.clip(t, 0, edges.size - 2)
    return t.astype(np.int32, copy=False)

def bin_midpoints(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])

# =========================
# METRICS
# =========================

def token_distribution_metrics(tokens: np.ndarray, n_bins: int) -> Dict[str, Any]:
    if tokens.size == 0:
        return {"n": 0, "status": "no_data"}

    counts = np.bincount(tokens, minlength=n_bins).astype(np.float64)
    used = int(np.sum(counts > 0))
    p = counts / max(counts.sum(), 1.0)
    p_nz = p[p > 0]
    entropy = float(-np.sum(p_nz * np.log(p_nz)))
    perplexity = float(np.exp(entropy))
    top10_mass = float(np.sort(p)[-10:].sum()) if n_bins >= 10 else float(np.sort(p).sum())

    return {
        "n": int(tokens.size),
        "used_bins": used,
        "used_bins_frac": float(used / max(n_bins, 1)),
        "entropy_nats": entropy,
        "perplexity": perplexity,
        "perplexity_frac": float(perplexity / max(n_bins, 1)),
        "top10_mass": top10_mass,
    }

def quantization_error(values: np.ndarray, tokens: np.ndarray, edges: np.ndarray) -> Dict[str, Any]:
    if values.size == 0:
        return {"status": "no_data"}

    mids = bin_midpoints(edges)
    recon = mids[tokens]  # midpoint reconstruction
    diff = values - recon
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    p95_abs = float(np.quantile(np.abs(diff), 0.95))
    return {
        "status": "ok",
        "mae": mae,
        "rmse": rmse,
        "p95_abs_err": p95_abs,
        "val_min": float(values.min()),
        "val_med": float(np.median(values)),
        "val_max": float(values.max()),
    }

# =========================
# SAMPLING
# =========================

def sample_column_values_from_files(
    col: str,
    files: List[Path],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Returns (values_finite, stats_about_missingness)
    """
    chunks: List[np.ndarray] = []
    total = 0

    files_with_col = 0
    files_missing_col = 0
    rows_seen = 0
    rows_with_value = 0
    n_nan_inf = 0

    for path in files:
        if total >= MAX_VALUES_PER_COL:
            break

        try:
            # read only needed column
            df = pl.read_parquet(path, columns=[col])
        except Exception:
            files_missing_col += 1
            continue

        files_with_col += 1
        s = df[col]
        rows_seen += df.height

        # convert to float64 for error metrics
        try:
            if s.dtype == pl.Boolean:
                arr = s.cast(pl.UInt8).to_numpy().astype(np.float64, copy=False)
            elif s.dtype.is_float():
                arr = s.to_numpy().astype(np.float64, copy=False)
            elif s.dtype.is_integer():
                arr = s.to_numpy().astype(np.float64, copy=False)
            else:
                # skip unsupported
                continue
        except Exception:
            continue

        if arr.size == 0:
            continue

        # subsample rows per file (fast)
        if arr.size > MAX_ROWS_PER_FILE:
            idx = rng.choice(arr.size, size=MAX_ROWS_PER_FILE, replace=False)
            arr = arr[idx]

        # count missing (nan/inf)
        finite = np.isfinite(arr)
        n_nan_inf += int((~finite).sum())
        arr = arr[finite]
        rows_with_value += int(arr.size)

        if arr.size == 0:
            continue

        remaining = MAX_VALUES_PER_COL - total
        if arr.size > remaining:
            idx = rng.choice(arr.size, size=int(remaining), replace=False)
            arr = arr[idx]

        chunks.append(arr)
        total += arr.size

    if not chunks:
        vals = np.array([], dtype=np.float64)
    else:
        vals = np.concatenate(chunks, axis=0)

    meta = {
        "files_considered": int(min(len(files), MAX_FILES)),
        "files_with_col": int(files_with_col),
        "files_missing_col": int(files_missing_col),
        "rows_seen": int(rows_seen),
        "values_collected_finite": int(vals.size),
        "nan_inf_dropped": int(n_nan_inf),
    }
    return vals, meta

# =========================
# MAIN
# =========================

def main():
    rng = np.random.default_rng(RNG_SEED)
    files = list_parquet_files()
    print(f"Found {len(files)} parquet files (capped by MAX_FILES={MAX_FILES}).")

    cfg = load_scalar_tokenizer_config(TOKENIZER_CONFIG)
    bin_edges: Dict[str, np.ndarray] = cfg["bin_edges"]
    n_bins = int(cfg["N_BINS"])

    cols = COLUMNS if COLUMNS is not None else sorted(bin_edges.keys())
    if not cols:
        raise RuntimeError("No columns to benchmark.")

    results: Dict[str, Any] = {
        "config": {
            "INPUT_DIR": INPUT_DIR,
            "FILE_PATTERN": FILE_PATTERN,
            "TOKENIZER_CONFIG": TOKENIZER_CONFIG,
            "MAX_FILES": MAX_FILES,
            "MAX_ROWS_PER_FILE": MAX_ROWS_PER_FILE,
            "MAX_VALUES_PER_COL": MAX_VALUES_PER_COL,
            "N_BINS": n_bins,
            "RNG_SEED": RNG_SEED,
        },
        "columns": {},
    }

    for col in tqdm(cols, desc="[bench] columns", unit="col"):
        edges = np.asarray(bin_edges[col], dtype=np.float64)
        if edges.ndim != 1 or edges.size != n_bins + 1:
            results["columns"][col] = {"status": "bad_edges", "edges_shape": list(edges.shape)}
            continue

        vals, meta = sample_column_values_from_files(col, files, rng=rng)
        if vals.size == 0:
            results["columns"][col] = {"status": "no_data", "meta": meta}
            continue

        tokens = encode_scalar_tokens(vals, edges)
        dist = token_distribution_metrics(tokens, n_bins=n_bins)
        qerr = quantization_error(vals, tokens, edges)

        results["columns"][col] = {
            "status": "ok",
            "meta": meta,
            "dist": dist,
            "quant_error_midpoint": qerr,
        }

        print(
            f"[{col}] n={dist['n']} used={dist['used_bins']}/{n_bins} "
            f"perp={dist['perplexity']:.1f} rmse={qerr.get('rmse', float('nan')):.4g}"
        )

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark report to {OUT_JSON}")

if __name__ == "__main__":
    main()