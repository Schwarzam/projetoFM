#!/usr/bin/env python3
from __future__ import annotations

import os
import time
import json
import random
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from tqdm.auto import tqdm
import joblib

# ---------------------------------------------------------------------
# THREAD LIMITS (avoid oversubscription)
# ---------------------------------------------------------------------
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("GOTO_NUM_THREADS", "1")

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
LATENT_DIR = "output_latents"          # expects .npz with key: "latents"
TOKENS_DIR = "image_tokens"           # expects .npz with key: tokens_flat or tokens_2d
FORCE_MODE = ""                       # "", "latents", "tokens"

CODEBOOK_PATH = "codebook_kmeans_2048.joblib"

MAX_FILES = 50
HOLDOUT_FRACTION = 0.2
RNG_SEED = 0

MAX_VECTORS_PER_FILE = 300_000
MAX_VECTORS_TOTAL_PER_SPLIT = 2_000_000
PRED_CHUNK = 200_000

MAX_TOKENS_PER_FILE = 2_000_000
TOPK = 10

# If True, we will save a float64-patched copy of the codebook alongside the original
SAVE_PATCHED_CODEBOOK = True
PATCHED_CODEBOOK_SUFFIX = ".float64.joblib"


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------
def list_npz_files(d: str, max_files: int) -> List[Path]:
    p = Path(d)
    if not p.exists():
        return []
    files = sorted(p.glob("*.npz"))
    if max_files and max_files > 0:
        files = files[:max_files]
    return files


def detect_mode() -> str:
    if FORCE_MODE in ("latents", "tokens"):
        return FORCE_MODE

    if list_npz_files(LATENT_DIR, MAX_FILES):
        return "latents"
    if list_npz_files(TOKENS_DIR, MAX_FILES):
        return "tokens"

    raise RuntimeError(
        f"No .npz found in LATENT_DIR='{LATENT_DIR}' or TOKENS_DIR='{TOKENS_DIR}'."
    )


# ---------------------------------------------------------------------
# Latents / Tokens loading
# ---------------------------------------------------------------------
def load_latents_npz(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True, mmap_mode="r")
    if "latents" not in data.files:
        raise KeyError(f"{path} missing 'latents'. Keys: {data.files}")
    return data["latents"]  # (N, C, H, W)


def latents_to_vectors(latents: np.ndarray) -> np.ndarray:
    if latents.ndim != 4:
        raise ValueError(f"Expected (N,C,H,W). Got {latents.shape}")
    N, C, H, W = latents.shape
    return latents.transpose(0, 2, 3, 1).reshape(-1, C)


def load_tokens_npz(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True, mmap_mode="r")
    if "tokens_flat" in data.files:
        return data["tokens_flat"].reshape(-1)
    if "tokens_2d" in data.files:
        return data["tokens_2d"].reshape(-1)
    raise KeyError(f"{path} has no tokens_flat/tokens_2d. Keys: {data.files}")


def sample_rows(x: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    if max_rows is None or max_rows <= 0 or n <= max_rows:
        return x
    idx = rng.choice(n, size=max_rows, replace=False)
    return x[idx]


def sample_tokens(x: np.ndarray, max_tokens: int, rng: np.random.Generator) -> np.ndarray:
    n = x.size
    if max_tokens is None or max_tokens <= 0 or n <= max_tokens:
        return x
    idx = rng.choice(n, size=max_tokens, replace=False)
    return x[idx]


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def dist_metrics_from_codes(codes: np.ndarray, codebook_size: int) -> Dict[str, Any]:
    codes = np.asarray(codes, dtype=np.int64)
    M = int(codes.size)
    if M == 0:
        return {
            "n": 0,
            "used_bins": 0,
            "used_bins_frac": 0.0,
            "entropy_nats": 0.0,
            "perplexity": 0.0,
            "perplexity_frac": 0.0,
            "top10_mass": 0.0,
        }

    counts = np.bincount(codes, minlength=codebook_size).astype(np.float64)
    p = counts / counts.sum()

    used = int((counts > 0).sum())
    used_frac = float(used / codebook_size)

    nz = p > 0
    ent = float(-(p[nz] * np.log(p[nz])).sum())
    perp = float(np.exp(ent))
    perp_frac = float(perp / codebook_size)

    k = min(TOPK, codebook_size)
    top_mass = float(np.sort(p)[-k:].sum())

    return {
        "n": M,
        "used_bins": used,
        "used_bins_frac": used_frac,
        "entropy_nats": ent,
        "perplexity": perp,
        "perplexity_frac": perp_frac,
        "top10_mass": top_mass,
    }


# ---------------------------------------------------------------------
# CRITICAL FIX: patch codebook to float64
# ---------------------------------------------------------------------
def patch_kmeans_to_float64(kmeans) -> None:
    """
    Make sklearn KMeans/MiniBatchKMeans safe for predict() in float64 path.
    This avoids: ValueError: expected 'const double' but got 'float'
    """
    kmeans.cluster_centers_ = np.ascontiguousarray(kmeans.cluster_centers_, dtype=np.float64)

    # Some sklearn versions carry these:
    if hasattr(kmeans, "_cluster_centers_"):
        kmeans._cluster_centers_ = np.ascontiguousarray(kmeans._cluster_centers_, dtype=np.float64)

    if hasattr(kmeans, "_centers"):
        kmeans._centers = np.ascontiguousarray(kmeans._centers, dtype=np.float64)

    # Distances/shift buffers (if present)
    if hasattr(kmeans, "_center_shift"):
        kmeans._center_shift = np.ascontiguousarray(kmeans._center_shift, dtype=np.float64)

    # Force dtype hint if present
    if hasattr(kmeans, "_dtype"):
        kmeans._dtype = np.float64


def encode_and_score_vectors(kmeans, vectors: np.ndarray, pred_chunk: int) -> Tuple[np.ndarray, Dict[str, float], float]:
    centers = np.ascontiguousarray(kmeans.cluster_centers_, dtype=np.float64)

    M = int(vectors.shape[0])
    D = int(centers.shape[1])
    if M == 0:
        return np.array([], dtype=np.int32), {"mse": 0.0, "rmse": 0.0, "mae": 0.0}, 0.0

    codes_list = []
    err_sq_sum = 0.0
    err_abs_sum = 0.0

    t0 = time.time()

    for i in range(0, M, pred_chunk):
        chunk = vectors[i:i + pred_chunk]

        # absolutely force float64 contiguous
        chunk64 = np.ascontiguousarray(chunk, dtype=np.float64)

        # sanity check (kept cheap)
        if chunk64.dtype != np.float64:
            raise RuntimeError(f"Internal error: chunk64 dtype={chunk64.dtype}, expected float64")

        c = kmeans.predict(chunk64).astype(np.int32, copy=False)
        codes_list.append(c)

        assigned = centers[c]  # (B, D) float64
        diff = chunk64 - assigned
        err_sq_sum += float(np.einsum("ij,ij->", diff, diff))
        err_abs_sum += float(np.abs(diff).sum())

    dt = max(time.time() - t0, 1e-12)
    vps = M / dt

    codes = np.concatenate(codes_list, axis=0)
    mse = err_sq_sum / (M * D)
    rmse = float(np.sqrt(mse))
    mae = err_abs_sum / (M * D)

    return codes, {"mse": float(mse), "rmse": rmse, "mae": float(mae)}, float(vps)


# ---------------------------------------------------------------------
# Streaming evaluation
# ---------------------------------------------------------------------
def stream_eval_latents(files: List[Path], kmeans, tag: str, rng: np.random.Generator) -> Dict[str, Any]:
    codebook_size = int(kmeans.n_clusters)

    total_vectors = 0
    all_codes = []
    mse_acc, rmse_acc, mae_acc, vps_acc = [], [], [], []

    for f in tqdm(files, desc=tag, unit="file"):
        lat = load_latents_npz(f)
        vec = latents_to_vectors(lat)

        vec = sample_rows(vec, MAX_VECTORS_PER_FILE, rng)

        remaining = MAX_VECTORS_TOTAL_PER_SPLIT - total_vectors
        if remaining <= 0:
            break
        if vec.shape[0] > remaining:
            vec = vec[:remaining]
        if vec.shape[0] == 0:
            continue

        codes, err, vps = encode_and_score_vectors(kmeans, vec, pred_chunk=PRED_CHUNK)

        all_codes.append(codes)
        mse_acc.append(err["mse"])
        rmse_acc.append(err["rmse"])
        mae_acc.append(err["mae"])
        vps_acc.append(vps)

        total_vectors += int(vec.shape[0])
        if total_vectors >= MAX_VECTORS_TOTAL_PER_SPLIT:
            break

    codes_all = np.concatenate(all_codes, axis=0) if all_codes else np.array([], dtype=np.int32)
    dist = dist_metrics_from_codes(codes_all, codebook_size)

    return {
        "tag": tag,
        "mode": "latents",
        "files": len(files),
        "vectors_total": int(total_vectors),
        "dist": dist,
        "quant_error": {
            "mse_mean": float(np.mean(mse_acc)) if mse_acc else 0.0,
            "rmse_mean": float(np.mean(rmse_acc)) if rmse_acc else 0.0,
            "mae_mean": float(np.mean(mae_acc)) if mae_acc else 0.0,
            "vectors_per_sec_mean": float(np.mean(vps_acc)) if vps_acc else 0.0,
        },
    }


def stream_eval_tokens(files: List[Path], kmeans, tag: str, rng: np.random.Generator) -> Dict[str, Any]:
    codebook_size = int(kmeans.n_clusters)

    total_tokens = 0
    all_codes = []

    for f in tqdm(files, desc=tag, unit="file"):
        tok = load_tokens_npz(f)
        tok = sample_tokens(tok, MAX_TOKENS_PER_FILE, rng)

        remaining = MAX_VECTORS_TOTAL_PER_SPLIT - total_tokens
        if remaining <= 0:
            break
        if tok.size > remaining:
            tok = tok[:remaining]
        if tok.size == 0:
            continue

        all_codes.append(np.asarray(tok, dtype=np.int64))
        total_tokens += int(tok.size)
        if total_tokens >= MAX_VECTORS_TOTAL_PER_SPLIT:
            break

    codes_all = np.concatenate(all_codes, axis=0) if all_codes else np.array([], dtype=np.int64)
    dist = dist_metrics_from_codes(codes_all, codebook_size)

    return {
        "tag": tag,
        "mode": "tokens",
        "files": len(files),
        "tokens_total": int(total_tokens),
        "dist": dist,
        "quant_error": None,
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    rng = np.random.default_rng(RNG_SEED)
    mode = detect_mode()

    input_dir = LATENT_DIR if mode == "latents" else TOKENS_DIR
    files = list_npz_files(input_dir, MAX_FILES)
    if not files:
        raise RuntimeError(f"No input files found in {input_dir}")

    print(f"[bench] mode={mode}")
    print(f"[bench] Using {len(files)} files (cap MAX_FILES={MAX_FILES})")
    print(f"[bench] Loading codebook: {CODEBOOK_PATH}")

    kmeans = joblib.load(CODEBOOK_PATH)
    print(f"[bench] codebook_size={int(kmeans.n_clusters)}  centers_dtype={kmeans.cluster_centers_.dtype}")

    # PATCH HERE (this is the key fix)
    patch_kmeans_to_float64(kmeans)
    print(f"[bench] patched centers_dtype={kmeans.cluster_centers_.dtype}")

    if SAVE_PATCHED_CODEBOOK:
        patched_path = str(CODEBOOK_PATH) + PATCHED_CODEBOOK_SUFFIX
        joblib.dump(kmeans, patched_path)
        print(f"[bench] saved patched codebook: {patched_path}")

    # Split files
    files_shuffled = files.copy()
    random.Random(RNG_SEED).shuffle(files_shuffled)

    n_hold = int(round(len(files_shuffled) * HOLDOUT_FRACTION))
    n_hold = max(1, n_hold) if len(files_shuffled) > 1 else 0

    holdout_files = files_shuffled[:n_hold]
    train_files = files_shuffled[n_hold:]

    print(f"[bench] train_files={len(train_files)} holdout_files={len(holdout_files)} (HOLDOUT_FRACTION={HOLDOUT_FRACTION})")

    if mode == "latents":
        train_res = stream_eval_latents(train_files, kmeans, "train_eval", rng)
        hold_res = stream_eval_latents(holdout_files, kmeans, "holdout_eval", rng) if holdout_files else None
    else:
        train_res = stream_eval_tokens(train_files, kmeans, "train_eval", rng)
        hold_res = stream_eval_tokens(holdout_files, kmeans, "holdout_eval", rng) if holdout_files else None

    results = {
        "config": {
            "mode": mode,
            "input_dir": input_dir,
            "CODEBOOK_PATH": CODEBOOK_PATH,
            "MAX_FILES": MAX_FILES,
            "HOLDOUT_FRACTION": HOLDOUT_FRACTION,
            "RNG_SEED": RNG_SEED,
            "MAX_VECTORS_PER_FILE": MAX_VECTORS_PER_FILE,
            "MAX_VECTORS_TOTAL_PER_SPLIT": MAX_VECTORS_TOTAL_PER_SPLIT,
            "PRED_CHUNK": PRED_CHUNK,
        },
        "train_eval": train_res,
        "holdout_eval": hold_res,
    }

    out_json = "benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[bench] Saved results to {out_json}")
    print("\n[bench] Summary train_eval:")
    print(json.dumps(train_res, indent=2))
    if hold_res is not None:
        print("\n[bench] Summary holdout_eval:")
        print(json.dumps(hold_res, indent=2))


if __name__ == "__main__":
    main()