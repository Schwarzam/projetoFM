#!/usr/bin/env python
import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from tqdm.auto import tqdm

from sklearn.cluster import MiniBatchKMeans
import joblib  # pip install joblib if needed

# =========================
# CONFIG
# =========================

# Directory where run_inference.py saved your latents
LATENT_DIR = "output_latents"

# Where to save the codebook
CODEBOOK_PATH = "codebook_kmeans_2048.joblib"

# Where to save the tokens
TOKENS_DIR = "tokens_for_transformer"

# Must match your autoencoder
LATENT_DIM = 2
LATENT_H = 24
LATENT_W = 24

# Codebook / K-means config
CODEBOOK_SIZE = 2048            # number of clusters (codebook entries)
KMEANS_BATCH_SIZE = 4096        # how many vectors per partial_fit
KMEANS_MAX_ITER = 50            # total iterations / passes
MAX_VECTORS_FOR_KMEANS = 2_000_000  # global cap of vectors used to fit codebook

# Sampling per file to avoid gigantic arrays
MAX_VECTORS_PER_FILE = 100_000

# Token dtype
TOKEN_DTYPE = np.uint16  # enough up to 65535 codes, smaller than int32

# =========================
# HELPERS
# =========================

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def list_latent_files(latent_dir: str) -> List[Path]:
    p = Path(latent_dir)
    files = sorted(p.glob("*.npz"))
    return files


def load_latents_from_npz(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[str]]:
    """
    Expects npz with:
      - latents: shape (N, LATENT_DIM, H, W)
      - (optional) ids: shape (N,)
      - (optional) id_col: str
    """
    data = np.load(path, allow_pickle=True)
    latents = data["latents"]  # (N, C, H, W)
    ids = data["ids"] if "ids" in data.files else None
    id_col = str(data["id_col"]) if "id_col" in data.files else None
    return latents, ids, id_col


def latents_to_vectors(latents: np.ndarray) -> np.ndarray:
    """
    (N, C, H, W) -> (N*H*W, C)
    """
    N, C, H, W = latents.shape
    assert C == LATENT_DIM, f"Expected LATENT_DIM={LATENT_DIM}, got {C}"
    return latents.transpose(0, 2, 3, 1).reshape(-1, C)


def sample_vectors(vectors: np.ndarray, max_samples: int) -> np.ndarray:
    """
    Randomly sample up to max_samples rows from vectors.
    """
    n = vectors.shape[0]
    if n <= max_samples:
        return vectors
    idx = np.random.choice(n, size=max_samples, replace=False)
    return vectors[idx]


# =========================
# CODEBOOK TRAINING
# =========================

def fit_codebook(files: List[Path]) -> MiniBatchKMeans:
    """
    Stream through latent files and fit a MiniBatchKMeans codebook.
    """
    print("\n[codebook] Fitting MiniBatchKMeans codebook...")
    kmeans = MiniBatchKMeans(
        n_clusters=CODEBOOK_SIZE,
        batch_size=KMEANS_BATCH_SIZE,
        max_iter=KMEANS_MAX_ITER,
        verbose=0,
        init="k-means++",
        n_init=1,  # we are streaming; 1 is enough here
    )

    total_used = 0
    # Shuffle files to avoid weird ordering biases
    files_shuffled = files.copy()
    random.shuffle(files_shuffled)

    for f in tqdm(files_shuffled, desc="Streaming latents for codebook"):
        latents, _, _ = load_latents_from_npz(f)
        vectors = latents_to_vectors(latents)  # (N*H*W, D)
        # sample per file
        vectors = sample_vectors(vectors, MAX_VECTORS_PER_FILE)

        # maybe cut globally
        if MAX_VECTORS_FOR_KMEANS is not None:
            remaining = MAX_VECTORS_FOR_KMEANS - total_used
            if remaining <= 0:
                break
            if vectors.shape[0] > remaining:
                vectors = vectors[:remaining]

        if vectors.shape[0] == 0:
            continue

        kmeans.partial_fit(vectors)
        total_used += vectors.shape[0]

        if MAX_VECTORS_FOR_KMEANS is not None and total_used >= MAX_VECTORS_FOR_KMEANS:
            break

    print(f"[codebook] Fitting done. Total vectors used: {total_used}")
    return kmeans


def load_or_fit_codebook(files: List[Path]) -> MiniBatchKMeans:
    if Path(CODEBOOK_PATH).exists():
        print(f"[codebook] Loading existing codebook from {CODEBOOK_PATH}")
        kmeans = joblib.load(CODEBOOK_PATH)
        return kmeans

    kmeans = fit_codebook(files)
    ensure_dir(Path(CODEBOOK_PATH).parent.as_posix())
    joblib.dump(kmeans, CODEBOOK_PATH)
    print(f"[codebook] Saved codebook to {CODEBOOK_PATH}")
    return kmeans


# =========================
# ENCODING / TOKENS
# =========================

def encode_latents_to_tokens(
    kmeans: MiniBatchKMeans,
    latents: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    latents: (N, C, H, W)
    returns:
      tokens_2d: (N, H, W) integer codes
      tokens_flat: (N, H*W) flattened sequences
    """
    N, C, H, W = latents.shape
    assert C == LATENT_DIM, f"Expected LATENT_DIM={LATENT_DIM}, got {C}"
    assert H == LATENT_H and W == LATENT_W, f"Expected ({LATENT_H},{LATENT_W}), got ({H},{W})"

    vectors = latents_to_vectors(latents)  # (N*H*W, C)

    # to avoid memory peak, we can predict in chunks
    B = 100_000  # chunk size; adjust if needed
    codes_list = []
    for i in range(0, vectors.shape[0], B):
        chunk = vectors[i:i+B]
        codes_chunk = kmeans.predict(chunk)
        codes_list.append(codes_chunk)

    codes = np.concatenate(codes_list, axis=0)  # (N*H*W,)
    tokens_2d = codes.reshape(N, H, W).astype(TOKEN_DTYPE)
    tokens_flat = tokens_2d.reshape(N, H * W)
    return tokens_2d, tokens_flat


def process_all_files():
    ensure_dir(TOKENS_DIR)

    files = list_latent_files(LATENT_DIR)
    if not files:
        print(f"No latent .npz files found in {LATENT_DIR}")
        return

    print("Found latent files:")
    for f in files:
        print("  -", f)

    # 1) Load or fit codebook
    kmeans = load_or_fit_codebook(files)

    # 2) Encode each file to tokens
    for f in files:
        print(f"\n[tokens] Processing {f.name}")
        latents, ids, id_col = load_latents_from_npz(f)

        tokens_2d, tokens_flat = encode_latents_to_tokens(kmeans, latents)

        out_name = f.stem.replace("_latents", "") + "_tokens.npz"
        out_path = Path(TOKENS_DIR) / out_name

        if ids is not None:
            np.savez_compressed(
                out_path,
                tokens_2d=tokens_2d,
                tokens_flat=tokens_flat,
                ids=ids,
                id_col=id_col if id_col is not None else "",
            )
        else:
            np.savez_compressed(
                out_path,
                tokens_2d=tokens_2d,
                tokens_flat=tokens_flat,
            )

        print(
            f"[tokens] Saved {out_path} "
            f"(tokens_2d shape={tokens_2d.shape}, tokens_flat shape={tokens_flat.shape})"
        )


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    process_all_files()