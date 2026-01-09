#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import logpool
import polars as pl
import torch
from tqdm.auto import tqdm

from astromodal.config import load_config
from astromodal.datasets.datacubes import load_datacube_files
from astromodal.models.autoencoder import AutoEncoder


# =========================
# User config
# =========================

ID_COL = "id"  # change if needed

BANDS = ["F378", "F395", "F410", "F430", "F515", "F660", "F861", "R", "I", "Z", "U", "G"]
CUT_COLS = [f"splus_cut_{b}" for b in BANDS]

IMG_SIZE = 96
NPIX = IMG_SIZE * IMG_SIZE

# rows per GPU forward (tune; 512-4096 depending on VRAM)
BATCH_ROWS = 1024

# how many rows to read/preprocess per chunk from parquet
CHUNK_ROWS = 32768

SKIP_EXISTING = True

# when True, run a background CPU thread preparing next chunk
PREFETCH = True


# =========================
# Preprocessing (same as dataset)
# =========================

def _fix_flat(v: Any, *, npix: int = NPIX) -> np.ndarray:
    """Return float32 flat array length npix. None/bad -> zeros, wrong len -> pad/truncate."""
    if v is None:
        return np.zeros(npix, dtype=np.float32)

    # common: list/tuple
    if isinstance(v, (list, tuple)):
        a = np.asarray(v, dtype=np.float32).ravel()
    else:
        try:
            a = np.asarray(v, dtype=np.float32).ravel()
        except Exception:
            return np.zeros(npix, dtype=np.float32)

    if a.size == npix:
        return a

    out = np.zeros(npix, dtype=np.float32)
    if a.size > 0:
        out[: min(npix, a.size)] = a[: min(npix, a.size)]
    return out


def arcsinh_norm_per_band(
    x: torch.Tensor,          # (C,H,W) float32
    m_valid: torch.Tensor,    # (C,H,W) float32
    eps: float = 1e-6,
    min_valid: int = 10,
    mad_mult: float = 3.0,
) -> torch.Tensor:
    """Robust per-band arcsinh normalization using valid pixels only (dataset-equivalent)."""
    C, H, W = x.shape
    x_out = torch.empty_like(x)

    for c in range(C):
        vals = x[c][m_valid[c] > 0.5]
        if vals.numel() < min_valid:
            vals = x[c].reshape(-1)

        if vals.numel() == 0:
            x_out[c] = x[c]
            continue

        med = vals.median()
        mad = (vals - med).abs().median()
        scale = mad_mult * mad + eps
        x_out[c] = torch.asinh((x[c] - med) / scale)

    return x_out


def _build_batch_from_lists(
    ids: List[Any],
    per_band_lists: List[List[Any]],  # len=C, each list len=B (each element list/array)
) -> Tuple[List[Any], torch.Tensor, torch.Tensor]:
    """
    Build (ids, x_norm, m_valid) for a batch.
    x_norm: (B,C,H,W) float32
    m_valid: (B,C,H,W) float32
    """
    B = len(ids)
    C = len(per_band_lists)

    # allocate numpy for speed
    x_np = np.zeros((B, C, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    m_np = np.zeros((B, C, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # fill
    for c in range(C):
        col_list = per_band_lists[c]
        # col_list length B
        for i in range(B):
            flat = _fix_flat(col_list[i])
            img = flat.reshape(IMG_SIZE, IMG_SIZE)

            # dataset validity: finite AND non-zero
            valid = np.isfinite(img) & (img != 0.0)

            # clean NaN/Inf -> 0
            img = img.copy()
            img[~np.isfinite(img)] = 0.0

            x_np[i, c] = img
            m_np[i, c] = valid.astype(np.float32)

    # to torch
    x = torch.from_numpy(x_np)   # (B,C,H,W)
    m = torch.from_numpy(m_np)   # (B,C,H,W)

    # apply arcsinh norm per object (same as dataset)
    # (loop over batch; still fast enough and keeps identical behavior)
    x_norm = torch.empty_like(x)
    for i in range(B):
        x_norm[i] = arcsinh_norm_per_band(x[i], m[i])

    return ids, x_norm, m


# =========================
# Model load
# =========================

def build_model_from_ckpt(ckpt_path: Path, device: torch.device) -> AutoEncoder:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})

    in_channels = int(cfg.get("in_channels", len(BANDS)))
    latent_dim = int(cfg.get("latent_dim", 2))

    model = AutoEncoder(in_channels=in_channels, latent_dim=latent_dim)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval().to(device)

    logpool.info(f"Loaded model from {ckpt_path} | in_channels={in_channels} latent_dim={latent_dim}")
    return model


# =========================
# Chunk loader + prefetch
# =========================

def load_and_prepare_chunk(
    in_path: Path,
    *,
    start: int,
    nrows: int,
) -> Tuple[List[Any], torch.Tensor, torch.Tensor]:
    """
    CPU side: read parquet slice -> build (ids, x_norm, m_valid) tensors on CPU.
    """
    df = pl.read_parquet(in_path, columns=[ID_COL] + CUT_COLS, use_pyarrow=True)

    if df.height == 0:
        return [], torch.empty(0), torch.empty(0)

    # slice
    chunk = df.slice(start, nrows)
    if chunk.height == 0:
        return [], torch.empty(0), torch.empty(0)

    ids = chunk[ID_COL].to_list()
    per_band_lists = [chunk[col].to_list() for col in CUT_COLS]  # C lists of len=B

    return _build_batch_from_lists(ids, per_band_lists)


@torch.no_grad()
def encode_one_file(
    *,
    model: AutoEncoder,
    in_path: Path,
    out_path: Path,
    device: torch.device,
    chunk_rows: int,
    batch_rows: int,
    prefetch: bool,
) -> None:
    # quick schema checks
    # (read minimal once to validate)
    head = pl.read_parquet(in_path, columns=[ID_COL] + CUT_COLS, n_rows=1)
    if ID_COL not in head.columns:
        raise KeyError(f"Missing '{ID_COL}' in {in_path.name}. Available: {head.columns}")
    for c in CUT_COLS:
        if c not in head.columns:
            raise KeyError(f"Missing '{c}' in {in_path.name}")

    # Read full df height cheaply (parquet metadata via polars)
    df_meta = pl.read_parquet(in_path, columns=[ID_COL], use_pyarrow=True)
    n_total = df_meta.height
    if n_total == 0:
        logpool.info(f"Skipping empty file: {in_path.name}")
        return

    out_ids: List[Any] = []
    out_latents: List[List[float]] = []

    # We'll read/prep in CHUNK_ROWS blocks, but run GPU forwards in smaller BATCH_ROWS.
    def chunk_iter_ranges():
        for start in range(0, n_total, chunk_rows):
            yield start, min(chunk_rows, n_total - start)

    ranges = list(chunk_iter_ranges())

    executor = ThreadPoolExecutor(max_workers=1) if prefetch else None
    fut = None

    def submit_next(ix: int):
        if executor is None:
            return None
        start, nrows = ranges[ix]
        return executor.submit(load_and_prepare_chunk, in_path, start=start, nrows=nrows)

    # prime first
    if prefetch:
        fut = submit_next(0)
        chunk_index = 0
        while chunk_index < len(ranges):
            ids, x_norm, m_valid = fut.result()
            chunk_index += 1
            fut = submit_next(chunk_index) if chunk_index < len(ranges) else None

            if len(ids) == 0:
                continue

            # now process this prepared chunk in GPU mini-batches
            for s in range(0, x_norm.shape[0], batch_rows):
                xb = x_norm[s:s+batch_rows].to(device, non_blocking=True)
                mb = m_valid[s:s+batch_rows].to(device, non_blocking=True)

                z_map = model.encode(xb)  # (B, latent_dim, 24, 24)
                z_flat = z_map.reshape(z_map.shape[0], -1).float().cpu()

                mv = mb > 0.5  # unused for encoding, but kept if you want later checks

                out_ids.extend(ids[s:s+batch_rows])
                out_latents.extend(z_flat.numpy().astype(np.float32).tolist())
    else:
        # no prefetch: load/prepare synchronously
        for (start, nrows) in ranges:
            ids, x_norm, m_valid = load_and_prepare_chunk(in_path, start=start, nrows=nrows)
            if len(ids) == 0:
                continue

            for s in range(0, x_norm.shape[0], batch_rows):
                xb = x_norm[s:s+batch_rows].to(device, non_blocking=True)
                z_map = model.encode(xb)
                z_flat = z_map.reshape(z_map.shape[0], -1).float().cpu()

                out_ids.extend(ids[s:s+batch_rows])
                out_latents.extend(z_flat.numpy().astype(np.float32).tolist())

    if executor is not None:
        executor.shutdown(wait=True)

    out_df = pl.DataFrame(
        {
            ID_COL: out_ids,
            "latent": out_latents,  # List[f32]
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(out_path)
    logpool.info(f"Wrote {out_path.name} (rows={out_df.height})")


# =========================
# Main
# =========================

def main():
    config = load_config("/home/schwarz/projetoFM/config.yaml")
    datacube_paths = config["datacubes_paths"]

    ckpt_path = Path(config["models_folder"]) / "autoencoder_model_silu.pth"
    out_dir = Path(config["models_folder"]) / "latents_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logpool.info(f"Device: {device}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_model_from_ckpt(ckpt_path, device=device)

    # get all files (train+val)
    train_files, val_files = load_datacube_files(
        datacubes_paths=datacube_paths,
        train_val_split=0.85,     # irrelevant; we want all
        nfiles_subsample=None,
        seed=42,
    )
    files = [Path(f) for f in (train_files + val_files)]
    logpool.info(f"Encoding {len(files)} parquet files -> {out_dir}")

    # speed-ups
    torch.backends.cudnn.benchmark = True

    for in_path in tqdm(files, desc="Encoding files"):
        out_path = out_dir / in_path.name  # exact same filename, no part-0000

        if SKIP_EXISTING and out_path.exists():
            continue

        try:
            encode_one_file(
                model=model,
                in_path=in_path,
                out_path=out_path,
                device=device,
                chunk_rows=CHUNK_ROWS,
                batch_rows=BATCH_ROWS,
                prefetch=PREFETCH,
            )
        except Exception as e:
            logpool.info(f"⚠️ Failed on {in_path.name}: {e}")

    logpool.info("Done.")


if __name__ == "__main__":
    main()