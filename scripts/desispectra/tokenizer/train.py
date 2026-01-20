#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from astromodal.config import load_config
from astromodal.datasets.desi_spectra import (
    LatentsPolarsDataset,
    collate_latents_pad_to_patch_multiple,
)
from astromodal.tokenizers.spectralrvq import SpectralPatchRVQ
from astromodal.tokenizers.rvq import ResidualVQ


# =============================================================================
# CONFIG
# =============================================================================
CONFIG_PATH = "/home/schwarz/projetoFM/config.yaml"

# Must match the **previous latent-encoding script**
LATENTS_SUBFOLDER = "spectra_latents_desi"   # inside hdd_folder
LATENT_COL = "latent"
ID_COL = "id"

# AE latent info
LATENT_DIM = 16

# RVQ / tokenizer
PATCH_SIZE = 2
NUM_STAGES = 3
CODEBOOK_SIZE = 1024
DECAY = 0.99

# Training
BATCH_SIZE = 128
NUM_WORKERS = 4
EPOCHS = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_NAME = "spectral_patch_rvq_desi.pth"
# =============================================================================


def main():
    config = load_config(CONFIG_PATH)

    latents_root = Path(config["hdd_folder"]) / LATENTS_SUBFOLDER
    assert latents_root.exists(), f"Latents folder not found: {latents_root}"

    # -------------------------------------------------------------------------
    # Load ALL latent parquet files
    # -------------------------------------------------------------------------
    latent_files = sorted(latents_root.glob("*.parquet"))
    assert len(latent_files) > 0, "No latent parquet files found"

    lat_df = None
    for f in tqdm(latent_files, desc="Loading latent files"):
        try:
            df = pl.read_parquet(f, columns=[ID_COL, LATENT_COL], use_pyarrow=True)
            if df.height == 0:
                continue
            lat_df = df if lat_df is None else pl.concat(
                [lat_df, df], how="vertical", rechunk=False
            )
        except Exception:
            pass

    assert lat_df is not None and lat_df.height > 0, "Empty latent dataframe"

    # -------------------------------------------------------------------------
    # Dataset + DataLoader
    # -------------------------------------------------------------------------
    ds_lat = LatentsPolarsDataset(
        lat_df,
        id_col=ID_COL,
        latent_col=LATENT_COL,
        latent_dim=LATENT_DIM,
        nested_layout="auto",
    )

    loader_lat = DataLoader(
        ds_lat,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=lambda batch: collate_latents_pad_to_patch_multiple(
            batch, patch_size=PATCH_SIZE
        ),
    )

    # -------------------------------------------------------------------------
    # Tokenizer
    # -------------------------------------------------------------------------
    P = PATCH_SIZE
    C = LATENT_DIM
    D = P * C

    rvq = ResidualVQ(
        dim=D,
        num_stages=NUM_STAGES,
        codebook_size=CODEBOOK_SIZE,
        decay=DECAY,
    ).to(DEVICE)

    tok = SpectralPatchRVQ(
        rvq=rvq,
        patch_size=P,
        channels=C,
    ).to(DEVICE)

    # -------------------------------------------------------------------------
    # Train (EMA-only)
    # -------------------------------------------------------------------------
    tok.train()
    for epoch in range(EPOCHS):
        stats = tok.train_epoch(loader_lat, device=DEVICE, update_ema=True)
        print(f"Epoch {epoch:02d} | {stats}")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    out_path = Path(config["models_folder"]) / OUT_NAME
    tok.save(out_path)
    print(f"Saved tokenizer to: {out_path}")


if __name__ == "__main__":
    main()