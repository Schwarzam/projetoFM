#!/usr/bin/env python3
# coding: utf-8

from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from astromodal.config import load_config
from astromodal.tokenizers.resvq import SpatialRVQ


def main():
    # -----------------------------
    # Configurable variables
    # -----------------------------
    CONFIG_PATH = "/home/schwarz/projetoFM/config.yaml"
    MAX_FILES = 51          # read up to this many parquet files (0..MAX_FILES-1); set None for all
    CHANNELS = 2
    H = 24
    W = 24
    NUM_STAGES = 1
    CODEBOOK_SIZE = 2048
    DECAY = 0.99
    BATCH_SIZE = 64
    EPOCHS = 10
    SHUFFLE = True
    DROP_LAST = True

    # -----------------------------
    # Load config and parquet latents
    # -----------------------------
    config = load_config(CONFIG_PATH)
    hddfolder = Path(config["hdd_folder"]) / "image_latents"

    all_latents = None
    for key, file in enumerate(hddfolder.glob("*.parquet")):
        latents = pl.read_parquet(file, use_pyarrow=True)
        all_latents = latents if all_latents is None else pl.concat([all_latents, latents])

        if MAX_FILES is not None and key == (MAX_FILES - 1):
            break

    # -----------------------------
    # To torch
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latents_tensor = torch.tensor(all_latents["latent"].to_numpy(), device=device)
    latents = latents_tensor

    print("mean ||z||:", latents.norm(dim=1).mean().item())
    print("std  ||z||:", latents.norm(dim=1).std().item())
    print("mean z:", latents.mean().item(), "std z:", latents.std().item())

    # -----------------------------
    # Train SpatialRVQ (EMA)
    # -----------------------------
    srvq = SpatialRVQ(
        channels=CHANNELS,
        num_stages=NUM_STAGES,
        codebook_size=CODEBOOK_SIZE,
        decay=DECAY,
    ).to(device)
    srvq.train()

    z_map = latents.view(-1, CHANNELS, H, W)
    loader = DataLoader(
        TensorDataset(z_map),
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        drop_last=DROP_LAST,
    )

    for epoch in range(EPOCHS):
        stats = srvq.train_epoch(loader, device=device)
        print(f"Epoch {epoch}: {stats}")

    srvq.save(Path(config['models_folder']) / "srvq_spluscuts_1stage_2048codebs.pth")

if __name__ == "__main__":
    main()