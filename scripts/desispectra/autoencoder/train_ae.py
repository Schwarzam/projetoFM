#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from astromodal.config import load_config
from tqdm import tqdm
import polars as pl
from pathlib import Path
from astromodal.datasets.datacubes import load_datacube_files

from astromodal.models.spectral_autoencoder import SpectraAutoEncoder
from astromodal.datasets.desi_spectra import DesiStitchedFluxOnlyDataset, desi_collate_pad_flux_only
from torch.utils.data import DataLoader
import torch


# =============================================================================
# CONFIG (set what you need here)
# =============================================================================
CONFIG_PATH = "/home/schwarz/projetoFM/config.yaml"
STRIPE = "STRIPE82-0002"          # used to probe schema (columns)
NFILES_SUBSAMPLE = 800
TRAIN_VAL_SPLIT = 0.9

MAG_R_MAX = 21
BATCH_SIZE = 64
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 16
BASE_CHANNELS = 64
NUM_DOWN = 4
KERNEL_SIZE = 5
LR = 3e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 10

MODEL_OUT_NAME = "spectra_autoencoder_desi.pth"

SCALER_PATH = Path("scalers") / "desi_flux_standard_scaler.npz"

# =============================================================================


def main():
    config = load_config(CONFIG_PATH)

    # Probe parquet schema to select only needed columns
    file_probe = config["datacubes_paths"].replace("*", STRIPE)
    header = pl.read_parquet(file_probe, n_rows=0)
    columns = [c for c in header.columns if "desi" in c] + ["id", "mag_psf_r"]

    # List train/val parquet files
    train_files, val_files = load_datacube_files(
        config["datacubes_paths"],
        train_val_split=TRAIN_VAL_SPLIT,
        nfiles_subsample=NFILES_SUBSAMPLE,
    )

    # Load train df
    train_df = None
    for f in tqdm(train_files, desc="Loading train files"):
        try:
            df = pl.read_parquet(f, columns=columns, use_pyarrow=True)
            df = df.filter(pl.col(columns[0]).is_not_null())
            df = df.filter(pl.col("mag_psf_r") < MAG_R_MAX)
            if df.height == 0:
                continue
            train_df = df if train_df is None else pl.concat([train_df, df], how="vertical", rechunk=False)
        except Exception:
            pass

    # Load val df
    val_df = None
    for f in tqdm(val_files, desc="Loading val files"):
        try:
            df = pl.read_parquet(f, columns=columns, use_pyarrow=True)
            df = df.filter(pl.col(columns[0]).is_not_null())
            df = df.filter(pl.col("mag_psf_r") < MAG_R_MAX)
            if df.height == 0:
                continue
            val_df = df if val_df is None else pl.concat([val_df, df], how="vertical", rechunk=False)
        except Exception:
            pass

    # Extra filter
    train_df = train_df.filter(pl.col("desi_TARGET_RA").is_not_null())
    val_df = val_df.filter(pl.col("desi_TARGET_RA").is_not_null())

    # Dataloaders
    scaler_path = Path(config["models_folder"]) / "scalers" / "desi_flux_standard_scaler.npz"
    ds_train = DesiStitchedFluxOnlyDataset(train_df, scaler_path=scaler_path)
    dl_train = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=desi_collate_pad_flux_only,
        pin_memory=True,
    )

    ds_val = DesiStitchedFluxOnlyDataset(val_df, scaler_path=scaler_path)
    dl_val = DataLoader(
        ds_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=desi_collate_pad_flux_only,
        pin_memory=True,
    )

    # Model + optimizer
    model = SpectraAutoEncoder(
        latent_dim=LATENT_DIM,
        base_channels=BASE_CHANNELS,
        num_down=NUM_DOWN,
        k=KERNEL_SIZE,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Train
    for epoch in range(EPOCHS):
        tr = model.train_epoch(dl_train, opt, DEVICE)
        va = model.validate(dl_val, DEVICE)
        print(f"Epoch {epoch:02d} | train={tr:.6f} | val={va:.6f}")

    # Save
    out_path = Path(config["models_folder"]) / MODEL_OUT_NAME
    model.save(out_path)
    print(f"Saved model to: {out_path}")


if __name__ == "__main__":
    main()