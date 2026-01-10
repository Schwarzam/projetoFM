#!/usr/bin/env python
# coding: utf-8
import logpool
import torch


from astromodal.config import load_config
from tqdm import tqdm
import polars as pl
import random

import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
import logpool

from pathlib import Path

from astromodal.datasets.datacubes import load_datacube_files
from astromodal.datasets.spluscuts import SplusCutoutsDataset
from astromodal.models.autoencoder import AutoEncoder


config = load_config("/home/schwarz/projetoFM/config.yaml")
datacube_paths = config['datacubes_paths']

columns = [
    "splus_cut_F378",
    "splus_cut_F395",
    "splus_cut_F410",
    "splus_cut_F430",
    "splus_cut_F515",
    "splus_cut_F660",
    "splus_cut_F861",
    "splus_cut_R",
    "splus_cut_I",
    "splus_cut_Z",
    "splus_cut_U",
    "splus_cut_G",
]

bands = ["F378", "F395", "F410", "F430", "F515", "F660", "F861", "R", "I", "Z", "U", "G"]
cutout_size = 96

batch_size = 1024
max_gpu_batch_size = 1024
num_epochs = 10
learning_rate = 1e-3
latent_dim = 2

model_output_path = Path(config['models_folder']) / "./autoencoder_model_silu.pth"

if __name__ == '__main__':
    train_files, val_files = load_datacube_files(
        datacubes_paths = datacube_paths,
        train_val_split = 0.85,
        nfiles_subsample = 30,
        seed = 42
    )

    train_df = None

    for f in tqdm(train_files, desc="Loading train files"):
        df = pl.read_parquet(f, columns=columns, use_pyarrow=True)
        df = df.filter(pl.col(columns[0]).is_not_null())

        if df.height == 0:
            continue

        train_df = df if train_df is None else pl.concat([train_df, df], how="vertical", rechunk=False)

    train_df = train_df.rechunk()

    val_df = None

    for f in tqdm(val_files, desc="Loading val files"):
        df = pl.read_parquet(f, columns=columns, use_pyarrow=True)
        df = df.filter(pl.col(columns[0]).is_not_null())

        if df.height == 0:
            continue

        val_df = df if val_df is None else pl.concat([val_df, df], how="vertical", rechunk=False)

    val_df = val_df.rechunk()


    train_dataset = SplusCutoutsDataset(
        train_df,
        bands=bands,
        img_size=cutout_size,
        return_valid_mask=True,
    )
    val_dataset = SplusCutoutsDataset(
        val_df,
        bands=bands,
        img_size=cutout_size,
        return_valid_mask=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=14,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max_gpu_batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
    )

    model = AutoEncoder(
        in_channels = len(bands),
        latent_dim = latent_dim,
        use_skips=False
    )

    model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        n_channels=bands,
        latent_dim=latent_dim,
        model_output_path=str(model_output_path),
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )






