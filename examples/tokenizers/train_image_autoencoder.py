#!/usr/bin/env python3
"""
Train image autoencoder for S-PLUS cutouts.

This trains the autoencoder from the astromodal package that compresses
96x96x12 images to 2x24x24 spatial latents.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set GPU

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import polars as pl

from astromodal.config import load_config
from astromodal.models import AutoEncoder  # From the package!
from astromodal.datasets import SplusDataset  # From the package!


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool = True,
    w_bg: float = 0.3,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        x_norm, m_obj, m_valid = batch
        x_norm = x_norm.to(device)
        m_obj = m_obj.to(device)
        m_valid = m_valid.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            x_recon, _ = model(x_norm)

            # Combined mask: object + valid pixels
            m_fg = m_obj & m_valid  # Foreground
            m_bg = (~m_obj) & m_valid  # Background

            # MSE loss with weighted foreground/background
            loss_fg = F.mse_loss(x_recon[m_fg], x_norm[m_fg])
            loss_bg = F.mse_loss(x_recon[m_bg], x_norm[m_bg])
            loss = loss_fg + w_bg * loss_bg

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    w_bg: float = 0.3,
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            x_norm, m_obj, m_valid = batch
            x_norm = x_norm.to(device)
            m_obj = m_obj.to(device)
            m_valid = m_valid.to(device)

            x_recon, _ = model(x_norm)

            m_fg = m_obj & m_valid
            m_bg = (~m_obj) & m_valid

            loss_fg = F.mse_loss(x_recon[m_fg], x_norm[m_fg])
            loss_bg = F.mse_loss(x_recon[m_bg], x_norm[m_bg])
            loss = loss_fg + w_bg * loss_bg

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    # Load configuration
    config = load_config()

    print("=" * 60)
    print("Training Image Autoencoder (from astromodal.models)")
    print("=" * 60)

    # Settings
    BATCH_SIZE = 64
    MAX_GPU_BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    USE_AMP = True
    W_BG = 0.3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Discover parquet files
    print(f"\nDiscovering datacube files from: {config.paths.datacube_template}")
    datacube_dir = Path(config.paths.datacube_template).parent
    pattern = Path(config.paths.datacube_template).name.replace("{field}", "*")
    parquet_files = sorted(datacube_dir.glob(pattern))

    if not parquet_files:
        raise RuntimeError(f"No parquet files found")

    print(f"Found {len(parquet_files)} datacube files")

    # Use first N fields for training
    train_files = parquet_files[:int(0.9 * len(parquet_files))]
    val_files = parquet_files[int(0.9 * len(parquet_files)):]

    print(f"Training on {len(train_files)} fields")
    print(f"Validating on {len(val_files)} fields")

    # Load training data
    print("\nLoading training data...")
    train_dfs = []
    for f in tqdm(train_files[:7], desc="Loading train"):  # Limit to 7 fields
        df = pl.read_parquet(f)
        df = df.filter(pl.col("splus_cut_F378").is_not_null())
        train_dfs.append(df)
    train_df = pl.concat(train_dfs)
    print(f"Training samples: {len(train_df):,}")

    # Load validation data
    print("Loading validation data...")
    val_dfs = []
    for f in tqdm(val_files[:2], desc="Loading val"):
        df = pl.read_parquet(f)
        df = df.filter(pl.col("splus_cut_F378").is_not_null())
        val_dfs.append(df)
    val_df = pl.concat(val_dfs) if val_dfs else train_df[:1000]
    print(f"Validation samples: {len(val_df):,}")

    # Create datasets using astromodal.datasets.SplusDataset
    train_dataset = SplusDataset(
        train_df,
        bands=config.tokenizers.image.bands,
        img_size=config.tokenizers.image.cutout_size,
    )
    val_dataset = SplusDataset(
        val_df,
        bands=config.tokenizers.image.bands,
        img_size=config.tokenizers.image.cutout_size,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=MAX_GPU_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Create model using astromodal.models.AutoEncoder
    print("\nInitializing AutoEncoder from astromodal.models...")
    model = AutoEncoder(
        in_channels=len(config.tokenizers.image.bands),
        out_channels=len(config.tokenizers.image.bands),
        latent_dim=config.tokenizers.image.latent_dim,
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer and scaler
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # Training loop
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Mixed precision: {USE_AMP}")
    print()

    best_val_loss = float('inf')
    output_path = Path(config.paths.image_autoencoder)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}")

        train_loss = train_epoch(
            model, train_loader, optimizer, device, scaler, USE_AMP, W_BG
        )
        val_loss = validate(model, val_loader, device, W_BG)

        print(f"  Train loss: {train_loss:.6f}")
        print(f"  Val loss:   {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'in_channels': len(config.tokenizers.image.bands),
                    'out_channels': len(config.tokenizers.image.bands),
                    'latent_dim': config.tokenizers.image.latent_dim,
                },
            }, output_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")

        print()

    print("=" * 60)
    print("✓ Autoencoder training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {output_path}")
    print()
    print("Next: Train KMeans codebook")
    print("  python examples/tokenizers/train_image_codebook.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
