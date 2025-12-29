#!/usr/bin/env python3
"""
Train KMeans codebook on image latents from trained autoencoder.

Uses the AutoEncoder from astromodal.models to extract latents,
then trains a KMeans codebook for discrete tokenization.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set GPU

from pathlib import Path
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.cluster import MiniBatchKMeans
import joblib

from astromodal.config import load_config
from astromodal.models import AutoEncoder  # From the package!
from astromodal.datasets import SplusDataset  # From the package!


def extract_latents(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int = 100_000,
) -> np.ndarray:
    """Extract latent representations from trained autoencoder."""
    model.eval()
    latents_list = []
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting latents"):
            if n_samples >= max_samples:
                break

            x_norm, m_obj, m_valid = batch
            x_norm = x_norm.to(device)

            # Encode to latents using autoencoder
            _, z = model(x_norm)  # z: (B, latent_dim, H, W)

            # Flatten spatial: (B, latent_dim, H, W) -> (B*H*W, latent_dim)
            B, C, H, W = z.shape
            z_flat = z.permute(0, 2, 3, 1).reshape(B * H * W, C)

            latents_list.append(z_flat.cpu().numpy())
            n_samples += z_flat.shape[0]

    latents = np.concatenate(latents_list, axis=0)
    if len(latents) > max_samples:
        latents = latents[:max_samples]

    return latents


def main():
    config = load_config()

    print("=" * 60)
    print("Training Image Codebook (KMeans)")
    print("=" * 60)

    # Check if autoencoder exists
    autoencoder_path = Path(config.paths.image_autoencoder)
    if not autoencoder_path.exists():
        print(f"\n❌ Autoencoder not found: {autoencoder_path}")
        print("\nTrain the autoencoder first:")
        print("  python examples/tokenizers/train_image_autoencoder.py")
        return

    print(f"\n✓ Found autoencoder: {autoencoder_path}")

    # Settings
    CODEBOOK_SIZE = config.tokenizers.image.codebook_size
    MAX_LATENTS = 200_000
    BATCH_SIZE = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Codebook size: {CODEBOOK_SIZE}")

    # Load autoencoder from astromodal.models
    print("\nLoading AutoEncoder from astromodal.models...")
    checkpoint = torch.load(autoencoder_path, map_location=device)

    model = AutoEncoder(
        in_channels=checkpoint['config']['in_channels'],
        out_channels=checkpoint['config']['out_channels'],
        latent_dim=checkpoint['config']['latent_dim'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Loaded (epoch {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.6f})")

    # Load data
    print("\nDiscovering datacube files...")
    datacube_dir = Path(config.paths.datacube_template).parent
    pattern = Path(config.paths.datacube_template).name.replace("{field}", "*")
    parquet_files = sorted(datacube_dir.glob(pattern))

    if not parquet_files:
        raise RuntimeError(f"No parquet files found")

    print(f"Found {len(parquet_files)} fields")

    # Load subset
    print("\nLoading data...")
    dfs = []
    for f in tqdm(parquet_files[:5], desc="Loading"):  # Use 5 fields
        df = pl.read_parquet(f)
        df = df.filter(pl.col("splus_cut_F378").is_not_null())
        dfs.append(df)
    combined_df = pl.concat(dfs)

    print(f"Total samples: {len(combined_df):,}")

    # Create dataset using astromodal.datasets
    dataset = SplusDataset(
        combined_df,
        bands=config.tokenizers.image.bands,
        img_size=config.tokenizers.image.cutout_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Extract latents
    print(f"\nExtracting latents (max {MAX_LATENTS:,})...")
    latents = extract_latents(model, dataloader, device, MAX_LATENTS)

    print(f"✓ Extracted {len(latents):,} latent vectors")
    print(f"  Shape: {latents.shape}")

    # Train KMeans
    print(f"\nTraining KMeans with {CODEBOOK_SIZE} clusters...")
    print("This may take 10-30 minutes...")

    kmeans = MiniBatchKMeans(
        n_clusters=CODEBOOK_SIZE,
        batch_size=1024,
        max_iter=100,
        random_state=0,
        verbose=1,
    )

    kmeans.fit(latents)

    print(f"\n✓ KMeans training complete")
    print(f"  Inertia: {kmeans.inertia_:.2f}")

    # Save codebook
    codebook_path = Path(config.paths.image_codebook)
    codebook_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(kmeans, codebook_path)

    print(f"\n✓ Codebook saved to: {codebook_path}")
    print(f"  Centroids shape: {kmeans.cluster_centers_.shape}")

    print()
    print("=" * 60)
    print("✓ Image tokenizer training complete!")
    print()
    print("Now encode images:")
    print("  python examples/tokenizers/encode_all.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
