#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Training with meaningful reconstruction loss

This script demonstrates how to use the astromodal training infrastructure
to train a transformer model with meaningful reconstruction losses instead
of standard next-token prediction.

Key features:
- Loads configuration from YAML
- Uses the Trainer class for clean training loop
- Supports both CE and meaningful losses
- Automatic checkpointing and resumption
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.astromodal.config import load_config
from src.astromodal.core import VocabSpec, build_vocab_spec
from src.astromodal.datasets import MultiFieldTokenDataset, collate_batch
from src.astromodal.training import (
    Trainer,
    set_seed,
    configure_cuda_performance,
)


def load_codebooks(config):
    """
    Load codebooks and bin centers for meaningful loss.

    Parameters
    ----------
    config : AstromodalConfig
        Configuration object

    Returns
    -------
    tuple
        (image_codebook, spectrum_codebook, scalar_centers) as numpy arrays
    """
    from src.astromodal.core.io import load_codebook, load_npz_key

    # Load image codebook
    img_codebook = load_codebook(
        config.paths.image_codebook,
        key_candidates=("centroids", "codebook", "centers"),
    )

    # Load spectrum codebook
    spec_codebook = load_npz_key(
        config.paths.spectrum_tokenizer_config,
        key_candidates=("centroids", "codebook", "centers"),
    )

    # Load scalar bin centers
    scalar_centers = load_npz_key(
        config.paths.scalar_tokenizer_config,
        key_candidates=("bin_centers", "centers", "quantile_centers", "centroids"),
    )

    # Validate shapes
    if img_codebook.shape[0] != config.vocab.image:
        raise ValueError(
            f"Image codebook size {img_codebook.shape[0]} != config {config.vocab.image}"
        )
    if spec_codebook.shape[0] != config.vocab.spectrum:
        raise ValueError(
            f"Spectrum codebook size {spec_codebook.shape[0]} != config {config.vocab.spectrum}"
        )
    if scalar_centers.shape[0] != config.vocab.scalar:
        raise ValueError(
            f"Scalar centers size {scalar_centers.shape[0]} != config {config.vocab.scalar}"
        )

    # Convert to float32 and flatten scalar centers
    img_codebook = img_codebook.astype(np.float32)
    spec_codebook = spec_codebook.astype(np.float32)
    scalar_centers = scalar_centers.reshape(-1).astype(np.float32)

    return img_codebook, spec_codebook, scalar_centers


def create_model(vocab: VocabSpec, config) -> nn.Module:
    """
    Create transformer model.

    Parameters
    ----------
    vocab : VocabSpec
        Vocabulary specification
    config : AstromodalConfig
        Configuration object

    Returns
    -------
    nn.Module
        Transformer model
    """
    from src.astromodal.models import CausalTransformerLM

    model = CausalTransformerLM(
        vocab_size=vocab.vocab_size,
        n_types=20,  # BOS + IMAGE + SCALAR + spectrum groups
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
        max_len=config.sequences.max_seq_len,
    )

    return model


def main():
    """Main training script."""
    # Load configuration
    config = load_config()  # Uses defaults.yaml from package

    # Set random seed for reproducibility
    set_seed(config.training.seed)

    # Configure CUDA performance settings
    configure_cuda_performance()

    # Build vocabulary specification
    vocab = build_vocab_spec(
        v_image=config.vocab.image,
        v_scalar=config.vocab.scalar,
        v_spectra=config.vocab.spectrum,
    )

    print(f"[setup] Vocabulary size: {vocab.vocab_size:,}")
    print(f"[setup] Image tokens: {vocab.base_image} - {vocab.base_image + config.vocab.image}")
    print(f"[setup] Scalar tokens: {vocab.base_scalar} - {vocab.base_scalar + config.vocab.scalar}")

    # Load codebooks for meaningful loss
    print("[setup] Loading codebooks...")
    img_codebook, spec_codebook, sca_centers = load_codebooks(config)
    print(f"[setup] Image codebook: {img_codebook.shape}")
    print(f"[setup] Spectrum codebook: {spec_codebook.shape}")
    print(f"[setup] Scalar centers: {sca_centers.shape}")

    # Convert to tensors
    img_codebook = torch.from_numpy(img_codebook)
    spec_codebook = torch.from_numpy(spec_codebook)
    sca_centers = torch.from_numpy(sca_centers)

    # Create dataset
    print("[setup] Creating dataset...")
    from src.astromodal.core.files import discover_fields

    fields = discover_fields(config.paths.datacube_template)
    print(f"[setup] Found {len(fields)} fields")

    dataset = MultiFieldTokenDataset(
        fields=fields,
        vocab=vocab,
        config=config,
        index_cache="cache_token_dataset_index.npz",
        lru_fields=8,
    )
    print(f"[setup] Dataset size: {len(dataset):,} samples")

    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=config.training.persistent_workers,
        prefetch_factor=config.training.prefetch_factor,
        collate_fn=lambda b: collate_batch(b, pad_id=vocab.pad_id),
        drop_last=True,
    )

    # Create model
    print("[setup] Creating model...")
    model = create_model(vocab, config)

    # Create trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        model=model,
        vocab=vocab,
        train_loader=train_loader,
        config=config.training,
        image_codebook=img_codebook,
        spectrum_codebook=spec_codebook,
        scalar_centers=sca_centers,
        device=device,
        output_dir=config.paths.output_dir,
        use_amp=True,
        use_ce=True,  # Use both CE and meaningful loss
        compile_model=False,  # Set to True for extra speed on PyTorch 2.0+
    )

    # Train
    print("\n" + "="*80)
    print("Starting training")
    print("="*80 + "\n")

    trainer.train(num_epochs=config.training.epochs)

    print("\n" + "="*80)
    print("Training complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
