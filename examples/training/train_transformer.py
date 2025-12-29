#!/usr/bin/env python3
"""
Train transformer with RECONSTRUCTION LOSS (not next-token prediction).

The model learns to reconstruct actual data values using:
- Image: Expected centroid (softmax over vocab) vs true centroid
- Scalar: Expected bin center vs true bin center
- Spectra: Expected centroid vs true centroid

This provides meaningful supervision beyond discrete token prediction.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set GPU

import torch
from astromodal.config import load_config
from astromodal.models import CausalTransformerLM
from astromodal.datasets import MultiFieldTokenDataset, collate_sequences
from astromodal.training import Trainer
from astromodal.core import build_vocab_spec

def main():
    # Load configuration
    config = load_config()

    print("=" * 60)
    print("Training Transformer with RECONSTRUCTION LOSS")
    print("=" * 60)
    print("\nThis model uses meaningful value reconstruction:")
    print("  - Images: Reconstruct actual image latent centroids")
    print("  - Scalars: Reconstruct actual scalar bin centers")
    print("  - Spectra: Reconstruct actual spectrum centroids")
    print()

    # Build vocabulary
    vocab_spec = build_vocab_spec(
        v_image=config.vocab.image,
        v_scalar=config.vocab.scalar,
        v_spectra=config.vocab.spectrum,
        spectrum_groups=config.tokenizers.spectrum.groups,
    )

    print(f"Vocabulary size: {vocab_spec.vocab_size}")

    # Create model
    print("\nInitializing transformer model...")

    # Calculate number of token types:
    # 0=PAD, 1=BOS, 2=EOS, 3=SEP, 4=MISS, 5=IMAGE, 6=SCALAR, 7+=SPECTRUM_GROUPS
    n_types = 5 + 1 + 1 + len(config.tokenizers.spectrum.groups)  # Special + image + scalar + spectra

    model = CausalTransformerLM(
        vocab_size=vocab_spec.vocab_size,
        n_types=n_types,
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
        max_len=config.sequences.max_seq_len,
    )

    # Discover fields
    print("Discovering fields...")
    from astromodal.core import discover_fields_from_datacubes
    fields = discover_fields_from_datacubes(config.paths.datacube_template)
    print(f"Found {len(fields)} fields")

    # Prepare spectrum groups mapping
    spectrum_groups = {}
    for group_name in config.tokenizers.spectrum.groups:
        spectrum_groups[group_name] = (
            f"tokens_{group_name}_flat",
            f"tokens_{group_name}_indptr"
        )

    # Create dataset
    print("Creating dataset...")
    dataset = MultiFieldTokenDataset(
        fields=fields,
        vocab=vocab_spec,
        datacube_template=config.paths.datacube_template,
        image_tokens_template=config.paths.image_tokens_template,
        scalar_tokens_template=config.paths.scalar_tokens_template,
        spectrum_tokens_template=config.paths.spectrum_tokens_template,
        spectrum_groups=spectrum_groups,
        mag_col=config.data.mag_filter['column'],
        mag_min=config.data.mag_filter['min'],
        mag_max=config.data.mag_filter['max'],
        magerr_max=config.data.magerr_max,
    )

    print(f"Dataset size: {len(dataset):,} sequences")

    # Create DataLoader
    print("Creating DataLoader...")
    from torch.utils.data import DataLoader
    from functools import partial

    # Create collate function with pad_id from vocab
    collate_fn = partial(collate_sequences, pad_id=vocab_spec.pad_id)

    train_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=config.training.persistent_workers if config.training.num_workers > 0 else False,
        collate_fn=collate_fn,
    )

    # Load codebooks for reconstruction loss
    print("Loading codebooks for reconstruction loss...")
    from astromodal.core import load_codebook, load_scalar_centers
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load image codebook
    image_codebook = load_codebook(config.paths.image_codebook)
    image_codebook = torch.from_numpy(image_codebook).to(device)
    print(f"  Image codebook: {image_codebook.shape}")

    # Load spectrum codebooks
    # The spectrum tokenizer stores codebooks in 'group_codebooks' key
    import numpy as np
    spectrum_config = np.load(config.paths.spectrum_tokenizer_config, allow_pickle=True)

    # For now, use the first group's codebook
    # The full implementation would need to handle multiple spectrum groups
    group_codebooks = spectrum_config['group_codebooks'].item()  # Dict of codebooks
    first_group = list(group_codebooks.keys())[0]
    spectrum_codebook = group_codebooks[first_group]
    spectrum_codebook = torch.from_numpy(spectrum_codebook).float().to(device)
    print(f"  Spectrum codebook ({first_group}): {spectrum_codebook.shape}")

    # Load scalar bin edges and compute centers
    # The scalar tokenizer stores bin_edges, not bin_centers
    scalar_config = np.load(config.paths.scalar_tokenizer_config, allow_pickle=True)
    bin_edges_dict = scalar_config['bin_edges'].item()  # Dict of column -> edges

    # Compute bin centers from edges for all columns
    # bin_centers[i] = (edges[i] + edges[i+1]) / 2
    all_centers = []
    for col_name in sorted(bin_edges_dict.keys()):
        edges = bin_edges_dict[col_name]
        centers = (edges[:-1] + edges[1:]) / 2  # Midpoints
        all_centers.append(centers)

    # Flatten all centers into a single array
    scalar_centers = np.concatenate(all_centers)
    scalar_centers = torch.from_numpy(scalar_centers).float().to(device)
    print(f"  Scalar centers: {scalar_centers.shape} (computed from bin_edges)")

    # Create trainer (uses reconstruction loss)
    print("\nInitializing trainer with reconstruction loss...")
    trainer = Trainer(
        model=model,
        vocab=vocab_spec,
        train_loader=train_loader,
        config=config.training,
        image_codebook=image_codebook,
        spectrum_codebook=spectrum_codebook,
        scalar_centers=scalar_centers,
        device=str(device),
        output_dir=config.paths.output_dir,
        use_amp=True,
        use_ce=True,  # Use both CE and reconstruction losses
    )

    # Train
    print(f"\nStarting training for {config.training.epochs} epochs...")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Output directory: {config.paths.output_dir}")
    print(f"Using reconstruction loss: YES")
    print()

    trainer.train()

    print("\n✓ Training complete!")
    print(f"Checkpoints saved to: {config.paths.output_dir}")

if __name__ == "__main__":
    main()
