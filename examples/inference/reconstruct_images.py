#!/usr/bin/env python3
"""
Example: Reconstruct masked images using trained AION model.

This demonstrates:
1. Loading a trained model
2. Loading a sample from the dataset
3. Masking image tokens
4. Reconstructing the masked regions
5. Visualizing the results
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.astromodal.inference import load_aion_model, prepare_observation, reconstruct_images
from src.astromodal.core import build_vocab_spec, discover_fields_from_datacubes
from src.astromodal.datasets import MultiFieldTokenDataset
from src.astromodal.models import AIONTransformer


# ============================================================
# CONFIG
# ============================================================

# Checkpoint to load
CHECKPOINT_PATH = "runs_aion_masked/ckpt_step_00005000.pt"  # Adjust to your checkpoint

# Data paths (same as training)
DATACUBE_TMPL = "/home/astrodados4/downloads/hypercube/datacube_{field}.parquet"
IMAGE_TOKENS_TMPL = "/home/schwarz/projetoFM/codecs/image_tokens/datacube_{field}_tokens.npz"
SCALAR_TOKENS_TMPL = "/home/schwarz/projetoFM/scalar_tokenizers/scalar_tokens/datacube_{field}_scalar_tokens.npz"
SPECTRUM_TOKENS_TMPL = "/home/schwarz/projetoFM/spectrum_tokenizers/spectrum_tokens/datacube_{field}_spectrum_tokens.npz"

# Token vocab sizes (same as training)
V_IMAGE = 2048
V_SCALAR = 1024
V_SPECTRA = 2048

SPECTRUM_GROUPS = {
    "gaiaxp_bp": ("tokens_gaiaxp_bp_flat", "tokens_gaiaxp_bp_indptr"),
    "gaiaxp_rp": ("tokens_gaiaxp_rp_flat", "tokens_gaiaxp_rp_indptr"),
    "desi_b": ("tokens_desi_b_flat", "tokens_desi_b_indptr"),
    "desi_r": ("tokens_desi_r_flat", "tokens_desi_r_indptr"),
    "desi_z": ("tokens_desi_z_flat", "tokens_desi_z_indptr"),
}

# Data filtering
MAG_MIN = 14.0
MAG_MAX = 22.0
MAGERR_MAX = 2.0
MAG_COL = "mag_pstotal_r"
MAGERR_COL = "err_mag_pstotal_r"

# Reconstruction settings
NUM_SAMPLES = 5  # Number of samples to reconstruct
MASK_RATIO = 0.5  # Fraction of image tokens to mask
NUM_ITERATIONS = 8  # Iterative refinement steps
TEMPERATURE = 1.0  # Sampling temperature

# Output directory
OUTPUT_DIR = "reconstruction_results"


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("AION Image Reconstruction Example")
    print("=" * 70)
    print()

    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Build vocabulary
    print("[1/6] Building vocabulary...")
    vocab = build_vocab_spec(
        v_image=V_IMAGE,
        v_scalar=V_SCALAR,
        v_spectra=V_SPECTRA,
        spectrum_groups=SPECTRUM_GROUPS,
    )
    print(f"  Vocabulary size: {vocab.vocab_size:,}")
    print(f"  Image token range: {vocab.image_offset} - {vocab.image_offset + vocab.image_size}")
    print()

    # Load model
    print("[2/6] Loading trained model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_aion_model(CHECKPOINT_PATH, device=device)
    print()

    # Load dataset
    print("[3/6] Loading dataset...")
    fields = discover_fields_from_datacubes(DATACUBE_TMPL)
    print(f"  Found {len(fields)} fields")

    dataset = MultiFieldTokenDataset(
        fields=fields,
        vocab=vocab,
        datacube_template=DATACUBE_TMPL,
        image_tokens_template=IMAGE_TOKENS_TMPL,
        scalar_tokens_template=SCALAR_TOKENS_TMPL,
        spectrum_tokens_template=SPECTRUM_TOKENS_TMPL,
        spectrum_groups=SPECTRUM_GROUPS,
        mag_col=MAG_COL,
        mag_min=MAG_MIN,
        mag_max=MAG_MAX,
        magerr_max=MAGERR_MAX,
    )
    print(f"  Dataset size: {len(dataset):,} sequences")
    print()

    # Select random samples
    print(f"[4/6] Selecting {NUM_SAMPLES} random samples...")
    indices = np.random.choice(len(dataset), NUM_SAMPLES, replace=False)
    print(f"  Sample indices: {indices.tolist()}")
    print()

    # Process each sample
    print(f"[5/6] Reconstructing masked images...")
    for idx in tqdm(indices, desc="Processing samples"):
        sample = dataset[idx]
        tokens = sample["tokens"]
        types = sample["types"]
        attn_mask = sample["attn_mask"]

        # Identify image token positions
        # Image tokens have type = 5 (special tokens: 0-4, image: 5, scalar: 6, spectra: 7+)
        image_positions = torch.where(types == 5)[0].tolist()

        if len(image_positions) == 0:
            print(f"  Warning: Sample {idx} has no image tokens, skipping")
            continue

        print(f"\n  Sample {idx}:")
        print(f"    Total tokens: {tokens.numel()}")
        print(f"    Image tokens: {len(image_positions)}")

        # Randomly mask some image positions
        num_to_mask = int(len(image_positions) * MASK_RATIO)
        mask_positions = np.random.choice(image_positions, num_to_mask, replace=False).tolist()
        print(f"    Masking: {num_to_mask} positions ({MASK_RATIO*100:.0f}%)")

        # Prepare observation
        obs = prepare_observation(
            tokens=tokens,
            types=types,
            mask_positions=mask_positions,
            pad_id=vocab.pad_id,
        )

        # Move to device
        for key in obs:
            obs[key] = obs[key].to(device)

        # Ground truth (for comparison)
        ground_truth = tokens[mask_positions]

        # Reconstruct
        print(f"    Reconstructing with {NUM_ITERATIONS} iterations...")
        predicted_tokens, history = reconstruct_images(
            model=model,
            obs_tokens=obs["obs_tokens"],
            obs_types=obs["obs_types"],
            obs_positions=obs["obs_positions"],
            obs_mask=obs["obs_mask"],
            tgt_positions=obs["tgt_positions"],
            tgt_types=obs["tgt_types"],
            tgt_mask=obs["tgt_mask"],
            vocab=vocab,
            temperature=TEMPERATURE,
            num_iterations=NUM_ITERATIONS,
        )

        # Compute accuracy
        predicted = predicted_tokens.squeeze(0).cpu()
        accuracy = (predicted == ground_truth).float().mean().item()
        print(f"    Reconstruction accuracy: {accuracy*100:.1f}%")

        # Save results
        result = {
            "sample_idx": idx,
            "tokens": tokens.cpu().numpy(),
            "types": types.cpu().numpy(),
            "mask_positions": mask_positions,
            "ground_truth": ground_truth.cpu().numpy(),
            "predicted": predicted.cpu().numpy(),
            "history": [h.squeeze(0).cpu().numpy() for h in history],
            "accuracy": accuracy,
        }
        np.savez(
            f"{OUTPUT_DIR}/reconstruction_sample_{idx:05d}.npz",
            **result
        )

    print()
    print("[6/6] Visualization...")
    print(f"  Results saved to: {OUTPUT_DIR}/")
    print()

    # Create summary plot for first sample
    print("Creating summary visualization...")
    first_result = np.load(f"{OUTPUT_DIR}/reconstruction_sample_{indices[0]:05d}.npz", allow_pickle=True)

    fig, axes = plt.subplots(2, NUM_ITERATIONS // 2, figsize=(16, 8))
    axes = axes.flatten()

    for i, hist in enumerate(first_result["history"]):
        ax = axes[i]
        # Visualize token IDs as a 1D signal
        ax.plot(first_result["ground_truth"], "b-", label="Ground Truth", alpha=0.5)
        ax.plot(hist, "r-", label=f"Iteration {i+1}", alpha=0.7)
        ax.set_title(f"Iteration {i+1}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Token ID")
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/reconstruction_progress.png", dpi=150)
    print(f"  Saved: {OUTPUT_DIR}/reconstruction_progress.png")

    print()
    print("=" * 70)
    print("✓ Image reconstruction complete!")
    print(f"✓ Results saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
