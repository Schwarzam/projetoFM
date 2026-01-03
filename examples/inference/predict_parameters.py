#!/usr/bin/env python3
"""
Example: Predict scalar parameters using trained AION model.

This demonstrates:
1. Loading a trained model
2. Using images and spectra to predict scalar parameters
3. Evaluating prediction accuracy
4. Comparing predicted vs. true values
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

from src.astromodal.inference import load_aion_model, prepare_observation, predict_parameters
from src.astromodal.core import build_vocab_spec, discover_fields_from_datacubes
from src.astromodal.datasets import MultiFieldTokenDataset


# ============================================================
# CONFIG
# ============================================================

# Checkpoint to load
CHECKPOINT_PATH = "runs_aion_h100/ckpt_step_00002000.pt"

# Data paths (same as training)
DATACUBE_TMPL = "/home/astrodados4/downloads/hypercube/datacube_{field}.parquet"
IMAGE_TOKENS_TMPL = "/home/schwarz/projetoFM/outputs/image_tokens/datacube_{field}_tokens.npz"
SCALAR_TOKENS_TMPL = "/home/schwarz/projetoFM/outputs/scalar_tokens/datacube_{field}_scalar_tokens.npz"
SPECTRUM_TOKENS_TMPL = "/home/schwarz/projetoFM/outputs/spectrum_tokens/datacube_{field}_spectrum_tokens.npz"

# Scalar tokenizer config (for bin centers)
SCALAR_TOKENIZER_CONFIG = "/home/schwarz/projetoFM/scalar_tokenizers/scalar_tokenizer_config.npz"

# Token vocab sizes
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

# Prediction settings
NUM_SAMPLES = 100  # Number of samples to evaluate
MASK_ALL_SCALARS = True  # If True, mask all scalar tokens; else mask 50%

# Output directory
OUTPUT_DIR = "parameter_prediction_results"


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("AION Parameter Prediction Example")
    print("=" * 70)
    print()

    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Build vocabulary
    print("[1/7] Building vocabulary...")
    vocab = build_vocab_spec(
        v_image=V_IMAGE,
        v_scalar=V_SCALAR,
        v_spectra=V_SPECTRA,
        spectrum_groups=SPECTRUM_GROUPS,
    )
    print(f"  Vocabulary size: {vocab.vocab_size:,}")
    print(f"  Scalar token range: {vocab.scalar_offset} - {vocab.scalar_offset + vocab.scalar_size}")
    print()

    # Load scalar bin centers
    print("[2/7] Loading scalar tokenizer configuration...")
    scalar_config = np.load(SCALAR_TOKENIZER_CONFIG, allow_pickle=True)
    bin_edges_dict = scalar_config['bin_edges'].item()

    # Compute bin centers from edges
    all_centers = []
    column_names = []
    for col_name in sorted(bin_edges_dict.keys()):
        edges = bin_edges_dict[col_name]
        centers = (edges[:-1] + edges[1:]) / 2
        all_centers.append(centers)
        column_names.append(col_name)

    scalar_bin_centers = np.concatenate(all_centers)
    print(f"  Total scalar bins: {len(scalar_bin_centers)}")
    print(f"  Scalar columns: {len(column_names)}")
    print(f"  Columns: {', '.join(column_names[:5])}{'...' if len(column_names) > 5 else ''}")
    print()

    # Load model
    print("[3/7] Loading trained model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_aion_model(CHECKPOINT_PATH, device=device)
    print()

    # Load dataset
    print("[4/7] Loading dataset...")
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
    print(f"[5/7] Selecting {NUM_SAMPLES} random samples...")
    indices = np.random.choice(len(dataset), NUM_SAMPLES, replace=False)
    print()

    # Evaluate predictions
    print(f"[6/7] Predicting scalar parameters...")
    results = []

    for idx in tqdm(indices, desc="Processing samples"):
        sample = dataset[idx]
        tokens = sample["tokens"]
        types = sample["types"]

        # Identify scalar token positions (type = 6)
        scalar_positions = torch.where(types == 6)[0].tolist()

        if len(scalar_positions) == 0:
            continue

        # Decide which scalars to mask
        if MASK_ALL_SCALARS:
            mask_positions = scalar_positions
        else:
            # Mask 50% of scalar tokens
            num_to_mask = max(1, len(scalar_positions) // 2)
            mask_positions = np.random.choice(scalar_positions, num_to_mask, replace=False).tolist()

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

        # Ground truth
        ground_truth_tokens = tokens[mask_positions]

        # Predict
        predictions = predict_parameters(
            model=model,
            obs_tokens=obs["obs_tokens"],
            obs_types=obs["obs_types"],
            obs_positions=obs["obs_positions"],
            obs_mask=obs["obs_mask"],
            tgt_positions=obs["tgt_positions"],
            tgt_types=obs["tgt_types"],
            tgt_mask=obs["tgt_mask"],
            vocab=vocab,
            scalar_bin_centers=scalar_bin_centers,
        )

        # Extract predictions
        predicted_tokens = predictions["predicted_tokens"].squeeze(0).cpu()
        predicted_values = predictions["predicted_values"].squeeze(0).cpu()

        # Compute accuracy
        token_accuracy = (predicted_tokens == ground_truth_tokens).float().mean().item()

        # Store results
        results.append({
            "sample_idx": idx,
            "num_scalars": len(mask_positions),
            "ground_truth_tokens": ground_truth_tokens.cpu().numpy(),
            "predicted_tokens": predicted_tokens.cpu().numpy(),
            "predicted_values": predicted_values.cpu().numpy(),
            "token_accuracy": token_accuracy,
        })

    print()
    print(f"  Processed {len(results)} samples")
    print()

    # Compute statistics
    print("[7/7] Computing statistics...")
    accuracies = [r["token_accuracy"] for r in results]
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    print(f"  Mean token accuracy: {mean_accuracy*100:.2f}% ± {std_accuracy*100:.2f}%")
    print(f"  Min accuracy: {np.min(accuracies)*100:.2f}%")
    print(f"  Max accuracy: {np.max(accuracies)*100:.2f}%")
    print()

    # Save all results
    results_file = f"{OUTPUT_DIR}/prediction_results.npz"
    np.savez(
        results_file,
        results=results,
        mean_accuracy=mean_accuracy,
        std_accuracy=std_accuracy,
        scalar_bin_centers=scalar_bin_centers,
        column_names=column_names,
    )
    print(f"  Results saved to: {results_file}")

    # Create visualization
    print()
    print("Creating visualizations...")

    # 1. Accuracy distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(accuracies, bins=20, edgecolor="black")
    ax.axvline(mean_accuracy, color="r", linestyle="--", label=f"Mean: {mean_accuracy*100:.1f}%")
    ax.set_xlabel("Token Accuracy")
    ax.set_ylabel("Count")
    ax.set_title("Parameter Prediction Accuracy Distribution")
    ax.legend()

    # 2. Example scatter plot (first 50 predictions from first sample)
    ax = axes[1]
    if len(results) > 0:
        first = results[0]
        gt = first["ground_truth_tokens"][:50]
        pred = first["predicted_tokens"][:50]
        ax.scatter(gt, pred, alpha=0.6)
        ax.plot([gt.min(), gt.max()], [gt.min(), gt.max()], "r--", label="Perfect prediction")
        ax.set_xlabel("Ground Truth Token ID")
        ax.set_ylabel("Predicted Token ID")
        ax.set_title("Example: Predicted vs. Ground Truth")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/prediction_summary.png", dpi=150)
    print(f"  Saved: {OUTPUT_DIR}/prediction_summary.png")

    print()
    print("=" * 70)
    print("✓ Parameter prediction complete!")
    print(f"✓ Mean accuracy: {mean_accuracy*100:.2f}%")
    print(f"✓ Results saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
