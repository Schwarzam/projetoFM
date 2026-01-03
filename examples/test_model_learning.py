#!/usr/bin/env python3
"""
Test if AION model is learning by checking reconstruction quality on masked tokens.

This script:
1. Loads a trained checkpoint
2. Takes samples from the dataset
3. Uses the same masking strategy as training
4. Compares model predictions with ground truth tokens
5. Reports accuracy metrics for different modalities

Usage:
    python examples/test_model_learning.py --checkpoint runs_aion_h100/ckpt_step_00002000.pt --n-samples 100
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.astromodal.models import AIONTransformer
from src.astromodal.core import build_vocab_spec, discover_fields_from_datacubes
from src.astromodal.datasets import MultiFieldTokenDataset, collate_sequences
from src.astromodal.training.masking import FourMMaskingStrategy
from functools import partial


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    config = ckpt.get("config", {})

    model = AIONTransformer(
        vocab_size=config["vocab_size"],
        n_types=config["n_types"],
        d_model=config["d_model"],
        n_enc_layers=config["n_enc_layers"],
        n_dec_layers=config["n_dec_layers"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        max_len=config["max_len"],
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    step = ckpt.get("global_step", '?')
    print(f"✓ Model loaded (step {step})")
    return model, config, step


def create_dataset():
    """Create dataset for testing."""
    # Same config as training
    DATACUBE_TMPL = "/home/astrodados4/downloads/hypercube/datacube_{field}.parquet"
    IMAGE_TOKENS_TMPL = "/home/schwarz/projetoFM/outputs/image_tokens/datacube_{field}_tokens.npz"
    SCALAR_TOKENS_TMPL = "/home/schwarz/projetoFM/outputs/scalar_tokens/datacube_{field}_scalar_tokens.npz"
    SPECTRUM_TOKENS_TMPL = "/home/schwarz/projetoFM/outputs/spectrum_tokens/datacube_{field}_spectrum_tokens.npz"

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

    MAG_MIN = 14.0
    MAG_MAX = 22.0
    MAGERR_MAX = 2.0
    MAG_COL = "mag_pstotal_r"
    MAGERR_COL = "err_mag_pstotal_r"

    vocab = build_vocab_spec(
        v_image=V_IMAGE,
        v_scalar=V_SCALAR,
        v_spectra=V_SPECTRA,
        spectrum_groups=SPECTRUM_GROUPS,
    )

    fields = discover_fields_from_datacubes(DATACUBE_TMPL)
    print(f"Found {len(fields)} fields")

    dataset = MultiFieldTokenDataset(
        fields=fields,
        vocab=vocab,
        datacube_template=DATACUBE_TMPL,
        image_tokens_template=IMAGE_TOKENS_TMPL,
        scalar_tokens_template=SCALAR_TOKENS_TMPL,
        spectrum_tokens_template=SPECTRUM_TOKENS_TMPL,
        spectrum_groups=SPECTRUM_GROUPS,
        mag_col=MAG_COL,
        magerr_col=MAGERR_COL,
        mag_min=MAG_MIN,
        mag_max=MAG_MAX,
        magerr_max=MAGERR_MAX,
        lru_fields=50,  # Small cache for testing
    )

    return dataset, vocab


def test_sample(model, sample, masking_strategy, vocab, device):
    """
    Test model on a single sample.

    Returns per-modality accuracy metrics.
    """
    # Prepare batch (single sample)
    tokens = sample["token_ids"].unsqueeze(0).to(device)  # (1, seq_len)
    types = sample["type_ids"].unsqueeze(0).to(device)
    attn_mask = torch.ones_like(tokens, dtype=torch.bool)

    # Apply masking (same as training)
    masked = masking_strategy.apply(tokens, types, attn_mask)

    # Forward pass
    with torch.no_grad():
        logits = model(
            obs_tokens=masked.obs_tokens,
            obs_types=masked.obs_types,
            obs_positions=masked.obs_positions,
            obs_mask=masked.obs_mask,
            tgt_types=masked.tgt_types,
            tgt_positions=masked.tgt_positions,
            tgt_mask=masked.tgt_mask,
        )  # (1, tgt_len, vocab_size)

    # Get predictions
    preds = logits.argmax(dim=-1).squeeze(0)  # (tgt_len,)
    gt_tokens = masked.tgt_tokens.squeeze(0)  # (tgt_len,)
    tgt_types = masked.tgt_types.squeeze(0)
    tgt_mask = masked.tgt_mask.squeeze(0)

    # Calculate per-modality accuracy
    results = {}

    # Build modality map dynamically from vocab
    modality_map = {5: "image"}  # IMAGE type

    # Add scalar types
    for col_name, type_id in vocab.scalar_type_map.items():
        modality_map[type_id] = f"scalar_{col_name}"

    # Add spectrum types (after scalars)
    spectrum_groups = ["gaiaxp_bp", "gaiaxp_rp", "desi_b", "desi_r", "desi_z"]
    if vocab.scalar_type_map:
        spectrum_start = max(vocab.scalar_type_map.values()) + 1
    else:
        spectrum_start = 7
    for i, group_name in enumerate(spectrum_groups):
        modality_map[spectrum_start + i] = f"spectrum_{group_name}"

    for type_id, mod_name in modality_map.items():
        # Find tokens of this modality in targets
        mod_mask = (tgt_types == type_id) & tgt_mask
        if mod_mask.sum() == 0:
            continue

        # Calculate accuracy
        correct = (preds[mod_mask] == gt_tokens[mod_mask]).sum().item()
        total = mod_mask.sum().item()
        accuracy = correct / total if total > 0 else 0.0

        # Calculate top-5 accuracy
        top5_preds = logits.squeeze(0)[mod_mask].topk(5, dim=-1).indices  # (n_tokens, 5)
        gt_expand = gt_tokens[mod_mask].unsqueeze(1)  # (n_tokens, 1)
        top5_correct = (top5_preds == gt_expand).any(dim=1).sum().item()
        top5_accuracy = top5_correct / total if total > 0 else 0.0

        results[mod_name] = {
            "n_tokens": total,
            "accuracy": accuracy,
            "top5_accuracy": top5_accuracy,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Test AION model learning")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of samples to test")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--input-budget", type=int, default=128, help="Input budget for masking")
    parser.add_argument("--output-budget", type=int, default=64, help="Output budget for masking")
    args = parser.parse_args()

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("\nAvailable checkpoints:")
        ckpt_dir = Path("runs_aion_h100")
        if ckpt_dir.exists():
            for ckpt in sorted(ckpt_dir.glob("ckpt_*.pt")):
                print(f"  {ckpt}")
        return

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, config, step = load_model(args.checkpoint, device)

    # Create dataset
    print("\nCreating dataset...")
    dataset, vocab = create_dataset()

    # Create masking strategy (same as training)
    masking_strategy = FourMMaskingStrategy(
        input_budget=args.input_budget,
        output_budget=args.output_budget,
        beta_alpha=0.5,
        beta_beta=1.0,
        pad_id=vocab.pad_id,
    )
    print(f"Masking: {args.input_budget} input budget, {args.output_budget} output budget\n")

    # Accumulate results
    all_results = defaultdict(lambda: {"n_tokens": 0, "correct": 0, "top5_correct": 0})

    print(f"Testing on {args.n_samples} samples...")
    print("=" * 70)

    for i in range(args.n_samples):
        # Get random sample
        idx = np.random.randint(0, len(dataset))
        sample = dataset[idx]

        # Test
        try:
            results = test_sample(model, sample, masking_strategy, vocab, device)

            # Accumulate
            for mod_name, metrics in results.items():
                all_results[mod_name]["n_tokens"] += metrics["n_tokens"]
                all_results[mod_name]["correct"] += int(metrics["accuracy"] * metrics["n_tokens"])
                all_results[mod_name]["top5_correct"] += int(metrics["top5_accuracy"] * metrics["n_tokens"])

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{args.n_samples} samples...")

        except Exception as e:
            print(f"  Error on sample {idx}: {e}")
            continue

    print("=" * 70)
    print("\nRESULTS")
    print("=" * 70)

    # Aggregate results by category
    aggregated = {
        "image": {"n_tokens": 0, "correct": 0, "top5_correct": 0},
        "scalar": {"n_tokens": 0, "correct": 0, "top5_correct": 0},
        "spectrum": {"n_tokens": 0, "correct": 0, "top5_correct": 0},
    }

    for mod_name, stats in all_results.items():
        if "scalar_" in mod_name:
            category = "scalar"
        elif "spectrum_" in mod_name:
            category = "spectrum"
        else:
            category = mod_name

        if category in aggregated:
            aggregated[category]["n_tokens"] += stats["n_tokens"]
            aggregated[category]["correct"] += stats["correct"]
            aggregated[category]["top5_correct"] += stats["top5_correct"]

    # Print aggregated results
    vocab_sizes = {"image": 2048, "scalar": 1024, "spectrum": 2048}

    for category in ["image", "scalar", "spectrum"]:
        stats = aggregated[category]
        if stats["n_tokens"] == 0:
            continue

        accuracy = stats["correct"] / stats["n_tokens"]
        top5_accuracy = stats["top5_correct"] / stats["n_tokens"]
        random_acc = 1.0 / vocab_sizes[category]

        print(f"\n{category.upper()}:")
        print(f"  Tokens tested: {stats['n_tokens']:,}")
        print(f"  Top-1 accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Top-5 accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)")
        print(f"  Random baseline: {random_acc:.4f} ({random_acc*100:.2f}%)")

        improvement = accuracy / random_acc if random_acc > 0 else 0
        if improvement > 10:
            print(f"  ✓ Model is learning! ({improvement:.1f}x better than random)")
        elif improvement > 3:
            print(f"  ~ Some learning ({improvement:.1f}x better than random)")
        else:
            print(f"  ✗ Not much better than random ({improvement:.1f}x)")

    print("\n" + "=" * 70)
    print(f"\nCheckpoint: {args.checkpoint} (step {step})")
    print("=" * 70)


if __name__ == "__main__":
    main()
