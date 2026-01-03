#!/usr/bin/env python3
"""
End-to-end inference from raw data.

This script demonstrates the full pipeline:
1. Load raw data from datacube (parquet)
2. Tokenize using trained tokenizers
3. Run model inference
4. Detokenize predictions back to physical values
5. Compare with ground truth

Usage:
    python examples/inference_from_raw.py \
        --checkpoint runs_aion_h100/ckpt_step_00010000.pt \
        --field STRIPE82-0001 \
        --sample-idx 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.astromodal.models import AIONTransformer
from src.astromodal.core import build_vocab_spec
from src.astromodal.tokenizers import ScalarTokenizer


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


def load_tokenizers():
    """Load all tokenizers."""
    print("\nLoading tokenizers...")

    # Load scalar tokenizer
    scalar_tokenizer = ScalarTokenizer.load("scalar_tokenizers/scalar_tokenizer_config.npz")

    print(f"✓ Scalar tokenizer: {len(scalar_tokenizer.bin_edges_)} columns")

    return scalar_tokenizer


def load_raw_sample(field: str, sample_idx: int):
    """Load raw data from datacube."""
    datacube_path = f"/home/astrodados4/downloads/hypercube/datacube_{field}.parquet"

    print(f"\nLoading raw sample from {datacube_path}")
    df = pd.read_parquet(datacube_path)

    # Apply magnitude filter
    mag_col = "mag_pstotal_r"
    df_filtered = df[(df[mag_col] > 14.0) & (df[mag_col] < 22.0)]

    if sample_idx >= len(df_filtered):
        print(f"Error: sample_idx {sample_idx} >= dataset size {len(df_filtered)}")
        return None

    row = df_filtered.iloc[sample_idx]
    print(f"✓ Loaded object: {row.get('id', 'N/A')}")
    return row


def tokenize_image(row, field: str):
    """Load and return pre-computed image tokens."""
    image_path = f"/home/schwarz/projetoFM/outputs/image_tokens/datacube_{field}_tokens.npz"

    if not Path(image_path).exists():
        print(f"Warning: Image tokens not found: {image_path}")
        return None

    # Load image tokens
    data = np.load(image_path, allow_pickle=True)
    ids = data["ids"]
    tokens_flat = data["tokens_flat"]

    # Find matching gaia_source_id
    gaia_id = row.get("gaia_source_id")
    if pd.isna(gaia_id):
        return None

    try:
        idx = np.where(ids == gaia_id)[0]
        if len(idx) > 0:
            return tokens_flat[idx[0]]  # Shape: (576,) for 24x24 image
    except:
        pass

    return None


def tokenize_scalars(row, scalar_tokenizer):
    """Tokenize scalar values from raw data."""
    # Get scalar columns that exist in both row and tokenizer
    available_cols = [c for c in scalar_tokenizer.bin_edges_.keys() if c in row.index]

    if not available_cols:
        return None, None

    # Extract values and tokenize
    values = []
    for col in available_cols:
        val = row[col]
        if pd.isna(val):
            val = scalar_tokenizer.col_means_.get(col, 0.0)
        values.append(val)

    values = np.array(values, dtype=np.float64)

    # Tokenize each column
    tokens = []
    for col, val in zip(available_cols, values):
        edges = scalar_tokenizer.bin_edges_[col]
        token = np.searchsorted(edges, val, side="right") - 1
        token = np.clip(token, 0, len(edges) - 2)
        tokens.append(token)

    return np.array(tokens, dtype=np.uint16), available_cols


def detokenize_scalars(tokens, col_names, scalar_tokenizer):
    """Convert scalar tokens back to physical values."""
    values = []
    for token, col in zip(tokens, col_names):
        edges = scalar_tokenizer.bin_edges_[col]
        # Use bin center as value estimate
        value = (edges[token] + edges[token + 1]) / 2
        values.append(value)
    return values


def predict_scalars_from_image(model, vocab, image_tokens, target_scalar_cols, device):
    """
    Given image tokens, predict scalar values.

    Returns predicted tokens for each scalar column.
    """
    # Prepare observed tokens (images)
    obs_tokens = torch.from_numpy(image_tokens).long() + vocab.base_image
    obs_tokens = obs_tokens[:576].unsqueeze(0).to(device)  # (1, 576)

    obs_types = torch.full_like(obs_tokens, 5)  # IMAGE type
    obs_positions = torch.arange(obs_tokens.size(1), device=device).unsqueeze(0)
    obs_mask = torch.ones_like(obs_tokens, dtype=torch.bool)

    # Prepare target (scalar columns we want to predict)
    n_scalars = len(target_scalar_cols)
    tgt_types = torch.zeros((1, n_scalars), dtype=torch.long, device=device)

    for i, col in enumerate(target_scalar_cols):
        type_id = vocab.get_scalar_type_id(col)
        tgt_types[0, i] = type_id

    tgt_positions = torch.arange(n_scalars, device=device).unsqueeze(0)
    tgt_mask = torch.ones((1, n_scalars), dtype=torch.bool, device=device)

    # Run model
    with torch.no_grad():
        logits = model(
            obs_tokens=obs_tokens,
            obs_types=obs_types,
            obs_positions=obs_positions,
            obs_mask=obs_mask,
            tgt_types=tgt_types,
            tgt_positions=tgt_positions,
            tgt_mask=tgt_mask,
        )  # (1, n_scalars, vocab_size)

    # Get predictions (subtract base_scalar to get bin indices)
    preds = logits.argmax(dim=-1).squeeze(0)  # (n_scalars,)
    pred_tokens = preds - vocab.base_scalar
    pred_tokens = torch.clamp(pred_tokens, 0, 1023)

    return pred_tokens.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Inference from raw data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--field", type=str, default="STRIPE82-0001", help="Field name")
    parser.add_argument("--sample-idx", type=int, default=100, help="Sample index in field")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model, config, step = load_model(args.checkpoint, device)

    # Load tokenizers
    scalar_tokenizer = load_tokenizers()

    # Build vocab
    print("\nBuilding vocabulary...")
    vocab = build_vocab_spec(
        v_image=2048,
        v_scalar=1024,
        v_spectra=2048,
    )

    # Load raw sample
    row = load_raw_sample(args.field, args.sample_idx)
    if row is None:
        return

    # Tokenize image
    print("\nTokenizing image...")
    image_tokens = tokenize_image(row, args.field)
    if image_tokens is None:
        print("Error: Could not load image tokens")
        return
    print(f"✓ Image tokens: {image_tokens.shape} (24x24 image)")

    # Tokenize scalars (ground truth)
    print("\nTokenizing scalars...")
    scalar_tokens_gt, scalar_cols = tokenize_scalars(row, scalar_tokenizer)
    if scalar_tokens_gt is None:
        print("Error: Could not tokenize scalars")
        return
    print(f"✓ Ground truth: {len(scalar_cols)} scalar columns")

    # Select columns to predict (e.g., magnitudes)
    mag_cols = [c for c in scalar_cols if c.startswith("mag_pstotal_")]
    if not mag_cols:
        mag_cols = scalar_cols[:5]  # Use first 5 if no mags

    print(f"\n{'='*70}")
    print(f"PREDICTION: Image → Scalars")
    print(f"{'='*70}")
    print(f"Predicting {len(mag_cols)} columns: {', '.join(mag_cols[:3])}...")

    # Run inference
    pred_tokens = predict_scalars_from_image(
        model, vocab, image_tokens, mag_cols, device
    )

    # Detokenize
    pred_values = detokenize_scalars(pred_tokens, mag_cols, scalar_tokenizer)

    # Get ground truth values
    gt_indices = [scalar_cols.index(c) for c in mag_cols]
    gt_tokens = scalar_tokens_gt[gt_indices]
    gt_values = detokenize_scalars(gt_tokens, mag_cols, scalar_tokenizer)

    # Show results
    print(f"\n{'Column':<25} {'Ground Truth':<15} {'Predicted':<15} {'Error':<10}")
    print("-" * 70)

    for col, gt_val, pred_val in zip(mag_cols, gt_values, pred_values):
        error = abs(pred_val - gt_val)
        status = "✓" if error < 0.5 else "✗"
        print(f"{col:<25} {gt_val:>14.4f} {pred_val:>14.4f} {error:>9.4f} {status}")

    # Summary statistics
    errors = np.abs(np.array(pred_values) - np.array(gt_values))
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Mean Absolute Error: {errors.mean():.4f}")
    print(f"Median Absolute Error: {np.median(errors):.4f}")
    print(f"Max Error: {errors.max():.4f}")
    print(f"Columns within 0.5 mag: {(errors < 0.5).sum()}/{len(errors)}")

    # Token-level accuracy
    gt_tokens_arr = gt_tokens.astype(int)
    pred_tokens_arr = pred_tokens.astype(int)
    token_acc = (gt_tokens_arr == pred_tokens_arr).mean()
    print(f"\nToken-level accuracy: {token_acc:.2%}")

    # Show raw data for reference
    print(f"\n{'='*70}")
    print(f"RAW DATA (first 3 scalar columns)")
    print(f"{'='*70}")
    for col in scalar_cols[:3]:
        print(f"{col}: {row[col]}")


if __name__ == "__main__":
    main()
