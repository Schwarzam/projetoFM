#!/usr/bin/env python3
"""
Debug inference to identify what's wrong.

This script checks:
1. Model checkpoint validity
2. Vocabulary consistency
3. Token/type ID mappings
4. Tokenizer consistency
5. Simple sanity checks
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.astromodal.models import AIONTransformer
from src.astromodal.core import build_vocab_spec
from src.astromodal.tokenizers import ScalarTokenizer


def check_checkpoint(checkpoint_path: str):
    """Check if checkpoint is valid and what step it's from."""
    print("="*80)
    print("1. CHECKING CHECKPOINT")
    print("="*80)

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print(f"\nCheckpoint keys: {list(ckpt.keys())}")
    print(f"\nGlobal step: {ckpt.get('global_step', 'N/A')}")
    print(f"Epoch: {ckpt.get('epoch', 'N/A')}")

    if 'config' in ckpt:
        config = ckpt['config']
        print(f"\nModel config:")
        for k, v in config.items():
            print(f"  {k}: {v}")

    if 'model' in ckpt:
        print(f"\nModel state dict keys (first 10):")
        for i, k in enumerate(list(ckpt['model'].keys())[:10]):
            print(f"  {k}: {ckpt['model'][k].shape}")

    # Check training metrics if available
    if 'train_loss' in ckpt:
        print(f"\nTraining metrics:")
        print(f"  Train loss: {ckpt['train_loss']:.4f}")
    if 'val_loss' in ckpt:
        print(f"  Val loss: {ckpt['val_loss']:.4f}")

    return ckpt


def check_vocab_consistency(ckpt):
    """Check if vocab in checkpoint matches what we're building."""
    print("\n" + "="*80)
    print("2. CHECKING VOCABULARY CONSISTENCY")
    print("="*80)

    # Get vocab from checkpoint config
    config = ckpt.get('config', {})
    ckpt_vocab_size = config.get('vocab_size', None)
    ckpt_n_types = config.get('n_types', None)

    print(f"\nCheckpoint vocab config:")
    print(f"  vocab_size: {ckpt_vocab_size}")
    print(f"  n_types: {ckpt_n_types}")

    # Build vocab as we do in inference
    vocab = build_vocab_spec(v_image=2048, v_scalar=1024, v_spectra=2048)

    print(f"\nInference vocab spec:")
    print(f"  vocab_size: {vocab.vocab_size}")
    print(f"  n_types: {vocab.n_types}")
    print(f"  base_image: {vocab.base_image}")
    print(f"  base_scalar: {vocab.base_scalar}")
    print(f"  base_spectrum: {vocab.base_spectrum}")

    # Check for mismatch
    if ckpt_vocab_size and ckpt_vocab_size != vocab.vocab_size:
        print(f"\n❌ MISMATCH: Checkpoint vocab_size ({ckpt_vocab_size}) != Inference vocab_size ({vocab.vocab_size})")
        return False

    if ckpt_n_types and ckpt_n_types != vocab.n_types:
        print(f"\n❌ MISMATCH: Checkpoint n_types ({ckpt_n_types}) != Inference n_types ({vocab.n_types})")
        return False

    print(f"\n✓ Vocabulary sizes match!")
    return True


def check_scalar_tokenizer():
    """Check scalar tokenizer."""
    print("\n" + "="*80)
    print("3. CHECKING SCALAR TOKENIZER")
    print("="*80)

    tokenizer = ScalarTokenizer.load("scalar_tokenizers/scalar_tokenizer_config.npz")

    print(f"\nScalar tokenizer:")
    print(f"  Columns: {len(tokenizer.bin_edges_)}")
    print(f"  First 5 columns: {list(tokenizer.bin_edges_.keys())[:5]}")

    # Check bin ranges for a few columns
    print(f"\nBin ranges (first 3 columns):")
    for col in list(tokenizer.bin_edges_.keys())[:3]:
        edges = tokenizer.bin_edges_[col]
        print(f"  {col}:")
        print(f"    Bins: {len(edges)-1}")
        print(f"    Range: [{edges[0]:.4f}, {edges[-1]:.4f}]")
        print(f"    Mean: {tokenizer.col_means_.get(col, 'N/A')}")

    return tokenizer


def check_model_outputs(checkpoint_path: str):
    """Check if model produces reasonable outputs."""
    print("\n" + "="*80)
    print("4. CHECKING MODEL OUTPUTS")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']

    model = AIONTransformer(
        vocab_size=config['vocab_size'],
        n_types=config['n_types'],
        d_model=config['d_model'],
        n_enc_layers=config['n_enc_layers'],
        n_dec_layers=config['n_dec_layers'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        max_len=config['max_len'],
    ).to(device)

    model.load_state_dict(ckpt['model'])
    model.eval()

    # Create dummy input (image tokens -> predict 5 scalars)
    vocab = build_vocab_spec(v_image=2048, v_scalar=1024, v_spectra=2048)

    # Observed: 100 random image tokens
    obs_tokens = torch.randint(vocab.base_image, vocab.base_image + 100, (1, 100), device=device)
    obs_types = torch.full_like(obs_tokens, 5)  # IMAGE type
    obs_positions = torch.arange(100, device=device).unsqueeze(0)
    obs_mask = torch.ones_like(obs_tokens, dtype=torch.bool)

    # Target: 5 scalar columns
    tgt_types = torch.tensor([[6, 7, 8, 9, 10]], dtype=torch.long, device=device)  # First 5 scalar types
    tgt_positions = torch.arange(5, device=device).unsqueeze(0)
    tgt_mask = torch.ones((1, 5), dtype=torch.bool, device=device)

    print(f"\nTest forward pass:")
    print(f"  Observed tokens: {obs_tokens.shape}")
    print(f"  Target tokens: {tgt_types.shape}")

    with torch.no_grad():
        logits = model(
            obs_tokens=obs_tokens,
            obs_types=obs_types,
            obs_positions=obs_positions,
            obs_mask=obs_mask,
            tgt_types=tgt_types,
            tgt_positions=tgt_positions,
            tgt_mask=tgt_mask,
        )

    print(f"\nOutput logits shape: {logits.shape}")
    print(f"Expected: (1, 5, {config['vocab_size']})")

    # Check if logits are reasonable
    logit_mean = logits.mean().item()
    logit_std = logits.std().item()
    logit_max = logits.max().item()
    logit_min = logits.min().item()

    print(f"\nLogit statistics:")
    print(f"  Mean: {logit_mean:.4f}")
    print(f"  Std:  {logit_std:.4f}")
    print(f"  Min:  {logit_min:.4f}")
    print(f"  Max:  {logit_max:.4f}")

    # Check predictions
    preds = logits.argmax(dim=-1)
    print(f"\nPredicted token IDs (raw): {preds[0].cpu().numpy()}")

    # Check if predictions are in scalar range
    scalar_preds = preds - vocab.base_scalar
    print(f"Predicted token IDs (shifted): {scalar_preds[0].cpu().numpy()}")

    # Check if they're all the same (bad sign)
    if len(torch.unique(preds)) == 1:
        print(f"\n❌ WARNING: All predictions are the same token ({preds[0, 0].item()})!")
        print(f"   This suggests the model is not trained or completely collapsed.")
        return False

    print(f"\n✓ Model produces {len(torch.unique(preds))} unique predictions")
    return True


def check_training_data_match():
    """Check if we're using the same tokenizers as training."""
    print("\n" + "="*80)
    print("5. CHECKING TRAINING DATA CONSISTENCY")
    print("="*80)

    # Try to load a sample and tokenize it
    try:
        df = pd.read_parquet("/home/astrodados4/downloads/hypercube/datacube_STRIPE82-0001.parquet")
        df_filtered = df[(df["mag_pstotal_r"] > 14.0) & (df["mag_pstotal_r"] < 22.0)]

        row = df_filtered.iloc[0]

        # Check what columns we have
        tokenizer = ScalarTokenizer.load("scalar_tokenizers/scalar_tokenizer_config.npz")
        available_cols = [c for c in tokenizer.bin_edges_.keys() if c in row.index]

        print(f"\nSample object has {len(available_cols)} scalar columns")
        print(f"Tokenizer has {len(tokenizer.bin_edges_)} columns")

        # Tokenize a few values
        print(f"\nTokenizing first 3 scalars:")
        for col in available_cols[:3]:
            val = row[col]
            edges = tokenizer.bin_edges_[col]
            token = np.searchsorted(edges, val, side="right") - 1
            token = np.clip(token, 0, len(edges) - 2)

            # Detokenize
            reconstructed = (edges[token] + edges[token + 1]) / 2
            error = abs(reconstructed - val)

            print(f"  {col}:")
            print(f"    Value: {val:.4f} -> Token: {token} -> Reconstructed: {reconstructed:.4f}")
            print(f"    Error: {error:.4f} ({error/abs(val)*100:.2f}%)")

    except Exception as e:
        print(f"\n❌ Error loading/tokenizing data: {e}")
        return False

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    print(f"\n{'#'*80}")
    print(f"# DEBUGGING INFERENCE ISSUES")
    print(f"# Checkpoint: {args.checkpoint}")
    print(f"{'#'*80}\n")

    # Run all checks
    ckpt = check_checkpoint(args.checkpoint)
    vocab_ok = check_vocab_consistency(ckpt)
    tokenizer = check_scalar_tokenizer()
    model_ok = check_model_outputs(args.checkpoint)
    data_ok = check_training_data_match()

    # Summary
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)

    if vocab_ok and model_ok and data_ok:
        print("\n✓ All checks passed!")
        print("\nPossible issues:")
        print("  1. Model might not be trained enough (check step number)")
        print("  2. Training might have used wrong loss/objective")
        print("  3. Learning rate might be too high (model diverged)")
        print("  4. Check training logs for issues")
    else:
        print("\n❌ Issues detected:")
        if not vocab_ok:
            print("  - Vocabulary mismatch between checkpoint and inference")
        if not model_ok:
            print("  - Model not producing reasonable outputs")
        if not data_ok:
            print("  - Data/tokenizer consistency issues")

    print()


if __name__ == "__main__":
    main()
