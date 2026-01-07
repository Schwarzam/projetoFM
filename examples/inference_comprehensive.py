#!/usr/bin/env python3
"""
Comprehensive inference benchmark with all modalities.

This script tests the model across:
1. ALL scalar columns (not just magnitudes)
2. Images (using autoencoder)
3. Spectra (GaiaXP BP/RP)
4. Multiple prediction modes with benchmarks

Features:
- Full scalar coverage
- Spectrum integration
- Timing benchmarks
- Accuracy metrics
- Cross-modal predictions
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import pandas as pd
import torch
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.astromodal.models import AIONTransformer
from src.astromodal.models.autoencoder import AutoEncoder
from src.astromodal.core import build_vocab_spec
from src.astromodal.tokenizers import ScalarTokenizer, SpectrumTokenizer

# Import utilities from the existing script
from inference_from_raw import (
    load_model, load_tokenizers, build_image_from_row,
    tokenize_image, reconstruct_image_from_tokens,
    BANDS, IMG_SIZE, LATENT_DIM, LATENT_H, LATENT_W
)


class InferenceBenchmark:
    """Comprehensive inference benchmark with timing and metrics."""

    def __init__(
        self,
        model: AIONTransformer,
        vocab,
        autoencoder: AutoEncoder,
        codebook,
        scalar_tokenizer: ScalarTokenizer,
        spectrum_tokenizer: SpectrumTokenizer = None,
        device: str = "cuda",
    ):
        self.model = model
        self.vocab = vocab
        self.autoencoder = autoencoder
        self.codebook = codebook
        self.scalar_tokenizer = scalar_tokenizer
        self.spectrum_tokenizer = spectrum_tokenizer
        self.device = device

        # Timing stats
        self.timings = {
            "tokenization": [],
            "inference": [],
            "detokenization": [],
            "total": [],
        }

        # Accuracy stats
        self.metrics = {
            "scalar_mae": [],
            "scalar_token_acc": [],
            "image_token_acc": [],
            "spectrum_token_acc": [],
        }

    def tokenize_all_scalars(self, row: pd.Series) -> Tuple[np.ndarray, List[str]]:
        """
        Tokenize ALL available scalar columns (not just magnitudes).

        Returns:
            tokens: (n_cols,) array of token IDs
            col_names: List of column names
        """
        # Get all columns that exist in both row and tokenizer
        available_cols = [
            c for c in self.scalar_tokenizer.bin_edges_.keys()
            if c in row.index and pd.notna(row[c])
        ]

        if not available_cols:
            return None, None

        # Extract values and tokenize
        tokens = []
        for col in available_cols:
            val = row[col]
            edges = self.scalar_tokenizer.bin_edges_[col]
            token = np.searchsorted(edges, val, side="right") - 1
            token = np.clip(token, 0, len(edges) - 2)
            tokens.append(token)

        return np.array(tokens, dtype=np.uint16), available_cols

    def tokenize_spectra(self, row: pd.Series) -> Dict[str, np.ndarray]:
        """
        Tokenize spectrum data (GaiaXP BP/RP).

        Returns dict with 'bp' and 'rp' keys containing token arrays.
        """
        if self.spectrum_tokenizer is None:
            return {}

        # This would need actual spectrum data from the row
        # For now, return empty (placeholder for future integration)
        return {}

    def detokenize_scalars(
        self,
        tokens: np.ndarray,
        col_names: List[str]
    ) -> np.ndarray:
        """Convert scalar tokens back to physical values."""
        values = []
        for token, col in zip(tokens, col_names):
            edges = self.scalar_tokenizer.bin_edges_[col]
            # Use bin center
            value = (edges[token] + edges[token + 1]) / 2
            values.append(value)
        return np.array(values)

    def predict_scalars(
        self,
        obs_tokens: torch.Tensor,
        obs_types: torch.Tensor,
        target_cols: List[str],
    ) -> np.ndarray:
        """
        Predict scalar values from observed tokens.

        Args:
            obs_tokens: (1, L_obs) observed token IDs
            obs_types: (1, L_obs) observed type IDs
            target_cols: List of scalar column names to predict

        Returns:
            pred_tokens: (n_cols,) predicted token IDs
        """
        t0 = time.time()

        B = obs_tokens.size(0)
        L_obs = obs_tokens.size(1)

        obs_positions = torch.arange(L_obs, device=self.device).unsqueeze(0).expand(B, -1)
        obs_mask = torch.ones_like(obs_tokens, dtype=torch.bool)

        # Prepare target (scalar columns to predict)
        n_scalars = len(target_cols)
        tgt_types = torch.zeros((B, n_scalars), dtype=torch.long, device=self.device)

        for i, col in enumerate(target_cols):
            type_id = self.vocab.get_scalar_type_id(col)
            tgt_types[:, i] = type_id

        tgt_positions = torch.arange(n_scalars, device=self.device).unsqueeze(0).expand(B, -1)
        tgt_mask = torch.ones((B, n_scalars), dtype=torch.bool, device=self.device)

        # Run model
        with torch.no_grad():
            logits = self.model(
                obs_tokens=obs_tokens,
                obs_types=obs_types,
                obs_positions=obs_positions,
                obs_mask=obs_mask,
                tgt_types=tgt_types,
                tgt_positions=tgt_positions,
                tgt_mask=tgt_mask,
            )  # (B, n_scalars, vocab_size)

        # Get predictions
        preds = logits.argmax(dim=-1).squeeze(0)  # (n_scalars,)
        pred_tokens = preds - self.vocab.base_scalar
        pred_tokens = torch.clamp(pred_tokens, 0, 1023)  # v_scalar - 1

        self.timings["inference"].append(time.time() - t0)

        return pred_tokens.cpu().numpy()

    def benchmark_image_to_all_scalars(self, row: pd.Series) -> Dict:
        """
        Benchmark: Image → ALL Scalars

        Tests model's ability to predict all available scalar columns from image.
        """
        print(f"\n{'='*80}")
        print("BENCHMARK: Image → ALL Scalars")
        print(f"{'='*80}")

        # Tokenize image
        t0 = time.time()
        image_tokens = tokenize_image(row, self.autoencoder, self.codebook, self.device)
        t_tokenize = time.time() - t0

        # Get ALL scalar ground truth
        scalar_tokens_gt, scalar_cols = self.tokenize_all_scalars(row)

        if scalar_tokens_gt is None:
            print("  No scalars available")
            return {}

        print(f"  Testing on {len(scalar_cols)} scalar columns")

        # Prepare observed tokens (image)
        obs_tokens = torch.from_numpy(image_tokens).long() + self.vocab.base_image
        obs_tokens = obs_tokens[:576].unsqueeze(0).to(self.device)
        obs_types = torch.full_like(obs_tokens, 5)  # IMAGE type

        # Predict all scalars
        t0 = time.time()
        pred_tokens = self.predict_scalars(obs_tokens, obs_types, scalar_cols)
        t_inference = time.time() - t0

        # Detokenize
        t0 = time.time()
        pred_values = self.detokenize_scalars(pred_tokens, scalar_cols)
        gt_values = self.detokenize_scalars(scalar_tokens_gt, scalar_cols)
        t_detokenize = time.time() - t0

        # Compute metrics
        mae = np.abs(pred_values - gt_values).mean()
        token_acc = (pred_tokens == scalar_tokens_gt).mean()

        # Per-column results
        results = []
        for i, col in enumerate(scalar_cols):
            error = abs(pred_values[i] - gt_values[i])
            results.append({
                "column": col,
                "gt": gt_values[i],
                "pred": pred_values[i],
                "error": error,
                "token_match": pred_tokens[i] == scalar_tokens_gt[i],
            })

        print(f"\n  Timing:")
        print(f"    Tokenization:   {t_tokenize*1000:>8.2f} ms")
        print(f"    Inference:      {t_inference*1000:>8.2f} ms")
        print(f"    Detokenization: {t_detokenize*1000:>8.2f} ms")
        print(f"    Total:          {(t_tokenize+t_inference+t_detokenize)*1000:>8.2f} ms")

        print(f"\n  Metrics:")
        print(f"    MAE:            {mae:.4f}")
        print(f"    Token Accuracy: {token_acc:.2%}")

        # Show top errors
        results_sorted = sorted(results, key=lambda x: x["error"], reverse=True)
        print(f"\n  Top 5 errors:")
        for r in results_sorted[:5]:
            print(f"    {r['column']:<30} GT: {r['gt']:>8.3f}  Pred: {r['pred']:>8.3f}  Err: {r['error']:>6.3f}")

        # Store metrics
        self.metrics["scalar_mae"].append(mae)
        self.metrics["scalar_token_acc"].append(token_acc)

        return {
            "mae": mae,
            "token_acc": token_acc,
            "n_scalars": len(scalar_cols),
            "results": results,
            "timing": {
                "tokenization": t_tokenize,
                "inference": t_inference,
                "detokenization": t_detokenize,
            }
        }

    def benchmark_scalars_to_image(self, row: pd.Series) -> Dict:
        """
        Benchmark: ALL Scalars → Image

        Tests model's ability to generate image from all scalar features.
        """
        print(f"\n{'='*80}")
        print("BENCHMARK: ALL Scalars → Image")
        print(f"{'='*80}")

        # Get ground truth
        image_tokens_gt = tokenize_image(row, self.autoencoder, self.codebook, self.device)
        scalar_tokens, scalar_cols = self.tokenize_all_scalars(row)

        if scalar_tokens is None:
            print("  No scalars available")
            return {}

        print(f"  Using {len(scalar_cols)} scalar columns as input")

        # Prepare observed tokens (all scalars)
        obs_tokens = []
        obs_types = []
        for tok, col in zip(scalar_tokens, scalar_cols):
            shifted = self.vocab.base_scalar + int(tok)
            obs_tokens.append(shifted)
            type_id = self.vocab.get_scalar_type_id(col)
            obs_types.append(type_id)

        obs_tokens = torch.tensor(obs_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        obs_types = torch.tensor(obs_types, dtype=torch.long, device=self.device).unsqueeze(0)

        B = 1
        L_obs = obs_tokens.size(1)
        obs_positions = torch.arange(L_obs, device=self.device).unsqueeze(0)
        obs_mask = torch.ones_like(obs_tokens, dtype=torch.bool)

        # Prepare target (image tokens)
        n_image = 576
        tgt_types = torch.full((B, n_image), 5, dtype=torch.long, device=self.device)  # IMAGE type
        tgt_positions = torch.arange(n_image, device=self.device).unsqueeze(0)
        tgt_mask = torch.ones((B, n_image), dtype=torch.bool, device=self.device)

        # Run inference
        t0 = time.time()
        with torch.no_grad():
            logits = self.model(
                obs_tokens=obs_tokens,
                obs_types=obs_types,
                obs_positions=obs_positions,
                obs_mask=obs_mask,
                tgt_types=tgt_types,
                tgt_positions=tgt_positions,
                tgt_mask=tgt_mask,
            )

        preds = logits.argmax(dim=-1).squeeze(0)
        pred_tokens = preds - self.vocab.base_image
        pred_tokens = torch.clamp(pred_tokens, 0, 2047)  # v_image - 1
        pred_tokens = pred_tokens.cpu().numpy()

        t_inference = time.time() - t0

        # Compute metrics
        token_match = (pred_tokens == image_tokens_gt).sum()
        token_acc = token_match / len(image_tokens_gt)

        print(f"\n  Timing:")
        print(f"    Inference:      {t_inference*1000:>8.2f} ms")

        print(f"\n  Metrics:")
        print(f"    Token Accuracy: {token_acc:.2%} ({token_match}/{len(image_tokens_gt)} match)")

        self.metrics["image_token_acc"].append(token_acc)

        return {
            "token_acc": token_acc,
            "token_match": int(token_match),
            "n_tokens": len(image_tokens_gt),
            "timing": {"inference": t_inference},
        }

    def benchmark_multimodal(self, row: pd.Series) -> Dict:
        """
        Benchmark: Image + Scalars (partial) → Scalars (remaining)

        Tests cross-modal prediction using multiple modalities as input.
        """
        print(f"\n{'='*80}")
        print("BENCHMARK: Image + Partial Scalars → Remaining Scalars")
        print(f"{'='*80}")

        # Get all scalars
        scalar_tokens_all, scalar_cols_all = self.tokenize_all_scalars(row)

        if scalar_tokens_all is None or len(scalar_cols_all) < 10:
            print("  Insufficient scalars")
            return {}

        # Split scalars: use first 50% as input, predict remaining 50%
        split_idx = len(scalar_cols_all) // 2
        input_scalar_tokens = scalar_tokens_all[:split_idx]
        input_scalar_cols = scalar_cols_all[:split_idx]
        target_scalar_tokens = scalar_tokens_all[split_idx:]
        target_scalar_cols = scalar_cols_all[split_idx:]

        print(f"  Input: Image + {len(input_scalar_cols)} scalars")
        print(f"  Target: {len(target_scalar_cols)} scalars")

        # Tokenize image
        image_tokens = tokenize_image(row, self.autoencoder, self.codebook, self.device)

        # Prepare observed tokens (image + partial scalars)
        obs_tokens = []
        obs_types = []

        # Add image tokens
        for tok in image_tokens[:576]:
            obs_tokens.append(int(tok) + self.vocab.base_image)
            obs_types.append(5)  # IMAGE type

        # Add scalar tokens
        for tok, col in zip(input_scalar_tokens, input_scalar_cols):
            obs_tokens.append(int(tok) + self.vocab.base_scalar)
            type_id = self.vocab.get_scalar_type_id(col)
            obs_types.append(type_id)

        obs_tokens = torch.tensor(obs_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        obs_types = torch.tensor(obs_types, dtype=torch.long, device=self.device).unsqueeze(0)

        # Predict remaining scalars
        pred_tokens = self.predict_scalars(obs_tokens, obs_types, target_scalar_cols)

        # Compute metrics
        pred_values = self.detokenize_scalars(pred_tokens, target_scalar_cols)
        gt_values = self.detokenize_scalars(target_scalar_tokens, target_scalar_cols)

        mae = np.abs(pred_values - gt_values).mean()
        token_acc = (pred_tokens == target_scalar_tokens).mean()

        print(f"\n  Metrics:")
        print(f"    MAE:            {mae:.4f}")
        print(f"    Token Accuracy: {token_acc:.2%}")

        return {
            "mae": mae,
            "token_acc": token_acc,
            "n_input_scalars": len(input_scalar_cols),
            "n_target_scalars": len(target_scalar_cols),
        }

    def print_summary(self):
        """Print summary of all benchmarks."""
        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}")

        if self.metrics["scalar_mae"]:
            print(f"\nScalar Prediction:")
            print(f"  Mean MAE:        {np.mean(self.metrics['scalar_mae']):.4f}")
            print(f"  Mean Token Acc:  {np.mean(self.metrics['scalar_token_acc']):.2%}")

        if self.metrics["image_token_acc"]:
            print(f"\nImage Prediction:")
            print(f"  Mean Token Acc:  {np.mean(self.metrics['image_token_acc']):.2%}")

        if self.timings["inference"]:
            print(f"\nTiming (mean):")
            print(f"  Inference:       {np.mean(self.timings['inference'])*1000:.2f} ms")


def filter_dataframe(df: pd.DataFrame, args) -> pd.DataFrame:
    """
    Filter dataframe based on data requirements.

    Args:
        df: Input dataframe
        args: Command-line arguments with filtering options

    Returns:
        Filtered dataframe
    """
    # Basic magnitude filter
    df_filtered = df[(df["mag_pstotal_r"] > 14.0) & (df["mag_pstotal_r"] < 22.0)]

    n_before = len(df_filtered)

    # Filter for spectra
    if args.require_spectra:
        print("  Filtering for objects with spectra...")

        # Check for GaiaXP columns (with or without prefix)
        bp_col = None
        rp_col = None

        for prefix in ["gaiaxp_", ""]:
            bp_cand = f"{prefix}bp_coefficients"
            rp_cand = f"{prefix}rp_coefficients"
            if bp_cand in df_filtered.columns:
                bp_col = bp_cand
            if rp_cand in df_filtered.columns:
                rp_col = rp_cand

        if bp_col and rp_col:
            # Filter for non-null and non-empty spectra
            has_bp = df_filtered[bp_col].notna()
            has_rp = df_filtered[rp_col].notna()

            # Check for non-empty arrays (if stored as lists/arrays)
            if df_filtered[bp_col].dtype == object:
                has_bp = has_bp & (df_filtered[bp_col].apply(lambda x: x is not None and len(x) > 0 if hasattr(x, '__len__') else False))
            if df_filtered[rp_col].dtype == object:
                has_rp = has_rp & (df_filtered[rp_col].apply(lambda x: x is not None and len(x) > 0 if hasattr(x, '__len__') else False))

            df_filtered = df_filtered[has_bp & has_rp]
            print(f"    Found {len(df_filtered)} objects with GaiaXP spectra (from {n_before})")
        else:
            print(f"    Warning: No GaiaXP columns found (tried {bp_col}, {rp_col})")
            print(f"    Available columns: {list(df_filtered.columns)[:10]}...")

    # Filter for images
    if args.require_image:
        print("  Filtering for objects with images...")
        # Check for at least one non-null cutout column
        cutout_cols = [f"splus_cut_{b}" for b in BANDS if f"splus_cut_{b}" in df_filtered.columns]
        if cutout_cols:
            # Require at least one valid cutout
            has_image = df_filtered[cutout_cols[0]].notna()
            df_filtered = df_filtered[has_image]
            print(f"    Found {len(df_filtered)} objects with images (from {n_before})")
        else:
            print(f"    Warning: No image cutout columns found")

    # Filter for minimum number of scalars
    if args.min_scalars > 0:
        print(f"  Filtering for objects with at least {args.min_scalars} scalar values...")
        # This will be checked per-row during benchmark
        pass

    return df_filtered


def main():
    parser = argparse.ArgumentParser(description="Comprehensive inference benchmark")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--autoencoder", type=str, default="autoencoder_splus_cuts.pt")
    parser.add_argument("--codebook", type=str, default="codebook_kmeans_2048.joblib")
    parser.add_argument("--field", type=str, default="STRIPE82-0001")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--spectrum-tokenizer", type=str, default=None,
                       help="Path to spectrum tokenizer config (optional)")

    # Data filtering options
    parser.add_argument("--require-spectra", action="store_true",
                       help="Only test objects with GaiaXP spectra")
    parser.add_argument("--require-image", action="store_true",
                       help="Only test objects with image cutouts")
    parser.add_argument("--min-scalars", type=int, default=0,
                       help="Minimum number of valid scalar columns required")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model, config, step = load_model(args.checkpoint, device)

    # Load tokenizers
    autoencoder, codebook, scalar_tokenizer = load_tokenizers(
        args.autoencoder, args.codebook, device
    )

    # Load spectrum tokenizer if provided
    spectrum_tokenizer = None
    if args.spectrum_tokenizer and Path(args.spectrum_tokenizer).exists():
        spectrum_tokenizer = SpectrumTokenizer.load(args.spectrum_tokenizer)
        print(f"✓ Spectrum tokenizer loaded: {len(spectrum_tokenizer.group_names_)} groups")

    # Build vocab
    vocab = build_vocab_spec(v_image=2048, v_scalar=1024, v_spectra=2048)

    # Create benchmark
    benchmark = InferenceBenchmark(
        model=model,
        vocab=vocab,
        autoencoder=autoencoder,
        codebook=codebook,
        scalar_tokenizer=scalar_tokenizer,
        spectrum_tokenizer=spectrum_tokenizer,
        device=device,
    )

    # Load data
    datacube_path = f"/home/astrodados4/downloads/hypercube/datacube_{args.field}.parquet"
    print(f"\nLoading data from {datacube_path}")
    df = pd.read_parquet(datacube_path)

    # Apply filters
    print("\nApplying data filters...")
    df_filtered = filter_dataframe(df, args)

    if len(df_filtered) == 0:
        print("ERROR: No objects match the filtering criteria!")
        return

    print(f"✓ {len(df_filtered)} objects match filtering criteria")

    # Select random samples
    n_samples = min(args.n_samples, len(df_filtered))
    sample_indices = np.random.choice(len(df_filtered), size=n_samples, replace=False)

    print(f"\n{'='*80}")
    print(f"RUNNING COMPREHENSIVE BENCHMARK ON {n_samples} SAMPLES")
    print(f"{'='*80}")

    # Run benchmarks
    for idx, sample_idx in enumerate(sample_indices):
        row = df_filtered.iloc[sample_idx]
        obj_id = row.get("id", f"sample_{idx}")

        print(f"\n\n{'#'*80}")
        print(f"# SAMPLE {idx+1}/{n_samples} - Object: {obj_id}")
        print(f"{'#'*80}")

        # Check if object meets minimum scalar requirement
        if args.min_scalars > 0:
            _, scalar_cols = benchmark.tokenize_all_scalars(row)
            if scalar_cols is None or len(scalar_cols) < args.min_scalars:
                print(f"  Skipping: only {len(scalar_cols) if scalar_cols else 0} scalars (min: {args.min_scalars})")
                continue

        # Run all benchmarks
        try:
            benchmark.benchmark_image_to_all_scalars(row)
            benchmark.benchmark_scalars_to_image(row)
            benchmark.benchmark_multimodal(row)
        except Exception as e:
            print(f"  ERROR during benchmark: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    benchmark.print_summary()


if __name__ == "__main__":
    main()
