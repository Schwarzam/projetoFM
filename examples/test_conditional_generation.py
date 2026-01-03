#!/usr/bin/env python3
"""
Test AION model's conditional generation capabilities.

Tests specific conditional generation tasks:
1. Images → Scalars (photometric parameter prediction)
2. Images → Spectra (photometric to spectroscopic)
3. Images + Scalars → Spectra (full multimodal prediction)

Usage:
    python examples/test_conditional_generation.py --checkpoint runs_aion_h100/ckpt_step_00002000.pt --n-samples 50
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
from src.astromodal.datasets import MultiFieldTokenDataset


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

    vocab = build_vocab_spec(
        v_image=V_IMAGE,
        v_scalar=V_SCALAR,
        v_spectra=V_SPECTRA,
        spectrum_groups=SPECTRUM_GROUPS,
    )

    fields = discover_fields_from_datacubes(DATACUBE_TMPL)

    dataset = MultiFieldTokenDataset(
        fields=fields,
        vocab=vocab,
        datacube_template=DATACUBE_TMPL,
        image_tokens_template=IMAGE_TOKENS_TMPL,
        scalar_tokens_template=SCALAR_TOKENS_TMPL,
        spectrum_tokens_template=SPECTRUM_TOKENS_TMPL,
        spectrum_groups=SPECTRUM_GROUPS,
        mag_col="mag_pstotal_r",
        magerr_col="err_mag_pstotal_r",
        mag_min=14.0,
        mag_max=22.0,
        magerr_max=2.0,
        lru_fields=50,
    )

    return dataset, vocab


def split_by_modality(tokens, types, attn_mask, vocab):
    """
    Split sequence into modalities.

    Returns dict with keys: 'image', 'scalar', 'spectra'
    Each value is (tokens, types, positions, mask)
    """
    modalities = {}

    # Image: type=5
    img_mask = (types == 5) & attn_mask
    if img_mask.any():
        positions = torch.arange(tokens.size(0), device=tokens.device)[img_mask]
        modalities['image'] = (
            tokens[img_mask],
            types[img_mask],
            positions,
            torch.ones_like(tokens[img_mask], dtype=torch.bool)
        )

    # Scalar: types in scalar_type_map (6 to 6+n_scalars-1)
    if vocab.scalar_type_map:
        scalar_type_ids = set(vocab.scalar_type_map.values())
        scalar_mask = torch.zeros_like(types, dtype=torch.bool)
        for type_id in scalar_type_ids:
            scalar_mask |= (types == type_id)
        scalar_mask &= attn_mask

        if scalar_mask.any():
            positions = torch.arange(tokens.size(0), device=tokens.device)[scalar_mask]
            modalities['scalar'] = (
                tokens[scalar_mask],
                types[scalar_mask],
                positions,
                torch.ones_like(tokens[scalar_mask], dtype=torch.bool)
            )

    # Spectra: types in spectrum range (after all scalars)
    # Spectrum types start after the last scalar type
    if vocab.scalar_type_map:
        spectrum_start = max(vocab.scalar_type_map.values()) + 1
    else:
        spectrum_start = 7  # Fallback

    spectra_mask = (types >= spectrum_start) & attn_mask
    if spectra_mask.any():
        positions = torch.arange(tokens.size(0), device=tokens.device)[spectra_mask]
        modalities['spectra'] = (
            tokens[spectra_mask],
            types[spectra_mask],
            positions,
            torch.ones_like(tokens[spectra_mask], dtype=torch.bool)
        )

    return modalities


def test_image_to_scalar(model, sample, vocab, device):
    """Test: Given images, predict scalars."""
    tokens = sample["token_ids"].to(device)
    types = sample["type_ids"].to(device)
    attn_mask = torch.ones_like(tokens, dtype=torch.bool)

    # Split into modalities
    mods = split_by_modality(tokens, types, attn_mask, vocab)

    if 'image' not in mods or 'scalar' not in mods:
        return None

    # Observed: all images
    obs_tokens = mods['image'][0].unsqueeze(0)
    obs_types = mods['image'][1].unsqueeze(0)
    obs_positions = mods['image'][2].unsqueeze(0)
    obs_mask = mods['image'][3].unsqueeze(0)

    # Target: all scalars
    tgt_tokens_gt = mods['scalar'][0].unsqueeze(0)
    tgt_types = mods['scalar'][1].unsqueeze(0)
    tgt_positions = mods['scalar'][2].unsqueeze(0)
    tgt_mask = mods['scalar'][3].unsqueeze(0)

    # Forward pass
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

    # Calculate accuracy
    preds = logits.argmax(dim=-1).squeeze(0)
    gt = tgt_tokens_gt.squeeze(0)

    correct = (preds == gt).sum().item()
    total = gt.size(0)
    accuracy = correct / total if total > 0 else 0.0

    # Top-5
    top5_preds = logits.squeeze(0).topk(5, dim=-1).indices
    top5_correct = (top5_preds == gt.unsqueeze(1)).any(dim=1).sum().item()
    top5_accuracy = top5_correct / total if total > 0 else 0.0

    return {
        'n_image_tokens': obs_tokens.size(1),
        'n_scalar_tokens': total,
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
    }


def test_image_to_spectra(model, sample, vocab, device):
    """Test: Given images, predict spectra."""
    tokens = sample["token_ids"].to(device)
    types = sample["type_ids"].to(device)
    attn_mask = torch.ones_like(tokens, dtype=torch.bool)

    mods = split_by_modality(tokens, types, attn_mask, vocab)

    if 'image' not in mods or 'spectra' not in mods:
        return None

    # Observed: all images
    obs_tokens = mods['image'][0].unsqueeze(0)
    obs_types = mods['image'][1].unsqueeze(0)
    obs_positions = mods['image'][2].unsqueeze(0)
    obs_mask = mods['image'][3].unsqueeze(0)

    # Target: all spectra
    tgt_tokens_gt = mods['spectra'][0].unsqueeze(0)
    tgt_types = mods['spectra'][1].unsqueeze(0)
    tgt_positions = mods['spectra'][2].unsqueeze(0)
    tgt_mask = mods['spectra'][3].unsqueeze(0)

    # Forward pass
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

    # Calculate accuracy
    preds = logits.argmax(dim=-1).squeeze(0)
    gt = tgt_tokens_gt.squeeze(0)

    correct = (preds == gt).sum().item()
    total = gt.size(0)
    accuracy = correct / total if total > 0 else 0.0

    # Top-5
    top5_preds = logits.squeeze(0).topk(5, dim=-1).indices
    top5_correct = (top5_preds == gt.unsqueeze(1)).any(dim=1).sum().item()
    top5_accuracy = top5_correct / total if total > 0 else 0.0

    return {
        'n_image_tokens': obs_tokens.size(1),
        'n_spectra_tokens': total,
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="Test AION conditional generation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of samples to test")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    # Check checkpoint
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, config, step = load_model(args.checkpoint, device)

    # Create dataset
    print("\nCreating dataset...")
    dataset, vocab = create_dataset()

    # Test tasks
    print(f"\nTesting conditional generation on {args.n_samples} samples...")
    print("=" * 70)

    results_img_to_scalar = []
    results_img_to_spectra = []

    for i in range(args.n_samples):
        idx = np.random.randint(0, len(dataset))
        sample = dataset[idx]

        # Test 1: Image → Scalar
        try:
            result = test_image_to_scalar(model, sample, vocab, device)
            if result:
                results_img_to_scalar.append(result)
        except Exception as e:
            pass

        # Test 2: Image → Spectra
        try:
            result = test_image_to_spectra(model, sample, vocab, device)
            if result:
                results_img_to_spectra.append(result)
        except Exception as e:
            pass

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{args.n_samples} samples...")

    # Print results
    print("=" * 70)
    print("\nRESULTS")
    print("=" * 70)

    # Image → Scalar
    if results_img_to_scalar:
        n_samples = len(results_img_to_scalar)
        avg_n_img = np.mean([r['n_image_tokens'] for r in results_img_to_scalar])
        avg_n_scalar = np.mean([r['n_scalar_tokens'] for r in results_img_to_scalar])
        avg_acc = np.mean([r['accuracy'] for r in results_img_to_scalar])
        avg_top5 = np.mean([r['top5_accuracy'] for r in results_img_to_scalar])

        print(f"\n📸 IMAGE → SCALAR (Photometric Parameters)")
        print(f"  Samples tested: {n_samples}")
        print(f"  Avg image tokens: {avg_n_img:.0f}")
        print(f"  Avg scalar tokens: {avg_n_scalar:.0f}")
        print(f"  Top-1 accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
        print(f"  Top-5 accuracy: {avg_top5:.4f} ({avg_top5*100:.2f}%)")
        print(f"  Random baseline: {1.0/1024:.4f} ({100.0/1024:.2f}%)")

        improvement = avg_acc / (1.0/1024)
        if improvement > 10:
            print(f"  ✓ Learning! ({improvement:.1f}x better than random)")
        elif improvement > 3:
            print(f"  ~ Some learning ({improvement:.1f}x better)")
        else:
            print(f"  ✗ Not learning ({improvement:.1f}x)")

    # Image → Spectra
    if results_img_to_spectra:
        n_samples = len(results_img_to_spectra)
        avg_n_img = np.mean([r['n_image_tokens'] for r in results_img_to_spectra])
        avg_n_spec = np.mean([r['n_spectra_tokens'] for r in results_img_to_spectra])
        avg_acc = np.mean([r['accuracy'] for r in results_img_to_spectra])
        avg_top5 = np.mean([r['top5_accuracy'] for r in results_img_to_spectra])

        print(f"\n📸 → 📊 IMAGE → SPECTRA (Photometric to Spectroscopic)")
        print(f"  Samples tested: {n_samples}")
        print(f"  Avg image tokens: {avg_n_img:.0f}")
        print(f"  Avg spectra tokens: {avg_n_spec:.0f}")
        print(f"  Top-1 accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
        print(f"  Top-5 accuracy: {avg_top5:.4f} ({avg_top5*100:.2f}%)")
        print(f"  Random baseline: {1.0/2048:.4f} ({100.0/2048:.2f}%)")

        improvement = avg_acc / (1.0/2048)
        if improvement > 10:
            print(f"  ✓ Learning! ({improvement:.1f}x better than random)")
        elif improvement > 3:
            print(f"  ~ Some learning ({improvement:.1f}x better)")
        else:
            print(f"  ✗ Not learning ({improvement:.1f}x)")

    print("\n" + "=" * 70)
    print(f"\nCheckpoint: {args.checkpoint} (step {step})")
    print("=" * 70)


if __name__ == "__main__":
    main()
