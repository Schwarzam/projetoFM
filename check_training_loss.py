#!/usr/bin/env python3
"""
Quick diagnostic to check if training loss is actually zero or just displayed as such.

Run this to see actual loss values with more precision.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from src.astromodal.models import AIONTransformer, count_parameters
from src.astromodal.training.masking import FourMMaskingStrategy
from src.astromodal.core import build_vocab_spec, discover_fields_from_datacubes
from src.astromodal.datasets import MultiFieldTokenDataset, collate_sequences
from functools import partial
from torch.utils.data import DataLoader

# Minimal config
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

DATACUBE_TMPL = "/home/astrodados4/downloads/hypercube/datacube_{field}.parquet"
IMAGE_TOKENS_TMPL = "/home/schwarz/projetoFM/codecs/image_tokens/datacube_{field}_tokens.npz"
SCALAR_TOKENS_TMPL = "/home/schwarz/projetoFM/scalar_tokenizers/scalar_tokens/datacube_{field}_scalar_tokens.npz"
SPECTRUM_TOKENS_TMPL = "/home/schwarz/projetoFM/spectrum_tokenizers/spectrum_tokens/datacube_{field}_spectrum_tokens.npz"

MAG_COL = "mag_pstotal_r"
MAG_MIN = 14.0
MAG_MAX = 22.0
MAGERR_MAX = 2.0

print("=" * 70)
print("Loss Diagnostic")
print("=" * 70)
print()

# Build vocab
print("[1/5] Building vocabulary...")
vocab = build_vocab_spec(
    v_image=V_IMAGE,
    v_scalar=V_SCALAR,
    v_spectra=V_SPECTRA,
    spectrum_groups=SPECTRUM_GROUPS,
)
print(f"  Vocab size: {vocab.vocab_size:,}")

# Create dataset
print("\n[2/5] Creating dataset...")
fields = discover_fields_from_datacubes(DATACUBE_TMPL)
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
print(f"  Dataset size: {len(dataset):,}")

# Create dataloader
print("\n[3/5] Creating dataloader...")
collate_fn = partial(collate_sequences, pad_id=vocab.pad_id)
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    drop_last=True,
)

# Create model
print("\n[4/5] Creating model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_types = 5 + 1 + 1 + len(SPECTRUM_GROUPS)

model = AIONTransformer(
    vocab_size=vocab.vocab_size,
    n_types=n_types,
    d_model=768,
    n_enc_layers=12,
    n_dec_layers=12,
    n_heads=12,
    d_ff=3072,
    dropout=0.1,
    max_len=2048,
).to(device)

total_params, _ = count_parameters(model)
print(f"  Model: {total_params/1e6:.1f}M parameters")

# Create masking strategy
masking_strategy = FourMMaskingStrategy(
    input_budget=256,
    output_budget=128,
    beta_alpha=0.5,
    beta_beta=1.0,
    pad_id=vocab.pad_id,
)

# Test on a few batches
print("\n[5/5] Testing loss on 10 batches...")
print()
print("Batch | Loss Value        | Obs Tokens | Tgt Tokens | Valid Targets")
print("-" * 70)

model.eval()
with torch.no_grad():
    for i, batch in enumerate(dataloader):
        if i >= 10:
            break

        tokens = batch["tokens"].to(device)
        types = batch["types"].to(device)
        attn_mask = batch["attn_mask"].to(device)

        # Apply masking
        masked = masking_strategy.apply(tokens, types, attn_mask)

        # Forward pass
        logits = model(
            obs_tokens=masked.obs_tokens,
            obs_types=masked.obs_types,
            obs_positions=masked.obs_positions,
            obs_mask=masked.obs_mask,
            tgt_types=masked.tgt_types,
            tgt_positions=masked.tgt_positions,
            tgt_mask=masked.tgt_mask,
        )

        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            masked.tgt_tokens.reshape(-1),
            ignore_index=vocab.pad_id,
        )

        # Count valid targets
        valid_targets = (masked.tgt_tokens != vocab.pad_id).sum().item()

        print(f"{i+1:5d} | {loss.item():17.10f} | {masked.obs_tokens.size(1):10d} | {masked.tgt_tokens.size(1):10d} | {valid_targets:13d}")

print()
print("=" * 70)
print("Interpretation:")
print("  - Random model should have loss ≈ ln(vocab_size) ≈ ln(13317) ≈ 9.5")
print("  - Loss << 9.5 means model is learning")
print("  - Loss ≈ 0.001 or lower is suspiciously low (possible bug)")
print("  - If all losses are exactly the same, there might be a bug")
print("=" * 70)
