#!/usr/bin/env python3
"""
Train AION-style Transformer with Masked Multimodal Modeling

This script implements the correct AION training approach:
1. Encoder-decoder architecture (NOT causal/autoregressive)
2. 4M-style masked modeling (NOT next-token prediction)
3. Random masking with input/output token budgets
4. Decoder queries without token embeddings

Key differences from previous approach:
- Uses AIONTransformer (encoder-decoder) instead of CausalTransformerLM
- Uses FourMMaskingStrategy instead of next-token prediction
- Loss only on masked tokens, not all positions
"""

from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import math
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Import your existing components
from src.astromodal.models import AIONTransformer, count_parameters
from src.astromodal.training.masking import FourMMaskingStrategy
from src.astromodal.core import build_vocab_spec, discover_fields_from_datacubes
from src.astromodal.datasets import MultiFieldTokenDataset, collate_sequences
from functools import partial

# ============================================================
# CONFIG
# ============================================================

# Paths
DATACUBE_TMPL = "/home/astrodados4/downloads/hypercube/datacube_{field}.parquet"
IMAGE_TOKENS_TMPL = "/home/schwarz/projetoFM/outputs/image_tokens/datacube_{field}_tokens.npz"
SCALAR_TOKENS_TMPL = "/home/schwarz/projetoFM/outputs/scalar_tokens/datacube_{field}_scalar_tokens.npz"
SPECTRUM_TOKENS_TMPL = "/home/schwarz/projetoFM/outputs/spectrum_tokens/datacube_{field}_spectrum_tokens.npz"

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
MAGERR_COL = "err_mag_pstotal_r"

# Model architecture (AION-B size: 300M params)
D_MODEL = 768
N_ENC_LAYERS = 12
N_DEC_LAYERS = 12
N_HEADS = 12
D_FF = 3072
DROPOUT = 0.1
MAX_SEQ_LEN = 2048

# Masking strategy (from AION paper Section 5.2)
INPUT_BUDGET = 256   # Number of observed tokens
OUTPUT_BUDGET = 128  # Number of target tokens to predict
BETA_ALPHA = 0.5     # Beta distribution parameter (skewed toward 0)
BETA_BETA = 1.0

# Training
OUT_DIR = "runs_aion_masked"
SEED = 42
BATCH_SIZE = 16
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

EPOCHS = 5
LR = 2e-4
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 2000
GRAD_CLIP = 1.0
LOG_EVERY = 50
SAVE_EVERY_STEPS = 5000

# Performance
USE_TORCH_COMPILE = False  # Set True if PyTorch 2.x
USE_AMP = True

# ============================================================
# UTILS
# ============================================================

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def lr_schedule(step: int, base_lr: float, warmup: int, total_steps: int = 200_000) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = step - warmup
    T = total_steps - warmup
    frac = min(1.0, t / max(1, T))
    cos = 0.5 * (1 + math.cos(math.pi * frac))
    return base_lr * (0.1 + 0.9 * cos)

# ============================================================
# MAIN TRAINING
# ============================================================

def train():
    set_seed(SEED)
    ensure_dir(OUT_DIR)

    print("=" * 70)
    print("AION-Style Masked Multimodal Modeling Training")
    print("=" * 70)
    print("\n✓ Using CORRECT approach:")
    print("  - Encoder-decoder architecture (NOT causal)")
    print("  - Masked modeling (NOT next-token prediction)")
    print("  - 4M-style masking with input/output budgets")
    print("  - Decoder queries WITHOUT token embeddings")
    print()

    # Enable performance optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Build vocabulary
    vocab = build_vocab_spec(
        v_image=V_IMAGE,
        v_scalar=V_SCALAR,
        v_spectra=V_SPECTRA,
        spectrum_groups=SPECTRUM_GROUPS,
    )
    print(f"[vocab] Total vocabulary size: {vocab.vocab_size:,}")
    print(f"  - Image tokens: {V_IMAGE}")
    print(f"  - Scalar tokens: {V_SCALAR}")
    print(f"  - Spectrum tokens: {V_SPECTRA} x {len(SPECTRUM_GROUPS)} groups")
    print(f"  - Special tokens: 5 (PAD, BOS, EOS, SEP, MISS)")

    # Discover fields
    fields = discover_fields_from_datacubes(DATACUBE_TMPL)
    if not fields:
        raise RuntimeError("No fields found")
    print(f"\n[data] Found {len(fields)} fields")

    # Create dataset
    print("[data] Creating dataset...")
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
    print(f"[data] Dataset size: {len(dataset):,} sequences")

    # Create dataloader
    collate_fn = partial(collate_sequences, pad_id=vocab.pad_id)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Calculate number of token types
    n_types = 5 + 1 + 1 + len(SPECTRUM_GROUPS)  # special + image + scalar + spectra

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[model] Creating AION-style encoder-decoder transformer...")
    print(f"[model] Device: {device}")

    model = AIONTransformer(
        vocab_size=vocab.vocab_size,
        n_types=n_types,
        d_model=D_MODEL,
        n_enc_layers=N_ENC_LAYERS,
        n_dec_layers=N_DEC_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
        max_len=MAX_SEQ_LEN,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"[model] Architecture:")
    print(f"  - d_model: {D_MODEL}")
    print(f"  - Encoder layers: {N_ENC_LAYERS}")
    print(f"  - Decoder layers: {N_DEC_LAYERS}")
    print(f"  - Attention heads: {N_HEADS}")
    print(f"  - Feedforward dim: {D_FF}")
    print(f"[model] Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"[model] Approximate size: {total_params / 1e6:.1f}M params")

    if USE_TORCH_COMPILE and device.type == "cuda":
        try:
            model = torch.compile(model, mode="max-autotune")
            print("[perf] torch.compile enabled")
        except Exception as e:
            print(f"[perf] torch.compile failed: {e}")

    # Create masking strategy
    masking_strategy = FourMMaskingStrategy(
        input_budget=INPUT_BUDGET,
        output_budget=OUTPUT_BUDGET,
        beta_alpha=BETA_ALPHA,
        beta_beta=BETA_BETA,
        pad_id=vocab.pad_id,
    )
    print(f"\n[masking] 4M-style strategy:")
    print(f"  - Input budget: {INPUT_BUDGET} tokens")
    print(f"  - Output budget: {OUTPUT_BUDGET} tokens")
    print(f"  - Beta distribution: α={BETA_ALPHA}, β={BETA_BETA}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")

    print(f"\n[training] Configuration:")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Learning rate: {LR}")
    print(f"  - Weight decay: {WEIGHT_DECAY}")
    print(f"  - Warmup steps: {WARMUP_STEPS}")
    print(f"  - Gradient clipping: {GRAD_CLIP}")
    print(f"  - Mixed precision: {USE_AMP and device.type == 'cuda'}")
    print(f"  - Output directory: {OUT_DIR}")
    print()

    # Training loop
    global_step = 0
    model.train()

    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", dynamic_ncols=True)

        for batch in pbar:
            tokens = batch["tokens"].to(device, non_blocking=True)
            types = batch["types"].to(device, non_blocking=True)
            attn_mask = batch["attn_mask"].to(device, non_blocking=True)

            # Apply 4M-style masking
            masked = masking_strategy.apply(tokens, types, attn_mask)

            # Update learning rate
            lr_now = lr_schedule(global_step, LR, WARMUP_STEPS)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
                logits = model(
                    obs_tokens=masked.obs_tokens,
                    obs_types=masked.obs_types,
                    obs_positions=masked.obs_positions,
                    obs_mask=masked.obs_mask,
                    tgt_types=masked.tgt_types,
                    tgt_positions=masked.tgt_positions,
                    tgt_mask=masked.tgt_mask,
                )

                # Compute loss only on valid target positions
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    masked.tgt_tokens.reshape(-1),
                    ignore_index=vocab.pad_id,
                )

            # Backward pass
            scaler.scale(loss).backward()

            if GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            # Logging
            if global_step % LOG_EVERY == 0:
                loss_val = loss.item()
                pbar.set_postfix({
                    "loss": f"{loss_val:.3f}",
                    "lr": f"{lr_now:.2e}",
                    "step": global_step,
                })

            # Save checkpoint
            if global_step % SAVE_EVERY_STEPS == 0:
                ckpt_path = Path(OUT_DIR) / f"ckpt_step_{global_step:08d}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if USE_AMP else None,
                    "global_step": global_step,
                    "epoch": epoch,
                    "config": {
                        "vocab_size": vocab.vocab_size,
                        "n_types": n_types,
                        "d_model": D_MODEL,
                        "n_enc_layers": N_ENC_LAYERS,
                        "n_dec_layers": N_DEC_LAYERS,
                        "n_heads": N_HEADS,
                        "d_ff": D_FF,
                        "dropout": DROPOUT,
                        "max_len": MAX_SEQ_LEN,
                    },
                }, ckpt_path)
                tqdm.write(f"[save] {ckpt_path}")

        # Save epoch checkpoint
        epoch_path = Path(OUT_DIR) / f"ckpt_epoch_{epoch:03d}.pt"
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        }, epoch_path)
        tqdm.write(f"[save] {epoch_path}")

    print("\n" + "=" * 70)
    print("✓ Training complete!")
    print(f"✓ Checkpoints saved to: {OUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    train()
