#!/usr/bin/env python3
"""
Optimized AION training for H100 GPU with performance improvements.

Key optimizations:
1. Larger batch size (64 vs 16) - better GPU utilization
2. torch.compile enabled
3. Better loss logging (more decimal places)
4. Faster DataLoader settings
5. Gradient accumulation support
6. Reduced epoch count with more frequent checkpoints
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

from src.astromodal.models import AIONTransformer, count_parameters
from src.astromodal.training.masking import FourMMaskingStrategy
from src.astromodal.core import build_vocab_spec, discover_fields_from_datacubes
from src.astromodal.datasets import MultiFieldTokenDataset, CachedTokenDataset, collate_sequences
from functools import partial

# ============================================================
# OPTIMIZED CONFIG FOR H100
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

# Masking strategy - Reduced for faster training
INPUT_BUDGET = 128  # Reduced from 256 - faster processing
OUTPUT_BUDGET = 64   # Reduced from 128 - faster processing
BETA_ALPHA = 0.5
BETA_BETA = 1.0

# Training - OPTIMIZED FOR H100
OUT_DIR = "runs_aion_h100"
SEED = 42
BATCH_SIZE = 256  # H100 can handle MUCH larger batches - maximize throughput
GRAD_ACCUM_STEPS = 1  # Accumulate gradients for effective batch size of 64
NUM_WORKERS = 16  # More workers for data loading (increase if still bottlenecked)
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 8  # More prefetching to keep GPU fed

# Reduce total steps by using fewer epochs or limiting dataset
MAX_STEPS = 100_000  # Stop after 100k steps instead of full epochs
EPOCHS = 1  # Just 1 epoch (will stop at MAX_STEPS anyway)

LR = 2e-4
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 2000
GRAD_CLIP = 1.0
LOG_EVERY = 10  # Log more frequently
SAVE_EVERY_STEPS = 2000  # Save checkpoints more often

# Performance - ALL ENABLED FOR H100
USE_TORCH_COMPILE = False  # Disable for now - has edge case bug with variable-length masking
USE_AMP = True  # Mixed precision
ENABLE_TF32 = True  # TF32 for faster matmul
USE_RAM_CACHE = False  # Disabled - too slow to cache 76M samples. Use optimized DataLoader instead.

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

def lr_schedule(step: int, base_lr: float, warmup: int, total_steps: int = 100_000) -> float:
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
    print("AION Training - H100 OPTIMIZED")
    print("=" * 70)
    print(f"\n🚀 Optimizations enabled:")
    print(f"  - Batch size: {BATCH_SIZE} (was 16)")
    print(f"  - Gradient accumulation: {GRAD_ACCUM_STEPS} steps")
    print(f"  - Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"  - torch.compile: {USE_TORCH_COMPILE}")
    print(f"  - Mixed precision: {USE_AMP}")
    print(f"  - TF32 matmul: {ENABLE_TF32}")
    print(f"  - DataLoader workers: {NUM_WORKERS}")
    print(f"  - Max training steps: {MAX_STEPS:,}")
    print()

    # Enable performance optimizations
    if torch.cuda.is_available() and ENABLE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
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

    # Discover fields
    fields = discover_fields_from_datacubes(DATACUBE_TMPL)
    if not fields:
        raise RuntimeError("No fields found")
    print(f"[data] Found {len(fields)} fields")

    # Create dataset
    print("[data] Creating dataset...")
    base_dataset = MultiFieldTokenDataset(
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
        lru_fields=200,  # Cache ALL fields in RAM (2444 fields fit easily in 1TB RAM!)
    )
    print(f"[data] Dataset size: {len(base_dataset):,} sequences")

    # Wrap with RAM cache if enabled
    if USE_RAM_CACHE:
        print("[data] RAM caching enabled - pre-loading all data...")
        dataset = CachedTokenDataset(base_dataset, verbose=True)
    else:
        dataset = base_dataset

    estimated_steps_per_epoch = len(dataset) // BATCH_SIZE
    print(f"[data] Estimated steps per epoch: {estimated_steps_per_epoch:,}")
    print(f"[data] Will train for max {MAX_STEPS:,} steps")

    # Create dataloader
    collate_fn = partial(collate_sequences, pad_id=vocab.pad_id)

    # Optimize DataLoader based on whether we're using RAM cache
    if USE_RAM_CACHE:
        # With RAM cache, use fewer workers (data is already in memory)
        dl_workers = 4
        dl_prefetch = 2
    else:
        # Without cache, use more workers to handle I/O
        dl_workers = NUM_WORKERS
        dl_prefetch = PREFETCH_FACTOR

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=dl_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS and dl_workers > 0,
        prefetch_factor=dl_prefetch if dl_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Calculate number of token types
    n_types = 5 + 1 + 1 + len(SPECTRUM_GROUPS)

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[model] Creating model on {device}...")

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
    print(f"[model] Parameters: {total_params/1e6:.1f}M")

    # Compile model if enabled
    if USE_TORCH_COMPILE and device.type == "cuda":
        try:
            print("[perf] Compiling model with torch.compile...")
            model = torch.compile(model, mode="max-autotune")
            print("[perf] ✓ Compilation successful")
        except Exception as e:
            print(f"[perf] ✗ Compilation failed: {e}")

    # Create masking strategy
    masking_strategy = FourMMaskingStrategy(
        input_budget=INPUT_BUDGET,
        output_budget=OUTPUT_BUDGET,
        beta_alpha=BETA_ALPHA,
        beta_beta=BETA_BETA,
        pad_id=vocab.pad_id,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")

    print(f"\n[training] Starting training...")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Learning rate: {LR}")
    print(f"  - Max steps: {MAX_STEPS:,}")
    print()

    # Training loop
    global_step = 0
    model.train()
    running_loss = 0.0
    start_time = time.time()

    # Profiling timers
    data_load_time = 0.0
    to_device_time = 0.0
    masking_time = 0.0
    forward_time = 0.0
    backward_time = 0.0

    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar):
            iter_start = time.time()

            if global_step >= MAX_STEPS:
                print(f"\n✓ Reached max steps ({MAX_STEPS}), stopping training")
                break

            # Time: data → device
            t0 = time.time()
            tokens = batch["tokens"].to(device, non_blocking=True)
            types = batch["types"].to(device, non_blocking=True)
            attn_mask = batch["attn_mask"].to(device, non_blocking=True)
            to_device_time += time.time() - t0

            # Time: masking
            t0 = time.time()
            masked = masking_strategy.apply(tokens, types, attn_mask)
            masking_time += time.time() - t0

            # Update learning rate
            lr_now = lr_schedule(global_step, LR, WARMUP_STEPS, MAX_STEPS)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            # Forward pass
            t0 = time.time()
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

                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    masked.tgt_tokens.reshape(-1),
                    ignore_index=vocab.pad_id,
                )
            forward_time += time.time() - t0

            # Scale loss for gradient accumulation
            loss = loss / GRAD_ACCUM_STEPS

            # Backward pass
            t0 = time.time()
            scaler.scale(loss).backward()
            backward_time += time.time() - t0

            # Update weights every GRAD_ACCUM_STEPS
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                if GRAD_CLIP > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

            # Track running loss
            running_loss += loss.item() * GRAD_ACCUM_STEPS

            # Logging
            if global_step % LOG_EVERY == 0 and global_step > 0:
                avg_loss = running_loss / LOG_EVERY
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed
                eta_seconds = (MAX_STEPS - global_step) / max(steps_per_sec, 0.001)
                eta_hours = eta_seconds / 3600

                pbar.set_postfix({
                    "loss": f"{avg_loss:.6f}",
                    "lr": f"{lr_now:.2e}",
                    "step": global_step,
                    "it/s": f"{steps_per_sec:.2f}",
                    "ETA": f"{eta_hours:.1f}h",
                })
                running_loss = 0.0

                # Print timing breakdown every 100 steps
                if global_step % 100 == 0:
                    total_time = time.time() - start_time
                    print(f"\n[PROFILE @ step {global_step}]")
                    print(f"  To-device: {to_device_time:.1f}s ({to_device_time/total_time*100:.1f}%)")
                    print(f"  Masking:   {masking_time:.1f}s ({masking_time/total_time*100:.1f}%)")
                    print(f"  Forward:   {forward_time:.1f}s ({forward_time/total_time*100:.1f}%)")
                    print(f"  Backward:  {backward_time:.1f}s ({backward_time/total_time*100:.1f}%)")
                    compute_time = to_device_time + masking_time + forward_time + backward_time
                    data_wait_time = total_time - compute_time
                    print(f"  DataLoader WAIT: {data_wait_time:.1f}s ({data_wait_time/total_time*100:.1f}%) ← BOTTLENECK!")
                    print()

            # Save checkpoint
            if global_step % SAVE_EVERY_STEPS == 0 and global_step > 0:
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

        if global_step >= MAX_STEPS:
            break

    # Save final checkpoint
    final_path = Path(OUT_DIR) / f"ckpt_final_{global_step:08d}.pt"
    torch.save({
        "model": model.state_dict(),
        "global_step": global_step,
    }, final_path)
    tqdm.write(f"[save] {final_path}")

    print("\n" + "=" * 70)
    print("✓ Training complete!")
    print(f"✓ Final step: {global_step:,}")
    print(f"✓ Checkpoints saved to: {OUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    train()
