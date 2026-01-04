#!/usr/bin/env python3
"""
AION training script (H100-ready) with the improvements we discussed:

✅ Improvements implemented
1) bf16-native mixed precision on H100 (no GradScaler needed)
2) Better memory control: microbatch + grad accumulation (keep effective batch)
3) Cheap shuffling that preserves cache locality: BlockShuffleSampler (no global shuffle)
4) Activation checkpointing (optional) to unlock bigger models
5) More accurate profiling: measures true DataLoader wait time
6) "Resume training" support:
   - auto-load latest checkpoint in OUT_DIR
   - saves RNG states so resume is deterministic
7) Optional "short training" mode:
   - train only N steps for quick validation runs
   - or stop after a time budget (hours)
8) Cleaner checkpoints: keep last K + periodic saves

Notes:
- Remove CUDA_VISIBLE_DEVICES inside script if you want multi-GPU later.
- This keeps your dataset in cache-friendly order, but still introduces randomness.
"""

from __future__ import annotations
import os

# ---- Thread limits (good for DataLoader CPU contention) ----
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

# Single GPU by default (set outside if you want multi-gpu later)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import math
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from src.astromodal.models import AIONTransformer, count_parameters
from src.astromodal.training.masking import FourMMaskingStrategy
from src.astromodal.core import build_vocab_spec, discover_fields_from_datacubes
from src.astromodal.datasets import MultiFieldTokenDataset, CachedTokenDataset, collate_sequences
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

# Model architecture (current ~0.2B-ish)
D_MODEL = 768
N_ENC_LAYERS = 12
N_DEC_LAYERS = 12
N_HEADS = 12
D_FF = 3072
DROPOUT = 0.1
MAX_SEQ_LEN = 2048

# Masking strategy
INPUT_BUDGET = 1024
OUTPUT_BUDGET = 128
BETA_ALPHA = 0.5
BETA_BETA = 1.0

# Output
OUT_DIR = "runs_1"
SEED = 42

# ----------------------------
# Training / memory knobs
# ----------------------------
# IMPORTANT: memory is driven by MICRO_BATCH, not effective batch.
MICRO_BATCH = 128            # was 256; reduces peak VRAM a lot
GRAD_ACCUM_STEPS = 2         # effective batch = 256 again
EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM_STEPS

# Dataloader
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 8

# Stop conditions
MAX_STEPS = 100_000          # full run
SHORT_RUN_STEPS = 2_000      # quick sanity run
USE_SHORT_RUN = False        # set True to quickly validate
MAX_TRAIN_HOURS: Optional[float] = None  # e.g. 6.0 to stop after 6h

# Optim
LR = 5e-5
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 2000
GRAD_CLIP = 1.0
LOG_EVERY = 10
SAVE_EVERY_STEPS = 4000

# Precision / perf
ENABLE_TF32 = True
USE_BF16 = True  # recommended on H100
USE_TORCH_COMPILE = False    # keep off due to variable-length masking edge cases

# Memory unlockers
USE_RAM_CACHE = False
USE_ACTIVATION_CHECKPOINTING = True  # big memory saver; slower

# Shuffle that doesn't kill locality
USE_BLOCK_SHUFFLE = True
BLOCK_SHUFFLE_SIZE = 200_000  # block size in sample indices (tune)
IN_BLOCK_SHUFFLE_BUFFER = 50_000  # extra mixing inside each block (0 disables)

# Checkpoint mgmt
KEEP_LAST_N_CHECKPOINTS = 3
AUTO_RESUME = True


# ============================================================
# UTILS
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def lr_schedule(step: int, base_lr: float, warmup: int, total_steps: int) -> float:
    """Cosine LR schedule with warmup; decays to 10% of LR."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = step - warmup
    T = max(1, total_steps - warmup)
    frac = min(1.0, t / T)
    cos = 0.5 * (1 + math.cos(math.pi * frac))
    return base_lr * (0.1 + 0.9 * cos)


def now_s() -> float:
    return time.time()


def _save_checkpoint(
    out_dir: Path,
    step: int,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
):
    ckpt_path = out_dir / f"ckpt_step_{step:08d}.pt"
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "epoch": epoch,
        "config": config,
        # RNG states for deterministic resume:
        "rng_py": random.getstate(),
        "rng_np": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


def _find_latest_ckpt(out_dir: Path) -> Optional[Path]:
    if not out_dir.exists():
        return None
    cks = sorted(out_dir.glob("ckpt_step_*.pt"))
    return cks[-1] if cks else None


def _cleanup_old_checkpoints(out_dir: Path, keep_last: int):
    cks = sorted(out_dir.glob("ckpt_step_*.pt"))
    if len(cks) <= keep_last:
        return
    for p in cks[:-keep_last]:
        try:
            p.unlink()
        except Exception:
            pass


def _load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    # Restore RNG for deterministic continuation
    try:
        random.setstate(ckpt["rng_py"])
        np.random.set_state(ckpt["rng_np"])
        torch.set_rng_state(ckpt["rng_torch"])
        if torch.cuda.is_available() and ckpt.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
    except Exception:
        pass
    return ckpt


# ============================================================
# SHUFFLE (CACHE-FRIENDLY)
# ============================================================

class BlockShuffleSampler(Sampler[int]):
    """
    Cache-friendly shuffling:
    - Create contiguous index blocks of size BLOCK_SHUFFLE_SIZE
    - Shuffle block order each epoch
    - Optionally do limited in-block shuffle via a small buffer

    This avoids global shuffle that destroys field locality, while still mixing enough.
    """

    def __init__(
        self,
        n: int,
        block_size: int,
        seed: int,
        in_block_buffer: int = 0,
    ):
        self.n = int(n)
        self.block_size = int(block_size)
        self.seed = int(seed)
        self.in_block_buffer = int(in_block_buffer)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)

        # block boundaries
        blocks = []
        for start in range(0, self.n, self.block_size):
            end = min(self.n, start + self.block_size)
            blocks.append((start, end))

        # shuffle block order
        rng.shuffle(blocks)

        if self.in_block_buffer <= 0:
            # no extra in-block mixing
            for (s, e) in blocks:
                yield from range(s, e)
            return

        # in-block buffered shuffle (streaming)
        buf: List[int] = []
        buffer_size = min(self.in_block_buffer, self.block_size)

        for (s, e) in blocks:
            for idx in range(s, e):
                buf.append(idx)
                if len(buf) >= buffer_size:
                    j = int(rng.integers(0, len(buf)))
                    yield buf.pop(j)

        # flush leftovers
        while buf:
            j = int(rng.integers(0, len(buf)))
            yield buf.pop(j)


# ============================================================
# OPTIONAL: Activation checkpointing enable
# ============================================================

def enable_activation_checkpointing(model: nn.Module):
    """
    Tries to enable activation checkpointing in a generic way.
    Works if your AIONTransformer defines a method or flag.
    If not available, silently does nothing.
    """
    # Common patterns:
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            return True
        except Exception:
            pass
    if hasattr(model, "set_gradient_checkpointing"):
        try:
            model.set_gradient_checkpointing(True)
            return True
        except Exception:
            pass
    if hasattr(model, "use_activation_checkpointing"):
        try:
            model.use_activation_checkpointing = True
            return True
        except Exception:
            pass
    return False


# ============================================================
# MAIN TRAINING
# ============================================================

def train():
    set_seed(SEED)
    out_dir = Path(OUT_DIR)
    ensure_dir(out_dir)

    # TF32 (matmul acceleration for fp32 ops)
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

    # Discover fields
    fields = discover_fields_from_datacubes(DATACUBE_TMPL)
    if not fields:
        raise RuntimeError("No fields found")

    # Dataset
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
        lru_fields=3000,
    )

    dataset = CachedTokenDataset(base_dataset, verbose=True) if USE_RAM_CACHE else base_dataset

    # Steps target
    total_steps_target = SHORT_RUN_STEPS if USE_SHORT_RUN else MAX_STEPS

    # Dataloader
    collate_fn = partial(collate_sequences, pad_id=vocab.pad_id)

    sampler = None
    if USE_BLOCK_SHUFFLE:
        sampler = BlockShuffleSampler(
            n=len(dataset),
            block_size=BLOCK_SHUFFLE_SIZE,
            seed=SEED,
            in_block_buffer=IN_BLOCK_SHUFFLE_BUFFER,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=MICRO_BATCH,
        shuffle=False if sampler is not None else False,  # keep dataset order unless sampler is used
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_types = vocab.n_types

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

    if USE_ACTIVATION_CHECKPOINTING:
        ok = enable_activation_checkpointing(model)
        print(f"[perf] activation checkpointing: {'enabled' if ok else 'not available'}")

    total_params, trainable_params = count_parameters(model)

    # Compile (optional)
    if USE_TORCH_COMPILE and device.type == "cuda":
        try:
            model = torch.compile(model, mode="max-autotune")
            print("[perf] torch.compile: enabled")
        except Exception as e:
            print(f"[perf] torch.compile failed: {e}")

    # Masking
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

    # Resume
    global_step = 0
    start_epoch = 1
    if AUTO_RESUME:
        latest = _find_latest_ckpt(out_dir)
        if latest is not None:
            ckpt = _load_checkpoint(latest, model, optimizer)
            global_step = int(ckpt.get("global_step", 0))
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            print(f"[resume] Loaded {latest.name} (step={global_step})")

    # Print summary
    print("=" * 70)
    print("AION Training - H100 (improved)")
    print("=" * 70)
    print(f"[vocab] vocab_size: {vocab.vocab_size:,}   n_types: {n_types}")
    print(f"[model] params: {total_params/1e6:.1f}M")
    print(f"[train] micro_batch: {MICRO_BATCH}   grad_accum: {GRAD_ACCUM_STEPS}   effective: {EFFECTIVE_BATCH}")
    print(f"[train] bf16: {USE_BF16 and device.type=='cuda'}   tf32: {ENABLE_TF32}")
    print(f"[train] max_steps: {total_steps_target:,}   warmup: {WARMUP_STEPS}")
    if MAX_TRAIN_HOURS is not None:
        print(f"[train] time budget: {MAX_TRAIN_HOURS} hours")
    print("=" * 70)

    # Training loop
    model.train()
    running_loss = 0.0
    train_start = now_s()
    last_log_t = now_s()

    # Profiling (true wait time)
    t_data_wait = 0.0
    t_to_device = 0.0
    t_masking = 0.0
    t_forward = 0.0
    t_backward = 0.0
    last_iter_end = now_s()

    # epoch loop (we loop until steps)
    epoch = start_epoch
    while global_step < total_steps_target:
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar):
            # measure how long we waited for dataloader to yield this batch
            t0 = now_s()
            t_data_wait += max(0.0, t0 - last_iter_end)

            # time budget stop
            if MAX_TRAIN_HOURS is not None:
                if (now_s() - train_start) > MAX_TRAIN_HOURS * 3600:
                    print(f"\n✓ Reached time budget ({MAX_TRAIN_HOURS}h), stopping.")
                    global_step = total_steps_target
                    break

            if global_step >= total_steps_target:
                break

            # move to GPU
            t1 = now_s()
            tokens = batch["tokens"].to(device, non_blocking=True)
            types = batch["types"].to(device, non_blocking=True)
            attn_mask = batch["attn_mask"].to(device, non_blocking=True)
            t_to_device += now_s() - t1

            # masking
            t1 = now_s()
            masked = masking_strategy.apply(tokens, types, attn_mask)
            t_masking += now_s() - t1

            # LR schedule based on global_step
            lr_now = lr_schedule(global_step, LR, WARMUP_STEPS, total_steps_target)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            # forward + loss
            t1 = now_s()
            autocast_enabled = (device.type == "cuda")
            autocast_dtype = torch.bfloat16 if (USE_BF16 and device.type == "cuda") else None

            if autocast_dtype is None:
                # fp32 (debug) or default autocast dtype
                ctx = torch.autocast(device_type="cuda", enabled=autocast_enabled)
            else:
                ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled)

            with ctx:
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
            t_forward += now_s() - t1

            # scale loss for accumulation
            loss = loss / GRAD_ACCUM_STEPS

            # backward
            t1 = now_s()
            loss.backward()
            t_backward += now_s() - t1

            # optimizer step
            do_step = ((batch_idx + 1) % GRAD_ACCUM_STEPS == 0)
            if do_step:
                if GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running_loss += loss.item() * GRAD_ACCUM_STEPS

            # logging
            if do_step and global_step % LOG_EVERY == 0 and global_step > 0:
                avg_loss = running_loss / LOG_EVERY
                running_loss = 0.0

                elapsed = now_s() - train_start
                steps_per_sec = global_step / max(1e-6, elapsed)
                eta_s = (total_steps_target - global_step) / max(1e-6, steps_per_sec)
                eta_h = eta_s / 3600.0

                # VRAM snapshot
                vram_gb = None
                if device.type == "cuda":
                    vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

                postfix = {
                    "loss": f"{avg_loss:.6f}",
                    "lr": f"{lr_now:.2e}",
                    "step": global_step,
                    "it/s": f"{steps_per_sec:.2f}",
                    "ETA(h)": f"{eta_h:.1f}",
                }
                if vram_gb is not None:
                    postfix["vram(G)"] = f"{vram_gb:.1f}"

                pbar.set_postfix(postfix)

            # profile print
            # if do_step and global_step > 0 and global_step % 100 == 0:
            #     total_t = now_s() - train_start
            #     compute_t = t_to_device + t_masking + t_forward + t_backward
            #     wait_t = max(0.0, t_data_wait)
            #     print(f"\n[PROFILE @ step {global_step}] total={total_t:.1f}s")
            #     print(f"  DataLoader WAIT: {wait_t:.1f}s ({wait_t/total_t*100:.1f}%)")
            #     print(f"  To-device:       {t_to_device:.1f}s ({t_to_device/total_t*100:.1f}%)")
            #     print(f"  Masking:         {t_masking:.1f}s ({t_masking/total_t*100:.1f}%)")
            #     print(f"  Forward:         {t_forward:.1f}s ({t_forward/total_t*100:.1f}%)")
            #     print(f"  Backward:        {t_backward:.1f}s ({t_backward/total_t*100:.1f}%)")
            #     print()

            # checkpoint
            if do_step and global_step > 0 and global_step % SAVE_EVERY_STEPS == 0:
                ckpt_path = _save_checkpoint(
                    out_dir=out_dir,
                    step=global_step,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    config={
                        "vocab_size": vocab.vocab_size,
                        "n_types": n_types,
                        "d_model": D_MODEL,
                        "n_enc_layers": N_ENC_LAYERS,
                        "n_dec_layers": N_DEC_LAYERS,
                        "n_heads": N_HEADS,
                        "d_ff": D_FF,
                        "dropout": DROPOUT,
                        "max_len": MAX_SEQ_LEN,
                        "micro_batch": MICRO_BATCH,
                        "grad_accum_steps": GRAD_ACCUM_STEPS,
                        "effective_batch": EFFECTIVE_BATCH,
                        "input_budget": INPUT_BUDGET,
                        "output_budget": OUTPUT_BUDGET,
                        "beta_alpha": BETA_ALPHA,
                        "beta_beta": BETA_BETA,
                        "lr": LR,
                        "weight_decay": WEIGHT_DECAY,
                        "warmup_steps": WARMUP_STEPS,
                        "bf16": USE_BF16,
                        "tf32": ENABLE_TF32,
                    },
                )
                tqdm.write(f"[save] {ckpt_path}")
                _cleanup_old_checkpoints(out_dir, KEEP_LAST_N_CHECKPOINTS)

            last_iter_end = now_s()

            # stop if reached
            if global_step >= total_steps_target:
                break

        epoch += 1

    # final checkpoint
    final_path = out_dir / f"ckpt_final_{global_step:08d}.pt"
    torch.save({"model": model.state_dict(), "global_step": global_step}, final_path)
    print(f"[save] {final_path}")

    print("\n" + "=" * 70)
    print("✓ Training complete!")
    print(f"✓ Final step: {global_step:,} / {total_steps_target:,}")
    print(f"✓ Checkpoints saved to: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    train()