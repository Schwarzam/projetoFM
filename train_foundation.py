#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a simple multimodal RVQ foundation model from final_all.parquet
and save checkpoint whenever the average epoch loss improves.

Expected columns in final_all.parquet (at least some of these):
- id
- image_codes
- desi_codes
- gaiaxp_codes OR bp_codes/rp_codes
- scalar_codes (preferred) OR fallback scalar cols

This is a minimal training script (single GPU / CPU).
"""

from __future__ import annotations

import os

# use CUDA GPU 1

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from pathlib import Path
import math
import copy
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Config
# ============================================================
FINAL_ALL_PATH = "final_all.parquet"
OUT_DIR = Path("./checkpoints_astromm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sampling / loader
N_SAMPLES = 30_000_000          # set lower for debugging
BATCH_SIZE = 1024               # reduce if OOM
NUM_WORKERS = 0
PIN_MEMORY = True
SEED = 42

# Training
EPOCHS = 15
LR = 3e-4
WEIGHT_DECAY = 1e-2
USE_AMP = torch.cuda.is_available()

# Model size (12.3M-ish)
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6
FF_MULT = 4
DROPOUT = 0.1

# Vocab sizes (adjust to your true codebooks)
IMAGE_VOCAB = 2048 + 2
DESI_VOCAB = 1024 + 2
GAIAXP_VOCAB = 1024 + 2
SCALAR_VOCAB = 1024 + 2

PAD_ID = 0
MASK_ID = 1

MASK_IDS = {
    "image": MASK_ID,
    "desi": MASK_ID,
    "gaiaxp": MASK_ID,
    "scalar": MASK_ID,
}

LOSS_WEIGHTS = {"image": 1.0, "desi": 1.0, "gaia": 1.0, "scalar": 0.5}

# Sequence sizes
L_IMAGE = 576
L_DESI = 244
L_GAIAXP = 55
L_SCALAR = 1
SEQ_LEN = L_IMAGE + L_DESI + L_GAIAXP + L_SCALAR

IMG_LEN = L_IMAGE
DESI_SHAPE = (L_DESI, 3)
GAIAXP_SHAPE = (L_GAIAXP, 3)
SCALAR_SHAPE = (L_SCALAR, 3)


# ============================================================
# Utils
# ============================================================
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} ({total/1e6:.2f}M)")
    print(f"Trainable params: {trainable:,} ({trainable/1e6:.2f}M)")
    return total, trainable


def to_fixed_array(x, shape, pad_value=0):
    if x is None:
        return np.full(shape, pad_value, dtype=np.int64)

    arr = np.asarray(x)
    try:
        arr = arr.astype(np.int64, copy=False)
    except Exception:
        arr = np.array(arr.tolist() if hasattr(arr, "tolist") else arr, dtype=np.int64)

    try:
        return arr.reshape(shape)
    except Exception:
        flat = arr.reshape(-1)
        out = np.full(int(np.prod(shape)), pad_value, dtype=np.int64)
        n = min(len(flat), out.size)
        out[:n] = flat[:n]
        return out.reshape(shape)


def has_real_content(x):
    if x is None:
        return False
    arr = np.asarray(x)
    return arr.size > 0


def build_scalar_rvq_from_row_dict(row, scalar_cols=None):
    if "scalar_codes" in row and row["scalar_codes"] is not None:
        return to_fixed_array(row["scalar_codes"], SCALAR_SHAPE, PAD_ID)

    scalar_cols = scalar_cols or []
    vals = []
    for c in scalar_cols[:3]:
        v = row.get(c)
        if v is None:
            vals.append(PAD_ID)
        elif isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v).reshape(-1)
            vals.append(int(arr[0]) if arr.size > 0 else PAD_ID)
        else:
            vals.append(int(v))
    vals = (vals + [PAD_ID, PAD_ID, PAD_ID])[:3]
    return np.array(vals, dtype=np.int64).reshape(SCALAR_SHAPE)


def build_gaiaxp_from_row_dict(row):
    if "gaiaxp_codes" in row and row["gaiaxp_codes"] is not None:
        return to_fixed_array(row["gaiaxp_codes"], GAIAXP_SHAPE, PAD_ID), True

    bp = row.get("bp_codes")
    rp = row.get("rp_codes")

    if bp is not None:
        return to_fixed_array(bp, GAIAXP_SHAPE, PAD_ID), True
    if rp is not None:
        return to_fixed_array(rp, GAIAXP_SHAPE, PAD_ID), True

    return np.full(GAIAXP_SHAPE, PAD_ID, dtype=np.int64), False


# ============================================================
# Dataset
# ============================================================
class PolarsMMRvqDataset(Dataset):
    def __init__(self, df: pl.DataFrame, indices=None, scalar_cols=None):
        self.scalar_cols = scalar_cols or []

        keep_cols = ["id"]
        for c in ["image_codes", "desi_codes", "gaiaxp_codes", "bp_codes", "rp_codes", "scalar_codes"]:
            if c in df.columns:
                keep_cols.append(c)
        keep_cols += [c for c in self.scalar_cols if c in df.columns]
        self.df = df.select(list(dict.fromkeys(keep_cols)))

        if indices is None:
            self.indices = np.arange(self.df.height, dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        ridx = int(self.indices[i])
        row_tuple = self.df.row(ridx)
        row = dict(zip(self.df.columns, row_tuple))

        raw_img = row.get("image_codes")
        has_image = has_real_content(raw_img)
        image_codes = to_fixed_array(raw_img, (IMG_LEN,), PAD_ID)

        raw_desi = row.get("desi_codes")
        has_desi = has_real_content(raw_desi)
        desi_codes = to_fixed_array(raw_desi, DESI_SHAPE, PAD_ID)

        gaiaxp_codes, has_gaiaxp = build_gaiaxp_from_row_dict(row)
        scalar_codes = build_scalar_rvq_from_row_dict(row, self.scalar_cols)

        return {
            "id": str(row["id"]),
            "image_codes": torch.tensor(image_codes, dtype=torch.long),
            "desi_codes": torch.tensor(desi_codes, dtype=torch.long),
            "gaiaxp_codes": torch.tensor(gaiaxp_codes, dtype=torch.long),
            "scalar_codes": torch.tensor(scalar_codes, dtype=torch.long),
            "has_image": torch.tensor(has_image, dtype=torch.bool),
            "has_desi": torch.tensor(has_desi, dtype=torch.bool),
            "has_gaiaxp": torch.tensor(has_gaiaxp, dtype=torch.bool),
        }


def collate_mm(batch):
    return {
        "id": [b["id"] for b in batch],
        "image_codes": torch.stack([b["image_codes"] for b in batch], dim=0),
        "desi_codes": torch.stack([b["desi_codes"] for b in batch], dim=0),
        "gaiaxp_codes": torch.stack([b["gaiaxp_codes"] for b in batch], dim=0),
        "scalar_codes": torch.stack([b["scalar_codes"] for b in batch], dim=0),
        "has_image": torch.stack([b["has_image"] for b in batch], dim=0),
        "has_desi": torch.stack([b["has_desi"] for b in batch], dim=0),
        "has_gaiaxp": torch.stack([b["has_gaiaxp"] for b in batch], dim=0),
    }


# ============================================================
# Model
# ============================================================
class RVQStageEmbed(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.e0 = nn.Embedding(vocab_size, d_model)
        self.e1 = nn.Embedding(vocab_size, d_model)
        self.e2 = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.e0(x[..., 0]) + self.e1(x[..., 1]) + self.e2(x[..., 2])


class SimpleAstroFM(nn.Module):
    def __init__(
        self,
        image_vocab_size: int,
        desi_vocab_size: int,
        gaiaxp_vocab_size: int,
        scalar_vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.image_emb = nn.Embedding(image_vocab_size, d_model)
        self.desi_emb = RVQStageEmbed(desi_vocab_size, d_model)
        self.gaiaxp_emb = RVQStageEmbed(gaiaxp_vocab_size, d_model)
        self.scalar_emb = RVQStageEmbed(scalar_vocab_size, d_model)

        self.missing_image = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.missing_desi = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.missing_gaiaxp = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.pos_emb = nn.Embedding(SEQ_LEN, d_model)
        self.mod_emb = nn.Embedding(4, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.image_head = nn.Linear(d_model, image_vocab_size)

        self.desi_head0 = nn.Linear(d_model, desi_vocab_size)
        self.desi_head1 = nn.Linear(d_model, desi_vocab_size)
        self.desi_head2 = nn.Linear(d_model, desi_vocab_size)

        self.gaia_head0 = nn.Linear(d_model, gaiaxp_vocab_size)
        self.gaia_head1 = nn.Linear(d_model, gaiaxp_vocab_size)
        self.gaia_head2 = nn.Linear(d_model, gaiaxp_vocab_size)

        self.scalar_head0 = nn.Linear(d_model, scalar_vocab_size)
        self.scalar_head1 = nn.Linear(d_model, scalar_vocab_size)
        self.scalar_head2 = nn.Linear(d_model, scalar_vocab_size)

    def forward(self, batch):
        device = batch["image_codes"].device
        B = batch["image_codes"].shape[0]

        x_img = self.image_emb(batch["image_codes"])
        x_desi = self.desi_emb(batch["desi_codes"])
        x_gaia = self.gaiaxp_emb(batch["gaiaxp_codes"])
        x_scalar = self.scalar_emb(batch["scalar_codes"])

        if "has_image" in batch:
            has_image = batch["has_image"].view(B, 1, 1)
            x_img = torch.where(has_image, x_img, self.missing_image.expand(B, L_IMAGE, self.d_model))
        if "has_desi" in batch:
            has_desi = batch["has_desi"].view(B, 1, 1)
            x_desi = torch.where(has_desi, x_desi, self.missing_desi.expand(B, L_DESI, self.d_model))
        if "has_gaiaxp" in batch:
            has_gaiaxp = batch["has_gaiaxp"].view(B, 1, 1)
            x_gaia = torch.where(has_gaiaxp, x_gaia, self.missing_gaiaxp.expand(B, L_GAIAXP, self.d_model))

        x = torch.cat([x_img, x_desi, x_gaia, x_scalar], dim=1)

        pos = torch.arange(SEQ_LEN, device=device).unsqueeze(0).expand(B, SEQ_LEN)
        mod_ids = torch.cat(
            [
                torch.zeros((B, L_IMAGE), dtype=torch.long, device=device),
                torch.ones((B, L_DESI), dtype=torch.long, device=device),
                torch.full((B, L_GAIAXP), 2, dtype=torch.long, device=device),
                torch.full((B, L_SCALAR), 3, dtype=torch.long, device=device),
            ],
            dim=1,
        )

        x = x + self.pos_emb(pos) + self.mod_emb(mod_ids)
        x = self.encoder(x)
        x = self.norm(x)

        s0 = 0
        s1 = L_IMAGE
        s2 = s1 + L_DESI
        s3 = s2 + L_GAIAXP
        s4 = s3 + L_SCALAR

        z_img = x[:, s0:s1]
        z_desi = x[:, s1:s2]
        z_gaia = x[:, s2:s3]
        z_scalar = x[:, s3:s4]

        return {
            "image_logits": self.image_head(z_img),
            "desi_logits": (self.desi_head0(z_desi), self.desi_head1(z_desi), self.desi_head2(z_desi)),
            "gaia_logits": (self.gaia_head0(z_gaia), self.gaia_head1(z_gaia), self.gaia_head2(z_gaia)),
            "scalar_logits": (
                self.scalar_head0(z_scalar),
                self.scalar_head1(z_scalar),
                self.scalar_head2(z_scalar),
            ),
        }


# ============================================================
# Loss / masking
# ============================================================
def make_masks(batch, p_image=0.15, p_desi=0.15, p_gaia=0.15, p_scalar=0.15):
    device = batch["image_codes"].device
    B = batch["image_codes"].shape[0]

    image_mask = (torch.rand(B, L_IMAGE, device=device) < p_image)
    desi_mask = (torch.rand(B, L_DESI, device=device) < p_desi)
    gaia_mask = (torch.rand(B, L_GAIAXP, device=device) < p_gaia)
    scalar_mask = (torch.rand(B, L_SCALAR, device=device) < p_scalar)

    image_mask &= batch["has_image"].view(B, 1)
    desi_mask &= batch["has_desi"].view(B, 1)
    gaia_mask &= batch["has_gaiaxp"].view(B, 1)

    return image_mask, desi_mask, gaia_mask, scalar_mask


def apply_input_masking(batch, image_mask, desi_mask, gaia_mask, scalar_mask, mask_ids):
    out = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}

    out["image_codes"][image_mask] = mask_ids["image"]
    out["desi_codes"][desi_mask] = mask_ids["desi"]
    out["gaiaxp_codes"][gaia_mask] = mask_ids["gaiaxp"]
    out["scalar_codes"][scalar_mask] = mask_ids["scalar"]

    return out


def masked_ce_loss(logits, targets, pos_mask):
    if pos_mask.sum() == 0:
        return logits.new_tensor(0.0)
    return F.cross_entropy(logits[pos_mask], targets[pos_mask])


def training_loss(model, batch, mask_ids, weights=None):
    if weights is None:
        weights = LOSS_WEIGHTS

    image_mask, desi_mask, gaia_mask, scalar_mask = make_masks(batch)
    masked_batch = apply_input_masking(batch, image_mask, desi_mask, gaia_mask, scalar_mask, mask_ids)
    out = model(masked_batch)

    tgt_img = batch["image_codes"]
    tgt_desi = batch["desi_codes"]
    tgt_gaia = batch["gaiaxp_codes"]
    tgt_scalar = batch["scalar_codes"]

    loss_img = masked_ce_loss(out["image_logits"], tgt_img, image_mask)

    d0, d1, d2 = out["desi_logits"]
    loss_desi = (
        masked_ce_loss(d0, tgt_desi[..., 0], desi_mask)
        + masked_ce_loss(d1, tgt_desi[..., 1], desi_mask)
        + masked_ce_loss(d2, tgt_desi[..., 2], desi_mask)
    ) / 3.0

    g0, g1, g2 = out["gaia_logits"]
    loss_gaia = (
        masked_ce_loss(g0, tgt_gaia[..., 0], gaia_mask)
        + masked_ce_loss(g1, tgt_gaia[..., 1], gaia_mask)
        + masked_ce_loss(g2, tgt_gaia[..., 2], gaia_mask)
    ) / 3.0

    s0, s1, s2 = out["scalar_logits"]
    loss_scalar = (
        masked_ce_loss(s0, tgt_scalar[..., 0], scalar_mask)
        + masked_ce_loss(s1, tgt_scalar[..., 1], scalar_mask)
        + masked_ce_loss(s2, tgt_scalar[..., 2], scalar_mask)
    ) / 3.0

    loss = (
        weights["image"] * loss_img
        + weights["desi"] * loss_desi
        + weights["gaia"] * loss_gaia
        + weights["scalar"] * loss_scalar
    )

    metrics = {
        "loss": float(loss.detach().cpu()),
        "loss_image": float(loss_img.detach().cpu()),
        "loss_desi": float(loss_desi.detach().cpu()),
        "loss_gaia": float(loss_gaia.detach().cpu()),
        "loss_scalar": float(loss_scalar.detach().cpu()),
        "n_mask_img": int(image_mask.sum().item()),
        "n_mask_desi": int(desi_mask.sum().item()),
        "n_mask_gaia": int(gaia_mask.sum().item()),
        "n_mask_scalar": int(scalar_mask.sum().item()),
    }
    return loss, metrics


# ============================================================
# Checkpointing
# ============================================================
def save_checkpoint(path: Path, model, optimizer, scaler, epoch, step, best_loss, extra=None):
    ckpt = {
        "epoch": epoch,
        "step": step,
        "best_loss": best_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": {
            "D_MODEL": D_MODEL,
            "NHEAD": NHEAD,
            "NUM_LAYERS": NUM_LAYERS,
            "FF_MULT": FF_MULT,
            "BATCH_SIZE": BATCH_SIZE,
            "LR": LR,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "IMAGE_VOCAB": IMAGE_VOCAB,
            "DESI_VOCAB": DESI_VOCAB,
            "GAIAXP_VOCAB": GAIAXP_VOCAB,
            "SCALAR_VOCAB": SCALAR_VOCAB,
            "MASK_IDS": MASK_IDS,
            "LOSS_WEIGHTS": LOSS_WEIGHTS,
        },
    }
    if extra is not None:
        ckpt["extra"] = extra
    torch.save(ckpt, path)


# ============================================================
# Main
# ============================================================
def main():
    set_seed(SEED)

    print(f"Loading: {FINAL_ALL_PATH}")
    final_all = pl.read_parquet(FINAL_ALL_PATH)
    print(f"final_all shape: {final_all.shape}")

    # Optional fallback scalar cols if scalar_codes doesn't exist
    scalar_cols = []
    if "scalar_codes" not in final_all.columns:
        scalar_cols = ["scalar_a_pixel_det", "scalar_b_pixel_det", "scalar_desi_flux_standard_scaler"]
        print("[WARN] scalar_codes column not found. Using fallback scalar cols:", scalar_cols)

    n = min(N_SAMPLES, final_all.height)
    rng = np.random.default_rng(SEED)
    sample_indices = rng.choice(final_all.height, size=n, replace=False)

    ds = PolarsMMRvqDataset(final_all, indices=sample_indices, scalar_cols=scalar_cols)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_mm,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = SimpleAstroFM(
        image_vocab_size=IMAGE_VOCAB,
        desi_vocab_size=DESI_VOCAB,
        gaiaxp_vocab_size=GAIAXP_VOCAB,
        scalar_vocab_size=SCALAR_VOCAB,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        ff_mult=FF_MULT,
        dropout=DROPOUT,
    ).to(device)

    count_parameters(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_epoch_loss = math.inf
    best_ckpt_path = OUT_DIR / "best_model.pt"
    last_ckpt_path = OUT_DIR / "last_model.pt"

    print("Starting training...")
    model.train()

    for epoch in range(EPOCHS):
        running = 0.0

        for step, batch in enumerate(dl):
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                loss, metrics = training_loss(model, batch, MASK_IDS, weights=LOSS_WEIGHTS)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += metrics["loss"]

            if step % 20 == 0:
                print(
                    f"epoch={epoch} step={step} "
                    f"loss={metrics['loss']:.4f} "
                    f"img={metrics['loss_image']:.4f} "
                    f"desi={metrics['loss_desi']:.4f} "
                    f"gaia={metrics['loss_gaia']:.4f} "
                    f"scalar={metrics['loss_scalar']:.4f} "
                    f"| masks img/desi/gaia/sc="
                    f"{metrics['n_mask_img']}/{metrics['n_mask_desi']}/"
                    f"{metrics['n_mask_gaia']}/{metrics['n_mask_scalar']}"
                )

        epoch_avg_loss = running / (step + 1)
        print(f"Epoch {epoch} avg loss: {epoch_avg_loss:.6f}")

        # Save last every epoch
        save_checkpoint(
            last_ckpt_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            step=step,
            best_loss=best_epoch_loss,
            extra={"epoch_avg_loss": epoch_avg_loss},
        )

        # Save best on improvement
        if epoch_avg_loss < best_epoch_loss:
            best_epoch_loss = epoch_avg_loss
            print(f"[BEST] New best loss: {best_epoch_loss:.6f} -> saving {best_ckpt_path}")
            save_checkpoint(
                best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                best_loss=best_epoch_loss,
                extra={"epoch_avg_loss": epoch_avg_loss},
            )

    print("Training finished.")
    print("Best checkpoint:", best_ckpt_path)
    print("Last checkpoint:", last_ckpt_path)


if __name__ == "__main__":
    main()