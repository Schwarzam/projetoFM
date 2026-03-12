#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a simple multimodal RVQ foundation model from final_all.parquet
and save checkpoint whenever the average epoch loss improves.

Adds "modality dropout" (full-modality masking) so the model learns tasks like:
  DESI <- (GaiaXP + scalar) when DESI is fully masked.

Also saves a PNG learning curve at the end.

Expected columns in final_all.parquet (at least some of these):
- id
- image_codes
- desi_codes
- gaiaxp_codes OR bp_codes/rp_codes
- scalar_codes (optional). If missing, we autodetect scalar_* columns and pack them.

Assumptions for autodetected scalar_* columns:
- All scalar_* columns have the SAME SHAPE (either scalar int per row, or list/array of 3 ints per row).
- Missing values are null or empty list; these become PAD.
"""

from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from pathlib import Path
import math
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================
# Config
# ============================================================
FINAL_ALL_PATH = "final_all.parquet"
OUT_DIR = Path("./checkpoints_astromm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sampling / loader
N_SAMPLES = 5_000_000
BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 8   # e.g. 8
NUM_WORKERS = 0
PIN_MEMORY = True
SEED = 42

# Training
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-2
USE_AMP = torch.cuda.is_available()

# Token-level MLM masking probs
P_IMAGE = 0.15
P_DESI = 0.15
P_GAIA = 0.15
P_SCALAR = 0.15

# Modality dropout (cross-modal objectives)
P_FULL_DESI = 0.30      # mask ALL DESI tokens for this fraction of rows (where DESI exists)
P_FULL_IMAGE = 0.10     # optionally mask ALL image tokens for some rows (where image exists)
PAIR_FULL_IMAGE_WITH_FULL_DESI = True  # if True, whenever DESI is fully masked and image exists, also mask all image

D_MODEL = 768
NHEAD = 12          # 768/12 = 64 per head
NUM_LAYERS = 10
FF_MULT = 4
DROPOUT = 0.1

# Vocab sizes (adjust to your true codebooks)
IMAGE_VOCAB = 2048 + 2
DESI_VOCAB = 1024 + 2
GAIAXP_VOCAB = 1024 + 2
SCALAR_VOCAB = 1024 + 2

PAD_ID = 0
MASK_ID = 1

MASK_IDS = {"image": MASK_ID, "desi": MASK_ID, "gaiaxp": MASK_ID, "scalar": MASK_ID}
LOSS_WEIGHTS = {"image": 1.0, "desi": 1.0, "gaia": 1.0, "scalar": 0.5}

# Sequence sizes
L_IMAGE = 576
L_DESI = 244
L_GAIAXP = 55

# Scalar sequence length is determined at runtime from scalar_* columns (if scalar_codes missing)
L_SCALAR = None  # runtime
SEQ_LEN = None   # runtime

IMG_LEN = L_IMAGE
DESI_SHAPE = (L_DESI, 3)
GAIAXP_SHAPE = (L_GAIAXP, 3)
SCALAR_SHAPE = None  # runtime


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


def build_scalar_from_row_dict(row, scalar_cols, scalar_shape):
    """
    Returns scalar_codes with shape (L_SCALAR, 3) where L_SCALAR=len(scalar_cols).
    Assumes each scalar_* column is either:
      - int code (single-stage): stored into stage0, stage1/2 PAD
      - list/array of 3 ints (3-stage RVQ): used directly
    """
    if "scalar_codes" in row and row["scalar_codes"] is not None:
        return to_fixed_array(row["scalar_codes"], scalar_shape, PAD_ID)

    Ls = scalar_shape[0]
    out = np.full(scalar_shape, PAD_ID, dtype=np.int64)

    for i, c in enumerate(scalar_cols[:Ls]):
        v = row.get(c)
        if v is None:
            continue

        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v).reshape(-1)
            if arr.size >= 3:
                out[i, 0] = int(arr[0])
                out[i, 1] = int(arr[1])
                out[i, 2] = int(arr[2])
            elif arr.size == 1:
                out[i, 0] = int(arr[0])
        else:
            out[i, 0] = int(v)

    return out


# ============================================================
# Dataset
# ============================================================
class PolarsMMRvqDataset(Dataset):
    def __init__(self, df: pl.DataFrame, indices=None, scalar_cols=None, scalar_shape=None):
        self.scalar_cols = scalar_cols or []
        self.scalar_shape = scalar_shape

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

        scalar_codes = build_scalar_from_row_dict(
            row=row,
            scalar_cols=self.scalar_cols,
            scalar_shape=self.scalar_shape,
        )

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
        seq_len: int,
        l_image: int,
        l_desi: int,
        l_gaia: int,
        l_scalar: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.l_image = l_image
        self.l_desi = l_desi
        self.l_gaia = l_gaia
        self.l_scalar = l_scalar
        self.seq_len = seq_len

        self.image_emb = nn.Embedding(image_vocab_size, d_model)
        self.desi_emb = RVQStageEmbed(desi_vocab_size, d_model)
        self.gaiaxp_emb = RVQStageEmbed(gaiaxp_vocab_size, d_model)
        self.scalar_emb = RVQStageEmbed(scalar_vocab_size, d_model)

        self.missing_image = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.missing_desi = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.missing_gaiaxp = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.pos_emb = nn.Embedding(seq_len, d_model)
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
            x_img = torch.where(has_image, x_img, self.missing_image.expand(B, self.l_image, self.d_model))
        if "has_desi" in batch:
            has_desi = batch["has_desi"].view(B, 1, 1)
            x_desi = torch.where(has_desi, x_desi, self.missing_desi.expand(B, self.l_desi, self.d_model))
        if "has_gaiaxp" in batch:
            has_gaiaxp = batch["has_gaiaxp"].view(B, 1, 1)
            x_gaia = torch.where(has_gaiaxp, x_gaia, self.missing_gaiaxp.expand(B, self.l_gaia, self.d_model))

        x = torch.cat([x_img, x_desi, x_gaia, x_scalar], dim=1)

        pos = torch.arange(self.seq_len, device=device).unsqueeze(0).expand(B, self.seq_len)
        mod_ids = torch.cat(
            [
                torch.zeros((B, self.l_image), dtype=torch.long, device=device),
                torch.ones((B, self.l_desi), dtype=torch.long, device=device),
                torch.full((B, self.l_gaia), 2, dtype=torch.long, device=device),
                torch.full((B, self.l_scalar), 3, dtype=torch.long, device=device),
            ],
            dim=1,
        )

        x = x + self.pos_emb(pos) + self.mod_emb(mod_ids)
        x = self.encoder(x)
        x = self.norm(x)

        s0 = 0
        s1 = self.l_image
        s2 = s1 + self.l_desi
        s3 = s2 + self.l_gaia
        s4 = s3 + self.l_scalar

        z_img = x[:, s0:s1]
        z_desi = x[:, s1:s2]
        z_gaia = x[:, s2:s3]
        z_scalar = x[:, s3:s4]

        return {
            "image_logits": self.image_head(z_img),
            "desi_logits": (self.desi_head0(z_desi), self.desi_head1(z_desi), self.desi_head2(z_desi)),
            "gaia_logits": (self.gaia_head0(z_gaia), self.gaia_head1(z_gaia), self.gaia_head2(z_gaia)),
            "scalar_logits": (self.scalar_head0(z_scalar), self.scalar_head1(z_scalar), self.scalar_head2(z_scalar)),
        }


# ============================================================
# Loss / masking
# ============================================================
def make_masks(
    batch,
    l_image,
    l_desi,
    l_gaia,
    l_scalar,
    p_image=0.15,
    p_desi=0.15,
    p_gaia=0.15,
    p_scalar=0.15,
    p_full_desi=0.30,
    p_full_image=0.10,
    pair_full_image_with_full_desi=True,
):
    device = batch["image_codes"].device
    B = batch["image_codes"].shape[0]

    # token-level masks
    image_mask = (torch.rand(B, l_image, device=device) < p_image)
    desi_mask = (torch.rand(B, l_desi, device=device) < p_desi)
    gaia_mask = (torch.rand(B, l_gaia, device=device) < p_gaia)
    scalar_mask = (torch.rand(B, l_scalar, device=device) < p_scalar)

    # only mask where modality exists
    image_mask &= batch["has_image"].view(B, 1)
    desi_mask &= batch["has_desi"].view(B, 1)
    gaia_mask &= batch["has_gaiaxp"].view(B, 1)

    # full DESI masking for some rows
    full_desi = (torch.rand(B, device=device) < p_full_desi) & batch["has_desi"]
    if full_desi.any():
        desi_mask[full_desi] = True

    # full image masking for some rows
    full_img = (torch.rand(B, device=device) < p_full_image) & batch["has_image"]
    if full_img.any():
        image_mask[full_img] = True

    # optionally: whenever we full-mask DESI, also full-mask image (forces Gaia+scalar -> DESI)
    if pair_full_image_with_full_desi:
        paired = full_desi & batch["has_image"]
        if paired.any():
            image_mask[paired] = True

    return image_mask, desi_mask, gaia_mask, scalar_mask


def apply_input_masking(batch, image_mask, desi_mask, gaia_mask, scalar_mask, mask_ids):
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    out["image_codes"][image_mask] = mask_ids["image"]
    out["desi_codes"][desi_mask] = mask_ids["desi"]
    out["gaiaxp_codes"][gaia_mask] = mask_ids["gaiaxp"]
    out["scalar_codes"][scalar_mask] = mask_ids["scalar"]
    return out


def masked_ce_loss(logits, targets, pos_mask):
    if pos_mask.sum() == 0:
        return logits.new_tensor(0.0)
    return F.cross_entropy(logits[pos_mask], targets[pos_mask])


def training_loss(model, batch, mask_ids, l_image, l_desi, l_gaia, l_scalar, weights=None):
    if weights is None:
        weights = LOSS_WEIGHTS

    image_mask, desi_mask, gaia_mask, scalar_mask = make_masks(
        batch,
        l_image=l_image,
        l_desi=l_desi,
        l_gaia=l_gaia,
        l_scalar=l_scalar,
        p_image=P_IMAGE,
        p_desi=P_DESI,
        p_gaia=P_GAIA,
        p_scalar=P_SCALAR,
        p_full_desi=P_FULL_DESI,
        p_full_image=P_FULL_IMAGE,
        pair_full_image_with_full_desi=PAIR_FULL_IMAGE_WITH_FULL_DESI,
    )

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
        "n_full_desi_rows": int(((desi_mask.sum(dim=1) == l_desi) & batch["has_desi"]).sum().item()),
        "n_full_img_rows": int(((image_mask.sum(dim=1) == l_image) & batch["has_image"]).sum().item()),
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
            "DROPOUT": DROPOUT,
            "BATCH_SIZE": BATCH_SIZE,
            "LR": LR,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "IMAGE_VOCAB": IMAGE_VOCAB,
            "DESI_VOCAB": DESI_VOCAB,
            "GAIAXP_VOCAB": GAIAXP_VOCAB,
            "SCALAR_VOCAB": SCALAR_VOCAB,
            "MASK_IDS": MASK_IDS,
            "LOSS_WEIGHTS": LOSS_WEIGHTS,
            "L_IMAGE": L_IMAGE,
            "L_DESI": L_DESI,
            "L_GAIAXP": L_GAIAXP,
            "L_SCALAR": L_SCALAR,
            "SEQ_LEN": SEQ_LEN,
            "P_IMAGE": P_IMAGE,
            "P_DESI": P_DESI,
            "P_GAIA": P_GAIA,
            "P_SCALAR": P_SCALAR,
            "P_FULL_DESI": P_FULL_DESI,
            "P_FULL_IMAGE": P_FULL_IMAGE,
            "PAIR_FULL_IMAGE_WITH_FULL_DESI": PAIR_FULL_IMAGE_WITH_FULL_DESI,
        },
    }
    if extra is not None:
        ckpt["extra"] = extra
    torch.save(ckpt, path)


# ============================================================
# Plot learning curves
# ============================================================
def plot_learning_curves(history, out_path: Path):
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    avg_loss = [h["epoch_avg_loss"] for h in history]
    img = [h["epoch_avg_loss_image"] for h in history]
    desi = [h["epoch_avg_loss_desi"] for h in history]
    gaia = [h["epoch_avg_loss_gaia"] for h in history]
    scalar = [h["epoch_avg_loss_scalar"] for h in history]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, avg_loss, label="total")
    plt.plot(epochs, img, label="image")
    plt.plot(epochs, desi, label="desi")
    plt.plot(epochs, gaia, label="gaia")
    plt.plot(epochs, scalar, label="scalar")
    plt.xlabel("epoch")
    plt.ylabel("avg loss")
    plt.title("Training learning curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    set_seed(SEED)

    print(f"Loading: {FINAL_ALL_PATH}")
    final_all = pl.read_parquet(FINAL_ALL_PATH)
    print(f"final_all shape: {final_all.shape}")

    # Detect scalar columns / L_SCALAR
    if "scalar_codes" in final_all.columns:
        first = final_all.select("scalar_codes").drop_nulls().head(1)
        if first.height == 0:
            print("[WARN] scalar_codes exists but all null; falling back to scalar_* autodetect.")
            scalar_cols = sorted([c for c in final_all.columns if c.startswith("scalar_") and c != "scalar_codes"])
            Ls = len(scalar_cols)
        else:
            ex = np.asarray(first["scalar_codes"][0])
            if ex.ndim == 2 and ex.shape[1] == 3:
                Ls = int(ex.shape[0])
            else:
                flat = ex.reshape(-1)
                Ls = int(max(1, flat.size // 3))
            scalar_cols = []
    else:
        scalar_cols = sorted([c for c in final_all.columns if c.startswith("scalar_") and c != "scalar_codes"])
        Ls = len(scalar_cols)

    if Ls <= 0:
        print("[WARN] No scalar codes found; setting L_SCALAR=1 with PAD.")
        Ls = 1

    global L_SCALAR, SEQ_LEN, SCALAR_SHAPE
    L_SCALAR = Ls
    SCALAR_SHAPE = (L_SCALAR, 3)
    SEQ_LEN = L_IMAGE + L_DESI + L_GAIAXP + L_SCALAR

    print(f"[INFO] L_SCALAR={L_SCALAR}  => SEQ_LEN={SEQ_LEN}")
    if scalar_cols:
        print(f"[INFO] Using {len(scalar_cols)} scalar_* columns (autodetected). Example:", scalar_cols[:10])

    # ============================================================
    # Tiered sampling: prefer (desi+gaia) then (gaia) then (desi) then (none)
    # ============================================================
    def _nonempty_list_expr(colname: str):
        c = pl.col(colname)
        expr = c.is_not_null()
        try:
            expr = expr & (c.list.len() > 0)
        except Exception:
            pass
        return expr

    has_desi_df = _nonempty_list_expr("desi_codes") if "desi_codes" in final_all.columns else pl.lit(False)

    if "gaiaxp_codes" in final_all.columns:
        has_gaia_df = _nonempty_list_expr("gaiaxp_codes")
    else:
        has_bp = _nonempty_list_expr("bp_codes") if "bp_codes" in final_all.columns else pl.lit(False)
        has_rp = _nonempty_list_expr("rp_codes") if "rp_codes" in final_all.columns else pl.lit(False)
        has_gaia_df = has_bp | has_rp

    m = final_all.select(
        has_desi_df.alias("has_desi"),
        has_gaia_df.alias("has_gaia"),
    ).to_dict(as_series=False)

    has_desi = np.asarray(m["has_desi"], dtype=bool)
    has_gaia = np.asarray(m["has_gaia"], dtype=bool)

    idx_both = np.where(has_desi & has_gaia)[0]
    idx_gaia = np.where(~has_desi & has_gaia)[0]
    idx_desi = np.where(has_desi & ~has_gaia)[0]
    idx_neither = np.where(~has_desi & ~has_gaia)[0]

    print(
        f"Counts: both={idx_both.size}, gaia_only={idx_gaia.size}, "
        f"desi_only={idx_desi.size}, neither={idx_neither.size}"
    )

    P_BOTH, P_GAIA_S, P_DESI_S, P_NONE = 0.60, 0.25, 0.10, 0.05
    ps = np.array([P_BOTH, P_GAIA_S, P_DESI_S, P_NONE], dtype=float)
    ps = ps / ps.sum()

    n_total = min(N_SAMPLES, final_all.height)
    n_targets = (ps * n_total).astype(int)
    n_targets[0] += (n_total - n_targets.sum())

    rng = np.random.default_rng(SEED)

    def pick_from(pool, k):
        k = int(min(k, pool.size))
        if k <= 0:
            return np.empty((0,), dtype=np.int64)
        return rng.choice(pool, size=k, replace=False)

    pick_both = pick_from(idx_both, n_targets[0])
    pick_gaia = pick_from(idx_gaia, n_targets[1])
    pick_desi = pick_from(idx_desi, n_targets[2])
    pick_none = pick_from(idx_neither, n_targets[3])

    sample_indices = np.concatenate([pick_both, pick_gaia, pick_desi, pick_none])

    remaining = n_total - sample_indices.size
    if remaining > 0:
        leftovers = []
        if idx_both.size > pick_both.size:
            leftovers.append(np.setdiff1d(idx_both, pick_both, assume_unique=False))
        if idx_gaia.size > pick_gaia.size:
            leftovers.append(np.setdiff1d(idx_gaia, pick_gaia, assume_unique=False))
        if idx_desi.size > pick_desi.size:
            leftovers.append(np.setdiff1d(idx_desi, pick_desi, assume_unique=False))
        if idx_neither.size > pick_none.size:
            leftovers.append(np.setdiff1d(idx_neither, pick_none, assume_unique=False))

        if leftovers:
            pool = np.concatenate(leftovers)
            k = min(remaining, pool.size)
            extra = rng.choice(pool, size=k, replace=False)
            sample_indices = np.concatenate([sample_indices, extra])

    rng.shuffle(sample_indices)

    print(f"Using {sample_indices.size} samples for training (target={n_total}).")
    print(f"Both:    {np.isin(sample_indices, idx_both).sum()} (target={n_targets[0]})")
    print(f"Gaia:    {np.isin(sample_indices, idx_gaia).sum()} (target={n_targets[1]})")
    print(f"DESI:    {np.isin(sample_indices, idx_desi).sum()} (target={n_targets[2]})")
    print(f"Neither: {np.isin(sample_indices, idx_neither).sum()} (target={n_targets[3]})")

    ds = PolarsMMRvqDataset(
        final_all,
        indices=sample_indices,
        scalar_cols=scalar_cols,
        scalar_shape=SCALAR_SHAPE,
    )

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
        seq_len=SEQ_LEN,
        l_image=L_IMAGE,
        l_desi=L_DESI,
        l_gaia=L_GAIAXP,
        l_scalar=L_SCALAR,
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
    best_ckpt_path = OUT_DIR / "best_model_100M.pt"
    last_ckpt_path = OUT_DIR / "last_model_100M.pt"

    history = []

    print("Starting training...")
    model.train()

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(EPOCHS):
        running = run_img = run_desi = run_gaia = run_scalar = 0.0

        for step, batch in enumerate(dl):
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                loss, metrics = training_loss(
                    model,
                    batch,
                    MASK_IDS,
                    l_image=L_IMAGE,
                    l_desi=L_DESI,
                    l_gaia=L_GAIAXP,
                    l_scalar=L_SCALAR,
                    weights=LOSS_WEIGHTS,
                )
                loss_scaled = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss_scaled).backward()

            # logging should use unscaled loss
            running += float(metrics["loss"])
            run_img += float(metrics["loss_image"])
            run_desi += float(metrics["loss_desi"])
            run_gaia += float(metrics["loss_gaia"])
            run_scalar += float(metrics["loss_scalar"])

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

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
                    f"{metrics['n_mask_gaia']}/{metrics['n_mask_scalar']} "
                    f"| full_rows desi/img={metrics['n_full_desi_rows']}/{metrics['n_full_img_rows']}"
                )

        # IMPORTANT: flush leftover grads if steps % accum != 0
        if (step + 1) % GRAD_ACCUM_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        denom = (step + 1)
        epoch_avg_loss = running / denom
        epoch_avg_img = run_img / denom
        epoch_avg_desi = run_desi / denom
        epoch_avg_gaia = run_gaia / denom
        epoch_avg_scalar = run_scalar / denom

        history.append({
            "epoch": epoch,
            "epoch_avg_loss": epoch_avg_loss,
            "epoch_avg_loss_image": epoch_avg_img,
            "epoch_avg_loss_desi": epoch_avg_desi,
            "epoch_avg_loss_gaia": epoch_avg_gaia,
            "epoch_avg_loss_scalar": epoch_avg_scalar,
        })

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
            extra={"epoch_avg_loss": epoch_avg_loss, "history": history},
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
                extra={"epoch_avg_loss": epoch_avg_loss, "history": history},
            )

    # Plot learning curve
    lc_path = OUT_DIR / "learning_curve.png"
    plot_learning_curves(history, lc_path)
    print("Saved learning curve:", lc_path)

    print("Training finished.")
    print("Best checkpoint:", best_ckpt_path)
    print("Last checkpoint:", last_ckpt_path)


if __name__ == "__main__":
    main()