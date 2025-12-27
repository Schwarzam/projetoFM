#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train an AION-like *token* Transformer over:
  - image tokens (KMeans codebook) keyed by Gaia source_id
  - scalar tokens (quantile bins) keyed by S-PLUS id
  - spectrum tokens (ragged CSR flat/indptr) keyed by S-PLUS id
    * supports GaiaXP BP/RP and DESI b/r/z if present in the NPZ

Model:
  - token embeddings (single global vocab via offsets)
  - token-type embeddings (modality + group)
  - positional embeddings
  - TransformerEncoder with causal mask (autoregressive LM)
  - loss: next-token prediction with PAD ignored

✅ Fixes your warnings:
  - uses torch.amp.autocast / torch.amp.GradScaler (no deprecation)
  - sets norm_first=False to avoid nested tensor warning

✅ Forces GPU #1:
  - sets CUDA_VISIBLE_DEVICES=1 before importing torch

Run:
  python3 train_transformer_tokens.py
"""

from __future__ import annotations

# ---------------------------
# FORCE GPU #1 (must be before torch import)
# ---------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Optional: avoid BLAS thread explosion
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import math
import time
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

# ---- input templates ----
FIELDS: List[str] = []  # if empty -> auto-discover from datacube files

DATACUBE_TMPL = "/home/astrodados4/downloads/hypercube/datacube_{field}.parquet"

# check if all these paths exist!
for field in FIELDS:
    dp = DATACUBE_TMPL.format(field=field)
    if not Path(dp).exists():
        FIELDS.remove(field)

IMAGE_TOKENS_TMPL   = "/home/schwarz/projetoFM/codecs/image_tokens/datacube_{field}_tokens.npz"
SCALAR_TOKENS_TMPL  = "/home/schwarz/projetoFM/scalar_tokenizers/datacube_{field}_scalar_tokens.npz"
SPECTRUM_TOKENS_TMPL= "/home/schwarz/projetoFM/spectrum_tokenizers/datacube_{field}_spectrum_tokens.npz"

# scalar tokenizer config only needed for label decoding (not for training)
SCALAR_TOKENIZER_CONFIG = "/home/schwarz/projetoFM/scalar_tokenizers/scalar_tokenizer_config.npz"

# ---- row filter (same as you used) ----
MAG_MIN = 14.0
MAG_MAX = 22.0
MAGERR_MAX = 2.0
MAG_COL = "mag_pstotal_r"
MAGERR_COL = "err_mag_pstotal_r"

# ---- token vocab sizes (must match how you generated them) ----
V_IMAGE   = 2048
V_SCALAR  = 1024  # N_BINS from scalar config; if different, update here
V_SPECTRA = 2048  # spectrum groups use KMeans 2048 in your generator

# ---- spectrum groups we may find in the NPZ (ragged CSR) ----
# keys are (flat_key, indptr_key) inside the NPZ
SPECTRUM_GROUPS = {
    "gaiaxp_bp": ("tokens_gaiaxp_bp_flat", "tokens_gaiaxp_bp_indptr"),
    "gaiaxp_rp": ("tokens_gaiaxp_rp_flat", "tokens_gaiaxp_rp_indptr"),
    "desi_b":    ("tokens_desi_b_flat",    "tokens_desi_b_indptr"),
    "desi_r":    ("tokens_desi_r_flat",    "tokens_desi_r_indptr"),
    "desi_z":    ("tokens_desi_z_flat",    "tokens_desi_z_indptr"),
}

# ---- sequence construction ----
# cap lengths so batches are stable
MAX_IMAGE_TOKENS  = 24 * 24         # typical
MAX_SCALAR_TOKENS = 512             # if your scalar_cols > 512, we’ll truncate
MAX_SPEC_TOKENS_PER_GROUP = 512     # cap ragged per group
MAX_SEQ_LEN = 2048                  # total cap after concatenation

# ---- training ----
OUT_DIR = "runs_tokens_transformer"
SEED = 0

BATCH_SIZE = 16
NUM_WORKERS = 4
PIN_MEMORY = True

EPOCHS = 5
LR = 2e-4
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 2000
GRAD_CLIP = 1.0
GRAD_ACCUM_STEPS = 1
LOG_EVERY = 50
SAVE_EVERY_STEPS = 5000

D_MODEL  = 768
N_LAYERS = 12
N_HEADS  = 12     # 768/12 = 64 per head
D_FF     = 3072   # 4x
DROPOUT  = 0.1


# ============================================================
# UTIL
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def norm_splus_id(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="ignore")
    return str(x)

def norm_gaia_id(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        return int(x)
    except Exception:
        return None

def discover_fields_from_datacubes(tmpl: str) -> List[str]:
    # e.g. tmpl ".../datacube_{field}.parquet"
    pat = tmpl.replace("{field}", "*")
    files = sorted(Path(pat).parent.glob(Path(pat).name))
    out = []
    for f in files:
        name = f.name
        # datacube_<FIELD>.parquet
        if name.startswith("datacube_") and name.endswith(".parquet"):
            out.append(name[len("datacube_"):-len(".parquet")])
    return sorted(out)

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================
# TOKEN OFFSETS + SPECIAL TOKENS
# ============================================================

@dataclass(frozen=True)
class VocabSpec:
    # global layout: [SPECIAL ...] + [IMAGE] + [SCALAR] + [SPECTRA group 0] + ...
    pad_id: int
    bos_id: int
    eos_id: int
    sep_id: int
    miss_id: int
    base_image: int
    base_scalar: int
    base_spectrum: Dict[str, int]
    vocab_size: int

def build_vocab_spec() -> VocabSpec:
    # special tokens
    PAD = 0
    BOS = 1
    EOS = 2
    SEP = 3
    MISS = 4
    next_id = 5

    base_image = next_id
    next_id += V_IMAGE

    base_scalar = next_id
    next_id += V_SCALAR

    base_spectrum = {}
    for g in SPECTRUM_GROUPS.keys():
        base_spectrum[g] = next_id
        next_id += V_SPECTRA

    return VocabSpec(
        pad_id=PAD,
        bos_id=BOS,
        eos_id=EOS,
        sep_id=SEP,
        miss_id=MISS,
        base_image=base_image,
        base_scalar=base_scalar,
        base_spectrum=base_spectrum,
        vocab_size=next_id,
    )


# ============================================================
# FIELD INDEXER (maps ids -> row indices, loads arrays once)
# ============================================================

class FieldIndex:
    """
    Loads:
      - image npz: ids=Gaia source_id, tokens_flat
      - scalar npz: ids=SPLUS id, scalar_tokens (N, n_cols)
      - spectrum npz: ids=SPLUS id, ragged groups (flat+indptr)
    Builds dict maps from id->row index so we can fetch quickly.
    """

    def __init__(self, field: str, vocab: VocabSpec):
        self.field = field
        self.vocab = vocab

        self.df = None  # filtered dataframe (metadata subset)

        # image
        self.img_tokens_flat: Optional[np.ndarray] = None
        self.img_id_to_row: Dict[int, int] = {}

        # scalar
        self.scalar_tokens: Optional[np.ndarray] = None
        self.scalar_cols: Optional[np.ndarray] = None
        self.splus_id_to_scalar_row: Dict[str, int] = {}

        # spectrum
        self.splus_id_to_spec_row: Dict[str, int] = {}
        self.spec_npz: Optional[np.lib.npyio.NpzFile] = None  # keep open
        self.spec_ids: Optional[np.ndarray] = None

        # available spectrum groups in this field
        self.spec_groups_present: Dict[str, Tuple[str, str]] = {}

        self._load()

    def _load(self):
        # ---- read & filter datacube ----
        df = pd.read_parquet(
            DATACUBE_TMPL.format(field=self.field),
            columns=["id", "ra", "dec", "gaia_source_id", MAG_COL, MAGERR_COL],
        )

        mask = (
            (df[MAG_COL] > MAG_MIN) & (df[MAG_COL] < MAG_MAX) &
            (df[MAGERR_COL] < MAGERR_MAX)
        )
        df = df.loc[mask].copy()
        df["id"] = df["id"].map(norm_splus_id)
        df["gaia_source_id"] = df["gaia_source_id"].map(norm_gaia_id)
        self.df = df

        # ---- image ----
        img_path = IMAGE_TOKENS_TMPL.format(field=self.field)
        if Path(img_path).exists():
            img = np.load(img_path, allow_pickle=True)
            if ("ids" in img.files) and ("tokens_flat" in img.files):
                ids = np.asarray(img["ids"], dtype=object)
                ids = [norm_gaia_id(x) for x in ids]
                toks = img["tokens_flat"]
                self.img_tokens_flat = toks
                self.img_id_to_row = {gid: i for i, gid in enumerate(ids) if gid is not None}
            img.close()

        # ---- scalar ----
        sca_path = SCALAR_TOKENS_TMPL.format(field=self.field)
        if Path(sca_path).exists():
            sca = np.load(sca_path, allow_pickle=True)
            if ("ids" in sca.files) and ("scalar_tokens" in sca.files) and ("scalar_cols" in sca.files):
                sids = np.asarray(sca["ids"], dtype=object)
                sids = [norm_splus_id(x) for x in sids]
                self.scalar_tokens = sca["scalar_tokens"]
                self.scalar_cols = np.asarray(sca["scalar_cols"], dtype=object)
                self.splus_id_to_scalar_row = {sid: i for i, sid in enumerate(sids) if sid}
            sca.close()

        # ---- spectrum ----
        spe_path = SPECTRUM_TOKENS_TMPL.format(field=self.field)
        if Path(spe_path).exists():
            spe = np.load(spe_path, allow_pickle=True)
            if "ids" in spe.files:
                sids = np.asarray(spe["ids"], dtype=object)
                sids = [norm_splus_id(x) for x in sids]
                self.spec_ids = np.array(sids, dtype=object)
                self.splus_id_to_spec_row = {sid: i for i, sid in enumerate(sids) if sid}
                self.spec_npz = spe  # keep open for flat/indptr slicing

                for g, (flat_k, ind_k) in SPECTRUM_GROUPS.items():
                    if flat_k in spe.files and ind_k in spe.files:
                        self.spec_groups_present[g] = (flat_k, ind_k)
            else:
                spe.close()

    def __del__(self):
        try:
            if self.spec_npz is not None:
                self.spec_npz.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return int(len(self.df)) if self.df is not None else 0

    def get_row(self, idx: int) -> Dict[str, Any]:
        """Return one sample (metadata + tokens)."""
        r = self.df.iloc[idx]
        sid: str = r["id"]
        gid: Optional[int] = r["gaia_source_id"]

        # image tokens (gaia keyed)
        img_tokens = None
        if gid is not None and self.img_tokens_flat is not None:
            j = self.img_id_to_row.get(gid, None)
            if j is not None:
                img_tokens = self.img_tokens_flat[j]

        # scalar tokens (splus keyed)
        scalar_tokens = None
        if self.scalar_tokens is not None:
            j = self.splus_id_to_scalar_row.get(sid, None)
            if j is not None:
                scalar_tokens = self.scalar_tokens[j]

        # spectrum tokens (ragged, splus keyed)
        spec = {}
        if self.spec_npz is not None:
            j = self.splus_id_to_spec_row.get(sid, None)
            if j is not None:
                for g, (flat_k, ind_k) in self.spec_groups_present.items():
                    flat = self.spec_npz[flat_k]
                    indp = self.spec_npz[ind_k]
                    a = int(indp[j])
                    b = int(indp[j + 1])
                    spec[g] = flat[a:b]
        return {
            "meta": {
                "field": self.field,
                "id": sid,
                "gaia_source_id": gid,
                "ra": float(r["ra"]),
                "dec": float(r["dec"]),
                MAG_COL: float(r[MAG_COL]),
                MAGERR_COL: float(r[MAGERR_COL]),
            },
            "image_tokens": img_tokens,
            "scalar_tokens": scalar_tokens,
            "spectrum_tokens": spec,  # dict[group]->1d tokens
        }


# ============================================================
# SEQUENCE BUILDER (handles missing data + masks)
# ============================================================

# token types (small set)
TT_BOS    = 0
TT_IMAGE  = 1
TT_SCALAR = 2
TT_SPEC_BASE = 10  # spec groups get TT_SPEC_BASE + index

def build_sequence(sample: Dict[str, Any], vocab: VocabSpec) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      token_ids: int64 [L]
      type_ids:  int64 [L]
    Autoregressive: we will predict next token.
    Missing modalities:
      - insert [MISS] with appropriate type_id
    """

    toks: List[int] = [vocab.bos_id]
    ttypes: List[int] = [TT_BOS]

    # --- image ---
    img = sample["image_tokens"]
    toks.append(vocab.sep_id); ttypes.append(TT_BOS)  # separator
    if img is None:
        toks.append(vocab.miss_id); ttypes.append(TT_IMAGE)
    else:
        arr = np.asarray(img, dtype=np.int64).reshape(-1)[:MAX_IMAGE_TOKENS]
        toks.extend((vocab.base_image + arr).tolist())
        ttypes.extend([TT_IMAGE] * arr.size)

    # --- scalar ---
    sca = sample["scalar_tokens"]
    toks.append(vocab.sep_id); ttypes.append(TT_BOS)
    if sca is None:
        toks.append(vocab.miss_id); ttypes.append(TT_SCALAR)
    else:
        arr = np.asarray(sca, dtype=np.int64).reshape(-1)[:MAX_SCALAR_TOKENS]
        arr = np.clip(arr, 0, V_SCALAR - 1)
        toks.extend((vocab.base_scalar + arr).tolist())
        ttypes.extend([TT_SCALAR] * arr.size)

    # --- spectrum groups ---
    spec: Dict[str, np.ndarray] = sample["spectrum_tokens"] or {}
    # fixed order so type_ids are stable
    spec_group_list = list(SPECTRUM_GROUPS.keys())
    for gi, g in enumerate(spec_group_list):
        tt = TT_SPEC_BASE + gi
        toks.append(vocab.sep_id); ttypes.append(TT_BOS)

        seq = spec.get(g, None)
        if seq is None or len(seq) == 0:
            toks.append(vocab.miss_id); ttypes.append(tt)
        else:
            arr = np.asarray(seq, dtype=np.int64).reshape(-1)[:MAX_SPEC_TOKENS_PER_GROUP]
            arr = np.clip(arr, 0, V_SPECTRA - 1)
            toks.extend((vocab.base_spectrum[g] + arr).tolist())
            ttypes.extend([tt] * arr.size)

    toks.append(vocab.eos_id); ttypes.append(TT_BOS)

    # global cap
    if len(toks) > MAX_SEQ_LEN:
        toks = toks[:MAX_SEQ_LEN]
        ttypes = ttypes[:MAX_SEQ_LEN]

    return np.asarray(toks, dtype=np.int64), np.asarray(ttypes, dtype=np.int64)


# ============================================================
# DATASET (multi-field)
# ============================================================

class MultiFieldTokenDataset(Dataset):
    def __init__(self, fields: List[str], vocab: VocabSpec):
        self.vocab = vocab
        self.field_indexes: List[FieldIndex] = []
        self.global_map: List[Tuple[int, int]] = []  # (field_i, row_i)

        for f in fields:
            fi = FieldIndex(f, vocab=vocab)
            n = len(fi)
            if n == 0:
                continue
            self.field_indexes.append(fi)
            field_i = len(self.field_indexes) - 1
            self.global_map.extend([(field_i, i) for i in range(n)])

        random.shuffle(self.global_map)
        print(f"[data] samples = {len(self.global_map)}")

    def __len__(self) -> int:
        return len(self.global_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fi, ri = self.global_map[idx]
        sample = self.field_indexes[fi].get_row(ri)
        token_ids, type_ids = build_sequence(sample, self.vocab)
        return {
            "token_ids": token_ids,
            "type_ids": type_ids,
        }


def collate_batch(batch: List[Dict[str, Any]], pad_id: int):
    lens = [len(x["token_ids"]) for x in batch]
    max_len = max(lens)

    B = len(batch)
    tokens = torch.full((B, max_len), pad_id, dtype=torch.long)
    types  = torch.zeros((B, max_len), dtype=torch.long)
    attn_mask = torch.zeros((B, max_len), dtype=torch.bool)  # True for real tokens

    for i, ex in enumerate(batch):
        t = torch.from_numpy(ex["token_ids"])
        y = torch.from_numpy(ex["type_ids"])
        L = t.numel()
        tokens[i, :L] = t
        types[i, :L] = y
        attn_mask[i, :L] = True

    return {
        "tokens": tokens,
        "types": types,
        "attn_mask": attn_mask,
    }


# ============================================================
# MODEL
# ============================================================

class CausalTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, n_types: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout: float, max_len: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_types = n_types
        self.d_model = d_model
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.type_emb = nn.Embedding(n_types, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,  # ✅ avoids nested tensor warning you saw
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.drop = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, types: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, L)
        types: (B, L)
        attn_mask: (B, L) bool, True where token is real (not pad)
        """
        B, L = tokens.shape
        if L > self.max_len:
            raise ValueError(f"Sequence length {L} > max_len {self.max_len}. Increase MAX_SEQ_LEN / model max_len.")

        pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)

        x = self.tok_emb(tokens) + self.type_emb(types) + self.pos_emb(pos)
        x = self.drop(x)

        # key padding mask: True where we should MASK (i.e. pad)
        key_padding_mask = ~attn_mask  # (B, L)

        # causal mask: True where future positions are masked
        causal_mask = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1)

        h = self.enc(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        h = self.ln(h)
        logits = self.head(h)  # (B, L, V)
        return logits


# ============================================================
# TRAIN LOOP
# ============================================================

def lr_schedule(step: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    # cosine decay to 10% of LR
    t = step - warmup
    T = 200_000  # you can tune; just needs to be "large"
    frac = min(1.0, t / max(1, T))
    cos = 0.5 * (1 + math.cos(math.pi * frac))
    return base_lr * (0.1 + 0.9 * cos)

def train():
    set_seed(SEED)
    ensure_dir(OUT_DIR)

    vocab = build_vocab_spec()
    n_types = TT_SPEC_BASE + len(SPECTRUM_GROUPS) + 5  # small safety margin

    # fields
    fields = FIELDS or discover_fields_from_datacubes(DATACUBE_TMPL)
    if not fields:
        raise RuntimeError("No fields found. Set FIELDS or fix DATACUBE_TMPL.")
    print(f"[data] using {len(fields)} fields")

    ds = MultiFieldTokenDataset(fields, vocab=vocab)

    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=lambda b: collate_batch(b, pad_id=vocab.pad_id),
        drop_last=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device} visible_cuda_devices={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    model = CausalTransformerLM(
        vocab_size=vocab.vocab_size,
        n_types=n_types,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
        max_len=MAX_SEQ_LEN,
    ).to(device)

    total_p, train_p = count_parameters(model)
    print(f"[model] d_model={D_MODEL} layers={N_LAYERS} heads={N_HEADS} ff={D_FF}")
    print(f"[model] vocab_size={vocab.vocab_size} n_types={n_types} max_len={MAX_SEQ_LEN}")
    print(f"[model] params_total={total_p:,} params_trainable={train_p:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # ✅ no deprecation

    global_step = 0
    model.train()

    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(dl, desc=f"epoch {epoch}/{EPOCHS}", dynamic_ncols=True)
        for it, batch in enumerate(pbar):
            # move
            tokens = batch["tokens"].to(device, non_blocking=True)
            types = batch["types"].to(device, non_blocking=True)
            attn_mask = batch["attn_mask"].to(device, non_blocking=True)

            # LM objective: predict next token
            x = tokens[:, :-1]
            x_types = types[:, :-1]
            x_mask = attn_mask[:, :-1]

            y = tokens[:, 1:].contiguous()

            lr_now = lr_schedule(global_step, LR, WARMUP_STEPS)
            for pg in opt.param_groups:
                pg["lr"] = lr_now

            with torch.amp.autocast("cuda", enabled=use_amp):  # ✅ no deprecation
                logits = model(x, x_types, x_mask)  # (B, L-1, V)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=vocab.pad_id,
                )

            loss_val = float(loss.detach().cpu().item())

            scaler.scale(loss / GRAD_ACCUM_STEPS).backward()

            if (it + 1) % GRAD_ACCUM_STEPS == 0:
                if GRAD_CLIP is not None and GRAD_CLIP > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

                global_step += 1

                if global_step % LOG_EVERY == 0:
                    pbar.set_postfix({"loss": f"{loss_val:.3f}", "lr": f"{lr_now:.2e}"})

                if global_step % SAVE_EVERY_STEPS == 0:
                    ckpt = {
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "scaler": scaler.state_dict() if use_amp else None,
                        "global_step": global_step,
                        "epoch": epoch,
                        "config": {
                            "D_MODEL": D_MODEL,
                            "N_LAYERS": N_LAYERS,
                            "N_HEADS": N_HEADS,
                            "D_FF": D_FF,
                            "MAX_SEQ_LEN": MAX_SEQ_LEN,
                            "vocab_size": vocab.vocab_size,
                            "offsets": {
                                "base_image": vocab.base_image,
                                "base_scalar": vocab.base_scalar,
                                "base_spectrum": vocab.base_spectrum,
                            },
                        },
                    }
                    outp = Path(OUT_DIR) / f"ckpt_step_{global_step:08d}.pt"
                    torch.save(ckpt, outp)
                    tqdm.write(f"[save] {outp}")

        # epoch end save
        outp = Path(OUT_DIR) / f"ckpt_epoch_{epoch:03d}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch, "global_step": global_step}, outp)
        tqdm.write(f"[save] {outp}")


if __name__ == "__main__":
    train()