#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark for your AION-like token Transformer trained by train_transformer_tokens.py.

Fixes vs your previous benchmark:
✅ Uses the SAME token file templates as training (no extra subfolders)
✅ Loads checkpoint STRICTLY (no silent random weights)
✅ Infers vocab_size, d_model, n_types, max_len, d_ff, n_layers from checkpoint tensors
✅ Keeps safe NPZ access (default NUM_WORKERS=0)

Run:
  python3 benchmark_transformer_fixed.py
"""

from __future__ import annotations

# ---------------------------
# FORCE GPU (must be before torch import)
# ---------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # change if you want another GPU

import math
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG (EDIT THIS)
# ============================================================

# ---- checkpoint (use a concrete file path) ----
CKPT_PATH = "runs_tokens_transformer/ckpt_step_01090000.pt"

# ---- data templates (MUST match training) ----
FIELDS: List[str] = [
    "HYDRA-0012",
    "HYDRA-0013",
]  # if empty -> auto-discover

DATACUBE_TMPL = "/home/astrodados4/downloads/hypercube/datacube_{field}.parquet"

IMAGE_TOKENS_TMPL   = "/home/schwarz/projetoFM/codecs/image_tokens/datacube_{field}_tokens.npz"
SCALAR_TOKENS_TMPL  = "/home/schwarz/projetoFM/scalar_tokenizers/datacube_{field}_scalar_tokens.npz"
SPECTRUM_TOKENS_TMPL= "/home/schwarz/projetoFM/spectrum_tokenizers/datacube_{field}_spectrum_tokens.npz"

# ---- row filter (should match training) ----
MAG_MIN = 14.0
MAG_MAX = 22.0
MAGERR_MAX = 2.0
MAG_COL = "mag_pstotal_r"
MAGERR_COL = "err_mag_pstotal_r"

# ---- token vocab sizes (must match training generator) ----
V_IMAGE   = 2048
V_SCALAR  = 1024
V_SPECTRA = 2048

# ---- spectrum groups order (MUST match training order) ----
SPECTRUM_GROUPS = {
    "gaiaxp_bp": ("tokens_gaiaxp_bp_flat", "tokens_gaiaxp_bp_indptr"),
    "gaiaxp_rp": ("tokens_gaiaxp_rp_flat", "tokens_gaiaxp_rp_indptr"),
    "desi_b":    ("tokens_desi_b_flat",    "tokens_desi_b_indptr"),
    "desi_r":    ("tokens_desi_r_flat",    "tokens_desi_r_indptr"),
    "desi_z":    ("tokens_desi_z_flat",    "tokens_desi_z_indptr"),
}

# ---- sequence caps (must match training caps) ----
MAX_IMAGE_TOKENS = 24 * 24
MAX_SCALAR_TOKENS = 512
MAX_SPEC_TOKENS_PER_GROUP = 512
MAX_SEQ_LEN_FALLBACK = 2048  # will be overridden by ckpt pos_emb if present

# ---- benchmark params ----
OUT_DIR = "bench_tokens_transformer"
SEED = 0
N_SAMPLES_LIMIT: Optional[int] = 50_000  # set None for all
BATCH_SIZE = 16

# safest default: 0 because np.load(npz) inside workers can trip zipfile overlap checks.
NUM_WORKERS = 0
PIN_MEMORY = True


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
    pat = tmpl.replace("{field}", "*")
    files = sorted(Path(pat).parent.glob(Path(pat).name))
    out = []
    for f in files:
        name = f.name
        if name.startswith("datacube_") and name.endswith(".parquet"):
            out.append(name[len("datacube_"):-len(".parquet")])
    return sorted(out)

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def entropy_from_counts(counts: np.ndarray) -> Tuple[float, float]:
    counts = counts.astype(np.float64)
    n = counts.sum()
    if n <= 0:
        return 0.0, 0.0
    p = counts[counts > 0] / n
    H = float(-(p * np.log(p)).sum())
    ppl = float(np.exp(H))
    return H, ppl

def topk_mass(counts: np.ndarray, k: int = 10) -> float:
    n = counts.sum()
    if n <= 0:
        return 0.0
    s = np.sort(counts)[::-1]
    return float(s[:k].sum() / n)

def infer_n_layers_from_state(state: dict) -> int:
    # keys look like: enc.layers.0.self_attn.in_proj_weight
    mx = -1
    for k in state.keys():
        if k.startswith("enc.layers."):
            rest = k[len("enc.layers."):]
            idx_str = rest.split(".", 1)[0]
            if idx_str.isdigit():
                mx = max(mx, int(idx_str))
    return (mx + 1) if mx >= 0 else 1

def infer_d_ff_from_state(state: dict) -> int:
    # linear1.weight shape: (d_ff, d_model)
    for k, v in state.items():
        if k.endswith("enc.layers.0.linear1.weight") and hasattr(v, "shape"):
            return int(v.shape[0])
    # fallback
    return 2048

def choose_n_heads(d_model: int) -> int:
    # prefer head_dim=64 if possible, else a divisor
    if d_model % 64 == 0:
        return d_model // 64
    for h in [16, 12, 8, 6, 4, 2, 1]:
        if d_model % h == 0:
            return h
    return 1


# ============================================================
# VOCAB SPEC (match training, use ckpt offsets when available)
# ============================================================

@dataclass(frozen=True)
class VocabSpec:
    pad_id: int
    bos_id: int
    eos_id: int
    sep_id: int
    miss_id: int
    base_image: int
    base_scalar: int
    base_spectrum: Dict[str, int]
    vocab_size: int

def build_vocab_spec_from_ckpt_or_default(ckpt: dict, ckpt_vocab_size: int) -> VocabSpec:
    # training used fixed specials
    PAD, BOS, EOS, SEP, MISS = 0, 1, 2, 3, 4

    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    offsets = cfg.get("offsets", None) if isinstance(cfg, dict) else None

    if isinstance(offsets, dict) and ("base_image" in offsets) and ("base_scalar" in offsets) and ("base_spectrum" in offsets):
        base_image = int(offsets["base_image"])
        base_scalar = int(offsets["base_scalar"])
        base_spectrum = {str(k): int(v) for k, v in offsets["base_spectrum"].items()}
        return VocabSpec(
            pad_id=PAD, bos_id=BOS, eos_id=EOS, sep_id=SEP, miss_id=MISS,
            base_image=base_image, base_scalar=base_scalar, base_spectrum=base_spectrum,
            vocab_size=int(ckpt_vocab_size),
        )

    # fallback: rebuild exactly like training script
    next_id = 5
    base_image = next_id; next_id += V_IMAGE
    base_scalar = next_id; next_id += V_SCALAR
    base_spectrum = {}
    for g in SPECTRUM_GROUPS.keys():
        base_spectrum[g] = next_id
        next_id += V_SPECTRA

    if next_id != ckpt_vocab_size:
        raise RuntimeError(
            f"Vocab mismatch: default layout gives vocab_size={next_id} "
            f"but checkpoint tok_emb has vocab_size={ckpt_vocab_size}. "
            f"Your ckpt likely used different offsets—save offsets in ckpt['config']['offsets']."
        )

    return VocabSpec(
        pad_id=PAD, bos_id=BOS, eos_id=EOS, sep_id=SEP, miss_id=MISS,
        base_image=base_image, base_scalar=base_scalar, base_spectrum=base_spectrum,
        vocab_size=int(next_id),
    )


# ============================================================
# TOKEN TYPES (match training)
# ============================================================

TT_BOS = 0
TT_IMAGE = 1
TT_SCALAR = 2
TT_SPEC_BASE = 10  # + group index in SPECTRUM_GROUPS order


# ============================================================
# MODEL (must match training module structure)
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
            norm_first=False,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, types: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        if L > self.max_len:
            raise ValueError(f"Sequence length {L} > max_len {self.max_len}")

        pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
        x = self.tok_emb(tokens) + self.type_emb(types) + self.pos_emb(pos)

        key_padding_mask = ~attn_mask
        causal_mask = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1)

        h = self.enc(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        h = self.ln(h)
        return self.head(h)


# ============================================================
# SAFE NPZ ACCESS (avoid overlapped entries errors)
# ============================================================

class LazySpectrumNPZ:
    """Open the spectrum npz lazily in the current process (important for DataLoader workers)."""
    def __init__(self, path: str):
        self.path = path
        self._npz = None

    def open(self):
        if self._npz is None:
            self._npz = np.load(self.path, allow_pickle=True)
        return self._npz

    def close(self):
        try:
            if self._npz is not None:
                self._npz.close()
        except Exception:
            pass
        self._npz = None


# ============================================================
# FIELD INDEXER
# ============================================================

class FieldIndex:
    def __init__(self, field: str):
        self.field = field
        self.df = None

        # image
        self.img_tokens_flat: Optional[np.ndarray] = None
        self.img_id_to_row: Dict[int, int] = {}

        # scalar
        self.scalar_tokens: Optional[np.ndarray] = None
        self.splus_id_to_scalar_row: Dict[str, int] = {}

        # spectrum (lazy)
        self.spe_lazy: Optional[LazySpectrumNPZ] = None
        self.splus_id_to_spec_row: Dict[str, int] = {}
        self.spec_groups_present: Dict[str, Tuple[str, str]] = {}

        self._load()

    def _load(self):
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

        # image
        ip = IMAGE_TOKENS_TMPL.format(field=self.field)
        if Path(ip).exists():
            npz = np.load(ip, allow_pickle=True)
            if "ids" in npz.files and "tokens_flat" in npz.files:
                ids = [norm_gaia_id(x) for x in np.asarray(npz["ids"], dtype=object)]
                toks = npz["tokens_flat"]
                self.img_tokens_flat = toks
                self.img_id_to_row = {gid: i for i, gid in enumerate(ids) if gid is not None}
            npz.close()

        # scalar
        sp = SCALAR_TOKENS_TMPL.format(field=self.field)
        if Path(sp).exists():
            npz = np.load(sp, allow_pickle=True)
            if "ids" in npz.files and "scalar_tokens" in npz.files:
                ids = [norm_splus_id(x) for x in np.asarray(npz["ids"], dtype=object)]
                self.scalar_tokens = npz["scalar_tokens"]
                self.splus_id_to_scalar_row = {sid: i for i, sid in enumerate(ids) if sid}
            npz.close()

        # spectrum (lazy open per process)
        xp = SPECTRUM_TOKENS_TMPL.format(field=self.field)
        if Path(xp).exists():
            npz = np.load(xp, allow_pickle=True)
            if "ids" in npz.files:
                ids = [norm_splus_id(x) for x in np.asarray(npz["ids"], dtype=object)]
                self.splus_id_to_spec_row = {sid: i for i, sid in enumerate(ids) if sid}
                for g, (fk, ik) in SPECTRUM_GROUPS.items():
                    if fk in npz.files and ik in npz.files:
                        self.spec_groups_present[g] = (fk, ik)
                self.spe_lazy = LazySpectrumNPZ(xp)
            npz.close()

    def __len__(self) -> int:
        return int(len(self.df))

    def get_row(self, idx: int) -> Dict[str, Any]:
        r = self.df.iloc[idx]
        sid = r["id"]
        gid = r["gaia_source_id"]

        img_tokens = None
        if gid is not None and self.img_tokens_flat is not None:
            j = self.img_id_to_row.get(gid, None)
            if j is not None:
                img_tokens = self.img_tokens_flat[j]

        scalar_tokens = None
        if self.scalar_tokens is not None:
            j = self.splus_id_to_scalar_row.get(sid, None)
            if j is not None:
                scalar_tokens = self.scalar_tokens[j]

        spec = {}
        if self.spe_lazy is not None and self.spec_groups_present:
            j = self.splus_id_to_spec_row.get(sid, None)
            if j is not None:
                spe = self.spe_lazy.open()
                for g, (fk, ik) in self.spec_groups_present.items():
                    flat = spe[fk]
                    indp = spe[ik]
                    a = int(indp[j]); b = int(indp[j + 1])
                    spec[g] = flat[a:b]

        return {
            "id": sid,
            "gaia_source_id": gid,
            "image_tokens": img_tokens,
            "scalar_tokens": scalar_tokens,
            "spectrum_tokens": spec,
        }

    def close(self):
        if self.spe_lazy is not None:
            self.spe_lazy.close()


# ============================================================
# SEQUENCE BUILDER (exactly training)
# ============================================================

def build_sequence(sample: Dict[str, Any], vocab: VocabSpec, max_seq_len: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[int, int]]]:
    toks: List[int] = [vocab.bos_id]
    ttypes: List[int] = [TT_BOS]
    seg: Dict[str, Tuple[int, int]] = {}

    def add_sep():
        toks.append(vocab.sep_id)
        ttypes.append(TT_BOS)

    # image
    add_sep()
    a = len(toks)
    img = sample["image_tokens"]
    if img is None:
        toks.append(vocab.miss_id); ttypes.append(TT_IMAGE)
    else:
        arr = np.asarray(img, dtype=np.int64).reshape(-1)[:MAX_IMAGE_TOKENS]
        toks.extend((vocab.base_image + arr).tolist())
        ttypes.extend([TT_IMAGE] * arr.size)
    seg["image"] = (a, len(toks))

    # scalar
    add_sep()
    a = len(toks)
    sca = sample["scalar_tokens"]
    if sca is None:
        toks.append(vocab.miss_id); ttypes.append(TT_SCALAR)
    else:
        arr = np.asarray(sca, dtype=np.int64).reshape(-1)[:MAX_SCALAR_TOKENS]
        arr = np.clip(arr, 0, V_SCALAR - 1)
        toks.extend((vocab.base_scalar + arr).tolist())
        ttypes.extend([TT_SCALAR] * arr.size)
    seg["scalar"] = (a, len(toks))

    # spectrum groups
    spec = sample["spectrum_tokens"] or {}
    for gi, g in enumerate(SPECTRUM_GROUPS.keys()):
        add_sep()
        a = len(toks)
        tt = TT_SPEC_BASE + gi
        seq = spec.get(g, None)
        if seq is None or len(seq) == 0:
            toks.append(vocab.miss_id); ttypes.append(tt)
        else:
            arr = np.asarray(seq, dtype=np.int64).reshape(-1)[:MAX_SPEC_TOKENS_PER_GROUP]
            arr = np.clip(arr, 0, V_SPECTRA - 1)
            toks.extend((vocab.base_spectrum[g] + arr).tolist())
            ttypes.extend([tt] * arr.size)
        seg[g] = (a, len(toks))

    toks.append(vocab.eos_id); ttypes.append(TT_BOS)

    if len(toks) > max_seq_len:
        toks = toks[:max_seq_len]
        ttypes = ttypes[:max_seq_len]

    return np.asarray(toks, np.int64), np.asarray(ttypes, np.int64), seg


# ============================================================
# DATASET
# ============================================================

class MultiFieldDataset(Dataset):
    def __init__(self, fields: List[str], vocab: VocabSpec, max_seq_len: int, limit: Optional[int] = None):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.fidx: List[FieldIndex] = []
        self.map: List[Tuple[int, int]] = []

        for f in fields:
            fi = FieldIndex(f)
            n = len(fi)
            if n == 0:
                continue
            self.fidx.append(fi)
            k = len(self.fidx) - 1
            self.map.extend([(k, i) for i in range(n)])

        random.shuffle(self.map)
        if limit is not None and len(self.map) > limit:
            self.map = self.map[:limit]
        print(f"[data] samples = {len(self.map)}")

    def __len__(self):
        return len(self.map)

    def __getitem__(self, idx: int):
        fi, ri = self.map[idx]
        s = self.fidx[fi].get_row(ri)
        token_ids, type_ids, seg = build_sequence(s, self.vocab, max_seq_len=self.max_seq_len)
        return {"token_ids": token_ids, "type_ids": type_ids, "segments": seg, "raw": s}

    def close(self):
        for fi in self.fidx:
            fi.close()


def collate(batch: List[Dict[str, Any]], pad_id: int):
    max_len = max(len(x["token_ids"]) for x in batch)
    B = len(batch)

    tokens = torch.full((B, max_len), pad_id, dtype=torch.long)
    types  = torch.zeros((B, max_len), dtype=torch.long)
    attn_mask = torch.zeros((B, max_len), dtype=torch.bool)

    segs = [x["segments"] for x in batch]
    raws = [x["raw"] for x in batch]

    for i, ex in enumerate(batch):
        t = torch.from_numpy(ex["token_ids"])
        y = torch.from_numpy(ex["type_ids"])
        L = t.numel()
        tokens[i, :L] = t
        types[i, :L] = y
        attn_mask[i, :L] = True

    return {"tokens": tokens, "types": types, "attn_mask": attn_mask, "segments": segs, "raws": raws}


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)
    ensure_dir(OUT_DIR)

    ckpt_path = Path(CKPT_PATH)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint format not understood (expected a state_dict or dict with key 'model').")

    # --- infer core shapes from checkpoint ---
    if "tok_emb.weight" not in state or "type_emb.weight" not in state or "pos_emb.weight" not in state:
        raise RuntimeError("Checkpoint missing tok_emb/type_emb/pos_emb weights. Are you benchmarking the right checkpoint?")

    ckpt_vocab_size = int(state["tok_emb.weight"].shape[0])
    ckpt_d_model    = int(state["tok_emb.weight"].shape[1])
    ckpt_n_types    = int(state["type_emb.weight"].shape[0])
    ckpt_max_len    = int(state["pos_emb.weight"].shape[0])
    ckpt_d_ff       = infer_d_ff_from_state(state)
    ckpt_n_layers   = infer_n_layers_from_state(state)
    ckpt_n_heads    = None

    # Prefer ckpt config if present, otherwise pick a safe head count
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if isinstance(cfg, dict) and "N_HEADS" in cfg:
        ckpt_n_heads = int(cfg["N_HEADS"])
    else:
        ckpt_n_heads = choose_n_heads(ckpt_d_model)

    if ckpt_d_model % ckpt_n_heads != 0:
        raise RuntimeError(f"Invalid heads: d_model={ckpt_d_model} not divisible by n_heads={ckpt_n_heads}")

    vocab = build_vocab_spec_from_ckpt_or_default(ckpt if isinstance(ckpt, dict) else {}, ckpt_vocab_size=ckpt_vocab_size)

    print(f"[ckpt] tok_emb={tuple(state['tok_emb.weight'].shape)} type_emb={tuple(state['type_emb.weight'].shape)} pos_emb={tuple(state['pos_emb.weight'].shape)}")
    print(f"[infer] vocab_size={ckpt_vocab_size} d_model={ckpt_d_model} n_types={ckpt_n_types} max_len={ckpt_max_len} d_ff={ckpt_d_ff} n_layers={ckpt_n_layers} n_heads={ckpt_n_heads}")
    print(f"[vocab] base_image={vocab.base_image} base_scalar={vocab.base_scalar} base_spectrum_keys={list(vocab.base_spectrum.keys())[:3]}...")

    model = CausalTransformerLM(
        vocab_size=ckpt_vocab_size,
        n_types=ckpt_n_types,
        d_model=ckpt_d_model,
        n_layers=ckpt_n_layers,
        n_heads=ckpt_n_heads,
        d_ff=ckpt_d_ff,
        dropout=0.0,
        max_len=ckpt_max_len,
    )

    # STRICT LOAD: if this fails, your benchmark is not matching training
    model.load_state_dict(state, strict=True)

    total_p, train_p = count_parameters(model)
    print(f"[model] params_total={total_p:,} params_trainable={train_p:,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"[env] device={device} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # ---- fields ----
    fields = FIELDS or discover_fields_from_datacubes(DATACUBE_TMPL)
    if not fields:
        raise RuntimeError("No fields found.")
    print(f"[data] fields={len(fields)}")

    ds = MultiFieldDataset(fields, vocab=vocab, max_seq_len=ckpt_max_len, limit=N_SAMPLES_LIMIT)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=lambda b: collate(b, pad_id=vocab.pad_id),
        drop_last=False,
    )

    # ---- accumulators ----
    sum_loss = 0.0
    sum_tokens = 0

    sum_correct = 0
    sum_total = 0

    sum_correct_nomiss = 0
    sum_total_nomiss = 0

    seg_names = ["image", "scalar"] + list(SPECTRUM_GROUPS.keys())
    seg_correct_nomiss = {k: 0 for k in seg_names}
    seg_total_nomiss = {k: 0 for k in seg_names}

    miss_counts = {k: 0 for k in seg_names}
    usage = {
        k: np.zeros((V_IMAGE if k == "image" else V_SCALAR if k == "scalar" else V_SPECTRA), dtype=np.int64)
        for k in seg_names
    }

    special_ids = torch.tensor([vocab.pad_id, vocab.bos_id, vocab.eos_id, vocab.sep_id], dtype=torch.long, device=device)

    # quick one-time sanity on first batch
    did_print_sanity = False

    pbar = tqdm(dl, desc="[bench]", dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            tokens = batch["tokens"].to(device, non_blocking=True)
            types = batch["types"].to(device, non_blocking=True)
            attn_mask = batch["attn_mask"].to(device, non_blocking=True)
            segs = batch["segments"]
            raws = batch["raws"]

            if not did_print_sanity and len(raws) > 0:
                r0 = raws[0]
                print("[sanity] first sample:", {
                    "img_none": r0["image_tokens"] is None,
                    "sca_none": r0["scalar_tokens"] is None,
                    "spec_keys": list((r0["spectrum_tokens"] or {}).keys()),
                })
                did_print_sanity = True

            x = tokens[:, :-1]
            x_types = types[:, :-1]
            x_mask = attn_mask[:, :-1]
            y = tokens[:, 1:]

            logits = model(x, x_types, x_mask)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=vocab.pad_id,
                reduction="sum",
            )

            valid = (y != vocab.pad_id)
            preds = logits.argmax(dim=-1)
            correct = (preds == y) & valid

            is_special = (y[..., None] == special_ids[None, None, :]).any(dim=-1)
            valid_nomiss = valid & (~is_special) & (y != vocab.miss_id)

            sum_loss += float(loss.item())
            sum_tokens += int(valid.sum().item())

            sum_correct += int(correct.sum().item())
            sum_total += int(valid.sum().item())

            sum_correct_nomiss += int(((preds == y) & valid_nomiss).sum().item())
            sum_total_nomiss += int(valid_nomiss.sum().item())

            # per-sample segment stats + missing fractions + usage
            B = tokens.shape[0]
            for i in range(B):
                raw = raws[i]
                if raw["image_tokens"] is None: miss_counts["image"] += 1
                if raw["scalar_tokens"] is None: miss_counts["scalar"] += 1
                for g in SPECTRUM_GROUPS.keys():
                    seq = (raw["spectrum_tokens"] or {}).get(g, None)
                    if seq is None or len(seq) == 0:
                        miss_counts[g] += 1

                seg = segs[i]
                for k in seg_names:
                    if k not in seg:
                        continue
                    a, b = seg[k]
                    ya = max(0, a - 1)
                    yb = max(0, b - 1)
                    if yb <= ya:
                        continue

                    yy = y[i, ya:yb]
                    pp = preds[i, ya:yb]

                    vv = (yy != vocab.pad_id)
                    is_sp = (yy[..., None] == special_ids[None, :]).any(dim=-1)
                    vv_nomiss = vv & (~is_sp) & (yy != vocab.miss_id)

                    seg_total_nomiss[k] += int(vv_nomiss.sum().item())
                    seg_correct_nomiss[k] += int(((pp == yy) & vv_nomiss).sum().item())

                    yy_np = yy[vv_nomiss].detach().cpu().numpy()
                    if yy_np.size > 0:
                        if k == "image":
                            local = yy_np - vocab.base_image
                            local = local[(local >= 0) & (local < V_IMAGE)]
                        elif k == "scalar":
                            local = yy_np - vocab.base_scalar
                            local = local[(local >= 0) & (local < V_SCALAR)]
                        else:
                            base = vocab.base_spectrum.get(k, None)
                            if base is None:
                                continue
                            local = yy_np - base
                            local = local[(local >= 0) & (local < V_SPECTRA)]
                        if local.size > 0:
                            usage[k] += np.bincount(local, minlength=usage[k].shape[0]).astype(np.int64)

            avg_loss = (sum_loss / max(1, sum_tokens))
            ppl = math.exp(min(50.0, avg_loss))  # clamp to avoid inf
            acc = sum_correct / max(1, sum_total)
            acc_nm = sum_correct_nomiss / max(1, sum_total_nomiss)
            pbar.set_postfix({"ppl": f"{ppl:.2f}", "acc": f"{acc*100:.2f}%", "acc_nomiss": f"{acc_nm*100:.2f}%"})

    avg_loss = sum_loss / max(1, sum_tokens)
    ppl = math.exp(min(50.0, avg_loss))
    acc = sum_correct / max(1, sum_total)
    acc_nm = sum_correct_nomiss / max(1, sum_total_nomiss)

    results = {
        "ckpt": str(CKPT_PATH),
        "samples": len(ds),
        "tokens": int(sum_tokens),
        "loss_per_token": float(avg_loss),
        "perplexity": float(ppl),
        "accuracy_including_miss": float(acc),
        "accuracy_excluding_special_and_miss": float(acc_nm),
        "coverage_missing_frac": {k: float(miss_counts[k] / max(1, len(ds))) for k in seg_names},
        "segment_accuracy_excluding_miss": {
            k: (float(seg_correct_nomiss[k] / seg_total_nomiss[k]) if seg_total_nomiss[k] > 0 else None)
            for k in seg_names
        },
        "segment_tokens_excluding_miss": {k: int(seg_total_nomiss[k]) for k in seg_names},
        "segment_entropy": {},
        "model": {
            "d_model": int(ckpt_d_model),
            "n_layers": int(ckpt_n_layers),
            "n_heads": int(ckpt_n_heads),
            "d_ff": int(ckpt_d_ff),
            "vocab_size": int(ckpt_vocab_size),
            "n_types": int(ckpt_n_types),
            "max_len": int(ckpt_max_len),
            "params_total": int(total_p),
        },
        "data": {
            "fields": fields,
            "limit": N_SAMPLES_LIMIT,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
        }
    }

    for k in seg_names:
        H, ppl_seg = entropy_from_counts(usage[k])
        results["segment_entropy"][k] = {
            "entropy_nats": float(H),
            "perplexity": float(ppl_seg),
            "top10_mass": float(topk_mass(usage[k], 10)),
            "used_bins": int((usage[k] > 0).sum()),
            "used_bins_frac": float((usage[k] > 0).sum() / usage[k].shape[0]),
        }

    ensure_dir(OUT_DIR)
    out_json = Path(OUT_DIR) / "bench_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"[out] wrote {out_json}")

    # ---- plots ----
    plt.figure()
    plt.bar(["loss/token", "ppl", "acc% (all)", "acc% (no miss)"], [avg_loss, ppl, acc*100.0, acc_nm*100.0])
    plt.title("Overall benchmark")
    plt.tight_layout()
    plt.savefig(Path(OUT_DIR) / "overall.png", dpi=180)
    plt.close()

    seg_acc = [
        (results["segment_accuracy_excluding_miss"][k] * 100.0) if results["segment_accuracy_excluding_miss"][k] is not None else 0.0
        for k in seg_names
    ]
    plt.figure(figsize=(10, 4))
    plt.bar(seg_names, seg_acc)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title("Next-token accuracy by segment (excluding MISS + specials)")
    plt.tight_layout()
    plt.savefig(Path(OUT_DIR) / "segment_accuracy_nomiss.png", dpi=180)
    plt.close()

    miss = [results["coverage_missing_frac"][k] * 100 for k in seg_names]
    plt.figure(figsize=(10, 4))
    plt.bar(seg_names, miss)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Missing fraction (%)")
    plt.title("Missing coverage by segment")
    plt.tight_layout()
    plt.savefig(Path(OUT_DIR) / "segment_missing.png", dpi=180)
    plt.close()

    for k in seg_names:
        cnt = usage[k]
        if cnt.sum() == 0:
            continue
        topk = 50
        idx = np.argsort(cnt)[::-1][:topk]
        plt.figure(figsize=(10, 4))
        plt.bar(np.arange(topk), cnt[idx])
        plt.title(f"Token usage top-{topk}: {k} (excluding MISS/special)")
        plt.xlabel("rank")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(Path(OUT_DIR) / f"usage_top{topk}_{k}.png", dpi=180)
        plt.close()

    ds.close()

    print(f"[done] outputs in: {OUT_DIR}")
    print(f"[summary] ppl={ppl:.2f} acc(all)={acc*100:.2f}% acc(no-miss)={acc_nm*100:.2f}%")

if __name__ == "__main__":
    main()