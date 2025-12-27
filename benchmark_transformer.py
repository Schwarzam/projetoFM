#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark for your AION-like token Transformer trained by train_transformer_tokens.py.

✅ Matches *training* vocab/type layout exactly:
  - Uses ckpt["config"]["offsets"] if present (base_image/base_scalar/base_spectrum)
  - Uses the fixed SPECTRUM_GROUPS order from training script
  - Uses n_types from checkpoint type_emb rows (no mismatch)
✅ Fixes NPZ BadZipFile in DataLoader workers:
  - Option A (default): NUM_WORKERS=0 (safe)
  - Option B: if you want workers, we open spectrum NPZ *per worker* lazily.
✅ Reports:
  - loss/token, perplexity
  - accuracy overall AND accuracy excluding {PAD,BOS,EOS,SEP,MISS}
  - per-segment accuracy excluding MISS
  - missing fraction per segment
  - token-usage stats (entropy/top10 mass)
✅ Forces GPU #1:
  - sets CUDA_VISIBLE_DEVICES=1 before torch import

Run:
  python3 benchmark_transformer.py

Tip:
  Set CKPT_PATH to a specific file, not a wildcard.
"""

from __future__ import annotations

# ---------------------------
# FORCE GPU #1 (must be before torch import)
# ---------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
CKPT_PATH = "runs_tokens_transformer/ckpt_epoch_003.pt"

# ---- data templates (must match training) ----
FIELDS: List[str] = [
    "HYDRA-0012",
    "HYDRA-0013",
    "HYDRA-0014",
    "HYDRA-0015",
    "HYDRA-0016",
    "HYDRA-0017",
]  # if empty -> auto-discover
DATACUBE_TMPL = "/home/astrodados4/downloads/hypercube/datacube_{field}.parquet"

IMAGE_TOKENS_TMPL    = "/home/schwarz/projetoFM/codecs/image_tokens/datacube_{field}_tokens.npz"
SCALAR_TOKENS_TMPL   = "/home/schwarz/projetoFM/scalar_tokenizers/scalar_tokens/datacube_{field}_scalar_tokens.npz"
SPECTRUM_TOKENS_TMPL = "/home/schwarz/projetoFM/spectrum_tokenizers/spectrum_tokens/datacube_{field}_spectrum_tokens.npz"

# ---- row filter (must match training) ----
MAG_MIN = 14.0
MAG_MAX = 22.0
MAGERR_MAX = 2.0
MAG_COL = "mag_pstotal_r"
MAGERR_COL = "err_mag_pstotal_r"

# ---- token vocab sizes (must match training) ----
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

# ---- sequence construction (must match training caps) ----
MAX_IMAGE_TOKENS = 24 * 24
MAX_SCALAR_TOKENS = 512
MAX_SPEC_TOKENS_PER_GROUP = 512
MAX_SEQ_LEN = 2048

# ---- benchmark params ----
OUT_DIR = "bench_tokens_transformer"
SEED = 0
N_SAMPLES_LIMIT: Optional[int] = 50_000  # set None for all
BATCH_SIZE = 16

# IMPORTANT: safest is 0 because np.load(npz) inside workers can trip zipfile overlap checks.
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


# ============================================================
# VOCAB SPEC (match training, with optional ckpt offsets)
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

def build_vocab_spec_from_training_or_ckpt(ckpt: dict) -> VocabSpec:
    # training used fixed specials
    PAD, BOS, EOS, SEP, MISS = 0, 1, 2, 3, 4

    offsets = None
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if isinstance(cfg, dict):
        offsets = cfg.get("offsets", None)

    if isinstance(offsets, dict) and ("base_image" in offsets) and ("base_scalar" in offsets) and ("base_spectrum" in offsets):
        base_image = int(offsets["base_image"])
        base_scalar = int(offsets["base_scalar"])
        base_spectrum_raw = offsets["base_spectrum"]
        base_spectrum = {str(k): int(v) for k, v in base_spectrum_raw.items()}

        vocab_size = int(cfg.get("vocab_size", 0)) or (max(base_spectrum.values(), default=base_scalar) + V_SPECTRA)
        # Make sure it includes the last group span
        if base_spectrum:
            last_base = max(base_spectrum.values())
            vocab_size = max(vocab_size, last_base + V_SPECTRA)
        return VocabSpec(
            pad_id=PAD, bos_id=BOS, eos_id=EOS, sep_id=SEP, miss_id=MISS,
            base_image=base_image, base_scalar=base_scalar, base_spectrum=base_spectrum,
            vocab_size=vocab_size,
        )

    # fallback: rebuild exactly like training script
    next_id = 5
    base_image = next_id; next_id += V_IMAGE
    base_scalar = next_id; next_id += V_SCALAR
    base_spectrum = {}
    for g in SPECTRUM_GROUPS.keys():
        base_spectrum[g] = next_id
        next_id += V_SPECTRA
    return VocabSpec(
        pad_id=PAD, bos_id=BOS, eos_id=EOS, sep_id=SEP, miss_id=MISS,
        base_image=base_image, base_scalar=base_scalar, base_spectrum=base_spectrum,
        vocab_size=next_id,
    )


# ============================================================
# TOKEN TYPES (match training)
# ============================================================

TT_BOS = 0
TT_IMAGE = 1
TT_SCALAR = 2
TT_SPEC_BASE = 10  # + group index in SPECTRUM_GROUPS order


def compute_n_types_from_ckpt_state(state_dict: dict) -> int:
    # training used: n_types = TT_SPEC_BASE + len(SPECTRUM_GROUPS) + 5
    # but safest is to read embedding rows from checkpoint.
    w = state_dict.get("type_emb.weight", None)
    if w is None:
        return TT_SPEC_BASE + len(SPECTRUM_GROUPS) + 5
    return int(w.shape[0])


# ============================================================
# MODEL (must match training)
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
            norm_first=False,  # match training (and avoids nested tensor warning)
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

        key_padding_mask = ~attn_mask  # True where pad
        causal_mask = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1)

        h = self.enc(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        h = self.ln(h)
        return self.head(h)


# ============================================================
# SAFE NPZ ACCESS (avoid overlapped entries errors)
# ============================================================

class LazySpectrumNPZ:
    """
    Opens the spectrum npz lazily in the *current process* (important for DataLoader workers).
    """
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
# FIELD INDEXER (match training loading semantics)
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

        # spectrum (lazy per-process)
        self.spe_path: Optional[str] = None
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

        # image npz (load eagerly, small enough)
        ip = IMAGE_TOKENS_TMPL.format(field=self.field)
        if Path(ip).exists():
            npz = np.load(ip, allow_pickle=True)
            if "ids" in npz.files and "tokens_flat" in npz.files:
                ids = [norm_gaia_id(x) for x in np.asarray(npz["ids"], dtype=object)]
                toks = npz["tokens_flat"]
                self.img_tokens_flat = toks
                self.img_id_to_row = {gid: i for i, gid in enumerate(ids) if gid is not None}
            npz.close()

        # scalar npz (load eagerly)
        sp = SCALAR_TOKENS_TMPL.format(field=self.field)
        if Path(sp).exists():
            npz = np.load(sp, allow_pickle=True)
            if "ids" in npz.files and "scalar_tokens" in npz.files:
                ids = [norm_splus_id(x) for x in np.asarray(npz["ids"], dtype=object)]
                self.scalar_tokens = npz["scalar_tokens"]
                self.splus_id_to_scalar_row = {sid: i for i, sid in enumerate(ids) if sid}
            npz.close()
            
            if self.scalar_tokens is not None:
                df_ids = self.df["id"].astype(str).map(norm_splus_id).values
                hit = sum((sid in self.splus_id_to_scalar_row) for sid in df_ids[:5000])
                print(f"[scalar] {self.field}: scalar_rows={len(self.scalar_tokens)} "
                    f"df_rows={len(self.df)} hits_first5k={hit}/5000 "
                    f"example_df_id={df_ids[0] if len(df_ids) else None} "
                    f"example_npz_id={next(iter(self.splus_id_to_scalar_row.keys()), None)}")
            else:
                print(f"[scalar] {self.field}: NOT LOADED (file missing or keys mismatch)")

        # spectrum npz (LAZY open in each process)
        xp = SPECTRUM_TOKENS_TMPL.format(field=self.field)
        if Path(xp).exists():
            # read just ids and which groups exist using a short open, then close
            npz = np.load(xp, allow_pickle=True)
            if "ids" in npz.files:
                ids = [norm_splus_id(x) for x in np.asarray(npz["ids"], dtype=object)]
                self.splus_id_to_spec_row = {sid: i for i, sid in enumerate(ids) if sid}
                for g, (fk, ik) in SPECTRUM_GROUPS.items():
                    if fk in npz.files and ik in npz.files:
                        self.spec_groups_present[g] = (fk, ik)
                self.spe_path = xp
                self.spe_lazy = LazySpectrumNPZ(xp)
            npz.close()

    def __len__(self) -> int:
        return int(len(self.df))

    def get_row(self, idx: int) -> Dict[str, Any]:
        r = self.df.iloc[idx]
        sid = r["id"]
        gid = r["gaia_source_id"]

        # image tokens
        img_tokens = None
        if gid is not None and self.img_tokens_flat is not None:
            j = self.img_id_to_row.get(gid, None)
            if j is not None:
                img_tokens = self.img_tokens_flat[j]

        # scalar tokens
        scalar_tokens = None
        if self.scalar_tokens is not None:
            j = self.splus_id_to_scalar_row.get(sid, None)
            if j is not None:
                scalar_tokens = self.scalar_tokens[j]

        # spectrum tokens (lazy open in this process)
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

        return {"id": sid, "gaia_source_id": gid, "image_tokens": img_tokens, "scalar_tokens": scalar_tokens, "spectrum_tokens": spec}

    def close(self):
        if self.spe_lazy is not None:
            self.spe_lazy.close()


# ============================================================
# SEQUENCE BUILDER (exactly training)
# ============================================================

def build_sequence(sample: Dict[str, Any], vocab: VocabSpec) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[int, int]]]:
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

    # spectrum groups (fixed order)
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

    if len(toks) > MAX_SEQ_LEN:
        toks = toks[:MAX_SEQ_LEN]
        ttypes = ttypes[:MAX_SEQ_LEN]

    return np.asarray(toks, np.int64), np.asarray(ttypes, np.int64), seg


# ============================================================
# DATASET
# ============================================================

class MultiFieldDataset(Dataset):
    def __init__(self, fields: List[str], vocab: VocabSpec, limit: Optional[int] = None):
        self.vocab = vocab
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
        token_ids, type_ids, seg = build_sequence(s, self.vocab)
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
# MAIN BENCH
# ============================================================

def main():
    set_seed(SEED)
    ensure_dir(OUT_DIR)

    # ---- load ckpt ----
    ckpt_path = Path(CKPT_PATH)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # ---- build vocab from ckpt or training fallback ----
    vocab = build_vocab_spec_from_training_or_ckpt(ckpt if isinstance(ckpt, dict) else {})
    expected_vocab = 5 + V_IMAGE + V_SCALAR + V_SPECTRA * len(SPECTRUM_GROUPS)
    print(f"[vocab] vocab_size={vocab.vocab_size} (expected {expected_vocab})")
    if vocab.vocab_size != expected_vocab:
        print("[warn] vocab_size differs from the expected training layout. If this is unintended, fix ckpt config offsets.")

    # ---- model config ----
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    d_model = int(cfg.get("D_MODEL", 512))
    n_layers = int(cfg.get("N_LAYERS", 8))
    n_heads = int(cfg.get("N_HEADS", 8))
    d_ff = int(cfg.get("D_FF", 2048))
    max_len = int(cfg.get("MAX_SEQ_LEN", MAX_SEQ_LEN))

    # safest: match n_types to checkpoint tensor shape
    n_types = compute_n_types_from_ckpt_state(state)

    model = CausalTransformerLM(
        vocab_size=vocab.vocab_size,
        n_types=n_types,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=0.0,
        max_len=max_len,
    )
    # strict=True should work if vocab/type sizes match; keep strict=False for safety,
    # but if you want hard failures, set strict=True.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[ckpt] missing keys: {missing[:8]}{'...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"[ckpt] unexpected keys: {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")

    total_p, train_p = count_parameters(model)
    print(f"[model] d_model={d_model} layers={n_layers} heads={n_heads} ff={d_ff}")
    print(f"[model] n_types={n_types} max_len={max_len} vocab_size={vocab.vocab_size}")
    print(f"[model] params_total={total_p:,} params_trainable={train_p:,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"[env] device={device} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if device == "cuda":
        print(f"[env] torch.cuda.current_device()={torch.cuda.current_device()} name={torch.cuda.get_device_name(torch.cuda.current_device())}")

    # ---- fields ----
    fields = FIELDS or discover_fields_from_datacubes(DATACUBE_TMPL)
    if not fields:
        raise RuntimeError("No fields found.")
    print(f"[data] fields={len(fields)}")

    ds = MultiFieldDataset(fields, vocab=vocab, limit=N_SAMPLES_LIMIT)
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

    # "meaningful" accuracy excludes special + MISS
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

    pbar = tqdm(dl, desc="[bench]", dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            tokens = batch["tokens"].to(device, non_blocking=True)
            types = batch["types"].to(device, non_blocking=True)
            attn_mask = batch["attn_mask"].to(device, non_blocking=True)
            segs = batch["segments"]
            raws = batch["raws"]

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

            # exclude specials + MISS
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

                    # usage from ground truth, excluding PAD/MISS/special
                    yy_np = yy[vv_nomiss].detach().cpu().numpy()
                    if yy_np.size > 0:
                        if k == "image":
                            local = yy_np - vocab.base_image
                            local = local[(local >= 0) & (local < V_IMAGE)]
                        elif k == "scalar":
                            local = yy_np - vocab.base_scalar
                            local = local[(local >= 0) & (local < V_SCALAR)]
                        else:
                            local = yy_np - vocab.base_spectrum[k]
                            local = local[(local >= 0) & (local < V_SPECTRA)]
                        if local.size > 0:
                            usage[k] += np.bincount(local, minlength=usage[k].shape[0]).astype(np.int64)

            avg_loss = (sum_loss / max(1, sum_tokens))
            ppl = math.exp(avg_loss)
            acc = sum_correct / max(1, sum_total)
            acc_nm = sum_correct_nomiss / max(1, sum_total_nomiss)
            pbar.set_postfix({"ppl": f"{ppl:.2f}", "acc": f"{acc*100:.2f}%", "acc_nomiss": f"{acc_nm*100:.2f}%"})

    # ---- finalize ----
    avg_loss = sum_loss / max(1, sum_tokens)
    ppl = math.exp(avg_loss)
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
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "vocab_size": vocab.vocab_size,
            "n_types": n_types,
            "max_len": max_len,
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
    # 1) overall metrics
    plt.figure()
    plt.bar(["loss/token", "ppl", "acc% (all)", "acc% (no miss)"], [avg_loss, ppl, acc*100.0, acc_nm*100.0])
    plt.title("Overall benchmark")
    plt.tight_layout()
    plt.savefig(Path(OUT_DIR) / "overall.png", dpi=180)
    plt.close()

    # 2) segment accuracy (no miss)
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

    # 3) missing coverage
    miss = [results["coverage_missing_frac"][k] * 100 for k in seg_names]
    plt.figure(figsize=(10, 4))
    plt.bar(seg_names, miss)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Missing fraction (%)")
    plt.title("Missing coverage by segment")
    plt.tight_layout()
    plt.savefig(Path(OUT_DIR) / "segment_missing.png", dpi=180)
    plt.close()

    # 4) token usage top-k per segment
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

    # close dataset resources
    ds.close()

    print(f"[done] outputs in: {OUT_DIR}")
    print(f"[summary] ppl={ppl:.2f} acc(all)={acc*100:.2f}% acc(no-miss)={acc_nm*100:.2f}%")

if __name__ == "__main__":
    main()