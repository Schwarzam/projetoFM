#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train an AION-like token Transformer, with an auxiliary "meaningful value" loss.

Main idea:
  - Keep CE loss on tokens (discrete LM objective)
  - Add differentiable reconstruction losses:
      * image: expected centroid (softmax over image-vocab) vs true centroid
      * scalar: expected bin center vs true bin center
      * spectra: expected centroid vs true centroid
  - To keep it fast, the auxiliary loss samples up to K positions per modality per batch.

Speed improvements (H100):
  - TF32 matmul enabled
  - flash-sdp enabled (where possible)
  - torch.compile on the model (optional)
  - cached causal masks
  - faster startup via index-cache: you scan fields once, then reuse a cached mapping

Run:
  python3 train_transformer_tokens_meaningful.py
"""

from __future__ import annotations

# ---------------------------
# GPU + perf knobs (must be before torch import)
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
import joblib
import pickle
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
FIELDS: List[str] = []  # if empty -> auto-discover

DATACUBE_TMPL = "/home/astrodados4/downloads/hypercube/datacube_{field}.parquet"

IMAGE_TOKENS_TMPL    = "/home/schwarz/projetoFM/codecs/image_tokens/datacube_{field}_tokens.npz"
SCALAR_TOKENS_TMPL   = "/home/schwarz/projetoFM/scalar_tokenizers/datacube_{field}_scalar_tokens.npz"
SPECTRUM_TOKENS_TMPL = "/home/schwarz/projetoFM/spectrum_tokenizers/datacube_{field}_spectrum_tokens.npz"

SCALAR_TOKENIZER_CONFIG = "/home/schwarz/projetoFM/scalar_tokenizers/scalar_tokenizer_config.npz"

# ---- IMPORTANT: CODEBOOKS / CENTERS for meaningful loss ----
# These must exist. Adjust paths + key names if needed.
IMAGE_CODEBOOK_PATH   = "/home/schwarz/projetoFM/codecs/codebook_kmeans_2048.joblib"
SPECTRA_CODEBOOK_PATH = "/home/schwarz/projetoFM/spectrum_tokenizers/spectrum_tokenizer_config.npz"
# scalar centers are loaded from SCALAR_TOKENIZER_CONFIG

# Expected keys inside codebook NPZ:
#  - image: one of ["centroids","codebook","centers"]
#  - spectra: one of ["centroids","codebook","centers"]
IMAGE_CODEBOOK_KEY_CANDIDATES   = ("centroids", "codebook", "centers")
SPECTRA_CODEBOOK_KEY_CANDIDATES = ("centroids", "codebook", "centers")
SCALAR_CENTERS_KEY_CANDIDATES   = ("bin_centers", "centers", "quantile_centers", "centroids")

# ---- row filter ----
MAG_MIN = 14.0
MAG_MAX = 22.0
MAGERR_MAX = 2.0
MAG_COL = "mag_pstotal_r"
MAGERR_COL = "err_mag_pstotal_r"

# ---- token vocab sizes ----
V_IMAGE   = 2048
V_SCALAR  = 1024
V_SPECTRA = 2048

SPECTRUM_GROUPS = {
    "gaiaxp_bp": ("tokens_gaiaxp_bp_flat", "tokens_gaiaxp_bp_indptr"),
    "gaiaxp_rp": ("tokens_gaiaxp_rp_flat", "tokens_gaiaxp_rp_indptr"),
    "desi_b":    ("tokens_desi_b_flat",    "tokens_desi_b_indptr"),
    "desi_r":    ("tokens_desi_r_flat",    "tokens_desi_r_indptr"),
    "desi_z":    ("tokens_desi_z_flat",    "tokens_desi_z_indptr"),
}

# ---- sequence caps ----
MAX_IMAGE_TOKENS  = 24 * 24
MAX_SCALAR_TOKENS = 512
MAX_SPEC_TOKENS_PER_GROUP = 512
MAX_SEQ_LEN = 2048

# ---- training ----
OUT_DIR = "runs_tokens_transformer_meaningful"
SEED = 0

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
GRAD_ACCUM_STEPS = 1
LOG_EVERY = 50
SAVE_EVERY_STEPS = 5000

# model
D_MODEL  = 768
N_LAYERS = 12
N_HEADS  = 12
D_FF     = 3072
DROPOUT  = 0.1

# ---- meaningful loss knobs ----
# total_loss = CE + ALPHA_IMG*L_img + ALPHA_SCA*L_sca + ALPHA_SPEC*L_spec
ALPHA_IMG  = 0.10
ALPHA_SCA  = 0.10
ALPHA_SPEC = 0.10

# sample up to K token positions per modality per batch for auxiliary loss (speed!)
AUX_SAMPLE_K = 4096

# compile for speed (PyTorch 2.x). If it errors, set False.
USE_TORCH_COMPILE = True

# startup speed: cache dataset index (field,row)
INDEX_CACHE_PATH = "cache_token_dataset_index.npz"


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

def _load_codebook(path: str,
                   key_candidates: Tuple[str, ...] = ("centroids", "codebook", "centers", "cluster_centers_", "clusters")) -> np.ndarray:
    """
    Load a codebook/centroids array from:
      - .npz  (keys like centroids/codebook/centers)
      - .joblib / .pkl (sklearn KMeans or dict-like)
    Returns: np.ndarray float32 with shape (K, D)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing: {path}")

    suffix = p.suffix.lower()

    # ---- NPZ ----
    if suffix == ".npz":
        z = np.load(str(p), allow_pickle=True)
        try:
            for k in key_candidates:
                if k in z.files:
                    arr = np.asarray(z[k])
                    if arr.ndim != 2:
                        raise ValueError(f"Key {k} in {path} has shape {arr.shape}, expected (K,D)")
                    return arr.astype(np.float32)
            raise KeyError(f"No key in {path} among {key_candidates}. Keys: {z.files}")
        finally:
            z.close()

    # ---- JOBLIB ----
    if suffix == ".joblib":
        obj = joblib.load(str(p))

        # sklearn KMeans-like
        if hasattr(obj, "cluster_centers_"):
            arr = np.asarray(obj.cluster_centers_)
            if arr.ndim != 2:
                raise ValueError(f"cluster_centers_ in {path} has shape {arr.shape}, expected (K,D)")
            return arr.astype(np.float32)

        # dict-like
        if isinstance(obj, dict):
            for k in key_candidates:
                if k in obj:
                    arr = np.asarray(obj[k])
                    if arr.ndim != 2:
                        raise ValueError(f"Key {k} in {path} has shape {arr.shape}, expected (K,D)")
                    return arr.astype(np.float32)

        raise TypeError(f"Don't know how to extract centroids from joblib object type={type(obj)} in {path}")

    # ---- PKL ----
    if suffix in (".pkl", ".pickle"):
        with open(str(p), "rb") as f:
            obj = pickle.load(f)

        if hasattr(obj, "cluster_centers_"):
            arr = np.asarray(obj.cluster_centers_)
            if arr.ndim != 2:
                raise ValueError(f"cluster_centers_ in {path} has shape {arr.shape}, expected (K,D)")
            return arr.astype(np.float32)

        if isinstance(obj, dict):
            for k in key_candidates:
                if k in obj:
                    arr = np.asarray(obj[k])
                    if arr.ndim != 2:
                        raise ValueError(f"Key {k} in {path} has shape {arr.shape}, expected (K,D)")
                    return arr.astype(np.float32)

        raise TypeError(f"Don't know how to extract centroids from pickle object type={type(obj)} in {path}")

    raise ValueError(f"Unsupported codebook format: {path} (expected .npz/.joblib/.pkl)")

def _load_npz_key(npz_path: str, key_candidates: Tuple[str, ...]) -> np.ndarray:
    if not Path(npz_path).exists():
        raise FileNotFoundError(f"Missing: {npz_path}")
    z = np.load(npz_path, allow_pickle=True)
    try:
        for k in key_candidates:
            if k in z.files:
                arr = z[k]
                return np.asarray(arr)
    finally:
        z.close()
    raise KeyError(f"No key in {npz_path} among {key_candidates}. Keys are: {z.files}")

def _load_scalar_centers(npz_path: str) -> np.ndarray:
    if not Path(npz_path).exists():
        raise FileNotFoundError(f"Missing: {npz_path}")
    z = np.load(npz_path, allow_pickle=True)
    try:
        for k in SCALAR_CENTERS_KEY_CANDIDATES:
            if k in z.files:
                c = np.asarray(z[k]).astype(np.float32)
                c = c.reshape(-1)
                return c
    finally:
        z.close()
    raise KeyError(f"No scalar centers key found in {npz_path}. Tried {SCALAR_CENTERS_KEY_CANDIDATES}.")


# ============================================================
# TOKEN OFFSETS + SPECIAL TOKENS
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

def build_vocab_spec() -> VocabSpec:
    PAD, BOS, EOS, SEP, MISS = 0, 1, 2, 3, 4
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

        # spectrum (keep open in main process; if workers break, set NUM_WORKERS=0)
        self.spec_npz: Optional[np.lib.npyio.NpzFile] = None
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
            z = np.load(ip, allow_pickle=True)
            try:
                if "ids" in z.files and "tokens_flat" in z.files:
                    ids = [norm_gaia_id(x) for x in np.asarray(z["ids"], dtype=object)]
                    self.img_tokens_flat = z["tokens_flat"]
                    self.img_id_to_row = {gid: i for i, gid in enumerate(ids) if gid is not None}
            finally:
                z.close()

        # scalar
        sp = SCALAR_TOKENS_TMPL.format(field=self.field)
        if Path(sp).exists():
            z = np.load(sp, allow_pickle=True)
            try:
                if "ids" in z.files and "scalar_tokens" in z.files:
                    ids = [norm_splus_id(x) for x in np.asarray(z["ids"], dtype=object)]
                    self.scalar_tokens = z["scalar_tokens"]
                    self.splus_id_to_scalar_row = {sid: i for i, sid in enumerate(ids) if sid}
            finally:
                z.close()

        # spectrum
        xp = SPECTRUM_TOKENS_TMPL.format(field=self.field)
        if Path(xp).exists():
            z = np.load(xp, allow_pickle=True)
            if "ids" in z.files:
                ids = [norm_splus_id(x) for x in np.asarray(z["ids"], dtype=object)]
                self.splus_id_to_spec_row = {sid: i for i, sid in enumerate(ids) if sid}
                for g, (fk, ik) in SPECTRUM_GROUPS.items():
                    if fk in z.files and ik in z.files:
                        self.spec_groups_present[g] = (fk, ik)
                self.spec_npz = z
            else:
                z.close()

    def __del__(self):
        try:
            if self.spec_npz is not None:
                self.spec_npz.close()
        except Exception:
            pass

    def __len__(self):
        return int(len(self.df)) if self.df is not None else 0

    def get_row(self, idx: int) -> Dict[str, Any]:
        r = self.df.iloc[idx]
        sid: str = r["id"]
        gid: Optional[int] = r["gaia_source_id"]

        img_tokens = None
        if gid is not None and self.img_tokens_flat is not None:
            j = self.img_id_to_row.get(gid)
            if j is not None:
                img_tokens = self.img_tokens_flat[j]

        scalar_tokens = None
        if self.scalar_tokens is not None:
            j = self.splus_id_to_scalar_row.get(sid)
            if j is not None:
                scalar_tokens = self.scalar_tokens[j]

        spec = {}
        if self.spec_npz is not None:
            j = self.splus_id_to_spec_row.get(sid)
            if j is not None:
                for g, (fk, ik) in self.spec_groups_present.items():
                    flat = self.spec_npz[fk]
                    indp = self.spec_npz[ik]
                    a = int(indp[j]); b = int(indp[j + 1])
                    spec[g] = flat[a:b]

        return {
            "image_tokens": img_tokens,
            "scalar_tokens": scalar_tokens,
            "spectrum_tokens": spec,
        }


# ============================================================
# SEQUENCE BUILDER
# ============================================================

TT_BOS = 0
TT_IMAGE = 1
TT_SCALAR = 2
TT_SPEC_BASE = 10

def build_sequence(sample: Dict[str, Any], vocab: VocabSpec) -> Tuple[np.ndarray, np.ndarray]:
    toks: List[int] = [vocab.bos_id]
    ttypes: List[int] = [TT_BOS]

    # image
    toks.append(vocab.sep_id); ttypes.append(TT_BOS)
    img = sample["image_tokens"]
    if img is None:
        toks.append(vocab.miss_id); ttypes.append(TT_IMAGE)
    else:
        arr = np.asarray(img, dtype=np.int64).reshape(-1)[:MAX_IMAGE_TOKENS]
        toks.extend((vocab.base_image + arr).tolist())
        ttypes.extend([TT_IMAGE] * arr.size)

    # scalar
    toks.append(vocab.sep_id); ttypes.append(TT_BOS)
    sca = sample["scalar_tokens"]
    if sca is None:
        toks.append(vocab.miss_id); ttypes.append(TT_SCALAR)
    else:
        arr = np.asarray(sca, dtype=np.int64).reshape(-1)[:MAX_SCALAR_TOKENS]
        arr = np.clip(arr, 0, V_SCALAR - 1)
        toks.extend((vocab.base_scalar + arr).tolist())
        ttypes.extend([TT_SCALAR] * arr.size)

    # spectrum groups
    spec: Dict[str, np.ndarray] = sample["spectrum_tokens"] or {}
    for gi, g in enumerate(SPECTRUM_GROUPS.keys()):
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

    if len(toks) > MAX_SEQ_LEN:
        toks = toks[:MAX_SEQ_LEN]
        ttypes = ttypes[:MAX_SEQ_LEN]

    return np.asarray(toks, np.int64), np.asarray(ttypes, np.int64)


# ============================================================
# DATASET with index-cache (fixes "takes forever to start")
# ============================================================

class MultiFieldTokenDataset(Dataset):
    """
    Startup issue before: you were loading ALL FieldIndex objects for 2444 fields upfront.
    Now:
      - We build (or load) an index list of (field, row) once.
      - We lazily instantiate FieldIndex per field on first access and keep a small LRU cache.
    """

    def __init__(self, fields: List[str], vocab: VocabSpec, index_cache: str = INDEX_CACHE_PATH, lru_fields: int = 8):
        self.vocab = vocab
        self.fields = fields
        self.index_cache = index_cache
        self.lru_fields = lru_fields

        self.index_field: List[str] = []
        self.index_row: np.ndarray = np.empty((0,), dtype=np.int32)

        self._field_cache: Dict[str, FieldIndex] = {}
        self._field_cache_order: List[str] = []

        self._load_or_build_index()

    def _load_or_build_index(self):
        p = Path(self.index_cache)
        if p.exists():
            z = np.load(str(p), allow_pickle=True)
            self.index_field = list(z["field"].astype(str))
            self.index_row = z["row"].astype(np.int32)
            print(f"[index] loaded {len(self.index_row)} samples from {p}")
            return

        print("[index] building index (first time; will be cached)...")
        t0 = time.time()
        fields_out = []
        rows_out = []

        for f in tqdm(self.fields, desc="[index-scan]", dynamic_ncols=True):
            dp = DATACUBE_TMPL.format(field=f)
            if not Path(dp).exists():
                continue
            # read just the needed columns to filter
            df = pd.read_parquet(dp, columns=[MAG_COL, MAGERR_COL])
            mask = ((df[MAG_COL] > MAG_MIN) & (df[MAG_COL] < MAG_MAX) & (df[MAGERR_COL] < MAGERR_MAX)).values
            idxs = np.nonzero(mask)[0]
            if idxs.size == 0:
                continue
            fields_out.extend([f] * int(idxs.size))
            rows_out.append(idxs.astype(np.int32))

        if rows_out:
            row_all = np.concatenate(rows_out, axis=0)
        else:
            row_all = np.empty((0,), dtype=np.int32)

        self.index_field = fields_out
        self.index_row = row_all

        # shuffle once here (and DataLoader shuffle=True can stay off if you want)
        perm = np.random.permutation(len(self.index_row))
        self.index_row = self.index_row[perm]
        self.index_field = [self.index_field[i] for i in perm.tolist()]

        np.savez_compressed(str(p), field=np.array(self.index_field, dtype=object), row=self.index_row)
        print(f"[index] built {len(self.index_row)} samples in {(time.time()-t0)/60:.1f} min; cached to {p}")

    def __len__(self):
        return int(len(self.index_row))

    def _get_field_index(self, field: str) -> FieldIndex:
        if field in self._field_cache:
            # refresh LRU
            if field in self._field_cache_order:
                self._field_cache_order.remove(field)
            self._field_cache_order.append(field)
            return self._field_cache[field]

        fi = FieldIndex(field)
        self._field_cache[field] = fi
        self._field_cache_order.append(field)

        # evict LRU if needed
        while len(self._field_cache_order) > self.lru_fields:
            old = self._field_cache_order.pop(0)
            try:
                del self._field_cache[old]
            except Exception:
                pass

        return fi

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        field = self.index_field[idx]
        row_i = int(self.index_row[idx])
        fi = self._get_field_index(field)
        sample = fi.get_row(row_i)
        token_ids, type_ids = build_sequence(sample, self.vocab)
        return {"token_ids": token_ids, "type_ids": type_ids}


def collate_batch(batch: List[Dict[str, Any]], pad_id: int):
    max_len = max(len(x["token_ids"]) for x in batch)
    B = len(batch)
    tokens = torch.full((B, max_len), pad_id, dtype=torch.long)
    types  = torch.zeros((B, max_len), dtype=torch.long)
    attn_mask = torch.zeros((B, max_len), dtype=torch.bool)
    for i, ex in enumerate(batch):
        t = torch.from_numpy(ex["token_ids"])
        y = torch.from_numpy(ex["type_ids"])
        L = t.numel()
        tokens[i, :L] = t
        types[i, :L] = y
        attn_mask[i, :L] = True
    return {"tokens": tokens, "types": types, "attn_mask": attn_mask}


# ============================================================
# MODEL (cached causal mask + optional compile)
# ============================================================

class CausalTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, n_types: int, d_model: int, n_layers: int, n_heads: int,
                 d_ff: int, dropout: float, max_len: int):
        super().__init__()
        self.vocab_size = vocab_size
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
        self.drop = nn.Dropout(dropout)

        self._mask_cache: Dict[int, torch.Tensor] = {}

    def _causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        m = self._mask_cache.get(L, None)
        if m is None or m.device != device:
            m = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
            self._mask_cache[L] = m
        return m

    def forward(self, tokens: torch.Tensor, types: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        if L > self.max_len:
            raise ValueError(f"Sequence length {L} > max_len {self.max_len}")

        pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
        x = self.tok_emb(tokens) + self.type_emb(types) + self.pos_emb(pos)
        x = self.drop(x)

        key_padding_mask = ~attn_mask
        causal_mask = self._causal_mask(L, tokens.device)

        h = self.enc(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        h = self.ln(h)
        return self.head(h)


# ============================================================
# LOSS: CE + meaningful auxiliary losses
# ============================================================

def _sample_positions(mask: torch.Tensor, k: int) -> torch.Tensor:
    """
    mask: (B, L) bool
    returns indices into flattened (B*L,) positions
    """
    idx = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)
    if idx.numel() == 0:
        return idx
    if idx.numel() <= k:
        return idx
    # uniform sample without replacement
    perm = torch.randperm(idx.numel(), device=idx.device)[:k]
    return idx[perm]

def meaningful_aux_losses(
    logits: torch.Tensor,  # (B, L, V)
    y: torch.Tensor,       # (B, L)
    vocab: VocabSpec,
    img_codebook: torch.Tensor,     # (V_IMAGE, Dimg)
    spec_codebook: torch.Tensor,    # (V_SPECTRA, Dspec)
    scalar_centers: torch.Tensor,   # (V_SCALAR,)
    aux_k: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Differentiable expected-centroid reconstruction losses.
    We only compute on a sampled subset of positions per modality for speed.
    """

    stats = {"aux_img": 0.0, "aux_sca": 0.0, "aux_spec": 0.0}
    device = logits.device
    B, L, V = logits.shape

    # ignore PAD/MISS/special in aux losses
    specials = torch.tensor([vocab.pad_id, vocab.bos_id, vocab.eos_id, vocab.sep_id, vocab.miss_id],
                            device=device, dtype=torch.long)
    is_special = (y[..., None] == specials[None, None, :]).any(dim=-1)

    # ---- IMAGE positions ----
    img_lo = vocab.base_image
    img_hi = vocab.base_image + V_IMAGE
    mask_img = (~is_special) & (y >= img_lo) & (y < img_hi)
    idx_img = _sample_positions(mask_img, aux_k)
    loss_img = torch.tensor(0.0, device=device)

    if idx_img.numel() > 0:
        y_flat = y.reshape(-1)[idx_img]
        # logits restricted to image vocab
        logits_img = logits.reshape(-1, V)[:, img_lo:img_hi][idx_img]  # (N, V_IMAGE)
        probs = torch.softmax(logits_img.float(), dim=-1).to(img_codebook.dtype)  # (N, V_IMAGE)
        pred = probs @ img_codebook  # (N, Dimg)
        true = img_codebook[(y_flat - img_lo).clamp(0, V_IMAGE - 1)]
        loss_img = F.mse_loss(pred, true)
        stats["aux_img"] = float(loss_img.detach().cpu().item())

    # ---- SCALAR positions ----
    sc_lo = vocab.base_scalar
    sc_hi = vocab.base_scalar + V_SCALAR
    mask_sca = (~is_special) & (y >= sc_lo) & (y < sc_hi)
    idx_sca = _sample_positions(mask_sca, aux_k)
    loss_sca = torch.tensor(0.0, device=device)

    if idx_sca.numel() > 0:
        y_flat = y.reshape(-1)[idx_sca]
        logits_sca = logits.reshape(-1, V)[:, sc_lo:sc_hi][idx_sca]  # (N, V_SCALAR)
        probs = torch.softmax(logits_sca.float(), dim=-1).to(scalar_centers.dtype)  # (N, V_SCALAR)
        pred = (probs * scalar_centers.unsqueeze(0)).sum(dim=-1)  # (N,)
        true = scalar_centers[(y_flat - sc_lo).clamp(0, V_SCALAR - 1)]
        loss_sca = F.mse_loss(pred, true)
        stats["aux_sca"] = float(loss_sca.detach().cpu().item())

    # ---- SPECTRA positions (all spectrum groups share V_SPECTRA; offsets differ) ----
    loss_spec_total = torch.tensor(0.0, device=device)
    n_used = 0
    for g, base in vocab.base_spectrum.items():
        sp_lo = base
        sp_hi = base + V_SPECTRA
        mask_sp = (~is_special) & (y >= sp_lo) & (y < sp_hi)
        idx_sp = _sample_positions(mask_sp, max(1, aux_k // max(1, len(vocab.base_spectrum))))
        if idx_sp.numel() == 0:
            continue

        y_flat = y.reshape(-1)[idx_sp]
        logits_sp = logits.reshape(-1, V)[:, sp_lo:sp_hi][idx_sp]  # (N, V_SPECTRA)
        probs = torch.softmax(logits_sp.float(), dim=-1).to(spec_codebook.dtype)
        pred = probs @ spec_codebook
        true = spec_codebook[(y_flat - sp_lo).clamp(0, V_SPECTRA - 1)]
        loss_g = F.mse_loss(pred, true)
        loss_spec_total = loss_spec_total + loss_g
        n_used += 1

    if n_used > 0:
        loss_spec_total = loss_spec_total / n_used
        stats["aux_spec"] = float(loss_spec_total.detach().cpu().item())

    return loss_img, loss_sca, loss_spec_total, stats


# ============================================================
# TRAIN LOOP
# ============================================================

def lr_schedule(step: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = step - warmup
    T = 200_000
    frac = min(1.0, t / max(1, T))
    cos = 0.5 * (1 + math.cos(math.pi * frac))
    return base_lr * (0.1 + 0.9 * cos)

def train():
    set_seed(SEED)
    ensure_dir(OUT_DIR)

    # perf knobs for H100
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    except Exception:
        pass

    vocab = build_vocab_spec()
    n_types = TT_SPEC_BASE + len(SPECTRUM_GROUPS) + 5

    # fields
    fields = FIELDS or discover_fields_from_datacubes(DATACUBE_TMPL)
    if not fields:
        raise RuntimeError("No fields found. Set FIELDS or fix DATACUBE_TMPL.")
    print(f"[data] fields={len(fields)}")

    # load codebooks / centers (CPU -> move to GPU once)
    img_cb_np  = _load_codebook(IMAGE_CODEBOOK_PATH)
    spec_cb_np = _load_npz_key(SPECTRA_CODEBOOK_PATH, SPECTRA_CODEBOOK_KEY_CANDIDATES).astype(np.float32)
    sc_cent_np = _load_scalar_centers(SCALAR_TOKENIZER_CONFIG).astype(np.float32)

    if img_cb_np.shape[0] != V_IMAGE:
        raise ValueError(f"IMAGE codebook first dim {img_cb_np.shape[0]} != V_IMAGE={V_IMAGE}")
    if spec_cb_np.shape[0] != V_SPECTRA:
        raise ValueError(f"SPECTRA codebook first dim {spec_cb_np.shape[0]} != V_SPECTRA={V_SPECTRA}")
    if sc_cent_np.shape[0] != V_SCALAR:
        raise ValueError(f"SCALAR centers dim {sc_cent_np.shape[0]} != V_SCALAR={V_SCALAR}")

    ds = MultiFieldTokenDataset(fields, vocab=vocab, index_cache=INDEX_CACHE_PATH, lru_fields=8)

    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS > 0),
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        collate_fn=lambda b: collate_batch(b, pad_id=vocab.pad_id),
        drop_last=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

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

    if USE_TORCH_COMPILE and device == "cuda":
        try:
            model = torch.compile(model, mode="max-autotune")
            print("[perf] torch.compile enabled")
        except Exception as e:
            print(f"[perf] torch.compile failed, continuing without it: {e}")

    total_p, train_p = count_parameters(model)
    print(f"[model] d_model={D_MODEL} layers={N_LAYERS} heads={N_HEADS} ff={D_FF}")
    print(f"[model] vocab_size={vocab.vocab_size} n_types={n_types} max_len={MAX_SEQ_LEN}")
    print(f"[model] params_total={total_p:,} params_trainable={train_p:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # move codebooks to GPU once
    img_codebook = torch.from_numpy(img_cb_np).to(device=device, dtype=torch.float16 if use_amp else torch.float32)
    spec_codebook = torch.from_numpy(spec_cb_np).to(device=device, dtype=torch.float16 if use_amp else torch.float32)
    scalar_centers = torch.from_numpy(sc_cent_np).to(device=device, dtype=torch.float16 if use_amp else torch.float32)

    global_step = 0
    model.train()

    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(dl, desc=f"epoch {epoch}/{EPOCHS}", dynamic_ncols=True)
        for it, batch in enumerate(pbar):
            tokens = batch["tokens"].to(device, non_blocking=True)
            types  = batch["types"].to(device, non_blocking=True)
            attn_mask = batch["attn_mask"].to(device, non_blocking=True)

            # next-token setup
            x = tokens[:, :-1]
            x_types = types[:, :-1]
            x_mask = attn_mask[:, :-1]
            y = tokens[:, 1:].contiguous()

            lr_now = lr_schedule(global_step, LR, WARMUP_STEPS)
            for pg in opt.param_groups:
                pg["lr"] = lr_now

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x, x_types, x_mask)  # (B, L-1, V)

                # CE (discrete)
                loss_ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=vocab.pad_id,
                )

                # meaningful auxiliary losses (differentiable expected centroid / expected center)
                loss_img, loss_sca, loss_spec, aux_stats = meaningful_aux_losses(
                    logits=logits,
                    y=y,
                    vocab=vocab,
                    img_codebook=img_codebook,
                    spec_codebook=spec_codebook,
                    scalar_centers=scalar_centers,
                    aux_k=AUX_SAMPLE_K,
                )

                loss = loss_ce + (ALPHA_IMG * loss_img) + (ALPHA_SCA * loss_sca) + (ALPHA_SPEC * loss_spec)

            loss_val = float(loss.detach().cpu().item())
            ce_val   = float(loss_ce.detach().cpu().item())

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
                    pbar.set_postfix({
                        "loss": f"{loss_val:.3f}",
                        "ce": f"{ce_val:.3f}",
                        "aux_img": f"{aux_stats['aux_img']:.3f}",
                        "aux_sca": f"{aux_stats['aux_sca']:.3f}",
                        "aux_spec": f"{aux_stats['aux_spec']:.3f}",
                        "lr": f"{lr_now:.2e}",
                    })

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
                            "loss": {
                                "ALPHA_IMG": ALPHA_IMG,
                                "ALPHA_SCA": ALPHA_SCA,
                                "ALPHA_SPEC": ALPHA_SPEC,
                                "AUX_SAMPLE_K": AUX_SAMPLE_K,
                            }
                        },
                    }
                    outp = Path(OUT_DIR) / f"ckpt_step_{global_step:08d}.pt"
                    torch.save(ckpt, outp)
                    tqdm.write(f"[save] {outp}")

        outp = Path(OUT_DIR) / f"ckpt_epoch_{epoch:03d}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch, "global_step": global_step}, outp)
        tqdm.write(f"[save] {outp}")


if __name__ == "__main__":
    train()