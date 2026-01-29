#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from astromodal.config import load_config
from astromodal.tokenizers.spectralrvq import SpectralPatchRVQ
from astromodal.scalers.scaler1d import StandardScaler1D


# =============================================================================
# CONFIG
# =============================================================================
CONFIG_PATH = "/home/schwarz/projetoFM/config.yaml"

OUT_SUBFOLDER = "scalar_codes"
OUT_SUFFIX = "_scalar_codes.parquet"

ID_COL = "id"

SCALERS_DIR = Path("/home/schwarz/projetoFM/outputs/scalers")
TOK_DIR     = Path("/home/schwarz/projetoFM/outputs/tokenizers")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8192

WRITE_MASK_COLUMNS = True
WRITE_RAW_COLUMNS  = False
# =============================================================================


def _infer_out_path(infile: Path, out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root / (infile.stem + OUT_SUFFIX)


def _load_scalar_columns(scalers_dir: Path, tok_dir: Path) -> List[str]:
    cols = set()
    for p in scalers_dir.glob("*.npz"):
        cols.add(p.stem)
    for p in tok_dir.glob("*.pt"):
        cols.add(p.stem)
    return sorted(cols)


def _safe_read_parquet_columns(
    path: Path,
    columns: List[str],
) -> pl.DataFrame:
    schema = pl.read_parquet_schema(path)
    available = set(schema.keys())

    cols_to_read = [c for c in columns if c in available]
    if ID_COL not in cols_to_read:
        cols_to_read = [ID_COL] + cols_to_read

    df = pl.read_parquet(path, columns=cols_to_read, use_pyarrow=True)

    missing = [c for c in columns if c not in df.columns]
    if missing:
        df = df.with_columns([pl.lit(None).alias(c) for c in missing])

    return df.select([ID_COL] + [c for c in columns if c != ID_COL])


def _load_scaler(col: str) -> Optional[StandardScaler1D]:
    p = SCALERS_DIR / f"{col}.npz"
    if not p.exists():
        return None
    return StandardScaler1D.load(p)


def _load_tokenizer(col: str, device: str) -> Optional[SpectralPatchRVQ]:
    p = TOK_DIR / f"{col}.pt"
    if not p.exists():
        return None
    tok = SpectralPatchRVQ.load_from_file(p, map_location=device)
    tok.eval().to(device)
    return tok


@torch.no_grad()
def _encode_scalar_column(
    values: np.ndarray,          # [N]
    scaler: StandardScaler1D,
    tok: SpectralPatchRVQ,
    *,
    device: str,
    batch_size: int,
) -> Tuple[List[Optional[List[int]]], np.ndarray]:
    v = values.astype(np.float32, copy=False)
    mask = np.isfinite(v)

    out_codes: List[Optional[List[int]]] = [None] * len(v)
    if scaler is None or tok is None:
        return out_codes, mask

    idx = np.where(mask)[0]
    if idx.size == 0:
        return out_codes, mask

    vn = scaler.transform_x(v[idx]).astype(np.float32)
    xn = torch.from_numpy(vn.reshape(-1, 1, 1))

    for j0 in range(0, xn.shape[0], batch_size):
        xb = xn[j0:j0 + batch_size].to(device=device, non_blocking=True)
        enc = tok.encode(xb, update_ema=False)
        codes = enc["codes"][:, 0, :].cpu().numpy().astype(np.int32)

        sl = idx[j0:j0 + codes.shape[0]]
        for i, ii in enumerate(sl):
            out_codes[int(ii)] = codes[i].tolist()

    return out_codes, mask


def encode_scalars_one_file(
    infile: Path,
    *,
    scalar_columns: List[str],
    device: str,
    batch_size: int,
) -> pl.DataFrame:

    df = _safe_read_parquet_columns(infile, [ID_COL] + scalar_columns)
    if df.height == 0:
        return pl.DataFrame({ID_COL: []})

    out: Dict[str, List] = {
        ID_COL: df[ID_COL].cast(pl.Utf8).to_list()
    }

    for col in scalar_columns:
        scaler = _load_scaler(col)
        tok = _load_tokenizer(col, device)

        s = df.select(pl.col(col).cast(pl.Float64, strict=False)).to_series()
        v = s.to_numpy().astype(np.float64, copy=False)

        codes, mask = _encode_scalar_column(
            v, scaler, tok, device=device, batch_size=batch_size
        )

        out[f"codes__{col}"] = codes
        if WRITE_MASK_COLUMNS:
            out[f"mask__{col}"] = mask.tolist()
        if WRITE_RAW_COLUMNS:
            out[f"raw__{col}"] = v.tolist()

    return pl.DataFrame(out)


def main():
    config = load_config(CONFIG_PATH)

    pattern = Path(config["datacubes_paths"])
    files = sorted(pattern.parent.glob(pattern.name))
    if not files:
        raise RuntimeError(f"No files matched: {pattern}")

    scalar_columns = _load_scalar_columns(SCALERS_DIR, TOK_DIR)
    if not scalar_columns:
        raise RuntimeError("No scalar columns found (no scalers/tokenizers).")

    out_root = Path(config["hdd_folder"]) / OUT_SUBFOLDER
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Files: {len(files)}")
    print(f"Scalar columns: {len(scalar_columns)}")

    for f in tqdm(files, desc="Encoding scalar columns"):
        try:
            df_out = encode_scalars_one_file(
                f,
                scalar_columns=scalar_columns,
                device=DEVICE,
                batch_size=BATCH_SIZE,
            )
            df_out.write_parquet(_infer_out_path(f, out_root))
        except Exception as e:
            print(f"[warn] {f.name}: {e}")

    print(f"Done. Output in {out_root}")


if __name__ == "__main__":
    main()