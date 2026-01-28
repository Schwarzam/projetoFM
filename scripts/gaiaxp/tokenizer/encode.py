#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import ast
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from astromodal.config import load_config
from astromodal.tokenizers.spectralrvq import SpectralPatchRVQ
from astromodal.tokenizers.rvq import ResidualVQ
from astromodal.scalers.standardvec import StandardScalerVec


# =============================================================================
# CONFIG
# =============================================================================
CONFIG_PATH = "/home/schwarz/projetoFM/config.yaml"

IN_SUBFOLDER = ""   # inside config["hdd_folder"]
IN_SUFFIX = ".parquet"

OUT_SUBFOLDER = "gaiaxp_codes_bp_rp"
OUT_SUFFIX = "_codes.parquet"

ID_COL = "id"
BP_COL = "gaiaxp_bp_coefficients"
RP_COL = "gaiaxp_rp_coefficients"

# Tokenizers (trained)
TOK_BP_PATH = "/home/schwarz/projetoFM/outputs/tokenizers/gaiaxp_spectral_rvq_bp.pt"
TOK_RP_PATH = "/home/schwarz/projetoFM/outputs/tokenizers/gaiaxp_spectral_rvq_rp.pt"

# Scalers (fitted)
SCALER_BP_PATH = "/home/schwarz/projetoFM/outputs/scalers/gaiaxp_scaler_bp.pkl.npz"
SCALER_RP_PATH = "/home/schwarz/projetoFM/outputs/scalers/gaiaxp_scaler_rp.pkl.npz"

L_EXPECTED = 55
PATCH_SIZE = 1   # must match training
CHANNELS = 1

BATCH_SIZE = 4096
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SKIP_BAD_ROWS = True
# =============================================================================


# -------------------------
# RVQ rebuild
# -------------------------
def make_rvq(cfg: dict) -> ResidualVQ:
    dim = cfg.get("dim", cfg.get("D", None))
    if dim is None:
        raise ValueError("RVQ config missing 'dim' or 'D'")
    return ResidualVQ(
        dim=int(dim),
        num_stages=int(cfg.get("num_stages", 3)),
        codebook_size=int(cfg.get("codebook_size", 1024)),
        decay=float(cfg.get("decay", 0.99)),
    )


def _infer_out_path(infile: Path, out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    name = infile.name
    if name.endswith(IN_SUFFIX):
        name = name[: -len(IN_SUFFIX)] + OUT_SUFFIX
    else:
        name = infile.stem + OUT_SUFFIX
    return out_root / name


def _parse_coeff_cell(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple, np.ndarray)):
                return list(v)
        except Exception:
            return None
    return None


def _df_to_coeff_matrices(df: pl.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Returns (ids, Xbp, Xrp) with shapes:
      ids: list[str] length N_valid
      Xbp, Xrp: float32 [N_valid,55]
    Keeps only rows where BOTH BP and RP are valid.
    """
    ids_out = []
    rows_bp = []
    rows_rp = []

    id_s = df[ID_COL].to_list()
    bp_s = df[BP_COL].to_list()
    rp_s = df[RP_COL].to_list()

    for i in range(len(id_s)):
        bp = _parse_coeff_cell(bp_s[i])
        rp = _parse_coeff_cell(rp_s[i])

        ok = (
            bp is not None and rp is not None and
            len(bp) == L_EXPECTED and len(rp) == L_EXPECTED
        )
        if not ok:
            if SKIP_BAD_ROWS:
                continue
            raise ValueError(f"Bad coeff at row {i}: bp_len={None if bp is None else len(bp)}, rp_len={None if rp is None else len(rp)}")

        bp = np.asarray(bp, dtype=np.float32)
        rp = np.asarray(rp, dtype=np.float32)

        if (not np.all(np.isfinite(bp))) or (not np.all(np.isfinite(rp))):
            if SKIP_BAD_ROWS:
                continue
            raise ValueError(f"Non-finite coeffs at row {i}")

        ids_out.append(str(id_s[i]))
        rows_bp.append(bp)
        rows_rp.append(rp)

    if not rows_bp:
        return [], np.zeros((0, L_EXPECTED), dtype=np.float32), np.zeros((0, L_EXPECTED), dtype=np.float32)

    Xbp = np.stack(rows_bp, axis=0).astype(np.float32)
    Xrp = np.stack(rows_rp, axis=0).astype(np.float32)
    return ids_out, Xbp, Xrp


class GaiaXPCoeffPairDataset(Dataset):
    """
    Returns:
      xbp_norm: [55,1] float32
      xrp_norm: [55,1] float32
      id: str
    """
    def __init__(self, ids: list[str], Xbp_norm: np.ndarray, Xrp_norm: np.ndarray):
        assert Xbp_norm.shape == Xrp_norm.shape
        assert Xbp_norm.ndim == 2 and Xbp_norm.shape[1] == L_EXPECTED
        self.ids = ids
        self.Xbp = Xbp_norm.astype(np.float32)
        self.Xrp = Xrp_norm.astype(np.float32)

    def __len__(self):
        return self.Xbp.shape[0]

    def __getitem__(self, i: int):
        bp = self.Xbp[i].reshape(L_EXPECTED, 1)
        rp = self.Xrp[i].reshape(L_EXPECTED, 1)
        return torch.from_numpy(bp), torch.from_numpy(rp), self.ids[i]


def collate_pair(batch):
    bp = torch.stack([b[0] for b in batch], dim=0)  # [B,55,1]
    rp = torch.stack([b[1] for b in batch], dim=0)  # [B,55,1]
    ids = [b[2] for b in batch]
    return bp, rp, ids


@torch.no_grad()
def encode_codes_one_file_bprp(
    infile: Path,
    tok_bp: SpectralPatchRVQ,
    tok_rp: SpectralPatchRVQ,
    scaler_bp: StandardScalerVec,
    scaler_rp: StandardScalerVec,
    device: str,
) -> pl.DataFrame:
    """
    Output columns:
      id
      bp_codes  (list: [55][R])
      rp_codes  (list: [55][R])
    """
    df = pl.read_parquet(infile, columns=[ID_COL, BP_COL, RP_COL], use_pyarrow=True)
    if df.height == 0:
        return pl.DataFrame({ID_COL: [], "bp_codes": [], "rp_codes": []})

    ids, Xbp, Xrp = _df_to_coeff_matrices(df)
    if len(ids) == 0:
        return pl.DataFrame({ID_COL: [], "bp_codes": [], "rp_codes": []})

    # normalize
    Xbp_n = scaler_bp.transform_x(Xbp).astype(np.float32)  # [N,55]
    Xrp_n = scaler_rp.transform_x(Xrp).astype(np.float32)  # [N,55]

    ds = GaiaXPCoeffPairDataset(ids, Xbp_n, Xrp_n)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_pair,
    )

    tok_bp.eval().to(device)
    tok_rp.eval().to(device)

    ids_out: list[str] = []
    bp_codes_out: list[list[list[int]]] = []
    rp_codes_out: list[list[list[int]]] = []

    for bp_norm, rp_norm, ids_batch in dl:
        bp_norm = bp_norm.to(device, non_blocking=True).float()
        rp_norm = rp_norm.to(device, non_blocking=True).float()

        enc_bp = tok_bp.encode(bp_norm, update_ema=False)
        enc_rp = tok_rp.encode(rp_norm, update_ema=False)

        bp_codes = enc_bp["codes"].detach().cpu().numpy().astype(np.int32)  # [B,55,R]
        rp_codes = enc_rp["codes"].detach().cpu().numpy().astype(np.int32)  # [B,55,R]

        B = bp_codes.shape[0]
        for i in range(B):
            ids_out.append(ids_batch[i])
            bp_codes_out.append(bp_codes[i].tolist())
            rp_codes_out.append(rp_codes[i].tolist())

    return pl.DataFrame({ID_COL: ids_out, "bp_codes": bp_codes_out, "rp_codes": rp_codes_out})


def main():
    config = load_config(CONFIG_PATH)
    
    in_root = Path(config["datacubes_paths"])
    
    out_root = Path(config["hdd_folder"]) / OUT_SUBFOLDER
    out_root.mkdir(parents=True, exist_ok=True)

    # load tokenizer + scalers
    tok_bp = SpectralPatchRVQ.load_from_file(Path(TOK_BP_PATH), rvq_ctor=make_rvq, map_location=DEVICE)
    tok_rp = SpectralPatchRVQ.load_from_file(Path(TOK_RP_PATH), rvq_ctor=make_rvq, map_location=DEVICE)

    scaler_bp = StandardScalerVec.load(Path(SCALER_BP_PATH))
    scaler_rp = StandardScalerVec.load(Path(SCALER_RP_PATH))

    files = sorted(Path(in_root).parent.glob(Path(in_root).name))
    if len(files) == 0:
        raise RuntimeError(f"No parquet files found in {in_root} matching '*{IN_SUFFIX}'")

    for f in tqdm(files, desc="Encoding GaiaXP BP+RP -> codes"):
        outfile = _infer_out_path(f, out_root)
        try:
            df_codes = encode_codes_one_file_bprp(
                f, tok_bp, tok_rp, scaler_bp, scaler_rp, DEVICE
            )
            df_codes.write_parquet(outfile)
        except Exception as e:
            print(f"[warn] failed {f.name}: {e}")
            continue

    print(f"Done. Wrote codes to: {out_root}")


if __name__ == "__main__":
    main()