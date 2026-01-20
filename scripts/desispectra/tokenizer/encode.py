#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from astromodal.config import load_config
from astromodal.datasets.datacubes import load_datacube_files
from astromodal.datasets.desi_spectra import LatentsPolarsDataset, collate_latents_pad_to_patch_multiple
from astromodal.tokenizers.spectralrvq import SpectralPatchRVQ
from astromodal.tokenizers.rvq import ResidualVQ


# =============================================================================
# CONFIG
# =============================================================================
CONFIG_PATH = "/home/schwarz/projetoFM/config.yaml"

# Input: folder produced by the AE-latents encoding script
IN_SUBFOLDER = "spectra_latents_desi"          # inside config["hdd_folder"]
IN_SUFFIX = "_latents.parquet"                # matches prior AE script output

# Output: where to write codes
OUT_SUBFOLDER = "spectra_codes_desi"          # inside config["hdd_folder"]
OUT_SUFFIX = "_codes.parquet"                 # same name pattern as before, but different suffix

# Tokenizer model checkpoint (trained in previous step)
TOK_NAME_DEFAULT = "spectral_patch_rvq_desi.pth"
TOK_PATH = None  # if None: Path(config["models_folder"]) / TOK_NAME_DEFAULT

# Latent storage info
LATENT_DIM = 16
LATENT_COL = "latent"
ID_COL = "id"

# Tok params (MUST match the trained tokenizer)
PATCH_SIZE = 2

# DataLoader
BATCH_SIZE = 256
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# If True, only process files corresponding to datacubes list (train+val);
# if False, process every parquet in IN_SUBFOLDER.
USE_DATACUBE_FILELIST = False
TRAIN_VAL_SPLIT = 0.9
NFILES_SUBSAMPLE = None
# =============================================================================


def _infer_out_path(infile: Path, out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    # keep same stem pattern as input, but swap suffix
    name = infile.name
    if name.endswith(IN_SUFFIX):
        name = name[: -len(IN_SUFFIX)] + OUT_SUFFIX
    else:
        name = infile.stem + OUT_SUFFIX
    return out_root / name


@torch.no_grad()
def encode_codes_one_latent_file(latent_file: Path, tok: SpectralPatchRVQ, device: str) -> pl.DataFrame:
    """
    Reads a single AE-latents parquet (id, latent), runs tokenizer encode,
    and writes per-object codes.

    Output DF columns:
      - id
      - codes  (nested list: [T][R] ints)
    """
    lat_df = pl.read_parquet(latent_file, columns=[ID_COL, LATENT_COL], use_pyarrow=True)
    if lat_df.height == 0:
        return pl.DataFrame({ID_COL: [], "codes": []})

    ds = LatentsPolarsDataset(
        lat_df,
        id_col=ID_COL,
        latent_col=LATENT_COL,
        latent_dim=LATENT_DIM,
        nested_layout="auto",
    )

    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=lambda batch: collate_latents_pad_to_patch_multiple(batch, patch_size=PATCH_SIZE),
    )

    tok.eval().to(device)

    ids_out = []
    codes_out = []

    for x_pad, mask_pad, lengths, ids in dl:
        # x_pad: [B, Lpad, C]
        x_pad = x_pad.to(device, non_blocking=True).float()

        enc = tok.encode(x_pad, update_ema=False)
        codes = enc["codes"]  # [B, T, R]

        codes_np = codes.detach().cpu().numpy()  # int-like
        B, T, R = codes_np.shape

        # lengths here are latent-lengths (before padding), but tok internally uses L_orig
        # which equals Lpad cropped to original L in patchify; we just store full T codes
        for i in range(B):
            ids_out.append(ids[i])
            codes_out.append(codes_np[i].astype(np.int32).tolist())  # [T][R]

    return pl.DataFrame({ID_COL: ids_out, "codes": codes_out})


def main():
    config = load_config(CONFIG_PATH)

    in_root = Path(config["hdd_folder"]) / IN_SUBFOLDER
    out_root = Path(config["hdd_folder"]) / OUT_SUBFOLDER

    if not in_root.exists():
        raise FileNotFoundError(f"Input latents folder not found: {in_root}")

    # Load tokenizer
    tok_path = Path(TOK_PATH) if TOK_PATH else Path(config["models_folder"]) / TOK_NAME_DEFAULT
    tok = SpectralPatchRVQ.load_from_file(tok_path, map_location=DEVICE)
    tok.eval().to(DEVICE)

    # Decide which latent files to process
    if USE_DATACUBE_FILELIST:
        train_files, val_files = load_datacube_files(
            config["datacubes_paths"],
            train_val_split=TRAIN_VAL_SPLIT,
            nfiles_subsample=NFILES_SUBSAMPLE,
        )
        # map datacube file -> latent file by name pattern used in AE script
        lat_files = []
        for f in (train_files + val_files):
            f = Path(f)
            lat_file = in_root / (f.stem + IN_SUFFIX)
            if lat_file.exists():
                lat_files.append(lat_file)
        files = lat_files
    else:
        files = sorted(in_root.glob(f"*{IN_SUFFIX}"))

    if len(files) == 0:
        raise RuntimeError(f"No input files found in {in_root} matching '*{IN_SUFFIX}'")

    # Process one-by-one and save with same naming pattern
    for lf in tqdm(files, desc="Tokenizing latent files"):
        outfile = _infer_out_path(Path(lf), out_root)
        try:
            df_codes = encode_codes_one_latent_file(Path(lf), tok, DEVICE)
            df_codes.write_parquet(outfile)
        except Exception:
            continue

    print(f"Done. Wrote codes to: {out_root}")


if __name__ == "__main__":
    main()