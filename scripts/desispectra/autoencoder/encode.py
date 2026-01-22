#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from astromodal.config import load_config
from tqdm import tqdm
import polars as pl
from pathlib import Path
import numpy as np
import torch

from astromodal.datasets.datacubes import load_datacube_files
from astromodal.models.spectral_autoencoder import SpectraAutoEncoder
from astromodal.datasets.desi_spectra import DesiSpectraDataset, desi_collate_pad_flux_only
from torch.utils.data import DataLoader


# =============================================================================
# CONFIG (edit what you need)
# =============================================================================
CONFIG_PATH = "/home/schwarz/projetoFM/config.yaml"

MODEL_PATH = None  # if None, uses Path(config["models_folder"]) / "spectra_autoencoder_desi.pth"
MODEL_NAME_DEFAULT = "spectra_autoencoder_desi.pth"

OUT_SUBFOLDER = "spectra_latents_desi"   # will be created inside hdd_folder or models_folder
OUT_ROOT_USE = "hdd_folder"             # "hdd_folder" or "models_folder"
OUT_SUFFIX = "_latents"                 # appended before .parquet

BATCH_SIZE = 128
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# If True, process only the subsampled train/val list; if False, glob all files from datacubes_paths
USE_LOAD_DATACUBE_FILES = True
NFILES_SUBSAMPLE = None   # e.g. 800 or None for all returned by helper
TRAIN_VAL_SPLIT = 0.9

# Filtering (keep consistent with training if you want)
MAG_R_MAX = 21


SCALER_PATH = Path("scalers") / "desi_flux_standard_scaler.npz"

# =============================================================================


def _infer_out_path(infile: Path, out_root: Path) -> Path:
    # same base name, different folder, plus suffix
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root / (infile.stem + OUT_SUFFIX + ".parquet")


@torch.no_grad()
def encode_one_file(
    infile: Path,
    model: SpectraAutoEncoder,
    device: str,
    columns: list[str],
) -> pl.DataFrame:
    """
    Reads one parquet file, builds a dataset over its rows, encodes latents,
    returns a Polars DF with columns: id, latent (flattened list[float]).
    """
    df = pl.read_parquet(infile, columns=columns, use_pyarrow=True)

    # basic filters (as in training)
    if "mag_psf_r" in df.columns:
        df = df.filter(pl.col("mag_psf_r") < MAG_R_MAX)
    if "desi_TARGET_RA" in df.columns:
        df = df.filter(pl.col("desi_TARGET_RA").is_not_null())

    # ensure has DESI content
    desi_cols = [c for c in df.columns if "desi" in c]
    if len(desi_cols) == 0 or df.height == 0:
        return pl.DataFrame({"id": [], "latent": []})

    ds = DesiSpectraDataset(df, scaler_path=Path(load_config(CONFIG_PATH)["models_folder"]) / SCALER_PATH)

    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=desi_collate_pad_flux_only,
        pin_memory=True,
    )

    ids_out = []
    latents_out = []

    model.eval().to(device)

    # We need ids aligned with dataset order. Dataset uses df.row(i).
    # We'll read ids in the same order and slice per batch.
    ids_all = df["id"].to_list()
    cursor = 0

    for batch in dl:
        # expected: x, mask, w, lengths   (stats optional, but return_stats=False)
        # handle both possible returns robustly
        if isinstance(batch, (tuple, list)):
            x = batch[0]
            lengths = batch[3] if len(batch) >= 4 else None
        else:
            x = batch
            lengths = None

        x = x.to(device, non_blocking=True).float()  # [B, L, 1]
        z = model.encode(x)                          # [B, latent_dim, T]

        z_np = z.detach().float().cpu().numpy()      # [B, C, T]
        B = z_np.shape[0]

        # align ids for this batch
        batch_ids = ids_all[cursor:cursor + B]
        cursor += B

        # store as flattened vectors (C*T) to keep a simple list column
        for i in range(B):
            ids_out.append(batch_ids[i])
            latents_out.append(z_np[i].reshape(-1).astype(np.float32).tolist())

    return pl.DataFrame({"id": ids_out, "latent": latents_out})


def main():
    config = load_config(CONFIG_PATH)

    # Resolve model path
    model_path = Path(MODEL_PATH) if MODEL_PATH else Path(config["models_folder"]) / MODEL_NAME_DEFAULT
    model = SpectraAutoEncoder.load_from_file(model_path, map_location=DEVICE)
    model.eval().to(DEVICE)

    # Probe schema once to select only needed columns
    probe_file = Path(config["datacubes_paths"].replace("*", "STRIPE82-0002"))
    header = pl.read_parquet(str(probe_file), n_rows=0)

    # Keep only needed columns: DESI + id + mag_psf_r (same logic you used)
    columns = [c for c in header.columns if "desi" in c] + ["id", "mag_psf_r"]

    # Determine input files list
    if USE_LOAD_DATACUBE_FILES:
        train_files, val_files = load_datacube_files(
            config["datacubes_paths"],
            train_val_split=TRAIN_VAL_SPLIT,
            nfiles_subsample=NFILES_SUBSAMPLE,
        )
        files = [Path(f) for f in (train_files + val_files)]
    else:
        # simple glob expansion
        files = sorted(Path().glob(config["datacubes_paths"]))

    # Output root folder
    out_root_base = Path(config[OUT_ROOT_USE])
    out_root = out_root_base / OUT_SUBFOLDER
    out_root.mkdir(parents=True, exist_ok=True)

    # Process each input file one-by-one, write corresponding output parquet
    for f in tqdm(files, desc="Encoding files"):
        infile = Path(f)
        outfile = _infer_out_path(infile, out_root)

        try:
            lat_df = encode_one_file(infile, model, DEVICE, columns)
            lat_df.write_parquet(outfile)
        except Exception as e:
            # skip failures but keep going
            continue


if __name__ == "__main__":
    main()