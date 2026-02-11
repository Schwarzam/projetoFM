#!/usr/bin/env python3
# coding: utf-8

from pathlib import Path

import polars as pl
import torch

from astromodal.config import load_config
from astromodal.tokenizers.spatialrvq import SpatialRVQ


def main():
    # -----------------------------
    # Configurable variables
    # -----------------------------
    CONFIG_PATH = "/home/schwarz/projetoFM/config.yaml"
    SRVQ_CKPT_NAME = "srvq_spluscuts_1stage_2048codebs.pth"

    INPUT_SUBFOLDER = "image_latents"          # folder with per-field latents parquet
    OUTPUT_SUBFOLDER = "image_codes"           # where to write codes parquet
    MAX_FILES = None                           # set int for a subset, or None for all

    CHANNELS = 2
    H = 24
    W = 24
    BATCH_SIZE = 256                           # chunk size for encoding
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Load config
    # -----------------------------
    config = load_config(CONFIG_PATH)

    in_folder = Path(config["hdd_folder"]) / INPUT_SUBFOLDER
    out_folder = Path(config["hdd_folder"]) / OUTPUT_SUBFOLDER
    out_folder.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(config["models_folder"]) / SRVQ_CKPT_NAME

    # -----------------------------
    # Load SRVQ
    # -----------------------------
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    srvq = SpatialRVQ.load_from_file(str(ckpt_path), map_location=device).to(device).eval()

    # -----------------------------
    # Encode all files
    # -----------------------------
    for i, file in enumerate(sorted(in_folder.glob("*.parquet"))):
        if MAX_FILES is not None and i >= MAX_FILES:
            break

        df = pl.read_parquet(file, use_pyarrow=True)

        lat = torch.tensor(df["latent"].to_numpy(), dtype=torch.float32)
        n = lat.shape[0]

        ids_out = []
        codes_out = []

        with torch.no_grad():
            for start in range(0, n, BATCH_SIZE):
                end = min(start + BATCH_SIZE, n)

                z_map = lat[start:end].to(device).view(-1, CHANNELS, H, W)
                out = srvq.encode(z_map)
                codes = out["codes"].detach().cpu().to(torch.int32).numpy()  # [B,H,W,R]
                codes = codes.reshape(codes.shape[0], -1)                   # [B, H*W*R]
                codes_out.extend(codes.tolist())                            # list[list[int]]

                ids_out.extend(df["id"][start:end].to_list())

        df_codes = pl.DataFrame(
            {
                "id": ids_out,
                "codes": pl.Series("codes", codes_out, dtype=pl.List(pl.Int32)),
            }
        )

        outfile = out_folder / file.name
        df_codes.write_parquet(outfile)

        print(f"[ok] {file.name} -> {outfile.name} ({len(ids_out)} rows)")

if __name__ == "__main__":
    main()