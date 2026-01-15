#!/usr/bin/env python3
# coding: utf-8

from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader

from astromodal.config import load_config
from astromodal.datatypes import SplusCuts
from astromodal.datasets.spluscuts import SplusCutoutsDataset
from astromodal.models.autoencoder import AutoEncoder

from tqdm import tqdm
import glob


def main():
    config = load_config("/home/schwarz/projetoFM/config.yaml")

    model_path = Path(config["models_folder"]) / "./autoencoder_model_silu.pth"
    model = AutoEncoder.load_from_file(str(model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    files = [Path(p) for p in glob.glob(config["datacubes_paths"])]
    
    # add tqdm progress bar
    for file in tqdm(
        files,
        desc="Processing datacube files",
        unit="file",
    ):
        # field should be basename without datacube_ and .parquet
        field = file.stem.replace("datacube_", "")
        path = str(file)
        
        outfile = Path(config["hdd_folder"]) / "image_latents" / f"{field}.parquet"
        if outfile.exists():
            continue

        bands = ["F378", "F395", "F410", "F430", "F515", "F660", "F861", "R", "I", "Z", "U", "G"]

        cut_cols = [f"splus_cut_{b}" for b in bands]

        columns = ["id", "mag_psf_r"] + cut_cols
        df = pl.read_parquet(path, columns=columns, use_pyarrow=True)
        
        df = df.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in cut_cols]))
        df = df.filter(pl.col("mag_psf_r") < 21)
        
        # if df is empty, skip
        if df.is_empty():
            continue
        
        bands = ["F378", "F395", "F410", "F430", "F515", "F660", "F861", "R", "I", "Z", "U", "G"]
        cutout_size = 96
        batch_size = 1024

        dataset = SplusCutoutsDataset(
            df,
            bands=bands,
            img_size=cutout_size,
            return_valid_mask=True,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=14,
            pin_memory=True,
        )

        ids_out = []
        latents_out = []

        model.eval()
        with torch.no_grad():
            for i, (x_norm, m_valid) in enumerate(loader):
                latents = model.encode(x_norm.to(device))
                latents = latents.reshape(latents.shape[0], -1)

                start = i * batch_size
                end = start + latents.shape[0]

                ids = df["id"][start:end].to_numpy()

                ids_out.extend(ids)
                latents_out.extend(latents.cpu().numpy())

        df_latents = pl.DataFrame(
            {
                "id": ids_out,
                "latent": latents_out,
            }
        )

        outfile = Path(config["hdd_folder"]) / "image_latents" / f"{field}.parquet"
        outfile.parent.mkdir(parents=True, exist_ok=True)

        df_latents.write_parquet(outfile)

if __name__ == "__main__":
    main()