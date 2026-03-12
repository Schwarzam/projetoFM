from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader

# your project imports
from astromodal.config import load_config
from astromodal.models.spectral_autoencoder import SpectraAutoEncoder
from astromodal.tokenizers.spectralrvq import SpectralPatchRVQ
from astromodal.tokenizers.rvq import ResidualVQ
from astromodal.datatypes.base import BaseDataType

def make_rvq(cfg: dict) -> ResidualVQ:
    dim = cfg.get("dim", cfg.get("D", None))
    if dim is None:
        raise ValueError("RVQ config missing 'dim'")
    return ResidualVQ(
        dim=int(dim),
        num_stages=int(cfg.get("num_stages", 3)),
        codebook_size=int(cfg.get("codebook_size", 1024)),
        decay=float(cfg.get("decay", 0.99)),
    )

# -----------------------------
# DESI DataType Handler
# -----------------------------
@dataclass
class DesiDataTypeConfig:
    config_path: str
    scaler_relpath: str = "scalers/desi_flux_standard_scaler.npz"  # relative to models_folder
    mag_r_max: float = 21.0

    # folders in hdd_folder
    latents_subfolder: str = "spectra_latents_desi"
    codes_subfolder: str = "spectra_codes_desi"

    # model file names in models_folder (defaults)
    ae_name: str = "spectra_autoencoder_desi.pth"
    tok_name: str = "spectral_patch_rvq_desi.pth"

    # tokenizer params
    patch_size: int = 2
    latent_dim: int = 16  # must match how LatentsPolarsDataset interprets data

    # dataloader
    batch_size: int = 256
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DesiDataType(BaseDataType):
    """
    One-stop pipeline + loader for DESI.
    - encode datacubes -> latents parquet (id, latent, L)
    - encode latents  -> codes parquet (id, codes, L)
    - make dataset/loader for codes
    - decode support via (codes, L)
    """
    def __init__(self, cfg: DesiDataTypeConfig):
        self.cfg = cfg
        self.config = load_config(cfg.config_path)
        self.device = cfg.device

        self.models_folder = Path(self.config["models_folder"])
        self.hdd_folder = Path(self.config["hdd_folder"])

        self.scaler_path = self.models_folder / cfg.scaler_relpath

        self.latents_root = self.hdd_folder / cfg.latents_subfolder
        self.codes_root = self.hdd_folder / cfg.codes_subfolder
        self.latents_root.mkdir(parents=True, exist_ok=True)
        self.codes_root.mkdir(parents=True, exist_ok=True)

        self._ae: Optional[SpectraAutoEncoder] = None
        self._tok: Optional[SpectralPatchRVQ] = None

    # -------- model loaders (cached) --------
    def ae(self) -> SpectraAutoEncoder:
        if self._ae is None:
            p = self.models_folder / self.cfg.ae_name
            self._ae = SpectraAutoEncoder.load_from_file(p, map_location=self.device)
            self._ae.eval().to(self.device)
        return self._ae

    def tok(self) -> SpectralPatchRVQ:
        if self._tok is None:
            p = self.models_folder / self.cfg.tok_name
            self._tok = SpectralPatchRVQ.load_from_file(p, rvq_ctor=make_rvq, map_location=self.device)
            self._tok.eval().to(self.device)
        return self._tok

    # -------- decode support: codes + L --------
    @torch.no_grad()
    def decode_codes_batch(self, codes: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """
        codes: [B, Tpad, R] with -1 padding
        L:     [B] original latent length (needed by decode)
        Returns reconstructed latents (shape depends on your tokenizer decode API).
        """
        tok = self.tok()

        # tok.decode expects codes WITHOUT padding; handle per-item
        outs = []
        for i in range(codes.shape[0]):
            # take only valid tokens (where first stage != -1)
            valid = (codes[i, :, 0] >= 0)
            c_i = codes[i, valid].unsqueeze(0).to(self.device)  # [1, Tvalid, R]
            L_i = int(L[i].item())
            out_i = tok.decode(c_i, L=L_i)  # <-- this is what you asked
            outs.append(out_i.detach().cpu())

        return torch.cat(outs, dim=0)