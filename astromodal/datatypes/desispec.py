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
from astromodal.datasets.desi_spectra import DesiSpectraDataset, desi_collate_pad_flux_only
from astromodal.datasets.desi_spectra import LatentsPolarsDataset, collate_latents_pad_to_patch_multiple
from astromodal.tokenizers.spectralrvq import SpectralPatchRVQ
from astromodal.tokenizers.rvq import ResidualVQ


# -----------------------------
# small utils
# -----------------------------
def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


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
# Dataset over codes parquet
# -----------------------------
class DesiCodesDataset(Dataset):
    """
    Reads a codes parquet already loaded as a Polars DF with columns:
      - id: str
      - codes: nested list [[...],[...],...] => [T][R]
      - L: int (latent length or original length used for decode)
    Returns python objects; collate will pad.
    """
    def __init__(self, df: pl.DataFrame, id_col: str = "id", codes_col: str = "codes", L_col: str = "L"):
        self.df = df
        self.id_col = id_col
        self.codes_col = codes_col
        self.L_col = L_col

        self.ids = self.df[self.id_col].to_list()

    def __len__(self) -> int:
        return self.df.height

    def __getitem__(self, i: int) -> Dict[str, Any]:
        _id = self.df[self.id_col][i]
        codes = self.df[self.codes_col][i]  # python nested list [T][R]
        L = self.df[self.L_col][i] if self.L_col in self.df.columns else None

        # normalize types
        _id = str(_id)
        L = int(L) if L is not None else None
        # codes -> np int32 [T,R]
        codes = np.array(codes, dtype=np.int32)

        return {"id": _id, "codes": codes, "L": L}


def desi_codes_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pads codes along T dimension:
      codes: [B, Tpad, R]
      mask:  [B, Tpad]  (True where valid)
      L:     [B]        (int, for decode)
    """
    B = len(batch)
    R = int(batch[0]["codes"].shape[1]) if B > 0 else 0
    Ts = [int(s["codes"].shape[0]) for s in batch]
    Tpad = max(Ts) if B > 0 else 0

    codes_pad = torch.full((B, Tpad, R), -1, dtype=torch.long)
    mask_pad = torch.zeros((B, Tpad), dtype=torch.bool)

    ids = []
    Ls = []

    for i, s in enumerate(batch):
        c = torch.from_numpy(s["codes"]).long()  # [T,R]
        t = c.shape[0]
        codes_pad[i, :t] = c
        mask_pad[i, :t] = True
        ids.append(s["id"])
        # fallback: if L not present, use t*patch_size later; but keep as None-safe
        Ls.append(int(s["L"]) if s["L"] is not None else -1)

    L_t = torch.tensor(Ls, dtype=torch.long)
    return {"id": ids, "codes": codes_pad, "mask": mask_pad, "L": L_t, "T": torch.tensor(Ts, dtype=torch.long)}


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


class DesiDataType:
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

    # -------- naming helpers --------
    def latents_path_for_datacube(self, datacube_file: Path) -> Path:
        return self.latents_root / f"{datacube_file.stem}_latents.parquet"

    def codes_path_for_latents(self, latents_file: Path) -> Path:
        name = latents_file.name
        if name.endswith("_latents.parquet"):
            name = name[: -len("_latents.parquet")] + "_codes.parquet"
        else:
            name = latents_file.stem + "_codes.parquet"
        return self.codes_root / name

    # -------- step 1: datacube -> latents (id, latent, L) --------
    @torch.no_grad()
    def encode_latents_file(self, datacube_file: Path, columns: List[str], batch_size: int = 128) -> pl.DataFrame:
        """
        Output DF:
          - id: str
          - latent: nested list [T][C]  (IMPORTANT: keep 2D so we preserve T)
          - L: int  (here L=T; length along time axis)
        """
        df = pl.read_parquet(datacube_file, columns=columns, use_pyarrow=True)

        # filters consistent with your script
        if "mag_psf_r" in df.columns:
            df = df.filter(pl.col("mag_psf_r") < self.cfg.mag_r_max)
        if "desi_TARGET_RA" in df.columns:
            df = df.filter(pl.col("desi_TARGET_RA").is_not_null())

        desi_cols = [c for c in df.columns if "desi" in c]
        if len(desi_cols) == 0 or df.height == 0:
            return pl.DataFrame({"id": [], "latent": [], "L": []})

        ds = DesiSpectraDataset(df, scaler_path=self.scaler_path)
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=desi_collate_pad_flux_only,
            pin_memory=True,
        )

        model = self.ae()

        ids_all = df["id"].cast(pl.Utf8).to_list()
        cursor = 0

        ids_out: List[str] = []
        latents_out: List[List[List[float]]] = []
        L_out: List[int] = []

        for batch in dl:
            # batch[0] = x, batch[3] = lengths (depending on collate)
            x = batch[0].to(self.device, non_blocking=True).float()  # [B, L, 1]
            z = model.encode(x)  # expect [B, C, T] or [B, T, C] depending on your AE

            z_np = z.detach().float().cpu().numpy()

            # Normalize to [B, T, C]
            if z_np.ndim != 3:
                raise ValueError(f"Unexpected latent shape: {z_np.shape}")

            if z_np.shape[1] == self.cfg.latent_dim:
                # [B, C, T] -> [B, T, C]
                z_np = np.transpose(z_np, (0, 2, 1))

            B = z_np.shape[0]
            batch_ids = ids_all[cursor:cursor + B]
            cursor += B

            for i in range(B):
                zz = z_np[i]  # [T, C]
                ids_out.append(str(batch_ids[i]))
                latents_out.append(zz.astype(np.float32).tolist())
                L_out.append(int(zz.shape[0]))  # L = T

        return pl.DataFrame({"id": ids_out, "latent": latents_out, "L": L_out})

    # -------- step 2: latents -> codes (id, codes, L) --------
    @torch.no_grad()
    def encode_codes_file(self, latents_file: Path) -> pl.DataFrame:
        """
        Reads latents parquet with (id, latent, L), runs tok.encode, saves:
          - id
          - codes: [Ttok][R]
          - L: original latent length (for decode)
        """
        lat_df = pl.read_parquet(latents_file, columns=["id", "latent", "L"], use_pyarrow=True)
        if lat_df.height == 0:
            return pl.DataFrame({"id": [], "codes": [], "L": []})

        ds = LatentsPolarsDataset(
            lat_df,
            id_col="id",
            latent_col="latent",
            latent_dim=self.cfg.latent_dim,
            nested_layout="auto",
        )

        dl = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: collate_latents_pad_to_patch_multiple(batch, patch_size=self.cfg.patch_size),
        )

        tok = self.tok()

        ids_out: List[str] = []
        codes_out: List[List[List[int]]] = []
        L_out: List[int] = []

        # Build quick id -> L map for exact per-object L
        # (LatentsPolarsDataset should preserve order, but map is safe)
        L_map = dict(zip(lat_df["id"].to_list(), lat_df["L"].to_list()))

        for x_pad, mask_pad, lengths, ids in dl:
            x_pad = x_pad.to(self.device, non_blocking=True).float()
            enc = tok.encode(x_pad, update_ema=False)
            codes = enc["codes"]  # [B, Ttok, R]

            codes_np = codes.detach().cpu().numpy().astype(np.int32)
            B = codes_np.shape[0]

            for i in range(B):
                _id = str(ids[i])
                ids_out.append(_id)
                codes_out.append(codes_np[i].tolist())
                # IMPORTANT: store original latent L (not padded)
                L_out.append(int(L_map.get(_id, -1)))

        return pl.DataFrame({"id": ids_out, "codes": codes_out, "L": L_out})

    # -------- I/O convenience: run full pipeline for one file --------
    def process_datacube_to_codes(self, datacube_file: Path, columns: List[str], ae_batch_size: int = 128) -> Path:
        lat_df = self.encode_latents_file(datacube_file, columns=columns, batch_size=ae_batch_size)
        lat_path = self.latents_path_for_datacube(datacube_file)
        lat_df.write_parquet(lat_path)

        codes_df = self.encode_codes_file(lat_path)
        codes_path = self.codes_path_for_latents(lat_path)
        codes_df.write_parquet(codes_path)
        return codes_path

    # -------- training loader over a codes parquet --------
    def make_codes_loader(self, codes_file: Path, batch_size: Optional[int] = None, shuffle: bool = False):
        df = pl.read_parquet(codes_file, columns=["id", "codes", "L"], use_pyarrow=True)
        ds = DesiCodesDataset(df, id_col="id", codes_col="codes", L_col="L")
        dl = DataLoader(
            ds,
            batch_size=(batch_size or self.cfg.batch_size),
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=desi_codes_collate,
        )
        return ds, dl

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