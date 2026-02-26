from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from astromodal.config import load_config
from astromodal.datatypes.base import BaseDataType
from astromodal.datasets.spluscuts import SplusCutoutsDataset
from astromodal.models.autoencoder import AutoEncoder
from astromodal.tokenizers.spatialrvq import SpatialRVQ


PathLike = Union[str, Path]
RawInput = Union[pl.DataFrame, PathLike]  # raw datacube (id + mag + cutouts)
CodesInput = Union[pl.DataFrame, PathLike, Sequence[int], Sequence[Sequence[int]]]  # file/df/single/many


@dataclass
class SplusCuts(BaseDataType):
    """
    Straight pipeline:
      raw -> AE.encode -> latents -> SRVQ.encode -> codes
      codes -> SRVQ.decode -> latents -> AE.decode -> recon

    encode(raw) returns: pl.DataFrame(id, codes[List[Int32]])
    decode(codes) returns: torch.Tensor [N, bands, 96, 96]
      - accepts: parquet path, dataframe, single code array, or list of code arrays
    """

    __metadata__ = {
        "id": "splus_cuts",
        "description": "S-PLUS photometric cuts data type",
        "bands": ["F378", "F395", "F410", "F430", "F515", "F660", "F861", "R", "I", "Z", "U", "G"],
        "cutout_size": 96,
        "columns": [
            "splus_cut_F378", "splus_cut_F395", "splus_cut_F410", "splus_cut_F430",
            "splus_cut_F515", "splus_cut_F660", "splus_cut_F861", "splus_cut_R",
            "splus_cut_I", "splus_cut_Z", "splus_cut_U", "splus_cut_G",
        ],
    }

    # config / checkpoints
    config_path: PathLike = "/home/schwarz/projetoFM/config.yaml"
    ae_ckpt: PathLike = "autoencoder_model_silu.pth"
    srvq_ckpt: PathLike = "srvq_spluscuts_1stage_2048codebs.pth"

    # same filters you used
    mag_col: str = "mag_psf_r"
    mag_cut: float = 21.0

    # perf
    ae_batch_size: int = 1024
    srvq_batch_size: int = 256
    num_workers: int = 14
    pin_memory: bool = True

    # latent map shape used by AE/SRVQ
    C: int = 2
    H: int = 24
    W: int = 24

    device: Optional[str] = None  # None => auto

    _cfg: Optional[Dict[str, Any]] = None
    _ae: Optional[AutoEncoder] = None
    _srvq: Optional[SpatialRVQ] = None

    # -------------------------
    # BaseDataType API
    # -------------------------
    def get_metadata(self) -> Dict[str, Any]:
        return dict(self.__metadata__)

    def encode(self, raw: RawInput) -> pl.DataFrame:
        """
        raw: parquet path or polars df with columns: id, mag_col, splus_cut_*
        return: df(id, codes)
        """
        df = self._load_raw(raw)
        df = self._filter_raw(df)
        if df.is_empty():
            return pl.DataFrame({"id": [], "codes": pl.Series("codes", [], dtype=pl.List(pl.Int32))})

        lat_flat = self._raw_to_latents_flat(df)       # [N, C*H*W] CPU
        codes = self._latents_to_codes(lat_flat)       # list[list[int]]

        return pl.DataFrame(
            {
                "id": df["id"].to_list(),
                "codes": pl.Series("codes", codes, dtype=pl.List(pl.Int32)),
            }
        )

    def decode(self, codes: CodesInput) -> torch.Tensor:
        """
        codes can be:
          - parquet path containing column "codes"
          - polars df containing column "codes"
          - a single flattened code array: Sequence[int]
          - many flattened code arrays: Sequence[Sequence[int]]

        return: torch.Tensor [N, bands, 96, 96]
        """
        codes_list = self._normalize_codes_input(codes)    # List[List[int]]
        if len(codes_list) == 0:
            return torch.empty((0, len(self.bands), self.cutout_size, self.cutout_size))

        z_map = self._codes_to_latent_maps(codes_list)     # [N,C,H,W] on device
        x_hat = self._ae_decode(z_map)                     # [N,bands,96,96] on device
        return x_hat

    # -------------------------
    # Convenience (optional)
    # -------------------------
    def encode_to_parquet(self, infile: PathLike, outfile: PathLike) -> None:
        df_codes = self.encode(infile)
        out = Path(outfile)
        out.parent.mkdir(parents=True, exist_ok=True)
        df_codes.write_parquet(out)

    # -------------------------
    # properties
    # -------------------------
    @property
    def bands(self) -> List[str]:
        return list(self.__metadata__["bands"])

    @property
    def cut_cols(self) -> List[str]:
        return list(self.__metadata__["columns"])

    @property
    def cutout_size(self) -> int:
        return int(self.__metadata__["cutout_size"])

    # -------------------------
    # internals: config/models
    # -------------------------
    def _ensure_cfg(self) -> Dict[str, Any]:
        if self._cfg is None:
            self._cfg = load_config(str(self.config_path))
        return self._cfg

    def _device(self) -> torch.device:
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(self.device)

    def _ckpt_path(self, ckpt: PathLike) -> Path:
        p = Path(ckpt)
        if p.is_absolute():
            return p
        cfg = self._ensure_cfg()
        return Path(cfg["models_folder"]) / p

    def _ensure_ae(self) -> AutoEncoder:
        if self._ae is None:
            ae = AutoEncoder.load_from_file(str(self._ckpt_path(self.ae_ckpt)))
            self._ae = ae.to(self._device()).eval()
        return self._ae

    def _ensure_srvq(self) -> SpatialRVQ:
        if self._srvq is None:
            dev = self._device()
            srvq = SpatialRVQ.load_from_file(str(self._ckpt_path(self.srvq_ckpt)), map_location=dev)
            self._srvq = srvq.to(dev).eval()
        return self._srvq

    # -------------------------
    # internals: IO + filtering
    # -------------------------
    def _load_raw(self, raw: RawInput) -> pl.DataFrame:
        if isinstance(raw, pl.DataFrame):
            return raw
        cols = ["id", self.mag_col] + self.cut_cols
        return pl.read_parquet(str(raw), columns=cols, use_pyarrow=True)

    def _filter_raw(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in self.cut_cols]))
        df = df.filter(pl.col(self.mag_col) < self.mag_cut)
        return df

    def _normalize_codes_input(self, codes: CodesInput) -> List[List[int]]:
        # parquet path
        if isinstance(codes, (str, Path)):
            df = pl.read_parquet(str(codes), use_pyarrow=True)
            return [list(x) for x in df["codes"].to_list()]

        # polars df
        if isinstance(codes, pl.DataFrame):
            return [list(x) for x in codes["codes"].to_list()]

        # many arrays (Sequence[Sequence[int]])
        if len(codes) == 0:  # type: ignore[arg-type]
            return []

        first = codes[0]  # type: ignore[index]
        # single array (Sequence[int])
        if isinstance(first, (int,)) or (hasattr(first, "__int__") and not isinstance(first, (list, tuple))):
            return [list(codes)]  # type: ignore[list-item]

        # list of arrays
        return [list(x) for x in codes]  # type: ignore[arg-type]

    # -------------------------
    # internals: raw -> latents -> codes
    # -------------------------
    def _raw_to_latents_flat(self, df: pl.DataFrame) -> torch.Tensor:
        ae = self._ensure_ae()
        dev = self._device()

        ds = SplusCutoutsDataset(
            df,
            bands=self.bands,
            img_size=self.cutout_size,
            return_valid_mask=True,
        )

        dl = DataLoader(
            ds,
            batch_size=self.ae_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for x_norm, m_valid in tqdm(dl, desc="AE encode", unit="batch"):
                z = ae.encode(x_norm.to(dev))               # [B,C,H,W]
                chunks.append(z.reshape(z.shape[0], -1).cpu())
        return torch.cat(chunks, dim=0).to(torch.float32)   # [N, C*H*W] CPU

    def _latents_to_codes(self, lat_flat: torch.Tensor) -> List[List[int]]:
        srvq = self._ensure_srvq()
        dev = self._device()

        n = lat_flat.shape[0]
        out_codes: List[List[int]] = []

        with torch.no_grad():
            for start in tqdm(range(0, n, self.srvq_batch_size), desc="SRVQ encode", unit="batch"):
                end = min(start + self.srvq_batch_size, n)
                z_map = lat_flat[start:end].to(dev).view(-1, self.C, self.H, self.W)
                out = srvq.encode(z_map)  # expects out["codes"] [B,H,W,R]
                c = out["codes"].detach().cpu().to(torch.int32).numpy()
                out_codes.extend(c.reshape(c.shape[0], -1).tolist())
        return out_codes

    # -------------------------
    # internals: codes -> latents -> recon
    # -------------------------
    def _codes_to_latent_maps(self, codes_flat: List[List[int]]) -> torch.Tensor:
        srvq = self._ensure_srvq()
        dev = self._device()

        codes_t = torch.tensor(codes_flat, dtype=torch.int64, device=dev)  # [N, L]
        n, L = codes_t.shape

        hw = self.H * self.W
        if L % hw != 0:
            raise ValueError(f"Codes length {L} not divisible by H*W={hw}")
        R = L // hw

        codes_4d = codes_t.view(n, self.H, self.W, R)  # [N,H,W,R]

        with torch.no_grad():
            # adjust ONLY if your SpatialRVQ uses a different method name
            z_q = srvq.decode(codes_4d)  # should return [N,C,H,W] or [N,H,W,C]

        if z_q.ndim != 4:
            raise ValueError(f"Unexpected SRVQ decode output shape: {tuple(z_q.shape)}")

        # normalize to [N,C,H,W]
        if z_q.shape[1] == self.C:
            return z_q
        if z_q.shape[-1] == self.C:
            return z_q.permute(0, 3, 1, 2).contiguous()

        raise ValueError(f"Unexpected SRVQ decode output shape: {tuple(z_q.shape)}")

    def _ae_decode(self, z_map: torch.Tensor) -> torch.Tensor:
        ae = self._ensure_ae()
        with torch.no_grad():
            x_hat = ae.decode(z_map)
        return x_hat