from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import polars as pl
import torch

from astromodal.config import load_config
from astromodal.datatypes.base import BaseDataType
from astromodal.scalers.scaler1d import StandardScaler1D
from astromodal.tokenizers.spectralrvq import SpectralPatchRVQ


@dataclass
class ScalarColumnDataTypeConfig:
    config_path: str

    # this instance handles exactly ONE scalar column
    col: str

    # relative to models_folder
    scalers_relpath: str = "outputs/scalers"
    tokenizers_relpath: str = "outputs/tokenizers"

    # input
    id_col: str = "id"

    # optional filter on DF (applied only if column exists)
    filter_mag_r_col: str = "mag_pstotal_r"
    filter_mag_r_max: float = 21.0  # set <=0 to disable

    # encode
    batch_size: int = 8192
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # code shape
    # if None -> use tokenizer rvq.num_stages
    # else -> force output to this R (truncate/pad -1)
    r: Optional[int] = None


class ScalarColumnDataType(BaseDataType):
    """
    RAM -> RAM datatype for a single scalar column.
    Owns:
      - scaler: <models_folder>/<scalers_relpath>/<col>.npz
      - tok   : <models_folder>/<tokenizers_relpath>/<col>.pt
    """

    def __init__(self, cfg: ScalarColumnDataTypeConfig):
        self.cfg = cfg
        self.config = load_config(cfg.config_path)
        self.device = cfg.device

        self.models_folder = Path(self.config["models_folder"])
        self.scaler_path = self.models_folder / cfg.scalers_relpath / f"{cfg.col}.npz"
        self.tok_path = self.models_folder / cfg.tokenizers_relpath / f"{cfg.col}.pt"

        self._scaler: Optional[StandardScaler1D] = None
        self._tok: Optional[SpectralPatchRVQ] = None
        self._R: Optional[int] = None

    # ------------------------------------------------------------------
    # resources (cached)
    # ------------------------------------------------------------------
    def scaler(self) -> Optional[StandardScaler1D]:
        if self._scaler is None:
            self._scaler = StandardScaler1D.load(self.scaler_path) if self.scaler_path.exists() else None
        return self._scaler

    def tok(self) -> Optional[SpectralPatchRVQ]:
        if self._tok is None:
            if self.tok_path.exists():
                t = SpectralPatchRVQ.load_from_file(self.tok_path, map_location=self.device)
                t.eval().to(self.device)
                self._tok = t
            else:
                self._tok = None
        return self._tok

    def R(self) -> int:
        if self._R is None:
            if self.cfg.r is not None:
                self._R = int(self.cfg.r)
            else:
                t = self.tok()
                if t is not None and hasattr(t, "rvq") and hasattr(t.rvq, "num_stages"):
                    self._R = int(t.rvq.num_stages)
                else:
                    self._R = 1
        return int(self._R)

    # ------------------------------------------------------------------
    # df selection (id + this column only)
    # ------------------------------------------------------------------
    def _select_df(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.cfg.id_col not in df.columns:
            raise ValueError(f"Missing id column '{self.cfg.id_col}'")

        # optional filter
        if self.cfg.filter_mag_r_max and float(self.cfg.filter_mag_r_max) > 0:
            c = self.cfg.filter_mag_r_col
            if c in df.columns:
                df = df.filter(pl.col(c) < float(self.cfg.filter_mag_r_max))

        if self.cfg.col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(self.cfg.col))

        return df.select([self.cfg.id_col, self.cfg.col])

    # ------------------------------------------------------------------
    # encode (RAM -> RAM)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_df(
        self,
        df: pl.DataFrame,
        *,
        batch_size: Optional[int] = None,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Returns:
          ids  : List[str]
          codes: LongTensor [B, R]  (-1 padded)
          mask : BoolTensor [B]
        """
        df = self._select_df(df)
        B = df.height
        R = self.R()
        bs = int(batch_size or self.cfg.batch_size)

        ids = df[self.cfg.id_col].cast(pl.Utf8).to_list()

        codes_out = torch.full((B, R), -1, dtype=torch.long)
        mask_out = torch.zeros((B,), dtype=torch.bool)

        scaler = self.scaler()
        tok = self.tok()
        if scaler is None or tok is None or B == 0:
            return ids, codes_out, mask_out

        v = df.select(pl.col(self.cfg.col).cast(pl.Float64, strict=False)).to_series().to_numpy()
        v = v.astype(np.float32, copy=False)
        finite = np.isfinite(v)
        if not np.any(finite):
            return ids, codes_out, mask_out

        idx = np.where(finite)[0]
        vn = scaler.transform_x(v[idx]).astype(np.float32)  # [M]
        x = torch.from_numpy(vn.reshape(-1, 1, 1))          # [M,1,1]

        for k0 in range(0, x.shape[0], bs):
            xb = x[k0:k0 + bs].to(self.device, non_blocking=True)
            enc = tok.encode(xb, update_ema=False)
            cb = enc["codes"][:, 0, :].detach().cpu()  # [b, Rtok]

            bsz, Rtok = cb.shape
            sl = idx[k0:k0 + bsz]

            r_take = min(Rtok, R)
            codes_out[sl, :r_take] = cb[:, :r_take].to(torch.long)
            mask_out[sl] = True

        return ids, codes_out, mask_out

    # ------------------------------------------------------------------
    # decode (RAM -> RAM)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode_codes(
        self,
        codes: torch.Tensor,  # [B, R]
        mask: torch.Tensor,   # [B]
    ) -> torch.Tensor:
        """
        Returns raw reconstructed scalar values: FloatTensor [B]
        NaN where mask=False or missing resources.
        """
        B, R = codes.shape
        out = torch.full((B,), float("nan"), dtype=torch.float32)

        scaler = self.scaler()
        tok = self.tok()
        if scaler is None or tok is None or B == 0:
            return out

        m = mask.detach().cpu().numpy().astype(bool)
        if not np.any(m):
            return out

        idx = np.where(m)[0]
        c = codes[idx, :].to(self.device)

        # use tokenizer's own stage count (or forced R); decode expects [M,1,R]
        c = c.unsqueeze(1)  # [M,1,R]
        xq_norm = tok.decode(c, L=1)  # typically [M,1,1]

        xq_norm_np = xq_norm.detach().cpu().numpy().reshape(-1).astype(np.float32)
        xq_raw = scaler.inverse_transform_x(xq_norm_np).astype(np.float32)

        out[idx] = torch.from_numpy(xq_raw)
        return out