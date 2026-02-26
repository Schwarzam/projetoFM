from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import ast
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from astromodal.config import load_config
from astromodal.datatypes.base import BaseDataType
from astromodal.scalers.standardvec import StandardScalerVec
from astromodal.tokenizers.spectralrvq import SpectralPatchRVQ
from astromodal.tokenizers.rvq import ResidualVQ


PathLike = Union[str, Path]
RawInput = Union[pl.DataFrame, PathLike]
# decode accepts: parquet path, df, single (bp_codes,rp_codes), or list of pairs
CodesInput = Union[
    PathLike,
    pl.DataFrame,
    Tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]],                 # single object: (bp_codes[55][R], rp_codes[55][R])
    Sequence[Tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]]],       # many objects
]


# -------------------------
# RVQ rebuild (same as your script)
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


@dataclass
class GaiaXP(BaseDataType):
    """
    Straight pipeline for GaiaXP coefficients (BP + RP):

      raw coeffs -> scaler -> SpectralPatchRVQ.encode -> codes
      codes -> SpectralPatchRVQ.decode -> de-scale -> coeffs

    encode(raw) returns: pl.DataFrame(id, bp_codes, rp_codes)
      bp_codes/rp_codes are List[List[Int32]] with shape [55][R]

    decode(codes) returns: (bp_coeffs, rp_coeffs) float32 arrays
      - accepts:
          * path to parquet with columns: id, bp_codes, rp_codes
          * polars df with same columns
          * single object: (bp_codes[55][R], rp_codes[55][R])
          * many objects: list[(bp_codes, rp_codes)]
      - returns:
          * if input is single object: (bp[55], rp[55])
          * else: (bp[N,55], rp[N,55])
    """

    __metadata__ = {
        "id": "gaiaxp_bprp",
        "description": "Gaia XP BP/RP coefficient datatype (55 coeffs each)",
        "L": 55,
        "channels": 1,
        "patch_size": 1,
        "raw_columns": ["gaiaxp_bp_coefficients", "gaiaxp_rp_coefficients"],
        "code_columns": ["bp_codes", "rp_codes"],
    }

    # config/paths
    config_path: PathLike = "/home/schwarz/projetoFM/config.yaml"

    tok_bp_path: PathLike = "/home/schwarz/projetoFM/outputs/spectra_tokenizers/gaiaxp_spectral_rvq_bp.pt"
    tok_rp_path: PathLike = "/home/schwarz/projetoFM/outputs/spectra_tokenizers/gaiaxp_spectral_rvq_rp.pt"

    scaler_bp_path: PathLike = "/home/schwarz/projetoFM/outputs/spectra_scalers/gaiaxp_scaler_bp.pkl.npz"
    scaler_rp_path: PathLike = "/home/schwarz/projetoFM/outputs/spectra_scalers/gaiaxp_scaler_rp.pkl.npz"

    # columns
    id_col: str = "id"
    bp_col: str = "gaiaxp_bp_coefficients"
    rp_col: str = "gaiaxp_rp_coefficients"

    # expected lengths
    L: int = 55
    channels: int = 1

    # perf
    batch_size: int = 4096
    num_workers: int = 2
    pin_memory: bool = True

    device: Optional[str] = None  # None => auto
    skip_bad_rows: bool = True

    _cfg: Optional[Dict[str, Any]] = None
    _tok_bp: Optional[SpectralPatchRVQ] = None
    _tok_rp: Optional[SpectralPatchRVQ] = None
    _scaler_bp: Optional[StandardScalerVec] = None
    _scaler_rp: Optional[StandardScalerVec] = None

    # -------------------------
    # BaseDataType API
    # -------------------------
    def get_metadata(self) -> Dict[str, Any]:
        return dict(self.__metadata__)

    def encode(self, raw: RawInput) -> pl.DataFrame:
        """
        raw: parquet path OR polars df with columns: id, bp_col, rp_col
        return: df(id, bp_codes, rp_codes)
        """
        df = self._load_raw(raw)
        if df.is_empty():
            return pl.DataFrame({self.id_col: [], "bp_codes": [], "rp_codes": []})

        ids, Xbp, Xrp = self._df_to_coeff_matrices(df)
        if len(ids) == 0:
            return pl.DataFrame({self.id_col: [], "bp_codes": [], "rp_codes": []})

        scaler_bp = self._ensure_scaler_bp()
        scaler_rp = self._ensure_scaler_rp()
        Xbp_n = scaler_bp.transform_x(Xbp).astype(np.float32)  # [N,55]
        Xrp_n = scaler_rp.transform_x(Xrp).astype(np.float32)  # [N,55]

        codes_bp, codes_rp = self._coeffs_to_codes(Xbp_n, Xrp_n)  # list of [55][R]

        return pl.DataFrame(
            {
                self.id_col: ids,
                "bp_codes": codes_bp,
                "rp_codes": codes_rp,
            }
        )

    def decode(self, codes: CodesInput):
        """
        Returns:
          - single input => (bp[55], rp[55])
          - many inputs  => (bp[N,55], rp[N,55])
        """
        single, pairs = self._normalize_codes_input(codes)  # List[(bp_codes, rp_codes)]
        if len(pairs) == 0:
            if single:
                return (
                    np.zeros((self.L,), dtype=np.float32),
                    np.zeros((self.L,), dtype=np.float32),
                )
            return (
                np.zeros((0, self.L), dtype=np.float32),
                np.zeros((0, self.L), dtype=np.float32),
            )

        Xbp_n, Xrp_n = self._codes_to_coeffs_norm(pairs)  # [N,55] float32
        scaler_bp = self._ensure_scaler_bp()
        scaler_rp = self._ensure_scaler_rp()
        Xbp = scaler_bp.inverse_transform_x(Xbp_n).astype(np.float32)
        Xrp = scaler_rp.inverse_transform_x(Xrp_n).astype(np.float32)

        if single:
            return Xbp[0], Xrp[0]
        return Xbp, Xrp

    # -------------------------
    # Internals: config/device
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

    # -------------------------
    # Internals: load models
    # -------------------------
    def _ensure_tok_bp(self) -> SpectralPatchRVQ:
        if self._tok_bp is None:
            dev = self._device()
            self._tok_bp = SpectralPatchRVQ.load_from_file(
                Path(self.tok_bp_path),
                rvq_ctor=make_rvq,
                map_location=dev,
            ).to(dev).eval()
        return self._tok_bp

    def _ensure_tok_rp(self) -> SpectralPatchRVQ:
        if self._tok_rp is None:
            dev = self._device()
            self._tok_rp = SpectralPatchRVQ.load_from_file(
                Path(self.tok_rp_path),
                rvq_ctor=make_rvq,
                map_location=dev,
            ).to(dev).eval()
        return self._tok_rp

    def _ensure_scaler_bp(self) -> StandardScalerVec:
        if self._scaler_bp is None:
            self._scaler_bp = StandardScalerVec.load(Path(self.scaler_bp_path))
        return self._scaler_bp

    def _ensure_scaler_rp(self) -> StandardScalerVec:
        if self._scaler_rp is None:
            self._scaler_rp = StandardScalerVec.load(Path(self.scaler_rp_path))
        return self._scaler_rp

    # -------------------------
    # Internals: raw IO
    # -------------------------
    def _load_raw(self, raw: RawInput) -> pl.DataFrame:
        if isinstance(raw, pl.DataFrame):
            return raw
        return pl.read_parquet(
            str(raw),
            columns=[self.id_col, self.bp_col, self.rp_col],
            use_pyarrow=True,
        )

    def _df_to_coeff_matrices(self, df: pl.DataFrame) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Keeps only rows where both BP and RP are valid length L and finite.
        Returns ids (list[str]), Xbp [N,L], Xrp [N,L].
        """
        ids_out: List[str] = []
        rows_bp: List[np.ndarray] = []
        rows_rp: List[np.ndarray] = []

        id_s = df[self.id_col].to_list()
        bp_s = df[self.bp_col].to_list()
        rp_s = df[self.rp_col].to_list()

        for i in range(len(id_s)):
            bp = _parse_coeff_cell(bp_s[i])
            rp = _parse_coeff_cell(rp_s[i])

            ok = (
                bp is not None and rp is not None and
                len(bp) == self.L and len(rp) == self.L
            )
            if not ok:
                if self.skip_bad_rows:
                    continue
                raise ValueError(
                    f"Bad coeff at row {i}: "
                    f"bp_len={None if bp is None else len(bp)}, "
                    f"rp_len={None if rp is None else len(rp)}"
                )

            bp_arr = np.asarray(bp, dtype=np.float32)
            rp_arr = np.asarray(rp, dtype=np.float32)

            if (not np.all(np.isfinite(bp_arr))) or (not np.all(np.isfinite(rp_arr))):
                if self.skip_bad_rows:
                    continue
                raise ValueError(f"Non-finite coeffs at row {i}")

            ids_out.append(str(id_s[i]))
            rows_bp.append(bp_arr)
            rows_rp.append(rp_arr)

        if not rows_bp:
            return [], np.zeros((0, self.L), dtype=np.float32), np.zeros((0, self.L), dtype=np.float32)

        Xbp = np.stack(rows_bp, axis=0).astype(np.float32)
        Xrp = np.stack(rows_rp, axis=0).astype(np.float32)
        return ids_out, Xbp, Xrp

    # -------------------------
    # Internals: coeffs -> codes
    # -------------------------
    class _PairDataset(Dataset):
        def __init__(self, Xbp_n: np.ndarray, Xrp_n: np.ndarray):
            self.Xbp = Xbp_n.astype(np.float32)  # [N,L]
            self.Xrp = Xrp_n.astype(np.float32)  # [N,L]

        def __len__(self) -> int:
            return self.Xbp.shape[0]

        def __getitem__(self, i: int):
            bp = self.Xbp[i].reshape(-1, 1)  # [L,1]
            rp = self.Xrp[i].reshape(-1, 1)  # [L,1]
            return torch.from_numpy(bp), torch.from_numpy(rp)

    @staticmethod
    def _collate_pair(batch):
        bp = torch.stack([b[0] for b in batch], dim=0)  # [B,L,1]
        rp = torch.stack([b[1] for b in batch], dim=0)  # [B,L,1]
        return bp, rp

    def _coeffs_to_codes(self, Xbp_n: np.ndarray, Xrp_n: np.ndarray) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
        """
        Input: normalized coeff matrices [N,55]
        Output: codes_bp and codes_rp as list of [55][R] per object.
        """
        tok_bp = self._ensure_tok_bp()
        tok_rp = self._ensure_tok_rp()
        dev = self._device()

        ds = self._PairDataset(Xbp_n, Xrp_n)
        dl = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_pair,
        )

        out_bp: List[List[List[int]]] = []
        out_rp: List[List[List[int]]] = []

        with torch.no_grad():
            for bp, rp in tqdm(dl, desc="GaiaXP encode", unit="batch"):
                bp = bp.to(dev, non_blocking=True).float()  # [B,L,1]
                rp = rp.to(dev, non_blocking=True).float()

                enc_bp = tok_bp.encode(bp, update_ema=False)
                enc_rp = tok_rp.encode(rp, update_ema=False)

                bp_codes = enc_bp["codes"].detach().cpu().numpy().astype(np.int32)  # [B,L,R]
                rp_codes = enc_rp["codes"].detach().cpu().numpy().astype(np.int32)  # [B,L,R]

                for i in range(bp_codes.shape[0]):
                    out_bp.append(bp_codes[i].tolist())
                    out_rp.append(rp_codes[i].tolist())

        return out_bp, out_rp

    # -------------------------
    # Internals: codes input normalize
    # -------------------------
    def _normalize_codes_input(self, codes: CodesInput) -> Tuple[bool, List[Tuple[List[List[int]], List[List[int]]]]]:
        """
        Returns: (single, pairs)
          pairs: list of (bp_codes, rp_codes) each [55][R]
        """
        # path
        if isinstance(codes, (str, Path)):
            df = pl.read_parquet(str(codes), use_pyarrow=True)
            bp_list = df["bp_codes"].to_list()
            rp_list = df["rp_codes"].to_list()
            pairs = [(list(map(list, bp_list[i])), list(map(list, rp_list[i]))) for i in range(len(bp_list))]
            return (False, pairs)

        # df
        if isinstance(codes, pl.DataFrame):
            bp_list = codes["bp_codes"].to_list()
            rp_list = codes["rp_codes"].to_list()
            pairs = [(list(map(list, bp_list[i])), list(map(list, rp_list[i]))) for i in range(len(bp_list))]
            return (False, pairs)

        # single pair (bp_codes, rp_codes)
        if isinstance(codes, tuple) and len(codes) == 2:
            bp, rp = codes
            return (True, [(list(map(list, bp)), list(map(list, rp)))])

        # many pairs
        pairs_out: List[Tuple[List[List[int]], List[List[int]]]] = []
        for bp, rp in codes:  # type: ignore[assignment]
            pairs_out.append((list(map(list, bp)), list(map(list, rp))))
        return (False, pairs_out)

    # -------------------------
    # Internals: codes -> normalized coeffs
    # -------------------------
    def _codes_to_coeffs_norm(
        self,
        pairs: List[Tuple[List[List[int]], List[List[int]]]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        pairs: list of (bp_codes, rp_codes) each [L][R]
        returns: Xbp_n, Xrp_n as float32 [N,L]
        """
        tok_bp = self._ensure_tok_bp()
        tok_rp = self._ensure_tok_rp()
        dev = self._device()

        bp_stack = np.asarray([bp for bp, _ in pairs], dtype=np.int64)  # [N,L,R]
        rp_stack = np.asarray([rp for _, rp in pairs], dtype=np.int64)  # [N,L,R]

        bp_t = torch.from_numpy(bp_stack).to(dev)
        rp_t = torch.from_numpy(rp_stack).to(dev)

        with torch.no_grad():
            # If your SpectralPatchRVQ uses a different method name, adjust these two lines only.
            xb_n = tok_bp.decode(bp_t, L=self.L)  # expected [N,L,1] or [N,L]
            xr_n = tok_rp.decode(rp_t, L=self.L)

        xb_n = xb_n.detach().cpu().float()
        xr_n = xr_n.detach().cpu().float()

        if xb_n.ndim == 3:
            xb_n = xb_n.squeeze(-1)  # [N,L]
        if xr_n.ndim == 3:
            xr_n = xr_n.squeeze(-1)

        if xb_n.shape[1] != self.L or xr_n.shape[1] != self.L:
            raise ValueError(f"Unexpected decoded shapes: bp={tuple(xb_n.shape)} rp={tuple(xr_n.shape)}")

        return xb_n.numpy().astype(np.float32), xr_n.numpy().astype(np.float32)