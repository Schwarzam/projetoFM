#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import numpy as np
import polars as pl
from tqdm.auto import tqdm

# =========================
# CONFIG
# =========================

INPUT_DIR = "/home/astrodados4/downloads/hypercube"
FILE_PATTERN = "datacube_*.parquet"

TOKENIZER_CONFIG = "spectrum_tokenizer_config.npz"
OUTPUT_DIR = "spectrum_tokens"

# Which ID column to prefer
PREFERRED_ID_COLS = ["gaia_source_id", "id"]

TOKEN_DTYPE = np.uint16  # fits up to 65535 codes

# If no ID exists, optionally store row indices for alignment
SAVE_ROW_INDEX_IF_NO_ID = True

# ZTF Option A precision floor (must match training choice reasonably)
ZTF_ERR_EPS = 1e-3

# If variable-length arrays appear, choose behavior:
PAD_TRUNCATE_LENGTH_MISMATCH = True


# =========================
# HELPERS
# =========================

def list_parquet_files() -> List[Path]:
    p = Path(INPUT_DIR)
    files = sorted(p.glob(FILE_PATTERN))
    if not files:
        raise RuntimeError(f"No parquet files found in {INPUT_DIR} matching {FILE_PATTERN}")
    return files


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_schema_fast(path: Path) -> List[str]:
    """Read parquet schema (columns) without loading rows."""
    df0 = pl.read_parquet(path, n_rows=0)
    return df0.columns


def choose_id_column(columns: List[str]) -> Optional[str]:
    for c in PREFERRED_ID_COLS:
        if c in columns:
            return c
    return None


def _to_str(x: Any) -> str:
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode()
    return str(x)


def _normalize_dict_keys(d: Dict[Any, Any]) -> Dict[str, Any]:
    return {_to_str(k): v for k, v in d.items()}


def load_tokenizer_config(config_path: str):
    """
    Loads:
      - group_names: list[str]
      - group_means: dict[str] -> np.ndarray (D,)
      - group_stds: dict[str] -> np.ndarray (D,)
      - group_codebooks: dict[str] -> np.ndarray (K,D)
      - group_columns: dict[str] -> group spec dict (kind, column names)
      - CODEBOOK_SIZE: int
    """
    data = np.load(config_path, allow_pickle=True)

    raw_names = data["group_names"]
    group_names = [_to_str(n) for n in raw_names]

    group_means = _normalize_dict_keys(data["group_means"].item())
    group_stds = _normalize_dict_keys(data["group_stds"].item())
    group_codebooks = _normalize_dict_keys(data["group_codebooks"].item())
    codebook_size = int(data["CODEBOOK_SIZE"])

    # group_columns is optional but recommended (from the training script I gave you)
    group_columns: Dict[str, Dict[str, Any]] = {}
    if "group_columns" in data.files:
        group_columns = _normalize_dict_keys(data["group_columns"].item())
        # normalize nested keys that may be bytes
        norm2 = {}
        for gname, spec in group_columns.items():
            if isinstance(spec, dict):
                spec2 = {}
                for k, v in spec.items():
                    kk = _to_str(k)
                    if isinstance(v, (bytes, np.bytes_)):
                        vv = v.decode()
                    else:
                        vv = v
                    spec2[kk] = vv
                norm2[gname] = spec2
            else:
                norm2[gname] = spec
        group_columns = norm2

    # cast arrays
    for k, v in list(group_means.items()):
        group_means[k] = np.asarray(v, dtype=np.float32)
    for k, v in list(group_stds.items()):
        group_stds[k] = np.asarray(v, dtype=np.float32)
    for k, v in list(group_codebooks.items()):
        group_codebooks[k] = np.asarray(v, dtype=np.float32)

    return group_names, group_means, group_stds, group_codebooks, group_columns, codebook_size


def _pad_or_truncate_1d(x: np.ndarray, L: int) -> np.ndarray:
    if x.size == L:
        return x
    if x.size > L:
        return x[:L]
    out = np.zeros(L, dtype=x.dtype)
    out[: x.size] = x
    return out


def _nearest_center_codes(x: np.ndarray, centers: np.ndarray, dtype=TOKEN_DTYPE) -> np.ndarray:
    """
    x: (L, D) float32
    centers: (K, D) float32
    returns codes: (L,) dtype
    Efficient: dist^2 = x^2 + c^2 - 2 x·c
    """
    x = np.asarray(x, dtype=np.float32)
    c = np.asarray(centers, dtype=np.float32)

    x2 = np.sum(x * x, axis=1, keepdims=True)            # (L,1)
    c2 = np.sum(c * c, axis=1, keepdims=True).T          # (1,K)
    xc = x @ c.T                                         # (L,K)
    dist2 = x2 + c2 - 2.0 * xc                           # (L,K)
    return np.argmin(dist2, axis=1).astype(dtype)         # (L,)


def compute_tokens_for_group_in_file(
    df: pl.DataFrame,
    group_spec: Dict[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """
    Supports kinds:
      - flux_ivar: feats=[flux, sqrt(ivar)]
      - value_only: feats=[value]
      - ztf_mag_err: feats=[mag, 1/magerr]
    """
    name = group_spec.get("name", "")
    kind = group_spec["kind"]

    n_rows = df.height
    if n_rows == 0:
        return np.zeros((0, 0), dtype=TOKEN_DTYPE)

    std = np.asarray(std, dtype=np.float32)
    std = np.where(std == 0, 1e-8, std)
    mean = np.asarray(mean, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)

    # pick columns depending on kind
    if kind == "flux_ivar":
        flux_col = group_spec["flux_col"]
        ivar_col = group_spec["ivar_col"]
        flux_series = df[flux_col]
        ivar_series = df[ivar_col]

        # determine L
        L = None
        for i in range(n_rows):
            arr = np.asarray(flux_series[i])
            if arr.size > 0:
                L = int(arr.size)
                break
        if L is None:
            return np.zeros((n_rows, 0), dtype=TOKEN_DTYPE)

        tokens = np.empty((n_rows, L), dtype=TOKEN_DTYPE)
        c2 = np.sum(centers * centers, axis=1, keepdims=True).T  # (1,K)

        for i in tqdm(range(n_rows), desc=f"  [tokens {name}] rows", leave=False):
            flux = np.asarray(flux_series[i], dtype=np.float32)
            if flux.size == 0:
                tokens[i, :] = 0
                continue

            if flux.size != L:
                if PAD_TRUNCATE_LENGTH_MISMATCH:
                    flux = _pad_or_truncate_1d(flux, L)
                else:
                    tokens[i, :] = 0
                    continue

            ivar = np.asarray(ivar_series[i], dtype=np.float32)
            if ivar.size != L:
                if PAD_TRUNCATE_LENGTH_MISMATCH:
                    ivar = _pad_or_truncate_1d(ivar, L)
                else:
                    tokens[i, :] = 0
                    continue

            ivar = np.clip(ivar, a_min=0.0, a_max=None)
            prec = np.sqrt(ivar)
            feats = np.stack([flux, prec], axis=-1)                # (L,2)
            feats_std = (feats - mean) / std                       # (L,2)

            # efficient nearest
            x = feats_std.astype(np.float32, copy=False)
            x2 = np.sum(x * x, axis=1, keepdims=True)              # (L,1)
            xc = x @ centers.T                                     # (L,K)
            dist2 = x2 + c2 - 2.0 * xc                             # (L,K)
            tokens[i, :] = np.argmin(dist2, axis=1).astype(TOKEN_DTYPE)

        return tokens

    elif kind == "value_only":
        flux_col = group_spec["flux_col"]
        series = df[flux_col]

        L = None
        for i in range(n_rows):
            arr = np.asarray(series[i])
            if arr.size > 0:
                L = int(arr.size)
                break
        if L is None:
            return np.zeros((n_rows, 0), dtype=TOKEN_DTYPE)

        tokens = np.empty((n_rows, L), dtype=TOKEN_DTYPE)
        c2 = np.sum(centers * centers, axis=1, keepdims=True).T

        for i in tqdm(range(n_rows), desc=f"  [tokens {name}] rows", leave=False):
            arr = np.asarray(series[i], dtype=np.float32)
            if arr.size == 0:
                tokens[i, :] = 0
                continue
            if arr.size != L:
                if PAD_TRUNCATE_LENGTH_MISMATCH:
                    arr = _pad_or_truncate_1d(arr, L)
                else:
                    tokens[i, :] = 0
                    continue

            feats = arr.reshape(-1, 1)                             # (L,1)
            feats_std = (feats - mean) / std                       # (L,1)

            x = feats_std.astype(np.float32, copy=False)
            x2 = np.sum(x * x, axis=1, keepdims=True)
            xc = x @ centers.T
            dist2 = x2 + c2 - 2.0 * xc
            tokens[i, :] = np.argmin(dist2, axis=1).astype(TOKEN_DTYPE)

        return tokens

    elif kind == "ztf_mag_err":
        mag_col = group_spec["mag_col"]
        err_col = group_spec["err_col"]
        mag_series = df[mag_col]
        err_series = df[err_col]

        L = None
        for i in range(n_rows):
            arr = np.asarray(mag_series[i])
            if arr.size > 0:
                L = int(arr.size)
                break
        if L is None:
            return np.zeros((n_rows, 0), dtype=TOKEN_DTYPE)

        tokens = np.empty((n_rows, L), dtype=TOKEN_DTYPE)
        c2 = np.sum(centers * centers, axis=1, keepdims=True).T

        for i in tqdm(range(n_rows), desc=f"  [tokens {name}] rows", leave=False):
            mag = np.asarray(mag_series[i], dtype=np.float32)
            err = np.asarray(err_series[i], dtype=np.float32)

            if mag.size == 0 or err.size == 0:
                tokens[i, :] = 0
                continue

            if mag.size != L:
                if PAD_TRUNCATE_LENGTH_MISMATCH:
                    mag = _pad_or_truncate_1d(mag, L)
                else:
                    tokens[i, :] = 0
                    continue
            if err.size != L:
                if PAD_TRUNCATE_LENGTH_MISMATCH:
                    err = _pad_or_truncate_1d(err, L)
                else:
                    tokens[i, :] = 0
                    continue

            # precision proxy
            err = np.clip(err, a_min=ZTF_ERR_EPS, a_max=None)
            prec = 1.0 / err

            feats = np.stack([mag, prec], axis=-1)                 # (L,2)
            feats_std = (feats - mean) / std                       # (L,2)

            x = feats_std.astype(np.float32, copy=False)
            x2 = np.sum(x * x, axis=1, keepdims=True)
            xc = x @ centers.T
            dist2 = x2 + c2 - 2.0 * xc
            tokens[i, :] = np.argmin(dist2, axis=1).astype(TOKEN_DTYPE)

        return tokens

    else:
        raise ValueError(f"Unknown group kind: {kind}")


# =========================
# MAIN ENCODING
# =========================

def encode_spectrum_tokens():
    files = list_parquet_files()
    print("Found parquet files:")
    for f in files:
        print("  -", f)

    group_names, group_means, group_stds, group_codebooks, group_columns, codebook_size = load_tokenizer_config(TOKENIZER_CONFIG)
    print(f"\nLoaded spectrum tokenizer config for groups: {list(group_names)} (K={codebook_size})")

    if not group_columns:
        raise RuntimeError(
            "Your tokenizer config does not contain 'group_columns'.\n"
            "Use the training script version that saves group_columns, or hardcode ZTF groups here."
        )

    ensure_dir(OUTPUT_DIR)

    for path in files:
        p = Path(path)
        print(f"\nProcessing file: {p.name}")

        # schema-only detect available groups + id
        try:
            cols_in_file = read_schema_fast(path)
        except Exception as e:
            print(f"  [warn] Could not read schema: {e}")
            continue

        cols_set = set(cols_in_file)
        id_col = choose_id_column(cols_in_file)

        # Decide which groups from tokenizer config are available in THIS file
        available_group_specs: List[Dict[str, Any]] = []
        for gname in group_names:
            gname = _to_str(gname)
            spec = group_columns.get(gname)
            if not isinstance(spec, dict):
                continue

            kind = spec.get("kind")
            spec = dict(spec)  # copy
            spec["name"] = gname

            if kind == "flux_ivar":
                need = [spec["flux_col"], spec["ivar_col"]]
            elif kind == "value_only":
                need = [spec["flux_col"]]
            elif kind == "ztf_mag_err":
                need = [spec["mag_col"], spec["err_col"]]
            else:
                continue

            if all(c in cols_set for c in need):
                available_group_specs.append(spec)

        if not available_group_specs:
            print("  -> no spectrum groups present in this file (per tokenizer config), skipping.")
            continue

        # Read only needed columns (unique preserve order)
        cols_to_read: List[str] = []
        if id_col is not None:
            cols_to_read.append(id_col)

        for spec in available_group_specs:
            kind = spec["kind"]
            if kind == "flux_ivar":
                cols_to_read.extend([spec["flux_col"], spec["ivar_col"]])
            elif kind == "value_only":
                cols_to_read.append(spec["flux_col"])
            elif kind == "ztf_mag_err":
                cols_to_read.extend([spec["mag_col"], spec["err_col"]])

        cols_to_read = list(dict.fromkeys(cols_to_read))
        df = pl.read_parquet(path, columns=cols_to_read)

        n_rows = df.height
        print(f"  -> rows: {n_rows}")
        print(f"  -> id_col: {id_col if id_col is not None else '(none)'}")
        print(f"  -> groups used: {[g['name'] for g in available_group_specs]}")

        out_data: Dict[str, Any] = {}

        # IDs / row_index
        if id_col is not None and id_col in df.columns:
            out_data["ids"] = df[id_col].to_numpy()
            out_data["id_col"] = np.array([id_col], dtype=object)
        else:
            out_data["ids"] = np.array([], dtype=np.int64)
            out_data["id_col"] = np.array([""], dtype=object)
            if SAVE_ROW_INDEX_IF_NO_ID:
                out_data["row_index"] = np.arange(n_rows, dtype=np.int64)

        # Encode each group
        for spec in available_group_specs:
            name = spec["name"]
            mean = group_means[name]
            std = group_stds[name]
            centers = group_codebooks[name]

            tokens = compute_tokens_for_group_in_file(df, spec, mean, std, centers)
            out_data[f"tokens_{name}"] = tokens

        out_name = p.stem + "_spectrum_tokens.npz"
        out_path = Path(OUTPUT_DIR) / out_name
        np.savez_compressed(out_path, **out_data)

        shapes = {k: v.shape for k, v in out_data.items() if isinstance(v, np.ndarray) and k.startswith("tokens_")}
        print(f"  -> saved {out_path} with token arrays: {shapes}")


if __name__ == "__main__":
    encode_spectrum_tokens()