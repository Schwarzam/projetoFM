#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train + save scalers/tokenizers for scalar columns from datacubes parquet files.

What this script does
---------------------
1) Builds train/val file lists from config["datacubes_paths"] using load_datacube_files()
2) Loads only EXPECTED_COLUMNS from each parquet, adding missing cols as nulls
3) Optional filter: mag_pstotal_r < 21 (configurable)
4) For EACH scalar column:
   - Fit StandardScaler1D on TRAIN finite values (optionally transform/clip)
   - Save scaler to .npz
   - If enough finite values: train a scalar SpectralPatchRVQ (L=1,C=1) and save best checkpoint
   - Record stats (n_finite, mean/std raw, best val mse/rmse, etc.)
5) Saves a summary table (parquet + csv) with per-column stats

NOTE (important):
- This script treats each scalar column independently (1D tokens).
- If you later decide to tokenize mags+errors jointly (recommended), that is a different training script
  (grouped, channels=2). This script still works for "non-photometry" scalars and parameter errors.

Run
---
python train_scalar_tokenizers.py \
  --config /home/schwarz/projetoFM/config.yaml \
  --out_root /home/schwarz/projetoFM/outputs/scalars_tokenizers \
  --nfiles 500 \
  --train_val_split 0.9 \
  --filter_mag_r 21

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Optional, Dict, Any, Tuple, List

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from astromodal.config import load_config
from astromodal.datasets.datacubes import load_datacube_files
from astromodal.scalers.scaler1d import StandardScaler1D
from astromodal.tokenizers.rvq import ResidualVQ
from astromodal.tokenizers.spectralrvq import SpectralPatchRVQ


# -------------------------
# bands / columns
# -------------------------

SPLUS_BANDS = [
    "u", "i", "r", "g", "z",
    "j0378", "j0395", "j0410", "j0430",
    "j0515", "j0660", "j0861",
]

CORE_COLUMNS = ["id", "ra", "dec"]

SCALAR_COLUMNS = [
    "ellipticity_det",
    "elongation_det",
    "a_pixel_det",
    "b_pixel_det",
    "theta_det",
    "fwhm_n_det",
    *[f"mag_pstotal_{b}" for b in SPLUS_BANDS],
    *[f"err_mag_pstotal_{b}" for b in SPLUS_BANDS],
    "gaia_parallax",
    "gaia_parallax_error",
    "gaia_pmra",
    "gaia_pmdec",
    "gaia_pmra_error",
    "gaia_pmdec_error",
    "gaia_phot_bp_mean_flux",
    "gaia_phot_rp_mean_flux",
    "gaia_phot_g_mean_flux",
    "gaia_phot_bp_mean_flux_error",
    "gaia_phot_rp_mean_flux_error",
    "gaia_phot_g_mean_flux_error",
    "gaia_teff_gspphot",
    "gaia_logg_gspphot",
    "gaia_mh_gspphot",
    "specz_z",
    "specz_e_z",
    "vista_yapermag6",
    "vista_yapermag6err",
    "vista_japermag6",
    "vista_japermag6err",
    "vista_hapermag6",
    "vista_hapermag6err",
    "vista_ksapermag6",
    "vista_ksapermag6err",
]

EXPECTED_COLUMNS = CORE_COLUMNS + SCALAR_COLUMNS


# -------------------------
# parquet loading with schema enforcement
# -------------------------

def read_parquet_with_schema(
    path: str | Path,
    *,
    expected_columns: Sequence[str],
) -> pl.DataFrame:
    """
    Reads a parquet file and guarantees that all expected_columns exist.
    Missing columns are added as nulls.
    """
    path = Path(path)

    schema = pl.read_parquet_schema(path)
    available = set(schema.keys())

    cols_to_read = [c for c in expected_columns if c in available]
    df = pl.read_parquet(path, columns=cols_to_read, use_pyarrow=True)

    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        df = df.with_columns([pl.lit(None).alias(c) for c in missing])

    return df.select(expected_columns)


def load_datacubes_from_filelist(
    files: Sequence[str | Path],
    *,
    expected_columns: Sequence[str],
    desc: str = "Loading datacubes",
) -> pl.DataFrame:
    """
    Load multiple parquet files given explicitly as a list.
    Guarantees that ALL expected_columns exist in the final DataFrame.
    """
    dfs: List[pl.DataFrame] = []

    for f in tqdm(files, desc=desc):
        f = Path(f)
        try:
            df = read_parquet_with_schema(f, expected_columns=expected_columns)
            dfs.append(df)
        except Exception as e:
            print(f"[skip] {f}: {e}")

    if not dfs:
        raise RuntimeError("No valid parquet files were loaded.")

    return pl.concat(dfs, how="vertical", rechunk=False)


# -------------------------
# scaler + dataloader helpers
# -------------------------

def _finite_values_from_df(df: pl.DataFrame, col: str) -> np.ndarray:
    """
    Returns finite float64 values from df[col], empty if col missing.
    """
    if col not in df.columns:
        return np.array([], dtype=np.float64)

    # polars may store as Int/Float/Null; coerce to numpy float64 and keep finite
    v = df.select(pl.col(col)).to_series().to_numpy()
    v = v.astype(np.float64, copy=False)
    v = v[np.isfinite(v)]
    return v


def fit_scaler_1d(
    train_df: pl.DataFrame,
    col: str,
    *,
    transform: str = "none",      # "none" | "asinh"
    asinh_scale: float = 1.0,
    clip_quantile: Optional[float] = None,  # e.g. 0.999
) -> StandardScaler1D:
    v = _finite_values_from_df(train_df, col)

    if v.size == 0:
        return StandardScaler1D(mean=0.0, std=1.0, transform=transform, asinh_scale=asinh_scale)

    if transform == "asinh":
        s0 = asinh_scale if asinh_scale > 0 else 1.0
        v = np.arcsinh(v / s0)

    if clip_quantile is not None and 0.0 < clip_quantile < 1.0 and v.size > 10:
        lo = np.quantile(v, 1.0 - clip_quantile)
        hi = np.quantile(v, clip_quantile)
        v = np.clip(v, lo, hi)

    mean = float(np.mean(v))
    std = float(np.std(v))
    if not np.isfinite(std) or std < 1e-12:
        std = 1.0

    return StandardScaler1D(mean=mean, std=std, transform=transform, asinh_scale=asinh_scale)


def make_scalar_loader(
    df: pl.DataFrame,
    col: str,
    scaler: StandardScaler1D,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """
    Yields (x_norm,) where x_norm is [B,1,1].
    Uses only finite values (drops null/nan/inf).
    """
    v = _finite_values_from_df(df, col).astype(np.float32)
    if v.size == 0:
        Xn = np.zeros((0, 1, 1), dtype=np.float32)
        return DataLoader(TensorDataset(torch.from_numpy(Xn)), batch_size=batch_size, shuffle=False, num_workers=0)

    vn = scaler.transform_x(v).astype(np.float32)
    Xn = vn.reshape(-1, 1, 1)
    ds = TensorDataset(torch.from_numpy(Xn))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


@torch.no_grad()
def eval_scalar_tok_unscaled_mse(
    tok: SpectralPatchRVQ,
    dl: DataLoader,
    scaler: StandardScaler1D,
    *,
    device: str,
    dtype_t: torch.dtype,
) -> Dict[str, float]:
    """
    Unscaled metrics in original units (via scaler.inverse_transform_x).
    """
    tok.eval().to(device)

    sum_mse = 0.0
    n = 0

    for (x_norm,) in dl:
        if x_norm.numel() == 0:
            continue

        x_norm = x_norm.to(device=device, dtype=dtype_t, non_blocking=True)  # [B,1,1]
        out = tok.encode(x_norm, update_ema=False)
        xq_norm = out["x_q"]  # [B,1,1]

        x_raw = scaler.inverse_transform_x(x_norm.detach().cpu().numpy().reshape(-1))
        xq_raw = scaler.inverse_transform_x(xq_norm.detach().cpu().numpy().reshape(-1))

        mse = float(np.mean((xq_raw - x_raw) ** 2))
        sum_mse += mse * x_norm.shape[0]
        n += x_norm.shape[0]

    mse = sum_mse / max(n, 1)
    rmse = float(np.sqrt(mse))
    return {
        "val_mse_unscaled": float(mse),
        "val_rmse_unscaled": float(rmse),
        "val_mse_x_1e6": float(mse * 1e6),
        "val_-log10_mse": float(-np.log10(mse + 1e-300)),
    }


def train_scalar_tokenizer(
    train_dl,
    val_dl,
    *,
    codebook_size: int,
    num_stages: int,
    decay: float,
    epochs: int,
    ema_epochs: int,
    device: str,
    save_path: Path,
    scaler,
    dtype_t: torch.dtype,
    # ---- early stopping knobs ----
    early_stop_patience: int = 6,      # stop after N epochs w/o improvement
    early_stop_min_delta: float = 0.0, # require improvement >= min_delta
    early_stop_warmup: int = 0,        # ignore ES before this epoch
) -> Tuple["SpectralPatchRVQ", List[Dict[str, Any]]]:
    """
    Train scalar tokenizer and save best checkpoint by val_mse_unscaled.

    Early stopping:
      - Tracks best val_mse_unscaled
      - If no improvement for `early_stop_patience` epochs (after warmup), stops.
      - Improvement means: new_mse < best_mse - min_delta
    """
    from astromodal.tokenizers.rvq import ResidualVQ
    from astromodal.tokenizers.spectralrvq import SpectralPatchRVQ

    P, C = 1, 1
    rvq = ResidualVQ(dim=P * C, num_stages=num_stages, codebook_size=codebook_size, decay=decay).to(device)
    tok = SpectralPatchRVQ(rvq=rvq, patch_size=P, channels=C).to(device)

    best_mse = float("inf")
    best_epoch = -1
    history: List[Dict[str, Any]] = []

    bad_epochs = 0

    tok.train()
    for epoch in range(epochs):
        update_ema = (epoch < int(ema_epochs))
        tr = tok.train_epoch(train_dl, device=device, update_ema=update_ema)

        va = eval_scalar_tok_unscaled_mse(tok, val_dl, scaler, device=device, dtype_t=dtype_t)

        cur_mse = float(va["val_mse_unscaled"])
        row = {
            "epoch": int(epoch),
            "train_loss_norm": float(tr["loss"]),
            "update_ema": bool(update_ema),
            **va,
        }
        history.append(row)

        improved = cur_mse < (best_mse - float(early_stop_min_delta))

        if improved:
            best_mse = cur_mse
            best_epoch = int(epoch)
            bad_epochs = 0

            tok.save(
                str(save_path),
                additional_info={
                    "best_epoch": best_epoch,
                    "best_val_mse_unscaled": float(best_mse),
                    "codebook_size": int(codebook_size),
                    "num_stages": int(num_stages),
                    "decay": float(decay),
                    "patch_size": int(P),
                    "channels": int(C),
                    "early_stop_patience": int(early_stop_patience),
                    "early_stop_min_delta": float(early_stop_min_delta),
                    "early_stop_warmup": int(early_stop_warmup),
                },
            )
        else:
            # only count patience after warmup
            if epoch >= int(early_stop_warmup):
                bad_epochs += 1

        print(
            f"Epoch {epoch:03d} | ema={int(update_ema)} | "
            f"train_loss(norm)={tr['loss']:.6g} | "
            f"val_mse(unscaled)={cur_mse:.6g} rmse={va['val_rmse_unscaled']:.6g} | "
            f"best_mse={best_mse:.6g} @ {best_epoch} | "
            f"pat={bad_epochs}/{early_stop_patience}"
        )

        if epoch >= int(early_stop_warmup) and bad_epochs >= int(early_stop_patience):
            print(
                f"[early-stop] no improvement for {early_stop_patience} epochs "
                f"(min_delta={early_stop_min_delta}). "
                f"Stopping at epoch {epoch}, best at {best_epoch} (mse={best_mse:.6g})."
            )
            break

    return tok, history

# -------------------------
# per-column pipeline
# -------------------------

def fit_and_save_all_scalars(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    *,
    scalar_columns: List[str],
    scalers_dir: Path,
    tok_dir: Path,
    codebook_size: int,
    num_stages: int,
    decay: float,
    epochs: int,
    batch_size: int,
    ema_epochs: int,
    min_finite: int,
    device: str,
    dtype_t: torch.dtype,
    transform_for_errors: str = "asinh",
    transform_for_values: str = "none",
    asinh_scale_default: float = 1.0,
    clip_quantile: Optional[float] = 0.999,
) -> pl.DataFrame:
    results: List[Dict[str, Any]] = []

    for col in tqdm(scalar_columns, desc="Fitting scalers + tokenizers"):
        is_error_col = ("err_" in col) or col.endswith("error") or col.endswith("err")

        transform = transform_for_errors if is_error_col else transform_for_values
        asinh_scale = asinh_scale_default

        # raw stats
        tr_vals = _finite_values_from_df(train_df, col)
        va_vals = _finite_values_from_df(val_df, col)

        tr_n = int(tr_vals.size)
        va_n = int(va_vals.size)

        tr_mean = float(np.mean(tr_vals)) if tr_n else float("nan")
        tr_std = float(np.std(tr_vals)) if tr_n else float("nan")
        va_mean = float(np.mean(va_vals)) if va_n else float("nan")
        va_std = float(np.std(va_vals)) if va_n else float("nan")

        # fit scaler on train
        scaler = fit_scaler_1d(
            train_df,
            col,
            transform=transform,
            asinh_scale=asinh_scale,
            clip_quantile=clip_quantile,
        )

        # save scaler
        scaler_path = scalers_dir / f"{col}.npz"
        scaler.save(scaler_path)

        # record base row
        row: Dict[str, Any] = {
            "col": col,
            "is_error_col": bool(is_error_col),
            "transform": str(transform),
            "asinh_scale": float(asinh_scale),
            "clip_quantile": float(clip_quantile) if clip_quantile is not None else None,
            "train_n_finite": tr_n,
            "val_n_finite": va_n,
            "train_mean_raw": tr_mean,
            "train_std_raw": tr_std,
            "val_mean_raw": va_mean,
            "val_std_raw": va_std,
            "scaler_mean": float(getattr(scaler, "mean", np.nan)),
            "scaler_std": float(getattr(scaler, "std", np.nan)),
            "saved_scaler": str(scaler_path),
            "saved_tokenizer": None,
            "skipped_tokenizer": True,
            "best_val_mse_unscaled": None,
            "best_val_rmse_unscaled": None,
            "best_epoch": None,
        }

        print(f"\n[{col}]")
        print(f"  train(raw): mean={tr_mean:.6g}, std={tr_std:.6g}, n={tr_n}")
        print(f"  val(raw):   mean={va_mean:.6g}, std={va_std:.6g}, n={va_n}")
        print(f"  scaler: mean={row['scaler_mean']:.6g}, std={row['scaler_std']:.6g}, transform={transform}")

        # decide train tokenizer
        if (col not in train_df.columns) or (tr_n < min_finite):
            print("  -> skip tokenizer (missing column or too few finite values). saved scaler only.")
            results.append(row)
            continue

        # build loaders
        train_dl = make_scalar_loader(train_df, col, scaler, batch_size=batch_size, shuffle=True)
        val_dl = make_scalar_loader(val_df, col, scaler, batch_size=batch_size, shuffle=False)

        # train tokenizer + save best
        tok_path = tok_dir / f"{col}.pt"
        tok, hist = train_scalar_tokenizer(
            train_dl,
            val_dl,
            codebook_size=codebook_size,
            num_stages=num_stages,
            decay=decay,
            epochs=epochs,
            ema_epochs=ema_epochs,
            device=device,
            save_path=tok_path,
            scaler=scaler,
            dtype_t=dtype_t,
        )

        # derive best from history
        best_rec = min(hist, key=lambda r: r["val_mse_unscaled"]) if hist else None

        row.update({
            "saved_tokenizer": str(tok_path),
            "skipped_tokenizer": False,
            "best_val_mse_unscaled": float(best_rec["val_mse_unscaled"]) if best_rec else None,
            "best_val_rmse_unscaled": float(best_rec["val_rmse_unscaled"]) if best_rec else None,
            "best_epoch": int(best_rec["epoch"]) if best_rec else None,
        })

        results.append(row)

    return pl.DataFrame(results)


# -------------------------
# CLI / main
# -------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    ap.add_argument("--nfiles", type=int, default=500, help="Subsample number of parquet files from datacubes_paths")
    ap.add_argument("--train_val_split", type=float, default=0.9)
    ap.add_argument("--filter_mag_r", type=float, default=21.0, help="If set, filter mag_pstotal_r < value. Use -1 to disable.")
    ap.add_argument("--device", type=str, default=None, help="cuda/cpu override")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--ema_epochs", type=int, default=10)
    ap.add_argument("--min_finite", type=int, default=200)
    ap.add_argument("--codebook_size", type=int, default=1024)
    ap.add_argument("--num_stages", type=int, default=3)
    ap.add_argument("--decay", type=float, default=0.99)
    ap.add_argument("--clip_quantile", type=float, default=0.999)
    ap.add_argument("--transform_for_errors", type=str, default="asinh", choices=["none", "asinh"])
    ap.add_argument("--transform_for_values", type=str, default="none", choices=["none", "asinh"])
    ap.add_argument("--asinh_scale", type=float, default=1.0)
    return ap.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out_root = Path(config["models_folder"])
    scalers_dir = out_root / "scalers"
    tok_dir = out_root / "tokenizers"
    scalers_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)

    # file lists
    train_files, val_files = load_datacube_files(
        config["datacubes_paths"],
        train_val_split=float(args.train_val_split),
        nfiles_subsample=int(args.nfiles) if args.nfiles and args.nfiles > 0 else None,
    )

    # load DFs
    train_df = load_datacubes_from_filelist(train_files, expected_columns=EXPECTED_COLUMNS, desc="Loading TRAIN datacubes")
    val_df = load_datacubes_from_filelist(val_files, expected_columns=EXPECTED_COLUMNS, desc="Loading VAL datacubes")

    # optional filter
    if args.filter_mag_r is not None and float(args.filter_mag_r) > 0:
        thr = float(args.filter_mag_r)
        if "mag_pstotal_r" in train_df.columns:
            train_df = train_df.filter(pl.col("mag_pstotal_r") < thr)
        if "mag_pstotal_r" in val_df.columns:
            val_df = val_df.filter(pl.col("mag_pstotal_r") < thr)

    print(f"\nTrain rows: {train_df.height:,} | Val rows: {val_df.height:,}")
    print(f"Device: {device}")
    print(f"Output: {out_root}")

    # train per-column
    summary_df = fit_and_save_all_scalars(
        train_df,
        val_df,
        scalar_columns=list(SCALAR_COLUMNS),
        scalers_dir=scalers_dir,
        tok_dir=tok_dir,
        codebook_size=int(args.codebook_size),
        num_stages=int(args.num_stages),
        decay=float(args.decay),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        ema_epochs=int(args.ema_epochs),
        min_finite=int(args.min_finite),
        device=device,
        dtype_t=torch.float32,
        transform_for_errors=str(args.transform_for_errors),
        transform_for_values=str(args.transform_for_values),
        asinh_scale_default=float(args.asinh_scale),
        clip_quantile=float(args.clip_quantile) if args.clip_quantile and args.clip_quantile > 0 else None,
    )

    # save summary
    summary_path_parquet = out_root / "scalar_fit_summary.parquet"
    summary_path_csv = out_root / "scalar_fit_summary.csv"
    summary_df.write_parquet(summary_path_parquet)
    summary_df.write_csv(summary_path_csv)

    print("\nDone.")
    print(f"Summary parquet: {summary_path_parquet}")
    print(f"Summary csv:     {summary_path_csv}")
    print(f"Scalers:         {scalers_dir}")
    print(f"Tokenizers:      {tok_dir}")


if __name__ == "__main__":
    main()