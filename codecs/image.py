#!/usr/bin/env python
import os
# pick GPU before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
from pathlib import Path
from typing import List, Sequence, Iterable, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

PARQUET_FILES: List[str] = [
    "/home/astrodados4/downloads/hypercube/datacube_HYDRA-0011.parquet",
    # add more here later
]

# RAM control: how many parquet files are loaded in memory at once
FILES_IN_MEMORY = 5

BANDS = [
    "F378", "F395", "F410", "F430",
    "F515", "F660", "F861",
    "U", "G", "R", "I", "Z",
]
IMG_SIZE = 96

# AION-like latent: more spatial resolution, fewer channels
LATENT_DIM = 4            # per-position latent dimension
BATCH_SIZE = 32
MAX_GPU_BATCH_SIZE = 32   # safety; set < BATCH_SIZE if VRAM is tight
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3

USE_AMP = True            # mixed precision
REQUIRE_F378_NOT_NULL = True

# loss weighting
W_BG = 0.3                # weight for background region
USE_SIGMA_IN_LOSS = False # keep False for now

# plotting config
PLOT_EPOCH_RESULTS = True
PLOT_BAND = "R"
PLOT_OUTPUT_DIR = "epoch_plots"
N_VISUALIZE_PER_EPOCH = 4

# =========================
# UTILS
# =========================

_GRID_CACHE = {}


def get_grid(H: int, W: int, device: torch.device):
    key = (H, W, device.type)
    if key not in _GRID_CACHE:
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        _GRID_CACHE[key] = (yy.float(), xx.float())
    return _GRID_CACHE[key]


def _to_image_torch(flat) -> torch.Tensor:
    """
    Convert flattened array-like into a square image (H, W).
    Assumes 96x96 = 9216 pixels, but works with any perfect square.
    """
    arr = torch.tensor(flat, dtype=torch.float32)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 1:
        n = arr.numel()
        side = int(math.isqrt(n))
        if side * side != n:
            raise ValueError(f"Cannot reshape length {n} into a square image")
        return arr.view(side, side)

    raise ValueError(f"Unexpected ndim={arr.ndim} for image data")


def elliptical_mask(H, W, x0, y0, a, b, theta, device="cpu", expand_factor=4.0):
    """
    Binary mask: 1 inside expanded ellipse, 0 outside.
    Uses cached coordinate grid for speed.
    """
    device = torch.device(device)
    yy, xx = get_grid(H, W, device=device)

    a_scaled = a * expand_factor
    b_scaled = b * expand_factor

    X = xx - x0
    Y = yy - y0

    ct = torch.cos(theta)
    st = torch.sin(theta)

    Xp = X * ct + Y * st
    Yp = -X * st + Y * ct

    mask = (Xp / a_scaled) ** 2 + (Yp / b_scaled) ** 2 <= 1.0
    return mask.float()


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def percentile_range(values: np.ndarray, p_lo=1.0, p_hi=99.0):
    flat = values.reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return float(0.0), float(1.0)
    v_lo = float(np.percentile(flat, p_lo))
    v_hi = float(np.percentile(flat, p_hi))
    if v_lo == v_hi:
        v_lo = float(flat.min())
        v_hi = float(flat.max())
        if v_lo == v_hi:
            v_hi = v_lo + 1.0
    return v_lo, v_hi


# =========================
# DATASET
# =========================

class SplusDataset(Dataset):
    """
    AION-style preprocessing:
      - build multi-band image (C,H,W)
      - compute valid-pixel mask per band
      - compute elliptical object mask
      - per-band robust arcsinh normalization (local background)
    """
    def __init__(self, df: pl.DataFrame):
        self.df = df
        self._len = df.height

    def __len__(self):
        return self._len

    @staticmethod
    def _arcsinh_norm_per_band(x: torch.Tensor, mask_valid: torch.Tensor, eps: float = 1e-6):
        """
        x: (C,H,W), mask_valid: (C,H,W)  (1.0 where pixels are valid flux)
        Per-band:
          - estimate background level & scale from valid pixels
          - apply arcsinh((x - median)/scale)
        """
        C, H, W = x.shape
        x_out = torch.zeros_like(x)
        for c in range(C):
            m = mask_valid[c] > 0.5
            vals = x[c][m]
            if vals.numel() < 10:
                vals = x[c].reshape(-1)
            if vals.numel() == 0:
                x_out[c] = x[c]
                continue

            med = vals.median()
            mad = (vals - med).abs().median()
            scale = mad * 3.0 + eps

            x_out[c] = torch.asinh((x[c] - med) / scale)
        return x_out

    def _build_single(self, idx: int):
        imgs = []
        masks_pix_binary = []

        # Build per-band images and basic validity masks from parquet row
        for band in BANDS:
            flat = self.df[f"splus_cut_{band}"][idx]
            img = _to_image_torch(flat)  # (H,W)

            valid = torch.isfinite(img) & (img != 0.0)
            img_clean = img.clone()
            img_clean[~torch.isfinite(img_clean)] = 0.0

            imgs.append(img_clean)
            masks_pix_binary.append(valid.float())

        x = torch.stack(imgs, dim=0)           # (C,H,W)
        m_pix_basic = torch.stack(masks_pix_binary, dim=0)

        C, H, W = x.shape
        device_t = torch.device("cpu")
        x = x.to(device_t)
        m_pix_basic = m_pix_basic.to(device_t)

        # Elliptical object mask
        x0 = torch.tensor(IMG_SIZE // 2, dtype=torch.float32, device=device_t)
        y0 = torch.tensor(IMG_SIZE // 2, dtype=torch.float32, device=device_t)

        a = torch.tensor(float(self.df["a_pixel_det"][idx]), device=device_t)
        b = torch.tensor(float(self.df["b_pixel_det"][idx]), device=device_t)
        th = torch.tensor(float(self.df["theta_det"][idx]), device=device_t)
        theta = th * math.pi / 180.0

        obj_mask = elliptical_mask(H, W, x0, y0, a, b, theta, device=device_t)
        obj_mask_full = obj_mask.unsqueeze(0).expand(C, H, W)

        # refined pixel mask: finite pixels inside ellipse
        m_pix = m_pix_basic * obj_mask_full

        # -------- AION-like normalization (per band, per object) --------
        x_norm = self._arcsinh_norm_per_band(x, m_pix_basic)

        return x_norm, m_pix, m_pix_basic  # x, object mask, basic valid mask

    def __getitem__(self, idx):
        return self._build_single(idx)


# =========================
# MODEL (AION-STYLE)
# =========================

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out = self.act(out + residual)
        return out


class AionStyleAutoEncoder(nn.Module):
    """
    Rough AION-inspired:
      - 1x1 projection: 12 → 64 channels
      - 2 downsampling blocks → 96 → 48 → 24 (spatial)
      - ResBlocks at each stage
      - bottleneck latent: (B, LATENT_DIM, 24, 24)
    """
    def __init__(self, in_channels: int = len(BANDS), latent_dim: int = LATENT_DIM):
        super().__init__()

        proj_channels = 64
        self.proj_channels = proj_channels

        # projection like AION's subsampled representation
        self.proj = nn.Conv2d(in_channels, proj_channels, kernel_size=1)

        # encoder
        self.enc_block1 = nn.Sequential(
            ResBlock(proj_channels),
            nn.Conv2d(proj_channels, 128, kernel_size=3, stride=2, padding=1),  # 96→48
            nn.ReLU(inplace=True),
        )
        self.enc_block2 = nn.Sequential(
            ResBlock(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),            # 48→24
            nn.ReLU(inplace=True),
            ResBlock(128),
        )

        # latent projection 128 → latent_dim
        self.to_latent = nn.Conv2d(128, latent_dim, kernel_size=1)

        # decoder: latent_dim→128→64→12
        self.from_latent = nn.Conv2d(latent_dim, 128, kernel_size=1)

        self.dec_block1 = nn.Sequential(
            ResBlock(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),    # 24→48
            nn.ReLU(inplace=True),
        )
        self.dec_block2 = nn.Sequential(
            ResBlock(64),
            nn.ConvTranspose2d(64, proj_channels, kernel_size=4, stride=2, padding=1),  # 48→96
            nn.ReLU(inplace=True),
            ResBlock(proj_channels),
        )

        self.out_conv = nn.Conv2d(proj_channels, in_channels, kernel_size=3, padding=1)

        # latent spatial dims: 96 / 4 = 24
        self.latent_H = IMG_SIZE // 4
        self.latent_W = IMG_SIZE // 4

    def encode(self, x):
        """
        x: (B, C, 96, 96)
        returns z_map: (B, LATENT_DIM, 24, 24)
        """
        h = self.proj(x)
        h = self.enc_block1(h)   # -> (B,128,48,48)
        h = self.enc_block2(h)   # -> (B,128,24,24)
        z_map = self.to_latent(h)
        return z_map

    def decode(self, z_map):
        """
        z_map: (B, LATENT_DIM, 24, 24)
        returns x_hat: (B, C, 96, 96)
        """
        h = self.from_latent(z_map)
        h = self.dec_block1(h)   # -> (B,64,48,48)
        h = self.dec_block2(h)   # -> (B,proj_channels,96,96)
        x_hat = self.out_conv(h) # -> (B,C,96,96)
        return x_hat

    def forward(self, x):
        z_map = self.encode(x)
        x_hat = self.decode(z_map)
        return x_hat, z_map


# =========================
# LOSS
# =========================

def reconstruction_loss(x, x_hat, m_obj, w_bg: float = W_BG, eps: float = 1e-6):
    """
    x, x_hat: (B,C,H,W) normalized flux
    m_obj: (B,C,H,W) object mask (ellipse ∩ valid-pixels)
    We focus on the object region but allow background to influence training.
    """
    diff2 = (x_hat - x) ** 2

    m_all = torch.ones_like(m_obj)
    m_bg = m_all - m_obj

    weights = w_bg * m_bg + 1.0 * m_obj

    num = (weights * diff2).sum()
    den = weights.sum() + eps
    return num / den


# =========================
# LOADING HELPERS
# =========================

def chunked(iterable: Sequence[str], n: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]


def load_parquet_chunk(
    file_paths: Sequence[str],
    use_columns: Sequence[str] = None
) -> pl.DataFrame:
    dfs = []
    for path in file_paths:
        if use_columns is not None:
            df = pl.read_parquet(path, columns=use_columns)
        else:
            df = pl.read_parquet(path)

        if REQUIRE_F378_NOT_NULL and "splus_cut_F378" in df.columns:
            df = df.filter(pl.col("splus_cut_F378").is_not_null())

        dfs.append(df)

    if len(dfs) == 1:
        return dfs[0]
    return pl.concat(dfs, how="vertical")


def estimate_total_rows(paths: Sequence[str]) -> int:
    total = 0
    for path in paths:
        df = pl.read_parquet(path, columns=["gaia_source_id"])
        if REQUIRE_F378_NOT_NULL:
            df = df.filter(pl.col("gaia_source_id").is_not_null())
        total += df.height
    return total


def build_loaders(device: str) -> List[Tuple[str, Sequence[str], DataLoader]]:
    loaders_info: List[Tuple[str, Sequence[str], DataLoader]] = []
    wanted_cols = (
        [f"splus_cut_{b}" for b in BANDS] +
        ["theta_det", "a_pixel_det", "b_pixel_det"]
    )

    physical_batch_size = min(BATCH_SIZE, MAX_GPU_BATCH_SIZE)

    for chunk_idx, file_chunk in enumerate(chunked(PARQUET_FILES, FILES_IN_MEMORY), start=1):
        print(f"\n[setup] Loading file chunk {chunk_idx} once:")
        for p in file_chunk:
            print(f"       - {p}")

        df = load_parquet_chunk(file_chunk, use_columns=wanted_cols)
        if df.height == 0:
            print("       (chunk empty after filtering, skipping)")
            continue

        dataset = SplusDataset(df)
        loader = DataLoader(
            dataset,
            batch_size=physical_batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=(device == "cuda"),
            persistent_workers=(device == "cuda" and 2 > 0),
        )

        chunk_label = f"{chunk_idx}"
        loaders_info.append((chunk_label, file_chunk, loader))

        print(f"       -> rows in this chunk: {df.height}")
        print(f"       -> physical batch size: {physical_batch_size}")

    if not loaders_info:
        raise RuntimeError("No non-empty chunks found after filtering.")
    return loaders_info


# =========================
# VISUALIZATION
# =========================

def visualize_epoch(model: nn.Module, device: str, epoch: int):
    if not PLOT_EPOCH_RESULTS:
        return

    ensure_dir(PLOT_OUTPUT_DIR)

    if len(PARQUET_FILES) == 0:
        print("No PARQUET_FILES defined, skipping visualization.")
        return

    df = pl.read_parquet(PARQUET_FILES[0])
    if REQUIRE_F378_NOT_NULL:
        df = df.filter(pl.col("splus_cut_F378").is_not_null())
    if df.height == 0:
        print("First parquet file is empty after filtering, skipping visualization.")
        return

    N_vis = min(N_VISUALIZE_PER_EPOCH, df.height)

    # reuse dataset logic
    ds = SplusDataset(df)

    xs = []
    m_objs = []
    for idx in range(N_vis):
        x_norm, m_obj, _ = ds[idx]
        xs.append(x_norm)
        m_objs.append(m_obj)

    x_batch = torch.stack(xs, dim=0).to(device)     # (N,C,H,W)
    m_batch = torch.stack(m_objs, dim=0).to(device)

    with torch.no_grad():
        model.eval()
        x_hat_batch, z_map = model(x_batch)
        x_hat_batch = x_hat_batch.detach().cpu()
        x_batch_cpu = x_batch.detach().cpu()
        m_batch_cpu = m_batch.detach().cpu()

    # band index
    if PLOT_BAND in BANDS:
        band_idx = BANDS.index(PLOT_BAND)
    else:
        band_idx = 0

    fig, axes = plt.subplots(
        N_vis, 3, figsize=(12, 4 * N_vis),
        squeeze=False
    )

    for row in range(N_vis):
        x_orig = x_batch_cpu[row]
        x_hat = x_hat_batch[row]
        m_obj = m_batch_cpu[row]

        img_orig = x_orig[band_idx].numpy()
        img_rec  = x_hat[band_idx].numpy()
        img_res  = img_rec - img_orig

        mask_band = m_obj[band_idx].numpy()
        valid_vals_orig = img_orig[mask_band > 0]
        valid_vals_rec  = img_rec[mask_band > 0]
        if valid_vals_orig.size == 0 or valid_vals_rec.size == 0:
            valid_vals_orig = img_orig
            valid_vals_rec  = img_rec

        vmin_o, vmax_o = percentile_range(valid_vals_orig, 1, 99)
        vmin_r, vmax_r = percentile_range(valid_vals_rec, 1, 99)
        vmin = min(vmin_o, vmin_r)
        vmax = max(vmax_o, vmax_r)

        valid_res = img_res[mask_band > 0]
        if valid_res.size == 0:
            valid_res = img_res
        res_amp = float(np.percentile(np.abs(valid_res.reshape(-1)), 99.0))
        if res_amp == 0.0:
            res_amp = 1.0

        ax0, ax1, ax2 = axes[row]

        im0 = ax0.imshow(img_orig, origin="lower", cmap="gray",
                         vmin=vmin, vmax=vmax)
        ax0.set_title(f"Original norm ({BANDS[band_idx]}) idx={row}")
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        im1 = ax1.imshow(img_rec, origin="lower", cmap="gray",
                         vmin=vmin, vmax=vmax)
        ax1.set_title("Reconstruction")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        im2 = ax2.imshow(img_res, origin="lower", cmap="bwr",
                         vmin=-res_amp, vmax=res_amp)
        ax2.set_title("Residual")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        for ax in (ax0, ax1, ax2):
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    out_path = Path(PLOT_OUTPUT_DIR) / f"epoch_{epoch:03d}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] Saved epoch {epoch} visualization to {out_path}")


# =========================
# TRAINING LOOP
# =========================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    try:
        total_rows = estimate_total_rows(PARQUET_FILES)
        print(f"Estimated total rows across all files: {total_rows}")
    except Exception as e:
        print("Could not estimate total rows:", e)

    loaders_info = build_loaders(device)

    model = AionStyleAutoEncoder(in_channels=len(BANDS), latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device == "cuda"))

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss_sum = 0.0
        epoch_n = 0

        print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===")

        for chunk_label, file_chunk, loader in loaders_info:
            print(f"\n  -> Epoch {epoch+1}, using preloaded file chunk {chunk_label}:")
            for p in file_chunk:
                print(f"     - {p}")

            with tqdm(loader, desc=f"Epoch {epoch+1} | Chunk {chunk_label}", unit="batch") as tbar:
                for x_norm, m_obj, m_valid in tbar:
                    x_norm = x_norm.to(device, non_blocking=True)
                    m_obj = m_obj.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)

                    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                        x_hat, z_map = model(x_norm)
                        loss_recon = reconstruction_loss(
                            x=x_norm,
                            x_hat=x_hat,
                            m_obj=m_obj,
                            w_bg=W_BG,
                        )
                        loss = loss_recon

                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    bsz = x_norm.size(0)
                    epoch_loss_sum += loss_recon.item() * bsz
                    epoch_n += bsz

                    tbar.set_postfix(loss=loss_recon.item())

        epoch_loss = epoch_loss_sum / epoch_n if epoch_n > 0 else float("nan")
        print(f"\nEpoch {epoch+1} finished. Mean recon loss = {epoch_loss:.4e}")

        visualize_epoch(model, device, epoch+1)

    out_path = "aion_style_autoencoder_splus.pt"
    torch.save(model.state_dict(), out_path)
    print(f"\nModel saved to {out_path}")


if __name__ == "__main__":
    main()