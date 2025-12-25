#!/usr/bin/env python
import os
# pick GPU before importing torch (same as training)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
from pathlib import Path
from typing import List, Sequence, Iterable, Tuple, Optional

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# =========================
# CONFIG
# =========================

# Folder containing your datacube parquet files
INPUT_DIR = "/home/astrodados4/downloads/hypercube"

# Where to save the latents
OUTPUT_DIR = "output_latents"

# Model weights (from your training script)
MODEL_PATH = "autoencoder_splus_cuts.pt"

# Parquet file pattern
FILE_PATTERN = "datacube_*.parquet"

BANDS = [
    "F378", "F395", "F410", "F430",
    "F515", "F660", "F861",
    "U", "G", "R", "I", "Z",
]
IMG_SIZE = 96
LATENT_DIM = 2          # must match training
BATCH_SIZE = 64
MAX_GPU_BATCH_SIZE = 32
REQUIRE_F378_NOT_NULL = True

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


# =========================
# DATASET (same as training)
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

            if flat is None:
                img = torch.zeros((IMG_SIZE, IMG_SIZE), dtype=torch.float32)
                valid = torch.zeros((IMG_SIZE, IMG_SIZE), dtype=torch.bool)
            else:
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
# MODEL (same as training)
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


class AutoEncoder(nn.Module):
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
# LOADING HELPERS
# =========================

def chunked(iterable: Sequence[str], n: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]


def load_parquet(
    path: str,
    use_columns: Sequence[str],
) -> pl.DataFrame:
    df = pl.read_parquet(path, columns=use_columns)

    if REQUIRE_F378_NOT_NULL and "splus_cut_F378" in df.columns:
        df = df.filter(pl.col("splus_cut_F378").is_not_null())

    return df


def build_loader_for_df(df: pl.DataFrame, device: str) -> DataLoader:
    physical_batch_size = min(BATCH_SIZE, MAX_GPU_BATCH_SIZE)
    dataset = SplusDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=physical_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
        persistent_workers=(device == "cuda" and 2 > 0),
    )
    return loader


def get_parquet_files(input_dir: str, pattern: str) -> List[Path]:
    p = Path(input_dir)
    files = sorted(p.glob(pattern))
    return files


# =========================
# INFERENCE
# =========================

def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    ensure_dir(OUTPUT_DIR)

    files = get_parquet_files(INPUT_DIR, FILE_PATTERN)
    if not files:
        print(f"No files found in {INPUT_DIR} matching pattern {FILE_PATTERN}")
        return

    print("Found parquet files:")
    for f in files:
        print("  -", f)

    # ---- Load model ----
    model = AutoEncoder(in_channels=len(BANDS), latent_dim=LATENT_DIM).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model weights from {MODEL_PATH}")

    wanted_cols = [f"splus_cut_{b}" for b in BANDS] + ["theta_det", "a_pixel_det", "b_pixel_det"]
    # Optionally add IDs if present
    maybe_id_cols = ["gaia_source_id", "id", "object_id"]

    for path in files:
        print(f"\nProcessing file: {path}")
        
        out_name = path.stem + "_latents.npz"
        out_path = Path(OUTPUT_DIR) / out_name
        
        if out_path.exists():
            print(f"  -> Output file {out_path} already exists, skipping.")
            continue
        
        # Try to include an ID column if it exists in the schema
        df_full = pl.read_parquet(path, n_rows=1)
        id_col: Optional[str] = None
        for col in maybe_id_cols:
            if col in df_full.columns:
                id_col = col
                break

        use_cols = wanted_cols.copy()
        if id_col is not None:
            use_cols = use_cols + [id_col]

        # Load full df with selected columns
        df = load_parquet(str(path), use_columns=use_cols)
        if df.height == 0:
            print("  -> File is empty after filtering, skipping.")
            continue

        print(f"  -> Rows after filtering: {df.height}")
        loader = build_loader_for_df(df, device)

        all_latents = []
        with torch.no_grad():
            for (x_norm, m_obj, m_valid) in tqdm(loader, desc=f"Inference {path.name}", unit="batch"):
                x_norm = x_norm.to(device, non_blocking=True)
                # we only need z_map for inference; we ignore reconstruction
                _, z_map = model(x_norm)   # (B, LATENT_DIM, 24, 24)
                all_latents.append(z_map.detach().cpu().numpy())

        latents = np.concatenate(all_latents, axis=0)  # (N, LATENT_DIM, 24, 24)
        print(f"  -> Latents shape: {latents.shape}")

        # collect IDs if we have them
        ids = None
        if id_col is not None and id_col in df.columns:
            ids = df[id_col].to_numpy()
            print(f"  -> Saved ID column: {id_col}, shape={ids.shape}")

        # Save to NPZ
        out_name = path.stem + "_latents.npz"
        out_path = Path(OUTPUT_DIR) / out_name

        if ids is not None:
            np.savez_compressed(out_path, latents=latents, ids=ids, id_col=id_col)
        else:
            np.savez_compressed(out_path, latents=latents)

        print(f"  -> Saved latents to {out_path}")


if __name__ == "__main__":
    run_inference()