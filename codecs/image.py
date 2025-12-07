#!/usr/bin/env python
import os
# 🔹 MUST be before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
from pathlib import Path
from typing import List, Sequence, Iterable

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")  # for headless servers
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

# List of parquet files to train on
PARQUET_FILES: List[str] = [
    "/home/astrodados4/downloads/hypercube/datacube_HYDRA-0011.parquet",
    "/home/astrodados4/downloads/hypercube/datacube_HYDRA-0012.parquet",
    "/home/astrodados4/downloads/hypercube/datacube_SPLUS-n04s06.parquet",
    "/home/astrodados4/downloads/hypercube/datacube_SPLUS-s19s51.parquet",
    "/home/astrodados4/downloads/hypercube/datacube_SPLUS-s20s06.parquet",
]

# Max number of parquet files to hold in memory at once
FILES_IN_MEMORY = 5   # N at a time

# Data / training config
BANDS = [
    "F378", "F395", "F410", "F430",
    "F515", "F660", "F861",
    "U", "G", "R", "I", "Z",
]
IMG_SIZE = 96
LATENT_DIM = 8          # per-position latent dimension (channels in latent map)
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

# If True, compute Sigma (slower). If False, use only m_pix in the loss.
USE_SIGMA_IN_LOSS = True

# Optional: minimum requirement to keep a row
REQUIRE_F378_NOT_NULL = True

# ---------- QINCo CONFIG ----------
QINCO_USE = True          # turn QINCo bottleneck on/off
QINCO_STEPS = 4           # M: number of residual quantization steps
QINCO_K = 1024             # K: centroids per step (vocab size)
QINCO_LAMBDA = 0.1        # weight of QINCo loss vs recon loss

# ---------- PLOTTING CONFIG ----------
PLOT_EPOCH_RESULTS = True
PLOT_BAND = "R"           # which band to visualize
PLOT_OUTPUT_DIR = "epoch_plots"


# =========================
# UTILS
# =========================

# Cache for grids (so we don't rebuild meshgrid every row)
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
    arr = torch.as_tensor(flat, dtype=torch.float32)

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
    st = torch.cos(theta) * 0 + torch.sin(theta)  # to keep numeric same but allow JIT? nah ignore :)
    ct = torch.cos(theta)
    st = torch.sin(theta)

    Xp =  X * ct + Y * st
    Yp = -X * st + Y * ct

    mask = (Xp / a_scaled) ** 2 + (Yp / b_scaled) ** 2 <= 1.0
    return mask.float()


def compute_sigma_detection_aware_single(x: torch.Tensor,
                                         m_pix: torch.Tensor,
                                         eps: float = 1e-3) -> torch.Tensor:
    """
    Per-object detection-aware Sigma.

    x    : (C, H, W) flux (on device: CPU or CUDA)
    m_pix: (C, H, W) mask of valid object pixels (0/1)
    """
    device = x.device
    C, H, W = x.shape
    Sigma = torch.zeros_like(x, device=device)

    for c in range(C):
        mask_in = m_pix[c] > 0.5
        mask_out = ~mask_in

        bg_vals = x[c][mask_out]
        if bg_vals.numel() < 10:
            bg_vals = x[c][torch.isfinite(x[c])]

        if bg_vals.numel() == 0:
            continue

        med_bg = bg_vals.median()
        mad_bg = (bg_vals - med_bg).abs().median()

        sigma_bg = mad_bg + eps

        obj_vals = x[c][mask_in]
        if obj_vals.numel() == 0:
            continue

        med_in = obj_vals.median()

        snr_band = (med_in - med_bg) / (sigma_bg + eps)
        snr_band = torch.clamp(snr_band, min=0.0)

        weight = snr_band / (snr_band + 1.0 + eps)

        base_invvar = 1.0 / (sigma_bg ** 2 + eps)
        Sigma[c][mask_in] = weight * base_invvar

    return Sigma


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def percentile_range(values: np.ndarray, p_lo=1.0, p_hi=99.0):
    """Safe percentile-based range for plotting."""
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
    Optimized dataset:
      - Polars is used only once to create numpy arrays.
      - No Sigma computation here (we do it per batch on GPU if needed).
    """
    def __init__(self, df: pl.DataFrame, compute_sigma: bool = False):
        self.compute_sigma = compute_sigma  # kept for API symmetry, not used

        # Pre-extract columns as numpy/object arrays (fast to index in workers)
        self.cuts = {
            band: df[f"splus_cut_{band}"].to_numpy()
            for band in BANDS
        }
        self.a_pixel = df["a_pixel_det"].to_numpy()
        self.b_pixel = df["b_pixel_det"].to_numpy()
        self.theta_det = df["theta_det"].to_numpy()

        self._len = df.height

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        imgs = []
        masks_pix_binary = []

        # Build per-band images and basic validity masks
        for band in BANDS:
            flat = self.cuts[band][idx]
            img = _to_image_torch(flat)  # (H,W)

            valid = torch.isfinite(img) & (img != 0.0)

            img_clean = img.clone()
            img_clean[~torch.isfinite(img_clean)] = 0.0

            imgs.append(img_clean)
            masks_pix_binary.append(valid.float())

        x = torch.stack(imgs, dim=0)           # (C, H, W)
        m_pix_basic = torch.stack(masks_pix_binary, dim=0)

        C, H, W = x.shape
        device_t = torch.device("cpu")

        x = x.to(device_t)
        m_pix_basic = m_pix_basic.to(device_t)

        # center of image
        x0 = torch.tensor(IMG_SIZE // 2, dtype=torch.float32, device=device_t)
        y0 = torch.tensor(IMG_SIZE // 2, dtype=torch.float32, device=device_t)

        a  = torch.tensor(float(self.a_pixel[idx]), device=device_t)
        b  = torch.tensor(float(self.b_pixel[idx]), device=device_t)
        th = torch.tensor(float(self.theta_det[idx]), device=device_t)
        theta = th * math.pi / 180.0

        obj_mask = elliptical_mask(H, W, x0, y0, a, b, theta, device=device_t)
        obj_mask_full = obj_mask.unsqueeze(0).expand(C, H, W)

        # refined pixel mask: finite pixels inside ellipse
        m_pix = m_pix_basic * obj_mask_full

        # per-band flag
        m_band = (m_pix.sum(dim=(1, 2)) > 0).float()

        # Sigma NOT computed here (done later if enabled)
        Sigma = torch.zeros_like(x)

        return x, Sigma, m_pix, m_band


# =========================
# QINCo MODULES
# =========================

class QINCoStep(nn.Module):
    """
    One QINCo residual step for vectors z ∈ R^D.
    Implements:
      - data-dependent codebook C_m(x_hat) via small MLP
      - nearest-centroid selection in residual space
      - updated reconstruction x_hat_new = x_hat + c_sel
    """
    def __init__(self, D: int, K: int, hidden_dim: int = 256, num_res_blocks: int = 2):
        super().__init__()
        self.D = D
        self.K = K

        # Base codebook C̄_m (K, D)
        self.base_codebook = nn.Parameter(torch.randn(K, D) * 0.1)

        # First projection of [x_hat, c̄_k] ∈ R^{2D} -> R^D
        self.concat_proj = nn.Linear(2 * D, D)

        # Residual MLP blocks on top of that
        blocks = []
        for _ in range(num_res_blocks):
            blocks.append(nn.Sequential(
                nn.Linear(D, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, D),
            ))
        self.blocks = nn.ModuleList(blocks)

    def forward_codebook(self, x_hat: torch.Tensor) -> torch.Tensor:
        """
        x_hat: (N, D) partial reconstruction
        returns:
          C: (N, K, D) data-dependent codebook
        """
        N, D = x_hat.shape
        x_exp = x_hat.unsqueeze(1).expand(-1, self.K, -1)              # (N, K, D)
        cbar = self.base_codebook.unsqueeze(0).expand(N, -1, -1)       # (N, K, D)
        concat = torch.cat([x_exp, cbar], dim=-1)                      # (N, K, 2D)

        C = self.concat_proj(concat)                                   # (N, K, D)
        for block in self.blocks:
            C = C + block(C)                                           # residual blocks

        return C                                                       # (N, K, D)

    def encode_step(self, x: torch.Tensor, x_hat: torch.Tensor):
        """
        One QINCo step:
          x     : (N, D) original vector
          x_hat : (N, D) current reconstruction

        returns:
          codes     : (N,) indices i_m
          x_hat_new : (N, D) updated reconstruction
          r         : (N, D) residual before step
          c_sel     : (N, D) selected centroid
        """
        N, D = x.shape

        # Build codebook C conditioned on current reconstruction
        C = self.forward_codebook(x_hat)                               # (N, K, D)

        # Residual r = x - x_hat
        r = x - x_hat                                                  # (N, D)

        # Distances to all centroids in residual space
        r_exp = r.unsqueeze(1).expand(-1, self.K, -1)                  # (N, K, D)
        dists = torch.sum((r_exp - C) ** 2, dim=-1)                    # (N, K)

        # Nearest centroid index
        codes = torch.argmin(dists, dim=-1)                            # (N,)

        # Gather chosen centroids c_sel
        C_flat = C.reshape(N * self.K, D)
        idx = codes + torch.arange(N, device=x.device) * self.K
        c_sel = C_flat[idx]                                            # (N, D)

        # Update reconstruction
        x_hat_new = x_hat + c_sel                                      # (N, D)

        return codes, x_hat_new, r, c_sel


class QINCoQuantizer(nn.Module):
    """
    Multi-step QINCo quantizer over latent vectors z ∈ R^D.

    During forward:
      - runs M residual steps
      - returns:
          z_q_st : straight-through quantized latent (N, D)
          codes  : (N, M)
          aux    : residuals & selected centroids for QINCo loss
    """
    def __init__(self, D: int, K: int = 256, M: int = 4):
        super().__init__()
        self.M = M
        self.steps = nn.ModuleList(
            [QINCoStep(D, K) for _ in range(M)]
        )

    def forward(self, z: torch.Tensor):
        """
        z: (N, D)
        returns:
          z_q_st : (N, D) straight-through quantized latent
          codes  : (N, M)
          aux    : dict for QINCo loss
        """
        N, D = z.shape
        x_hat = torch.zeros_like(z)

        codes_all = []
        residuals = []
        selected_centroids = []

        for step in self.steps:
            codes, x_hat, r, c_sel = step.encode_step(z, x_hat)
            codes_all.append(codes)
            residuals.append(r)
            selected_centroids.append(c_sel)

        codes_all = torch.stack(codes_all, dim=-1)  # (N, M)

        # Reconstruction from QINCo:
        z_q = x_hat                                   # (N, D)

        # Straight-through estimator: forward uses z_q, backward flows to z
        z_q_st = z + (z_q - z).detach()

        aux = {
            "residuals": residuals,         # list of M tensors (N, D)
            "centroids": selected_centroids # list of M tensors (N, D)
        }

        return z_q_st, codes_all, aux


class QINCoQuantizerSpatial(nn.Module):
    """
    Apply QINCo to each spatial location of a latent map z_map (B, D, H, W).

    Returns:
      z_q_map : (B, D, H, W)
      codes   : (B, H, W, M) integer codes in [0..K-1]
      aux     : inner aux dict for QINCo loss
    """
    def __init__(self, D: int, H: int, W: int, K: int = 256, M: int = 4):
        super().__init__()
        self.D = D
        self.H = H
        self.W = W
        self.inner = QINCoQuantizer(D=D, K=K, M=M)

    def forward(self, z_map: torch.Tensor):
        """
        z_map: (B, D, H, W)
        """
        B, D, H, W = z_map.shape
        assert D == self.D
        assert H == self.H and W == self.W

        # (B, D, H, W) -> (B, H, W, D) -> (N, D)
        z_flat = z_map.permute(0, 2, 3, 1).reshape(-1, D)   # (N, D), N = B*H*W

        z_q_flat, codes_flat, aux = self.inner(z_flat)      # (N, D), (N, M)

        # back to spatial map
        z_q_map = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)
        codes = codes_flat.view(B, H, W, -1)                     # (B, H, W, M)

        return z_q_map, codes, aux


def qinco_loss(aux) -> torch.Tensor:
    """
    Simple QINCo loss: sum over steps of MSE(residual, selected centroid).
    """
    residuals = aux["residuals"]
    centroids = aux["centroids"]

    loss = 0.0
    for r, c_sel in zip(residuals, centroids):
        loss = loss + F.mse_loss(c_sel, r)

    return loss


# =========================
# MODEL
# =========================

class ConvAutoEncoder(nn.Module):
    def __init__(self, in_channels: int = len(BANDS), latent_dim: int = LATENT_DIM):
        super().__init__()

        # Encoder: 96→48→24→12
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 96 -> 48

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 48 -> 24

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24 -> 12
        )

        # After 3x MaxPool2d(2), 96 -> 12
        self.latent_H = IMG_SIZE // 8
        self.latent_W = IMG_SIZE // 8

        # Project 128 channels to latent_dim at each spatial location
        self.to_latent = nn.Conv2d(128, latent_dim, kernel_size=1)
        self.from_latent = nn.Conv2d(latent_dim, 128, kernel_size=1)

        # QINCo quantizer on latent map (B, latent_dim, H, W)
        if QINCO_USE:
            self.qinco = QINCoQuantizerSpatial(
                D=latent_dim,
                H=self.latent_H,
                W=self.latent_W,
                K=QINCO_K,
                M=QINCO_STEPS,
            )
        else:
            self.qinco = None

        # Decoder: 12→24→48→96
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 12 -> 24
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),   # 24 -> 48
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, in_channels, kernel_size=2, stride=2),  # 48 -> 96
            # no activation: regress flux directly
        )

    def encode(self, x):
        """
        x: (B, C, H, W)
        returns z_map: (B, latent_dim, H_lat, W_lat)
        """
        h = self.encoder_conv(x)                     # (B,128,H_lat,W_lat)
        z_map = self.to_latent(h)                    # (B,D,H_lat,W_lat)
        return z_map

    def decode(self, z_map):
        """
        z_map: (B, latent_dim, H_lat, W_lat)
        returns x_hat: (B, C, H, W)
        """
        h = self.from_latent(z_map)                  # (B,128,H_lat,W_lat)
        x_hat = self.decoder(h)                      # (B,C,96,96)
        return x_hat

    def forward(self, x):
        """
        Returns:
          x_hat : reconstructed image (B,C,H,W)
          z_map : original continuous latent map (B,D,H_lat,W_lat)
          codes : (B,H_lat,W_lat,M) int codes (or None if QINCo disabled)
          q_aux : dict for QINCo loss (or None)
        """
        z_map = self.encode(x)

        if self.qinco is not None:
            z_q_map, codes, q_aux = self.qinco(z_map)
            x_hat = self.decode(z_q_map)
        else:
            codes, q_aux = None, None
            x_hat = self.decode(z_map)

        return x_hat, z_map, codes, q_aux


def reconstruction_loss(x, x_hat, m_pix, Sigma=None, eps=1e-6):
    """
    x, x_hat, m_pix, Sigma: (B, C, H, W)
    - m_pix: 0/1 pixel validity mask
    - Sigma: optional per-pixel inverse variance (0 inside object)

    The loss is computed ONLY where m_pix == 1 (and optionally weighted by Sigma).
    """
    diff2 = (x_hat - x) ** 2

    if Sigma is not None:
        weights = Sigma * m_pix
    else:
        weights = m_pix

    num = (weights * diff2).sum()
    den = weights.sum() + eps

    return num / den


# =========================
# TRAINING HELPERS
# =========================

def chunked(iterable: Sequence[str], n: int) -> Iterable[Sequence[str]]:
    """Yield lists of length <= n from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]


def load_parquet_chunk(file_paths: Sequence[str]) -> pl.DataFrame:
    dfs = []
    for path in file_paths:
        df = pl.read_parquet(path)
        if REQUIRE_F378_NOT_NULL:
            df = df.filter(pl.col("splus_cut_F378").is_not_null())
        dfs.append(df)
    if len(dfs) == 1:
        return dfs[0]
    return pl.concat(dfs, how="vertical")


def estimate_total_rows(paths: Sequence[str]) -> int:
    """
    Rough estimate of total rows (for info/progress printing).
    Uses only a small cheap column.
    """
    total = 0
    for path in paths:
        df = pl.read_parquet(path, columns=["a_pixel_det"])
        if REQUIRE_F378_NOT_NULL:
            df = df.filter(pl.col("a_pixel_det").is_not_null())
        total += df.height
    return total


def visualize_epoch(model: nn.Module, device: str, epoch: int):
    """
    Plot original, reconstruction, and residual for a single example
    from the first parquet file and save as PNG.

    - Rebuilds one sample from polars directly (for simplicity).
    - Uses same elliptical+validity mask logic.
    """
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

    # ----- reconstruct one object using same logic as dataset -----
    idx = 0
    with torch.no_grad():
        imgs = []
        masks_pix_binary = []
        for band in BANDS:
            flat = df[f"splus_cut_{band}"][idx]
            img = _to_image_torch(flat)
            valid = torch.isfinite(img) & (img != 0.0)
            img_clean = img.clone()
            img_clean[~torch.isfinite(img_clean)] = 0.0
            imgs.append(img_clean)
            masks_pix_binary.append(valid.float())

        x = torch.stack(imgs, dim=0)           # (C,H,W)
        m_pix_basic = torch.stack(masks_pix_binary, dim=0)

        C, H, W = x.shape
        device_cpu = torch.device("cpu")
        x = x.to(device_cpu)
        m_pix_basic = m_pix_basic.to(device_cpu)

        x0 = torch.tensor(IMG_SIZE // 2, dtype=torch.float32, device=device_cpu)
        y0 = torch.tensor(IMG_SIZE // 2, dtype=torch.float32, device=device_cpu)
        a  = torch.tensor(float(df["a_pixel_det"][idx]), device=device_cpu)
        b  = torch.tensor(float(df["b_pixel_det"][idx]), device=device_cpu)
        th = torch.tensor(float(df["theta_det"][idx]), device=device_cpu)
        theta = th * math.pi / 180.0

        obj_mask = elliptical_mask(H, W, x0, y0, a, b, theta, device=device_cpu)
        obj_mask_full = obj_mask.unsqueeze(0).expand(C, H, W)
        m_pix = m_pix_basic * obj_mask_full

        x_batch = x.unsqueeze(0).to(device)          # (1,C,H,W)
        m_pix_batch = m_pix.unsqueeze(0).to(device)  # (1,C,H,W)

        x_in = x_batch * m_pix_batch

        model.eval()
        x_hat_batch, z_map, codes, q_aux = model(x_in)
        x_hat = x_hat_batch[0].detach().cpu()        # (C,H,W)

    x_orig = x                                    # (C,H,W)
    m_pix_cpu = m_pix                             # (C,H,W)

    # Pick band index
    if PLOT_BAND in BANDS:
        band_idx = BANDS.index(PLOT_BAND)
    else:
        band_idx = 0

    img_orig = x_orig[band_idx].numpy()          # (H,W)
    img_rec  = x_hat[band_idx].numpy()           # (H,W)
    mask_band = m_pix_cpu[band_idx].numpy()      # (H,W), 0/1

    # Apply mask: only object region + valid pixels
    img_orig_masked = img_orig * mask_band
    img_rec_masked  = img_rec  * mask_band
    img_residual    = (img_rec - img_orig) * mask_band

    # Compute plotting ranges using percentiles over masked pixels
    valid_vals_orig = img_orig_masked[mask_band > 0]
    valid_vals_rec  = img_rec_masked[mask_band > 0]
    if valid_vals_orig.size == 0 or valid_vals_rec.size == 0:
        valid_vals_orig = img_orig
        valid_vals_rec  = img_rec

    vmin_o, vmax_o = percentile_range(valid_vals_orig, 1, 99)
    vmin_r, vmax_r = percentile_range(valid_vals_rec, 1, 99)
    vmin = min(vmin_o, vmin_r)
    vmax = max(vmax_o, vmax_r)

    # Residual range: symmetric around zero
    valid_res = img_residual[mask_band > 0]
    if valid_res.size == 0:
        valid_res = img_residual
    res_amp = float(np.percentile(np.abs(valid_res.reshape(-1)), 99.0))
    if res_amp == 0.0:
        res_amp = 1.0

    # For plotting, set outside mask to NaN so it doesn't show
    img_orig_plot = img_orig_masked.copy()
    img_rec_plot  = img_rec_masked.copy()
    img_res_plot  = img_residual.copy()

    img_orig_plot[mask_band <= 0] = np.nan
    img_rec_plot[mask_band <= 0]  = np.nan
    img_res_plot[mask_band <= 0]  = np.nan

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ax0, ax1, ax2 = axes

    im0 = ax0.imshow(img_orig_plot, origin="lower", cmap="gray",
                     vmin=vmin, vmax=vmax)
    ax0.set_title(f"Original ({BANDS[band_idx]})")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    im1 = ax1.imshow(img_rec_plot, origin="lower", cmap="gray",
                     vmin=vmin, vmax=vmax)
    ax1.set_title("Reconstruction")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(img_res_plot, origin="lower", cmap="bwr",
                     vmin=-res_amp, vmax=res_amp)
    ax2.set_title("Residual (masked only)")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    for ax in axes:
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

    # Optional: estimate total rows for info
    try:
        total_rows = estimate_total_rows(PARQUET_FILES)
        print(f"Estimated total rows across all files: {total_rows}")
    except Exception as e:
        print("Could not estimate total rows:", e)
        total_rows = None

    model = ConvAutoEncoder(in_channels=len(BANDS), latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss_sum = 0.0
        epoch_n = 0

        print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===")
        for chunk_idx, file_chunk in enumerate(chunked(PARQUET_FILES, FILES_IN_MEMORY), start=1):
            print(f"\n  -> Loading file chunk {chunk_idx}:")
            for p in file_chunk:
                print(f"     - {p}")

            df = load_parquet_chunk(file_chunk)
            if df.height == 0:
                print("  (chunk empty after filtering, skipping)")
                continue

            # Dataset now lightweight (no Sigma, no polars in __getitem__)
            dataset = SplusDataset(df, compute_sigma=False)
            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=4,                # tune based on CPU cores
                pin_memory=(device == "cuda"),
                persistent_workers=True,
            )

            with tqdm(loader, desc=f"Epoch {epoch+1} | Chunk {chunk_idx}", unit="batch") as tbar:
                for x, _, m_pix, m_band in tbar:
                    x = x.to(device, non_blocking=True)          # (B,C,H,W)
                    m_pix = m_pix.to(device, non_blocking=True)

                    # Optional Sigma on GPU
                    if USE_SIGMA_IN_LOSS:
                        Sigma_list = []
                        for i in range(x.size(0)):
                            Sigma_i = compute_sigma_detection_aware_single(
                                x[i], m_pix[i]
                            )
                            Sigma_list.append(Sigma_i)
                        Sigma_for_loss = torch.stack(Sigma_list, dim=0)  # (B,C,H,W)
                    else:
                        Sigma_for_loss = None

                    # Encoder sees only object pixels
                    x_in = x * m_pix

                    optimizer.zero_grad()

                    # Forward with QINCo bottleneck
                    x_hat, z_map, codes, q_aux = model(x_in)

                    # Reconstruction loss (masked by m_pix, optionally Sigma)
                    loss_recon = reconstruction_loss(
                        x=x,
                        x_hat=x_hat,
                        m_pix=m_pix,
                        Sigma=Sigma_for_loss,
                    )

                    # QINCo regularization
                    if QINCO_USE and q_aux is not None:
                        loss_q = qinco_loss(q_aux)
                        loss = loss_recon + QINCO_LAMBDA * loss_q
                    else:
                        loss_q = torch.tensor(0.0, device=device)
                        loss = loss_recon

                    loss.backward()
                    optimizer.step()

                    batch_size = x.size(0)
                    epoch_loss_sum += loss_recon.item() * batch_size
                    epoch_n += batch_size

                    tbar.set_postfix(
                        loss=loss_recon.item(),
                        qinco=loss_q.item() if QINCO_USE else 0.0
                    )

        epoch_loss = epoch_loss_sum / epoch_n if epoch_n > 0 else float("nan")
        print(f"\nEpoch {epoch+1} finished. Mean recon loss = {epoch_loss:.4e}")

        # Visualization for this epoch
        visualize_epoch(model, device, epoch+1)

    # Save model at the end
    out_path = "conv_autoencoder_splus_qinco_spatial.pt" if QINCO_USE else "conv_autoencoder_splus_spatial.pt"
    torch.save(model.state_dict(), out_path)
    print(f"\nModel saved to {out_path}")


if __name__ == "__main__":
    main()