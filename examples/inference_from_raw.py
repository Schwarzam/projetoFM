#!/usr/bin/env python3
"""
End-to-end inference from raw data.

This script demonstrates the full pipeline:
1. Load raw data from datacube (parquet)
2. Tokenize using trained tokenizers (autoencoder + codebook for images)
3. Run model inference
4. Detokenize predictions back to physical values
5. Compare with ground truth

Usage:
    python examples/inference_from_raw.py \
        --checkpoint runs_aion_h100/ckpt_step_00010000.pt \
        --field STRIPE82-0001 \
        --sample-idx 100 \
        --autoencoder autoencoder_splus_cuts.pt \
        --codebook codebook_kmeans_2048.joblib
"""

import argparse
import math
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.astromodal.models import AIONTransformer
from src.astromodal.models.autoencoder import AutoEncoder
from src.astromodal.core import build_vocab_spec
from src.astromodal.tokenizers import ScalarTokenizer

# Image configuration (must match training)
BANDS = [
    "F378", "F395", "F410", "F430",
    "F515", "F660", "F861",
    "U", "G", "R", "I", "Z",
]
IMG_SIZE = 96
LATENT_DIM = 2
LATENT_H = 24
LATENT_W = 24


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    model = AIONTransformer(
        vocab_size=config["vocab_size"],
        n_types=config["n_types"],
        d_model=config["d_model"],
        n_enc_layers=config["n_enc_layers"],
        n_dec_layers=config["n_dec_layers"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        max_len=config["max_len"],
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    step = ckpt.get("global_step", '?')
    print(f"✓ Model loaded (step {step})")
    return model, config, step


def load_tokenizers(autoencoder_path: str, codebook_path: str, device: str = "cuda"):
    """Load all tokenizers (autoencoder, codebook, scalar)."""
    print("\nLoading tokenizers...")

    # Load autoencoder
    print(f"Loading autoencoder from {autoencoder_path}")
    autoencoder = AutoEncoder(in_channels=len(BANDS), latent_dim=LATENT_DIM).to(device)
    state_dict = torch.load(autoencoder_path, map_location=device)

    # Remap keys if needed (old flat structure -> new encoder/decoder structure)
    if "proj.weight" in state_dict and "encoder.proj.weight" not in state_dict:
        print("  Remapping old state dict format to new structure...")
        new_state_dict = {}
        for key, value in state_dict.items():
            # Encoder keys
            if key.startswith(("proj.", "enc_block1.", "enc_block2.", "to_latent.")):
                new_state_dict[f"encoder.{key}"] = value
            # Decoder keys
            elif key.startswith(("from_latent.", "dec_block1.", "dec_block2.", "out_conv.")):
                new_state_dict[f"decoder.{key}"] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()
    print(f"✓ Autoencoder loaded")

    # Load codebook (K-Means model)
    print(f"Loading codebook from {codebook_path}")
    codebook = joblib.load(codebook_path)
    print(f"✓ Codebook loaded ({codebook.n_clusters} clusters)")

    # Load scalar tokenizer
    scalar_tokenizer = ScalarTokenizer.load("scalar_tokenizers/scalar_tokenizer_config.npz")
    print(f"✓ Scalar tokenizer: {len(scalar_tokenizer.bin_edges_)} columns")

    return autoencoder, codebook, scalar_tokenizer


def load_raw_sample(field: str, sample_idx: int):
    """Load raw data from datacube with all necessary columns for image processing."""
    datacube_path = f"/home/astrodados4/downloads/hypercube/datacube_{field}.parquet"

    print(f"\nLoading raw sample from {datacube_path}")

    # Columns needed for image tokenization
    cutout_cols = [f"splus_cut_{b}" for b in BANDS]
    ellipse_cols = ["a_pixel_det", "b_pixel_det", "theta_det"]
    id_cols = ["id", "gaia_source_id"]
    mag_cols = ["mag_pstotal_r"]

    # Load with needed columns
    all_cols = cutout_cols + ellipse_cols + id_cols + mag_cols
    df = pd.read_parquet(datacube_path)  # Load all first to check available columns

    # Apply magnitude filter
    mag_col = "mag_pstotal_r"
    df_filtered = df[(df[mag_col] > 14.0) & (df[mag_col] < 22.0)]

    if sample_idx >= len(df_filtered):
        print(f"Error: sample_idx {sample_idx} >= dataset size {len(df_filtered)}")
        return None

    row = df_filtered.iloc[sample_idx]
    print(f"✓ Loaded object: {row.get('id', 'N/A')}")
    return row


def _to_image_torch(flat) -> torch.Tensor:
    """
    Convert flattened array-like into a square image (H, W).
    Assumes 96x96 = 9216 pixels, but works with any perfect square.
    """
    if flat is None:
        return torch.zeros((IMG_SIZE, IMG_SIZE), dtype=torch.float32)

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
    """
    device = torch.device(device)
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    yy = yy.float()
    xx = xx.float()

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


def arcsinh_norm_per_band(x: torch.Tensor, mask_valid: torch.Tensor, eps: float = 1e-6):
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


def build_image_from_row(row, device="cpu"):
    """
    Build multi-band image from raw datacube row with AION-style preprocessing.
    Returns normalized image tensor (C, H, W).
    """
    imgs = []
    masks_pix_binary = []

    # Build per-band images and basic validity masks
    for band in BANDS:
        col = f"splus_cut_{band}"
        flat = row.get(col)

        # Check if flat is None or a scalar NA value (not an array)
        if flat is None or (not isinstance(flat, (list, np.ndarray)) and pd.isna(flat)):
            img = torch.zeros((IMG_SIZE, IMG_SIZE), dtype=torch.float32)
            valid = torch.zeros((IMG_SIZE, IMG_SIZE), dtype=torch.bool)
        else:
            try:
                img = _to_image_torch(flat)  # (H,W)
                valid = torch.isfinite(img) & (img != 0.0)
            except Exception as e:
                # If conversion fails, use zeros
                print(f"  Warning: Could not convert {band}, using zeros: {e}")
                img = torch.zeros((IMG_SIZE, IMG_SIZE), dtype=torch.float32)
                valid = torch.zeros((IMG_SIZE, IMG_SIZE), dtype=torch.bool)

        img_clean = img.clone()
        img_clean[~torch.isfinite(img_clean)] = 0.0

        imgs.append(img_clean)
        masks_pix_binary.append(valid.float())

    x = torch.stack(imgs, dim=0)  # (C,H,W)
    m_pix_basic = torch.stack(masks_pix_binary, dim=0)

    C, H, W = x.shape
    device_t = torch.device(device)
    x = x.to(device_t)
    m_pix_basic = m_pix_basic.to(device_t)

    # Elliptical object mask
    x0 = torch.tensor(IMG_SIZE // 2, dtype=torch.float32, device=device_t)
    y0 = torch.tensor(IMG_SIZE // 2, dtype=torch.float32, device=device_t)

    a = torch.tensor(float(row.get("a_pixel_det", 10.0)), device=device_t)
    b = torch.tensor(float(row.get("b_pixel_det", 10.0)), device=device_t)
    th = torch.tensor(float(row.get("theta_det", 0.0)), device=device_t)
    theta = th * math.pi / 180.0

    obj_mask = elliptical_mask(H, W, x0, y0, a, b, theta, device=device_t)
    obj_mask_full = obj_mask.unsqueeze(0).expand(C, H, W)

    # Refined pixel mask: finite pixels inside ellipse
    m_pix = m_pix_basic * obj_mask_full

    # AION-like normalization (per band, per object)
    x_norm = arcsinh_norm_per_band(x, m_pix_basic)

    return x_norm


def tokenize_image(row, autoencoder, codebook, device="cuda"):
    """
    Tokenize image from raw datacube row using autoencoder + codebook.

    Pipeline:
    1. Build multi-band image from row
    2. Encode through autoencoder to get latents
    3. Quantize latents using codebook (K-Means) to get discrete tokens

    Returns tokens as 1D array of shape (LATENT_H * LATENT_W,) = (576,)
    """
    # Build normalized image
    x_norm = build_image_from_row(row, device=device)  # (C, H, W)

    # Add batch dimension and encode
    x_batch = x_norm.unsqueeze(0)  # (1, C, H, W)

    with torch.no_grad():
        latents = autoencoder.encode(x_batch)  # (1, LATENT_DIM, LATENT_H, LATENT_W)

    # Convert to numpy and reshape for codebook
    latents_np = latents.cpu().numpy()  # (1, LATENT_DIM, LATENT_H, LATENT_W)
    N, C, H, W = latents_np.shape
    vectors = latents_np.transpose(0, 2, 3, 1).reshape(-1, C)  # (H*W, LATENT_DIM)

    # Quantize using codebook (K-Means predict)
    tokens = codebook.predict(vectors)  # (H*W,)

    return tokens.astype(np.uint16)


def tokenize_scalars(row, scalar_tokenizer):
    """Tokenize scalar values from raw data."""
    # Get scalar columns that exist in both row and tokenizer
    available_cols = [c for c in scalar_tokenizer.bin_edges_.keys() if c in row.index]

    if not available_cols:
        return None, None

    # Extract values and tokenize
    values = []
    for col in available_cols:
        val = row[col]
        if pd.isna(val):
            val = scalar_tokenizer.col_means_.get(col, 0.0)
        values.append(val)

    values = np.array(values, dtype=np.float64)

    # Tokenize each column
    tokens = []
    for col, val in zip(available_cols, values):
        edges = scalar_tokenizer.bin_edges_[col]
        token = np.searchsorted(edges, val, side="right") - 1
        token = np.clip(token, 0, len(edges) - 2)
        tokens.append(token)

    return np.array(tokens, dtype=np.uint16), available_cols


def detokenize_scalars(tokens, col_names, scalar_tokenizer):
    """Convert scalar tokens back to physical values."""
    values = []
    for token, col in zip(tokens, col_names):
        edges = scalar_tokenizer.bin_edges_[col]
        # Use bin center as value estimate
        value = (edges[token] + edges[token + 1]) / 2
        values.append(value)
    return values


def predict_scalars_from_image(model, vocab, image_tokens, target_scalar_cols, device):
    """
    Given image tokens, predict scalar values.

    Returns predicted tokens for each scalar column.
    """
    # Prepare observed tokens (images)
    obs_tokens = torch.from_numpy(image_tokens).long() + vocab.base_image
    obs_tokens = obs_tokens[:576].unsqueeze(0).to(device)  # (1, 576)

    obs_types = torch.full_like(obs_tokens, 5)  # IMAGE type
    obs_positions = torch.arange(obs_tokens.size(1), device=device).unsqueeze(0)
    obs_mask = torch.ones_like(obs_tokens, dtype=torch.bool)

    # Prepare target (scalar columns we want to predict)
    n_scalars = len(target_scalar_cols)
    tgt_types = torch.zeros((1, n_scalars), dtype=torch.long, device=device)

    for i, col in enumerate(target_scalar_cols):
        type_id = vocab.get_scalar_type_id(col)
        tgt_types[0, i] = type_id

    tgt_positions = torch.arange(n_scalars, device=device).unsqueeze(0)
    tgt_mask = torch.ones((1, n_scalars), dtype=torch.bool, device=device)

    # Run model
    with torch.no_grad():
        logits = model(
            obs_tokens=obs_tokens,
            obs_types=obs_types,
            obs_positions=obs_positions,
            obs_mask=obs_mask,
            tgt_types=tgt_types,
            tgt_positions=tgt_positions,
            tgt_mask=tgt_mask,
        )  # (1, n_scalars, vocab_size)

    # Get predictions (subtract base_scalar to get bin indices)
    preds = logits.argmax(dim=-1).squeeze(0)  # (n_scalars,)
    pred_tokens = preds - vocab.base_scalar
    pred_tokens = torch.clamp(pred_tokens, 0, 1023)

    return pred_tokens.cpu().numpy()


def predict_image_from_scalars(model, vocab, scalar_tokens, scalar_cols, device):
    """
    Given scalar tokens, predict image tokens.

    Returns predicted image tokens (576,).
    """
    # Prepare observed tokens (scalars)
    obs_tokens = []
    obs_types = []
    for i, (tok, col) in enumerate(zip(scalar_tokens, scalar_cols)):
        shifted = vocab.base_scalar + int(tok)
        obs_tokens.append(shifted)
        type_id = vocab.get_scalar_type_id(col)
        obs_types.append(type_id)

    obs_tokens = torch.tensor(obs_tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, n_scalars)
    obs_types = torch.tensor(obs_types, dtype=torch.long, device=device).unsqueeze(0)
    obs_positions = torch.arange(obs_tokens.size(1), device=device).unsqueeze(0)
    obs_mask = torch.ones_like(obs_tokens, dtype=torch.bool)

    # Prepare target (image tokens we want to predict)
    n_image_tokens = 576  # 24x24
    tgt_types = torch.full((1, n_image_tokens), 5, dtype=torch.long, device=device)  # IMAGE type
    tgt_positions = torch.arange(n_image_tokens, device=device).unsqueeze(0)
    tgt_mask = torch.ones((1, n_image_tokens), dtype=torch.bool, device=device)

    # Run model
    with torch.no_grad():
        logits = model(
            obs_tokens=obs_tokens,
            obs_types=obs_types,
            obs_positions=obs_positions,
            obs_mask=obs_mask,
            tgt_types=tgt_types,
            tgt_positions=tgt_positions,
            tgt_mask=tgt_mask,
        )  # (1, 576, vocab_size)

    # Get predictions (subtract base_image to get codebook indices)
    preds = logits.argmax(dim=-1).squeeze(0)  # (576,)
    pred_tokens = preds - vocab.base_image
    pred_tokens = torch.clamp(pred_tokens, 0, 2047)

    return pred_tokens.cpu().numpy()


def reconstruct_image_from_tokens(tokens, codebook, autoencoder, device):
    """
    Reconstruct image from discrete tokens via codebook and autoencoder decoder.

    tokens: (576,) discrete token IDs
    Returns: (C, H, W) reconstructed image
    """
    # Get continuous latents from codebook
    codebook_vectors = codebook.cluster_centers_  # (n_clusters, LATENT_DIM)
    latent_vectors = codebook_vectors[tokens]  # (576, LATENT_DIM)

    # Reshape to spatial format
    latents = latent_vectors.reshape(LATENT_H, LATENT_W, LATENT_DIM)  # (24, 24, 2)
    latents = latents.transpose(2, 0, 1)  # (2, 24, 24)
    latents = torch.from_numpy(latents).float().unsqueeze(0).to(device)  # (1, 2, 24, 24)

    # Decode through autoencoder
    with torch.no_grad():
        reconstructed = autoencoder.decode(latents)  # (1, C, H, W)

    return reconstructed.squeeze(0).cpu().numpy()  # (C, H, W)


def tokens_to_latents(tokens, codebook):
    """Convert discrete tokens to continuous latents via codebook lookup."""
    codebook_vectors = codebook.cluster_centers_  # (n_clusters, LATENT_DIM)
    latent_vectors = codebook_vectors[tokens]  # (576, LATENT_DIM)

    # Reshape to spatial format
    latents = latent_vectors.reshape(LATENT_H, LATENT_W, LATENT_DIM)  # (24, 24, 2)
    latents = latents.transpose(2, 0, 1)  # (2, 24, 24)

    return latents


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def make_rgb_composite(image_array, r_idx=8, g_idx=9, i_idx=10):
    """
    Create RGB composite from multi-band image.

    Default indices: R (idx 8), G (idx 9), I (idx 10) from BANDS list.
    """
    r = image_array[r_idx]
    g = image_array[g_idx]
    b = image_array[i_idx]

    # Stack and normalize each channel independently
    rgb = np.stack([r, g, b], axis=-1)

    # Robust percentile normalization per channel
    for i in range(3):
        channel = rgb[:, :, i]
        vmin = np.percentile(channel, 1)
        vmax = np.percentile(channel, 99)
        rgb[:, :, i] = np.clip((channel - vmin) / (vmax - vmin + 1e-8), 0, 1)

    return rgb


def plot_scalar_predictions(gt_values, pred_values, col_names, save_path=None):
    """Plot scalar prediction comparison as bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(col_names))
    width = 0.35

    ax.bar(x - width/2, gt_values, width, label='Ground Truth', alpha=0.8)
    ax.bar(x + width/2, pred_values, width, label='Predicted', alpha=0.8)

    ax.set_xlabel('Scalar Column')
    ax.set_ylabel('Value')
    ax.set_title('Scalar Predictions vs Ground Truth')
    ax.set_xticks(x)
    ax.set_xticklabels(col_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_image_comparison(original_img, reconstructed_img, title="Image Comparison", save_path=None):
    """
    Plot original vs reconstructed image (RGB composite and individual bands).

    original_img: (C, H, W) numpy array
    reconstructed_img: (C, H, W) numpy array
    """
    # Create RGB composites
    rgb_orig = make_rgb_composite(original_img)
    rgb_recon = make_rgb_composite(reconstructed_img)

    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(title, fontsize=16)

    # Row 1: RGB composite + 3 individual bands (original)
    axes[0, 0].imshow(rgb_orig)
    axes[0, 0].set_title('Original RGB (R,G,I)')
    axes[0, 0].axis('off')

    for i, (band_idx, band_name) in enumerate([(3, 'F430'), (8, 'R'), (10, 'I')]):
        img_band = original_img[band_idx]
        vmin, vmax = np.percentile(img_band, [1, 99])
        axes[0, i+1].imshow(img_band, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, i+1].set_title(f'Original {band_name}')
        axes[0, i+1].axis('off')

    # Row 2: RGB composite + 3 individual bands (reconstructed)
    axes[1, 0].imshow(rgb_recon)
    axes[1, 0].set_title('Reconstructed RGB')
    axes[1, 0].axis('off')

    for i, (band_idx, band_name) in enumerate([(3, 'F430'), (8, 'R'), (10, 'I')]):
        img_band = reconstructed_img[band_idx]
        vmin, vmax = np.percentile(img_band, [1, 99])
        axes[1, i+1].imshow(img_band, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, i+1].set_title(f'Reconstructed {band_name}')
        axes[1, i+1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_token_map(tokens, title="Token Map", save_path=None):
    """
    Visualize discrete token IDs as a 2D map.

    tokens: (H*W,) or (H, W) array of token IDs
    """
    if tokens.ndim == 1:
        # Reshape to 2D
        tokens_2d = tokens.reshape(LATENT_H, LATENT_W)
    else:
        tokens_2d = tokens

    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(tokens_2d, cmap='tab20c', interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')

    plt.colorbar(im, ax=ax, label='Token ID')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_all_bands(image_array, title="All Bands", save_path=None):
    """Plot all 12 bands in a grid."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(title, fontsize=16)

    for i, (ax, band_name) in enumerate(zip(axes.flat, BANDS)):
        img_band = image_array[i]
        vmin, vmax = np.percentile(img_band, [1, 99])
        im = ax.imshow(img_band, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(band_name)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_latent_space(latents, title="Latent Space", save_path=None):
    """
    Visualize latent representation.

    latents: (LATENT_DIM, H, W) array
    """
    fig, axes = plt.subplots(1, LATENT_DIM, figsize=(8*LATENT_DIM, 8))
    fig.suptitle(title, fontsize=16)

    if LATENT_DIM == 1:
        axes = [axes]

    for i in range(LATENT_DIM):
        im = axes[i].imshow(latents[i], cmap='RdBu_r', interpolation='nearest')
        axes[i].set_title(f'Latent Channel {i}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Inference from raw data with multiple prediction modes")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--autoencoder", type=str, default="autoencoder_splus_cuts.pt",
                        help="Path to trained autoencoder")
    parser.add_argument("--codebook", type=str, default="codebook_kmeans_2048.joblib",
                        help="Path to trained codebook (K-Means)")
    parser.add_argument("--field", type=str, default="STRIPE82-0001", help="Field name")
    parser.add_argument("--sample-idx", type=int, default=100, help="Sample index in field")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--mode", type=str, default="image-to-scalar",
                        choices=["image-to-scalar", "scalar-to-image", "reconstruct-image", "all"],
                        help="Prediction mode")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of random samples to test (for 'all' mode)")
    parser.add_argument("--plot", action="store_true",
                        help="Enable plotting/visualization")
    parser.add_argument("--plot-dir", type=str, default="inference_plots",
                        help="Directory to save plots")
    args = parser.parse_args()

    # Create plot directory if needed
    if args.plot:
        Path(args.plot_dir).mkdir(parents=True, exist_ok=True)
        print(f"Plots will be saved to: {args.plot_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model, config, step = load_model(args.checkpoint, device)

    # Load tokenizers (autoencoder, codebook, scalar)
    autoencoder, codebook, scalar_tokenizer = load_tokenizers(
        args.autoencoder, args.codebook, device
    )

    # Build vocab
    print("\nBuilding vocabulary...")
    vocab = build_vocab_spec(
        v_image=2048,
        v_scalar=1024,
        v_spectra=2048,
    )

    # Run predictions based on mode
    if args.mode == "all":
        print(f"\n{'='*80}")
        print(f"RUNNING ALL PREDICTION MODES ON {args.n_samples} RANDOM SAMPLES")
        print(f"{'='*80}")

        # Load all samples from field
        datacube_path = f"/home/astrodados4/downloads/hypercube/datacube_{args.field}.parquet"
        df = pd.read_parquet(datacube_path)
        df_filtered = df[(df["mag_pstotal_r"] > 14.0) & (df["mag_pstotal_r"] < 22.0)]

        # Select random samples
        n_samples = min(args.n_samples, len(df_filtered))
        sample_indices = np.random.choice(len(df_filtered), size=n_samples, replace=False)

        for idx_num, sample_idx in enumerate(sample_indices):
            row = df_filtered.iloc[sample_idx]

            print(f"\n\n{'#'*80}")
            print(f"# SAMPLE {idx_num + 1}/{n_samples} - Object: {row.get('id', 'N/A')}")
            print(f"{'#'*80}")

            # Tokenize
            image_tokens = tokenize_image(row, autoencoder, codebook, device)
            scalar_tokens_gt, scalar_cols = tokenize_scalars(row, scalar_tokenizer)

            if image_tokens is None or scalar_tokens_gt is None:
                print("  Skipping (missing data)")
                continue

            mag_cols = [c for c in scalar_cols if c.startswith("mag_pstotal_")][:5]
            if not mag_cols:
                mag_cols = scalar_cols[:5]

            # Mode 1: Image → Scalars
            print(f"\n[1] IMAGE → SCALARS")
            print("-" * 80)
            pred_tokens = predict_scalars_from_image(model, vocab, image_tokens, mag_cols, device)
            pred_values = detokenize_scalars(pred_tokens, mag_cols, scalar_tokenizer)
            gt_indices = [scalar_cols.index(c) for c in mag_cols]
            gt_values = detokenize_scalars(scalar_tokens_gt[gt_indices], mag_cols, scalar_tokenizer)

            for col, gt_val, pred_val in zip(mag_cols, gt_values, pred_values):
                error = abs(pred_val - gt_val)
                status = "✓" if error < 0.5 else "✗"
                print(f"  {col:<20} GT: {gt_val:>8.3f}  Pred: {pred_val:>8.3f}  Err: {error:>6.3f} {status}")

            # Plot scalar predictions
            if args.plot:
                obj_id = str(row.get('id', idx_num)).replace('b\'', '').replace('\'', '')
                plot_path = f"{args.plot_dir}/sample_{idx_num+1}_{obj_id}_scalars.png"
                plot_scalar_predictions(gt_values, pred_values, mag_cols, save_path=plot_path)

            # Mode 2: Scalars → Image
            print(f"\n[2] SCALARS → IMAGE")
            print("-" * 80)
            pred_img_tokens = predict_image_from_scalars(
                model, vocab, scalar_tokens_gt[:10], scalar_cols[:10], device
            )
            # Calculate token accuracy
            token_match = (pred_img_tokens == image_tokens).sum()
            token_acc = token_match / len(image_tokens)
            print(f"  Predicted {len(pred_img_tokens)} image tokens")
            print(f"  Token accuracy: {token_acc:.2%} ({token_match}/{len(image_tokens)} match)")

            # Reconstruct predicted image
            reconstructed = reconstruct_image_from_tokens(pred_img_tokens, codebook, autoencoder, device)
            print(f"  Reconstructed image shape: {reconstructed.shape}")

            # Mode 3: Reconstruct original image
            print(f"\n[3] RECONSTRUCT ORIGINAL IMAGE")
            print("-" * 80)
            original_reconstructed = reconstruct_image_from_tokens(image_tokens, codebook, autoencoder, device)
            print(f"  Reconstructed image shape: {original_reconstructed.shape}")

            # Compare original and reconstructed in first 3 bands
            for i, band in enumerate(BANDS[:3]):
                orig_mean = original_reconstructed[i].mean()
                print(f"  {band}: mean = {orig_mean:.4f}")

            # Plot visualizations
            if args.plot:
                obj_id = str(row.get('id', idx_num)).replace('b\'', '').replace('\'', '')

                # Plot original image tokens
                plot_path = f"{args.plot_dir}/sample_{idx_num+1}_{obj_id}_tokens_original.png"
                plot_token_map(image_tokens, title="Original Image Tokens", save_path=plot_path)

                # Plot predicted image tokens
                plot_path = f"{args.plot_dir}/sample_{idx_num+1}_{obj_id}_tokens_predicted.png"
                plot_token_map(pred_img_tokens, title="Predicted Image Tokens (from scalars)", save_path=plot_path)

                # Plot original vs reconstructed image
                # Need to get original image first
                original_img = build_image_from_row(row, device=device).cpu().numpy()
                plot_path = f"{args.plot_dir}/sample_{idx_num+1}_{obj_id}_reconstruction.png"
                plot_image_comparison(
                    original_img, original_reconstructed,
                    title=f"Sample {idx_num+1}: Original vs Reconstructed",
                    save_path=plot_path
                )

                # Plot latent space
                latents = tokens_to_latents(image_tokens, codebook)
                plot_path = f"{args.plot_dir}/sample_{idx_num+1}_{obj_id}_latents.png"
                plot_latent_space(latents, title="Latent Space Representation", save_path=plot_path)

    else:
        # Single sample mode
        row = load_raw_sample(args.field, args.sample_idx)
        if row is None:
            return

        # Tokenize image using autoencoder + codebook
        print("\nTokenizing image...")
        image_tokens = tokenize_image(row, autoencoder, codebook, device)
        if image_tokens is None:
            print("Error: Could not tokenize image")
            return
        print(f"✓ Image tokens: {image_tokens.shape} (24x24 = {LATENT_H}x{LATENT_W} tokens)")

        # Tokenize scalars (ground truth)
        print("\nTokenizing scalars...")
        scalar_tokens_gt, scalar_cols = tokenize_scalars(row, scalar_tokenizer)
        if scalar_tokens_gt is None:
            print("Error: Could not tokenize scalars")
            return
        print(f"✓ Ground truth: {len(scalar_cols)} scalar columns")

        # Select columns to predict
        mag_cols = [c for c in scalar_cols if c.startswith("mag_pstotal_")]
        if not mag_cols:
            mag_cols = scalar_cols[:5]

        if args.mode == "image-to-scalar":
            print(f"\n{'='*70}")
            print(f"MODE: Image → Scalars")
            print(f"{'='*70}")
            print(f"Predicting {len(mag_cols)} columns: {', '.join(mag_cols[:3])}...")

            # Run inference
            pred_tokens = predict_scalars_from_image(model, vocab, image_tokens, mag_cols, device)
            pred_values = detokenize_scalars(pred_tokens, mag_cols, scalar_tokenizer)

            # Get ground truth
            gt_indices = [scalar_cols.index(c) for c in mag_cols]
            gt_tokens = scalar_tokens_gt[gt_indices]
            gt_values = detokenize_scalars(gt_tokens, mag_cols, scalar_tokenizer)

            # Show results
            print(f"\n{'Column':<25} {'Ground Truth':<15} {'Predicted':<15} {'Error':<10}")
            print("-" * 70)
            for col, gt_val, pred_val in zip(mag_cols, gt_values, pred_values):
                error = abs(pred_val - gt_val)
                status = "✓" if error < 0.5 else "✗"
                print(f"{col:<25} {gt_val:>14.4f} {pred_val:>14.4f} {error:>9.4f} {status}")

            # Summary
            errors = np.abs(np.array(pred_values) - np.array(gt_values))
            print(f"\nMean Absolute Error: {errors.mean():.4f}")
            print(f"Token-level accuracy: {(gt_tokens == pred_tokens).mean():.2%}")

            # Plot
            if args.plot:
                obj_id = str(row.get('id', 'sample')).replace('b\'', '').replace('\'', '')
                plot_path = f"{args.plot_dir}/image_to_scalar_{obj_id}.png"
                plot_scalar_predictions(gt_values, pred_values, mag_cols, save_path=plot_path)

        elif args.mode == "scalar-to-image":
            print(f"\n{'='*70}")
            print(f"MODE: Scalars → Image")
            print(f"{'='*70}")

            # Use first 10 scalar values as input
            input_scalars = scalar_tokens_gt[:10]
            input_cols = scalar_cols[:10]

            print(f"Using {len(input_scalars)} scalar columns as input:")
            for col in input_cols[:5]:
                print(f"  - {col}")

            # Predict image tokens
            pred_img_tokens = predict_image_from_scalars(model, vocab, input_scalars, input_cols, device)

            # Calculate accuracy
            token_match = (pred_img_tokens == image_tokens).sum()
            token_acc = token_match / len(image_tokens)

            print(f"\nImage token prediction:")
            print(f"  Predicted tokens: {len(pred_img_tokens)}")
            print(f"  Token accuracy: {token_acc:.2%} ({token_match}/{len(image_tokens)} match)")

            # Reconstruct image
            reconstructed = reconstruct_image_from_tokens(pred_img_tokens, codebook, autoencoder, device)
            print(f"  Reconstructed image shape: {reconstructed.shape}")
            print(f"\nFirst 3 bands statistics:")
            for i, band in enumerate(BANDS[:3]):
                print(f"  {band}: mean={reconstructed[i].mean():.4f}, std={reconstructed[i].std():.4f}")

            # Plot
            if args.plot:
                obj_id = str(row.get('id', 'sample')).replace('b\'', '').replace('\'', '')

                # Plot token comparison
                plot_path = f"{args.plot_dir}/scalar_to_image_{obj_id}_tokens_gt.png"
                plot_token_map(image_tokens, title="Ground Truth Image Tokens", save_path=plot_path)

                plot_path = f"{args.plot_dir}/scalar_to_image_{obj_id}_tokens_pred.png"
                plot_token_map(pred_img_tokens, title="Predicted Image Tokens", save_path=plot_path)

                # Plot image comparison
                original_img = build_image_from_row(row, device=device).cpu().numpy()
                plot_path = f"{args.plot_dir}/scalar_to_image_{obj_id}_comparison.png"
                plot_image_comparison(
                    original_img, reconstructed,
                    title="Scalars → Image: Original vs Predicted",
                    save_path=plot_path
                )

                # Plot all bands of reconstructed image
                plot_path = f"{args.plot_dir}/scalar_to_image_{obj_id}_all_bands.png"
                plot_all_bands(reconstructed, title="Predicted Image (All Bands)", save_path=plot_path)

        elif args.mode == "reconstruct-image":
            print(f"\n{'='*70}")
            print(f"MODE: Reconstruct Image (Autoencoder)")
            print(f"{'='*70}")

            # Reconstruct from tokens
            reconstructed = reconstruct_image_from_tokens(image_tokens, codebook, autoencoder, device)

            print(f"Original image tokens: {image_tokens.shape}")
            print(f"Reconstructed image: {reconstructed.shape}")
            print(f"\nBand statistics:")
            for i, band in enumerate(BANDS):
                print(f"  {band}: mean={reconstructed[i].mean():.4f}, std={reconstructed[i].std():.4f}")

            # Plot
            if args.plot:
                obj_id = str(row.get('id', 'sample')).replace('b\'', '').replace('\'', '')

                # Plot tokens
                plot_path = f"{args.plot_dir}/reconstruct_{obj_id}_tokens.png"
                plot_token_map(image_tokens, title="Image Tokens", save_path=plot_path)

                # Plot latent space
                latents = tokens_to_latents(image_tokens, codebook)
                plot_path = f"{args.plot_dir}/reconstruct_{obj_id}_latents.png"
                plot_latent_space(latents, title="Latent Space", save_path=plot_path)

                # Plot original vs reconstructed
                original_img = build_image_from_row(row, device=device).cpu().numpy()
                plot_path = f"{args.plot_dir}/reconstruct_{obj_id}_comparison.png"
                plot_image_comparison(
                    original_img, reconstructed,
                    title="Original vs Autoencoder Reconstruction",
                    save_path=plot_path
                )

                # Plot all bands
                plot_path = f"{args.plot_dir}/reconstruct_{obj_id}_original_all_bands.png"
                plot_all_bands(original_img, title="Original Image (All Bands)", save_path=plot_path)

                plot_path = f"{args.plot_dir}/reconstruct_{obj_id}_reconstructed_all_bands.png"
                plot_all_bands(reconstructed, title="Reconstructed Image (All Bands)", save_path=plot_path)


if __name__ == "__main__":
    main()
