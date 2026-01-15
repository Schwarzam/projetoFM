"""
AutoEncoder model for image tokenization in astromodal.

AION-inspired conv autoencoder with spatial latents:
- input:  (B, C, 96, 96)
- latent: (B, latent_dim, 24, 24)
- output: (B, C, 96, 96)

This version adds:
- GroupNorm for stability
- optional skip connections (keeps same latent interface)
- better init

You can build a codebook on the encoder output z_map as usual.
"""

from __future__ import annotations

from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn

from tqdm import tqdm
import torch.nn.functional as F 

import torch
import logpool

# -------------------------
# Helpers
# -------------------------

def _gn(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    # pick groups that divide channels
    g = min(num_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class ConvGNAct(nn.Module):
    """Conv2d -> GroupNorm -> ReLU."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=True)
        self.gn = _gn(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class ResBlock(nn.Module):
    """
    Residual block with two 3x3 convolutions + GroupNorm + ReLU.

    Output shape = input shape.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.gn1 = _gn(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.gn2 = _gn(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return self.act(h + x)


# -------------------------
# Encoder / Decoder
# -------------------------

class Encoder(nn.Module):
    """
    Encoder: (B, C, 96, 96) -> (B, latent_dim, 24, 24)

    Returns z_map and (optionally) skip features for decoder.
    """
    def __init__(self, in_channels: int, latent_dim: int, proj_channels: int = 64, use_skips: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.proj_channels = proj_channels
        self.use_skips = use_skips

        # 1x1 projection to feature space
        self.proj = nn.Conv2d(in_channels, proj_channels, kernel_size=1, bias=True)

        # Stage 1: 96 -> 48
        self.stage1 = nn.Sequential(
            ResBlock(proj_channels),
            ConvGNAct(proj_channels, 128, k=3, s=2, p=1),  # downsample
        )

        # Stage 2: 48 -> 24
        self.stage2 = nn.Sequential(
            ResBlock(128),
            ConvGNAct(128, 128, k=3, s=2, p=1),           # downsample
            ResBlock(128),
        )

        # Project to latent
        self.to_latent = nn.Conv2d(128, latent_dim, kernel_size=1, bias=True)

        self.latent_H = 24
        self.latent_W = 24

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        h0 = self.proj(x)        # (B, proj_channels, 96, 96)
        h1 = self.stage1(h0)     # (B, 128, 48, 48)
        h2 = self.stage2(h1)     # (B, 128, 24, 24)
        z_map = self.to_latent(h2)  # (B, latent_dim, 24, 24)

        if not self.use_skips:
            return z_map, None

        skips = {
            "h0": h0,  # 96x96
            "h1": h1,  # 48x48
            "h2": h2,  # 24x24 (pre-latent features)
        }
        return z_map, skips


class Decoder(nn.Module):
    """
    Decoder: (B, latent_dim, 24, 24) -> (B, out_channels, 96, 96)
    Optionally uses skips from the encoder.
    """
    def __init__(self, latent_dim: int, out_channels: int, proj_channels: int = 64, use_skips: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.proj_channels = proj_channels
        self.use_skips = use_skips

        self.from_latent = nn.Conv2d(latent_dim, 128, kernel_size=1, bias=True)

        # 24 -> 48
        self.up1 = nn.Sequential(
            ResBlock(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True),
            _gn(64),
            nn.ReLU(inplace=True),
        )
        # fuse skip at 48 if enabled
        self.fuse1 = nn.Sequential(
            ConvGNAct(64 + 128, 64, k=3, s=1, p=1),
            ResBlock(64),
        ) if use_skips else None

        # 48 -> 96
        self.up2 = nn.Sequential(
            ResBlock(64),
            nn.ConvTranspose2d(64, proj_channels, kernel_size=4, stride=2, padding=1, bias=True),
            _gn(proj_channels),
            nn.ReLU(inplace=True),
            ResBlock(proj_channels),
        )
        # fuse skip at 96 if enabled
        self.fuse2 = nn.Sequential(
            ConvGNAct(proj_channels + proj_channels, proj_channels, k=3, s=1, p=1),
            ResBlock(proj_channels),
        ) if use_skips else None

        self.out_conv = nn.Conv2d(proj_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, z_map: torch.Tensor, skips: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        h = self.from_latent(z_map)   # (B, 128, 24, 24)

        h = self.up1(h)               # (B, 64, 48, 48)

        if self.use_skips and (skips is not None) and ("h1" in skips):
            h = torch.cat([h, skips["h1"]], dim=1)   # (B, 64+128, 48, 48)
            h = self.fuse1(h)

        h = self.up2(h)               # (B, proj_channels, 96, 96)

        if self.use_skips and (skips is not None) and ("h0" in skips):
            h = torch.cat([h, skips["h0"]], dim=1)   # (B, proj+proj, 96, 96)
            h = self.fuse2(h)

        x_hat = self.out_conv(h)      # (B, out_channels, 96, 96)
        return x_hat

def _gaussian_kernel2d(sigma: float, device, dtype):
    # small, fixed kernel; you can increase k for bigger sigma
    radius = int(3 * sigma + 0.5)
    k = 2 * radius + 1
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g1 = torch.exp(-(x**2) / (2 * sigma**2))
    g1 = g1 / g1.sum()
    g2 = g1[:, None] * g1[None, :]
    g2 = g2 / g2.sum()
    return g2  # (k,k)

def _gaussian_blur(x, sigma: float):
    # x: (B,C,H,W)
    B, C, H, W = x.shape
    k2 = _gaussian_kernel2d(sigma, x.device, x.dtype)
    k = k2.shape[0]
    w = k2.view(1, 1, k, k).repeat(C, 1, 1, 1)  # depthwise
    return F.conv2d(x, w, padding=k//2, groups=C)

def _laplacian(x):
    # x: (B,C,H,W) -> (B,C,H,W)
    B, C, H, W = x.shape
    k = torch.tensor([[0,  1, 0],
                    [1, -4, 1],
                    [0,  1, 0]], device=x.device, dtype=x.dtype)
    w = k.view(1, 1, 3, 3).repeat(C, 1, 1, 1)  # depthwise
    return F.conv2d(x, w, padding=1, groups=C)

class AutoEncoder(nn.Module):
    """
    AutoEncoder with spatial latents.

    - encode(x) -> z_map (B, latent_dim, 24, 24)
    - decode(z_map) -> x_hat (B, C, 96, 96)
    - forward(x) -> (x_hat, z_map)
    """
    def __init__(
        self,
        in_channels: int,
        latent_dim: int = 2,
        proj_channels: int = 64,
        use_skips: bool = False,   # keep False if you really don't want skips
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.proj_channels = proj_channels
        self.use_skips = use_skips

        self.encoder = Encoder(in_channels, latent_dim, proj_channels, use_skips=use_skips)
        self.decoder = Decoder(latent_dim, in_channels, proj_channels, use_skips=use_skips)

        self.latent_H = self.encoder.latent_H
        self.latent_W = self.encoder.latent_W

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_map, _skips = self.encoder(x)
        return z_map

    def decode(self, z_map: torch.Tensor) -> torch.Tensor:
        # decode without skips (use forward(x) if you want skips)
        return self.decoder(z_map, skips=None)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_map, skips = self.encoder(x)
        x_hat = self.decoder(z_map, skips=skips if self.use_skips else None)
        return x_hat, z_map



    def train_epoch(
        self,
        dataloader,
        optimizer,
        device,
        lam_lowfreq: float = 0.0,
        lam_points: float = 0.10,
        sigma_lowfreq: float = 1.5,
    ) -> float:
        self.train()
        total_loss = 0.0
        n_batches = 0

        for x_norm, m_valid in tqdm(dataloader, desc="Training", leave=False):
            x_norm = x_norm.to(device, non_blocking=True).float()
            m_valid = m_valid.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            x_recon, _ = self(x_norm)

            mv = (m_valid > 0.5).unsqueeze(1) if m_valid.ndim == 3 else (m_valid > 0.5)
            mv = mv.to(x_norm.dtype)

            # --- base recon loss (your original)
            if mv.any():
                recon = F.mse_loss(x_recon[mv.bool()], x_norm[mv.bool()])
            else:
                recon = x_recon.sum() * 0.0

            # --- residual structure regularizers
            r = (x_recon - x_norm) * mv  # (B,C,H,W)

            # 1) low-frequency structure should be ~0 (noise averages out after blur)
            r_blur = _gaussian_blur(r, sigma=sigma_lowfreq)
            lowfreq = (r_blur**2).mean()

            # 2) point-like / sharp structures: laplacian spikes on stars/edges
            r_lap = _laplacian(r)
            points = (r_lap**2).mean()

            loss = recon + lam_lowfreq * lowfreq + lam_points * points

            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            n_batches += 1

        return total_loss / max(n_batches, 1)


    @torch.no_grad()
    def validate(
        self,
        dataloader,
        device,
    ) -> float:
        self.eval()
        total_loss = 0.0
        n_batches = 0

        for x_norm, m_valid in tqdm(dataloader, desc="Validating", leave=False):
            x_norm = x_norm.to(device, non_blocking=True).float()
            m_valid = m_valid.to(device, non_blocking=True)

            x_recon, _ = self(x_norm)

            mv = m_valid > 0.5
            if mv.any():
                loss = F.mse_loss(x_recon[mv], x_norm[mv])
            else:
                loss = x_recon.sum() * 0.0

            total_loss += float(loss.detach().cpu())
            n_batches += 1

        return total_loss / max(n_batches, 1)
    
    def train_model(
        self,
        train_loader,
        val_loader,
        n_channels,
        latent_dim: int,
        model_output_path: str,
        num_epochs: int = 100,
        learning_rate: float = 1e-3
    ):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self = self.to(device)
        best_val_loss = float("inf")

        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
        }

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_loader, optimizer, device)
            val_loss = self.validate(val_loader, device)

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            logpool.info(
                f"Epoch {epoch:03d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                self.save(
                    model_output_path,
                    additional_info={
                        "epoch": epoch,
                        "history": history,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "in_channels": len(n_channels),
                        "out_channels": len(n_channels),
                        "latent_dim": latent_dim,
                    })
                logpool.info(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
    
    def save(
        self,
        model_output_path: str,
        additional_info: Optional[dict] = None,
    ):
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                ## add here the additional info you want to save
                **(additional_info if additional_info is not None else {}),
            },
            model_output_path,
        )
    
    def load(
        self,
        model_input_path: str,
        map_location: Optional[torch.device] = None,
    ):
        checkpoint = torch.load(model_input_path, map_location=map_location)
        self.load_state_dict(checkpoint["model_state_dict"])
        logpool.info(f"Loaded model from {model_input_path}")
        
    @staticmethod
    def load_from_file(
        model_input_path: str,
        map_location: Optional[torch.device] = None,
        use_skips: bool = False,
    ) -> AutoEncoder:
        checkpoint = torch.load(model_input_path, map_location=map_location)
        model = AutoEncoder(
            in_channels=checkpoint.get("in_channels", 12),
            latent_dim=checkpoint.get("latent_dim", 2),
            use_skips=use_skips,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        logpool.info(f"Loaded model from {model_input_path}")
        return model