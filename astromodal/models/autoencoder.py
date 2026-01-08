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