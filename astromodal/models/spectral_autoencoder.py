import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from tqdm import tqdm

import logpool

def _gn1d(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    g = min(num_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class ConvGNAct1D(nn.Module):
    """Conv1d -> GroupNorm -> SiLU."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 5, s: int = 1, p: Optional[int] = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=True)
        self.gn = _gn1d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class ResBlock1D(nn.Module):
    """Residual block: Conv1d -> GN -> SiLU -> Conv1d -> GN + skip."""
    def __init__(self, channels: int, k: int = 5):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=k, padding=p, bias=True)
        self.gn1 = _gn1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=k, padding=p, bias=True)
        self.gn2 = _gn1d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return self.act(h + x)


class SpectraEncoder(nn.Module):
    """
    Expects x in [B, L, C] and internally converts to [B, C, L].

    Output: z in [B, latent_dim, T]
    """
    def __init__(
        self,
        in_channels: int = 2,
        latent_dim: int = 16,
        base_channels: int = 64,
        num_down: int = 4,
        k: int = 5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.num_down = num_down

        self.proj = nn.Conv1d(in_channels, base_channels, kernel_size=1, bias=True)

        blocks = []
        ch = base_channels
        for i in range(num_down):
            blocks += [
                ResBlock1D(ch, k=k),
                ConvGNAct1D(ch, ch * 2 if i < num_down - 1 else ch, k=k, s=2),  # stride=2 downsample
            ]
            if i < num_down - 1:
                ch = ch * 2

        self.trunk = nn.Sequential(*blocks)
        self.to_latent = nn.Conv1d(ch, latent_dim, kernel_size=1, bias=True)

        self.downsample_factor = 2 ** num_down

    def forward(self, x_blc: torch.Tensor) -> torch.Tensor:
        # x_blc: [B,L,C] -> [B,C,L]
        if x_blc.ndim != 3:
            raise ValueError(f"Expected x [B,L,C], got {tuple(x_blc.shape)}")
        x = x_blc.transpose(1, 2).contiguous()
        h = self.proj(x)
        h = self.trunk(h)
        z = self.to_latent(h)
        return z


class SpectraDecoder(nn.Module):
    """
    Input:  z [B, latent_dim, T]
    Output: x_hat [B, C, Lhat]  (Conv1d layout)
    """
    def __init__(
        self,
        out_channels: int = 2,
        latent_dim: int = 16,
        base_channels: int = 64,
        num_down: int = 4,
        k: int = 5,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.num_down = num_down

        ch = base_channels * (2 ** (num_down - 1)) if num_down > 1 else base_channels
        self.from_latent = nn.Conv1d(latent_dim, ch, kernel_size=1, bias=True)

        blocks = []
        for i in range(num_down):
            next_ch = ch // 2 if i < num_down - 1 else base_channels
            blocks += [
                ResBlock1D(ch, k=k),
                nn.ConvTranspose1d(
                    ch,
                    next_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=True,
                ),
                _gn1d(next_ch),
                nn.SiLU(inplace=True),
            ]
            ch = next_ch

        self.trunk = nn.Sequential(*blocks)
        self.out_conv = nn.Conv1d(base_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        h = self.trunk(h)
        x_hat = self.out_conv(h)  # [B,C,Lhat]
        return x_hat


class SpectraAutoEncoder(nn.Module):
    """
    Public interface matches your loader:

      Input x:  [B, L, 2]
      Output:   [B, L, 2]

    mask from loader:
      mask: [B, L]  (bool)
    """
    def __init__(
        self,
        in_channels: int = 2,
        latent_dim: int = 16,
        base_channels: int = 64,
        num_down: int = 4,
        k: int = 5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.num_down = num_down
        self.k = k

        self.encoder = SpectraEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            num_down=num_down,
            k=k,
        )
        self.decoder = SpectraDecoder(
            out_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            num_down=num_down,
            k=k,
        )

        self.downsample_factor = self.encoder.downsample_factor
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,C]
        return self.encoder(x)

    def decode(self, z: torch.Tensor, L_out: Optional[int] = None) -> torch.Tensor:
        # decoder returns [B, 2, L']
        x_hat = self.decoder(z)
        Lh = x_hat.shape[-1]

        if L_out is not None:
            if Lh > L_out:
                x_hat = x_hat[..., :L_out]
            elif Lh < L_out:
                x_hat = F.pad(x_hat, (0, L_out - Lh))

        # ALWAYS return [B, L, 2]
        return x_hat.transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)                 # [B,latent_dim,T]
        x_hat = self.decode(z, L_out=x.shape[1])  # [B,L,C]
        return x_hat, z

    # -------------------------
    # Loss + training (FIXED)
    # -------------------------

    def compute_loss(
        self,
        x: torch.Tensor,                    # [B, L, 2]
        mask: Optional[torch.Tensor] = None # [B, L] bool (True=valid)
    ) -> torch.Tensor:
        x_hat, _ = self(x)                  # x_hat: [B, L, 2]

        if mask is None:
            return F.mse_loss(x_hat, x)

        # mask: [B,L] -> [B,L,1] and broadcast over channels
        m = mask.to(dtype=torch.bool, device=x.device).unsqueeze(-1)   # [B,L,1]
        m_f = m.to(dtype=x.dtype)                                      # float for math

        diff2 = (x_hat - x).pow(2)                                     # [B,L,2]
        num = (diff2 * m_f).sum()                                      # sum over valid pixels & channels
        den = m_f.sum() * x.shape[-1]                                  # valid_pixels * channels

        # avoid div0 if a batch is fully masked
        return num / (den.clamp_min(1.0))

    def train_epoch(self, dataloader, optimizer, device) -> float:
        self.train()
        total = 0.0
        n = 0

        for batch in tqdm(dataloader, desc="Training", leave=False):
            # your collate returns (x_pad, mask_pad, lengths)
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                x, mask, _lengths = batch
            elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                x, mask = batch
            else:
                x, mask = batch, None

            x = x.to(device, non_blocking=True).float()         # [B,L,2]
            mask = mask.to(device, non_blocking=True) if mask is not None else None

            optimizer.zero_grad(set_to_none=True)
            loss = self.compute_loss(x, mask=mask)
            loss.backward()
            optimizer.step()

            bs = x.shape[0]
            total += float(loss.detach().cpu()) * bs
            n += bs

        return total / max(n, 1)

    @torch.no_grad()
    def validate(self, dataloader, device) -> float:
        self.eval()
        total = 0.0
        n = 0

        for batch in tqdm(dataloader, desc="Validating", leave=False):
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                x, mask, _lengths = batch
            elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                x, mask = batch
            else:
                x, mask = batch, None

            x = x.to(device, non_blocking=True).float()
            mask = mask.to(device, non_blocking=True) if mask is not None else None

            loss = self.compute_loss(x, mask=mask)

            bs = x.shape[0]
            total += float(loss.detach().cpu()) * bs
            n += bs

        return total / max(n, 1)

    def save(self, path: str, additional_info: Optional[dict] = None):
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "in_channels": self.in_channels,
                "latent_dim": self.latent_dim,
                "base_channels": self.base_channels,
                "num_down": self.num_down,
                "k": self.k,
                **(additional_info or {}),
            },
            path,
        )

    @staticmethod
    def load_from_file(path: str, map_location=None) -> "SpectraAutoEncoder":
        ckpt = torch.load(path, map_location=map_location)
        model = SpectraAutoEncoder(
            in_channels=ckpt.get("in_channels", 2),
            latent_dim=ckpt.get("latent_dim", 16),
            base_channels=ckpt.get("base_channels", 64),
            num_down=ckpt.get("num_down", 4),
            k=ckpt.get("k", 5),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        logpool.info(f"Loaded spectra AE from {path}")
        return model