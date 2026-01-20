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
    Expects x in [B, L, 1] and returns z in [B, latent_dim, T].
    """
    def __init__(self, in_channels: int = 1, latent_dim: int = 16, base_channels: int = 64, num_down: int = 4, k: int = 5):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, base_channels, kernel_size=1, bias=True)

        blocks: List[nn.Module] = []
        ch = base_channels
        for i in range(num_down):
            blocks += [
                ResBlock1D(ch, k=k),
                ConvGNAct1D(ch, ch * 2 if i < num_down - 1 else ch, k=k, s=2),
            ]
            if i < num_down - 1:
                ch *= 2

        self.trunk = nn.Sequential(*blocks)
        self.to_latent = nn.Conv1d(ch, latent_dim, kernel_size=1, bias=True)
        self.downsample_factor = 2 ** num_down

    def forward(self, x_blc: torch.Tensor) -> torch.Tensor:
        if x_blc.ndim != 3:
            raise ValueError(f"Expected x [B,L,1], got {tuple(x_blc.shape)}")
        x = x_blc.transpose(1, 2).contiguous()  # [B,1,L] -> conv layout
        h = self.proj(x)
        h = self.trunk(h)
        z = self.to_latent(h)
        return z

class SpectraDecoder(nn.Module):
    """
    z [B, latent_dim, T] -> x_hat [B, 1, Lhat]
    """
    def __init__(self, out_channels: int = 1, latent_dim: int = 16, base_channels: int = 64, num_down: int = 4, k: int = 5):
        super().__init__()
        ch = base_channels * (2 ** (num_down - 1)) if num_down > 1 else base_channels
        self.from_latent = nn.Conv1d(latent_dim, ch, kernel_size=1, bias=True)

        blocks: List[nn.Module] = []
        for i in range(num_down):
            next_ch = ch // 2 if i < num_down - 1 else base_channels
            blocks += [
                ResBlock1D(ch, k=k),
                nn.ConvTranspose1d(ch, next_ch, kernel_size=4, stride=2, padding=1, bias=True),
                _gn1d(next_ch),
                nn.SiLU(inplace=True),
            ]
            ch = next_ch

        self.trunk = nn.Sequential(*blocks)
        self.out_conv = nn.Conv1d(base_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        h = self.trunk(h)
        return self.out_conv(h)

class SpectraAutoEncoder(nn.Module):
    """
    Flux-only AE.
      Input:  x [B, L, 1] (normalized flux)
      Output: x_hat [B, L, 1] (normalized flux)

    Training uses:
      mask [B,L] bool
      w    [B,L] float (weights derived from ivar; zeros for invalid)
    """
    def __init__(self, latent_dim: int = 16, base_channels: int = 64, num_down: int = 4, k: int = 5):
        super().__init__()
        self.in_channels = 1
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.num_down = num_down
        self.k = k

        self.encoder = SpectraEncoder(in_channels=1, latent_dim=latent_dim, base_channels=base_channels, num_down=num_down, k=k)
        self.decoder = SpectraDecoder(out_channels=1, latent_dim=latent_dim, base_channels=base_channels, num_down=num_down, k=k)

        self.downsample_factor = self.encoder.downsample_factor
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor, L_out: Optional[int] = None) -> torch.Tensor:
        x_hat = self.decoder(z)  # [B,1,Lhat]
        Lh = x_hat.shape[-1]
        if L_out is not None:
            if Lh > L_out:
                x_hat = x_hat[..., :L_out]
            elif Lh < L_out:
                x_hat = F.pad(x_hat, (0, L_out - Lh))
        return x_hat.transpose(1, 2).contiguous()  # [B,L,1]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z, L_out=x.shape[1])
        return x_hat, z

    def compute_loss(
        self,
        x: torch.Tensor,                      # [B,L,1]
        mask: Optional[torch.Tensor] = None,  # [B,L] bool
        w: Optional[torch.Tensor] = None,     # [B,L] float (weights)
    ) -> torch.Tensor:
        x_hat, _ = self(x)

        if mask is None and w is None:
            return F.mse_loss(x_hat, x)

        # valid mask
        if mask is None:
            m = torch.isfinite(x[..., 0])
        else:
            m = mask.to(dtype=torch.bool, device=x.device)

        # weights
        if w is None:
            ww = m.to(dtype=x.dtype, device=x.device)  # uniform weights on valid
        else:
            ww = w.to(dtype=x.dtype, device=x.device)
            ww = ww * m.to(dtype=x.dtype)

        # compute weighted MSE on flux channel
        diff2 = (x_hat - x).pow(2)[..., 0]  # [B,L]
        num = (diff2 * ww).sum()
        den = ww.sum().clamp_min(1e-6)
        return num / den

    def train_epoch(self, dataloader, optimizer, device) -> float:
        self.train()
        total = 0.0
        n = 0

        for batch in tqdm(dataloader, desc="Training", leave=False):
            # batch: (x, mask, w, lengths, [stats])
            if isinstance(batch, (tuple, list)) and (len(batch) == 5 or len(batch) == 6):
                x, mask, w, lengths = batch[0], batch[1], batch[2], batch[3]
            else:
                raise ValueError("Expected batch from desi_collate_pad_flux_only")

            x = x.to(device, non_blocking=True).float()
            mask = mask.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            loss = self.compute_loss(x, mask=mask, w=w)
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
            if isinstance(batch, (tuple, list)) and (len(batch) == 5 or len(batch) == 6):
                x, mask, w, lengths = batch[0], batch[1], batch[2], batch[3]
            else:
                raise ValueError("Expected batch from desi_collate_pad_flux_only")

            x = x.to(device, non_blocking=True).float()
            mask = mask.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True).float()

            loss = self.compute_loss(x, mask=mask, w=w)

            bs = x.shape[0]
            total += float(loss.detach().cpu()) * bs
            n += bs

        return total / max(n, 1)

    def save(self, path: str, additional_info: Optional[dict] = None):
        torch.save(
            {
                "model_state_dict": self.state_dict(),
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
            latent_dim=ckpt.get("latent_dim", 16),
            base_channels=ckpt.get("base_channels", 64),
            num_down=ckpt.get("num_down", 4),
            k=ckpt.get("k", 5),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        logpool.info(f"Loaded spectra AE (flux-only) from {path}")
        return model