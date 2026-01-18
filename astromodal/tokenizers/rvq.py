import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

# -------------------------
# ResidualVQ with decode from codes
# -------------------------

class ResidualVQ(nn.Module):
    """
    Residual Vector Quantization (RVQ) with EMA codebook updates.

    z:      [B, D]
    codes:  [B, R]
    z_q:    [B, D]
    """
    def __init__(
        self,
        dim: int,
        num_stages: int = 6,
        codebook_size: int = 512,
        decay: float = 0.99,
        eps: float = 1e-5,
        kmeans_init: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.R = num_stages
        self.K = codebook_size
        self.decay = decay
        self.eps = eps

        embed = torch.randn(self.R, self.K, self.dim) / (self.dim ** 0.5)
        self.register_buffer("embed", embed)

        self.register_buffer("ema_cluster_size", torch.zeros(self.R, self.K))
        self.register_buffer("ema_embed_sum", torch.zeros(self.R, self.K, self.dim))

        self.kmeans_init = kmeans_init
        self._initted = False

    def _nearest_code(self, residual: torch.Tensor, codebook: torch.Tensor):
        x2 = (residual ** 2).sum(dim=1, keepdim=True)          # [B,1]
        e2 = (codebook ** 2).sum(dim=1).unsqueeze(0)           # [1,K]
        xe = residual @ codebook.t()                           # [B,K]
        dist = x2 + e2 - 2 * xe                                # [B,K]
        codes = dist.argmin(dim=1)                             # [B]
        quant = codebook[codes]                                # [B,D]
        return codes, quant

    @torch.no_grad()
    def _ema_update(self, stage: int, x_stage_in: torch.Tensor, codes: torch.Tensor):
        # x_stage_in is the input to this stage (before subtracting quant)
        onehot = F.one_hot(codes, num_classes=self.K).type_as(x_stage_in)  # [B,K]
        cluster_size = onehot.sum(dim=0)                                    # [K]
        embed_sum = onehot.t() @ x_stage_in                                 # [K,D]

        self.ema_cluster_size[stage].mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.ema_embed_sum[stage].mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        n = self.ema_cluster_size[stage].sum()
        smoothed = (self.ema_cluster_size[stage] + self.eps) / (n + self.K * self.eps) * n
        new_embed = self.ema_embed_sum[stage] / smoothed.unsqueeze(1).clamp_min(self.eps)
        self.embed[stage].copy_(new_embed)

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct z_q from codes only.

        codes: [B, R] int64
        returns z_q: [B, D]
        """
        assert codes.ndim == 2 and codes.shape[1] == self.R, f"Expected [B,{self.R}] got {tuple(codes.shape)}"
        B = codes.shape[0]
        z_q = torch.zeros(B, self.dim, device=codes.device, dtype=self.embed.dtype)
        for s in range(self.R):
            z_q = z_q + self.embed[s][codes[:, s]]
        return z_q

    def forward(self, z: torch.Tensor, update_ema: bool = True) -> Dict[str, Any]:
        assert z.ndim == 2 and z.shape[1] == self.dim, f"Expected [B,{self.dim}] got {tuple(z.shape)}"

        residual = z
        z_q = torch.zeros_like(z)
        all_codes = []

        for s in range(self.R):
            codebook = self.embed[s]  # [K,D]
            codes, quant = self._nearest_code(residual, codebook)
            all_codes.append(codes)

            x_stage_in = residual  # input to stage before subtract
            z_q = z_q + quant
            residual = residual - quant

            if self.training and update_ema:
                self._ema_update(s, x_stage_in, codes)

        codes = torch.stack(all_codes, dim=1)  # [B,R]
        recon_loss = F.mse_loss(z_q, z)

        with torch.no_grad():
            usage = []
            for s in range(self.R):
                u = torch.bincount(codes[:, s], minlength=self.K).float()
                usage.append(u / u.sum().clamp_min(1.0))
            usage = torch.stack(usage, dim=0)  # [R,K]

        return {
            "codes": codes,
            "z_q": z_q,
            "residual": residual,
            "recon_loss": recon_loss,
            "usage": usage,
        }

