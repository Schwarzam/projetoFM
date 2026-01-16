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


# -------------------------
# SpatialRVQ with train/save/load/encode/decode
# -------------------------

class SpatialRVQ(nn.Module):
    """
    Applies ResidualVQ to each spatial location of z_map.

    z_map: [B,C,H,W]
    codes: [B,H,W,R]
    z_q:   [B,C,H,W]
    """
    def __init__(self, channels: int, num_stages=6, codebook_size=512, decay=0.99, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.rvq = ResidualVQ(
            dim=channels,
            num_stages=num_stages,
            codebook_size=codebook_size,
            decay=decay,
            eps=eps,
        )

    @property
    def R(self) -> int:
        return self.rvq.R

    @property
    def K(self) -> int:
        return self.rvq.K
    
    def forward(self, z_map: torch.Tensor, update_ema: bool = True):
        # during eval() we typically do NOT want EMA updates
        if not self.training:
            update_ema = False
        return self.encode(z_map, update_ema=update_ema)

    def encode(self, z_map: torch.Tensor, update_ema: bool = False) -> Dict[str, Any]:
        """
        Returns quantization outputs including integer codes.

        z_map: [B,C,H,W]
        """
        assert z_map.ndim == 4, f"Expected [B,C,H,W], got {tuple(z_map.shape)}"
        B, C, H, W = z_map.shape
        assert C == self.channels, f"Expected C={self.channels}, got {C}"

        z_tok = z_map.permute(0, 2, 3, 1).reshape(B * H * W, C)
        out = self.rvq(z_tok, update_ema=update_ema)

        z_q_tok = out["z_q"].reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        codes = out["codes"].reshape(B, H, W, self.rvq.R).contiguous()

        return {
            "codes": codes,            # [B,H,W,R]
            "z_q": z_q_tok,            # [B,C,H,W]
            "recon_loss": F.mse_loss(z_q_tok, z_map),
            "usage": out["usage"],     # [R,K]
        }

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode integer codes back into a quantized latent map.

        codes: [B,H,W,R] int64
        returns z_q: [B,C,H,W]
        """
        assert codes.ndim == 4, f"Expected [B,H,W,R], got {tuple(codes.shape)}"
        B, H, W, R = codes.shape
        assert R == self.rvq.R, f"Expected R={self.rvq.R}, got {R}"

        codes_flat = codes.reshape(B * H * W, R).to(torch.long)
        zq_flat = self.rvq.decode_codes(codes_flat)             # [B*H*W, C]
        z_q = zq_flat.reshape(B, H, W, self.channels).permute(0, 3, 1, 2).contiguous()
        return z_q

    def compute_loss(self, z_map: torch.Tensor, update_ema: bool = True) -> Dict[str, Any]:
        """
        Main objective for offline RVQ training: MSE(z_q, z_map).

        Returns dict with loss + helpful metrics.
        """
        out = self.encode(z_map, update_ema=update_ema)
        loss = out["recon_loss"]
        with torch.no_grad():
            z = z_map.flatten(1)
            zq = out["z_q"].flatten(1)
            mean_sq_L2 = ((z - zq) ** 2).sum(dim=1).mean()
            per_dim_MSE = ((z - zq) ** 2).mean()
        out.update({"loss": loss, "mean_sq_L2": mean_sq_L2, "per_dim_MSE": per_dim_MSE})
        return out

    def train_epoch(self, dataloader, device=None, optimizer=None, update_ema=True):
        self.train()
        if device is not None:
            self.to(device)

        dev = self.rvq.embed.device  # <-- works even with no parameters

        n = 0
        loss_sum = 0.0
        mse_sum = 0.0
        l2_sum = 0.0

        for batch in dataloader:
            z_map = batch[0] if isinstance(batch, (tuple, list)) else batch
            z_map = z_map.to(dev, non_blocking=True)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            out = self.compute_loss(z_map, update_ema=update_ema)

            if optimizer is not None:
                out["loss"].backward()
                optimizer.step()

            bs = z_map.shape[0]
            n += bs
            loss_sum += float(out["loss"].detach().cpu()) * bs
            mse_sum  += float(out["per_dim_MSE"].detach().cpu()) * bs
            l2_sum   += float(out["mean_sq_L2"].detach().cpu()) * bs

        return {
            "loss": loss_sum / max(n, 1),
            "per_dim_MSE": mse_sum / max(n, 1),
            "mean_sq_L2": l2_sum / max(n, 1),
        }

    # ---------- checkpointing ----------
    def save(self, path: str, additional_info: Optional[dict] = None) -> None:
        payload = {
            "model_state_dict": self.state_dict(),
            "channels": self.channels,
            "num_stages": self.rvq.R,
            "codebook_size": self.rvq.K,
            "decay": self.rvq.decay,
            "eps": self.rvq.eps,
        }
        if additional_info is not None:
            payload.update(additional_info)
        torch.save(payload, path)

    def load(self, path: str, map_location: Optional[torch.device] = None) -> dict:
        ckpt = torch.load(path, map_location=map_location)
        self.load_state_dict(ckpt["model_state_dict"])
        return ckpt

    @staticmethod
    def load_from_file(path: str, map_location: Optional[torch.device] = None) -> "SpatialRVQ":
        ckpt = torch.load(path, map_location=map_location)
        model = SpatialRVQ(
            channels=ckpt["channels"],
            num_stages=ckpt["num_stages"],
            codebook_size=ckpt["codebook_size"],
            decay=ckpt.get("decay", 0.99),
            eps=ckpt.get("eps", 1e-5),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return model