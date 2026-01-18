import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from astromodal.tokenizers.rvq import ResidualVQ

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