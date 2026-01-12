import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualVQ(nn.Module):
    """
    Residual Vector Quantization (RVQ) with EMA codebook updates.
    Offline-friendly: you can train this directly on frozen latents.

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
        kmeans_init: bool = False,   # optional hook
    ):
        super().__init__()
        self.dim = dim
        self.R = num_stages
        self.K = codebook_size
        self.decay = decay
        self.eps = eps

        # Codebooks: [R, K, D]
        embed = torch.randn(self.R, self.K, self.dim) / (self.dim ** 0.5)
        self.register_buffer("embed", embed)

        # EMA buffers
        self.register_buffer("ema_cluster_size", torch.zeros(self.R, self.K))
        self.register_buffer("ema_embed_sum", torch.zeros(self.R, self.K, self.dim))

        self.kmeans_init = kmeans_init
        self._initted = False

    @torch.no_grad()
    def init_from_data(self, data_loader, device=None, max_batches=200):
        """
        Optional: initialize each stage with KMeans-like seeding by sampling data.
        Good if your latents are not N(0,1).
        """
        if self._initted:
            return

        device = device or next(self.parameters()).device
        xs = []
        for i, batch in enumerate(data_loader):
            x = batch.to(device) if torch.is_tensor(batch) else batch[0].to(device)
            xs.append(x)
            if i + 1 >= max_batches:
                break
        x = torch.cat(xs, dim=0)  # [N, D]

        # Simple init: pick random points for each stage
        # (You can replace with actual k-means if you want.)
        for s in range(self.R):
            idx = torch.randperm(x.shape[0], device=device)[: self.K]
            self.embed[s] = x[idx].clone()
        self._initted = True

    def _nearest_code(self, residual: torch.Tensor, codebook: torch.Tensor):
        """
        residual: [B, D]
        codebook: [K, D]
        returns:
          codes: [B]
          quant: [B, D]
        """
        # compute squared L2 distances efficiently:
        # ||x - e||^2 = ||x||^2 + ||e||^2 - 2 x.e
        x2 = (residual ** 2).sum(dim=1, keepdim=True)          # [B, 1]
        e2 = (codebook ** 2).sum(dim=1).unsqueeze(0)           # [1, K]
        xe = residual @ codebook.t()                           # [B, K]
        dist = x2 + e2 - 2 * xe                                # [B, K]
        codes = dist.argmin(dim=1)                             # [B]
        quant = codebook[codes]                                # [B, D]
        return codes, quant

    @torch.no_grad()
    def _ema_update(self, stage: int, residual: torch.Tensor, codes: torch.Tensor):
        """
        EMA update for one stage.
        residual: [B, D]
        codes:    [B]
        """
        # one-hot counts: [B, K]
        onehot = F.one_hot(codes, num_classes=self.K).type_as(residual)  # float
        cluster_size = onehot.sum(dim=0)                                  # [K]
        embed_sum = onehot.t() @ residual                                 # [K, D]

        self.ema_cluster_size[stage].mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.ema_embed_sum[stage].mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        # Laplace smoothing for stability
        n = self.ema_cluster_size[stage].sum()
        smoothed = (self.ema_cluster_size[stage] + self.eps) / (n + self.K * self.eps) * n

        new_embed = self.ema_embed_sum[stage] / smoothed.unsqueeze(1).clamp_min(self.eps)
        self.embed[stage].copy_(new_embed)

    def forward(self, z: torch.Tensor, update_ema: bool = True):
        """
        z: [B, D]
        """
        assert z.ndim == 2 and z.shape[1] == self.dim, f"Expected [B,{self.dim}] got {tuple(z.shape)}"

        residual = z
        z_q = torch.zeros_like(z)
        all_codes = []

        for s in range(self.R):
            codebook = self.embed[s]  # [K, D]
            codes, quant = self._nearest_code(residual, codebook)
            all_codes.append(codes)

            z_q = z_q + quant
            residual = residual - quant

            if self.training and update_ema:
                self._ema_update(s, residual + quant, codes)  # update using the input to this stage

        codes = torch.stack(all_codes, dim=1)  # [B, R]
        # reconstruction loss (what you optimize offline)
        recon_loss = F.mse_loss(z_q, z)

        # Optional stats: dead codes detection
        with torch.no_grad():
            usage = []
            for s in range(self.R):
                u = torch.bincount(codes[:, s], minlength=self.K).float()
                usage.append(u / u.sum().clamp_min(1.0))
            usage = torch.stack(usage, dim=0)  # [R, K]

        return {
            "codes": codes,
            "z_q": z_q,
            "residual": residual,
            "recon_loss": recon_loss,
            "usage": usage,
        }

class SpatialRVQ(nn.Module):
    """
    Applies your existing ResidualVQ to each spatial location of a latent map.

    z_map: [B, C, H, W]
    returns:
      codes: [B, H, W, R]   (ints)
      z_q:   [B, C, H, W]
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

    def forward(self, z_map: torch.Tensor, update_ema: bool = True):
        assert z_map.ndim == 4, f"Expected [B,C,H,W], got {tuple(z_map.shape)}"
        B, C, H, W = z_map.shape
        assert C == self.channels

        # (B,C,H,W) -> (B*H*W, C)
        z_tok = z_map.permute(0, 2, 3, 1).reshape(B * H * W, C)

        out = self.rvq(z_tok, update_ema=update_ema)

        # tokens -> map
        z_q_tok = out["z_q"].reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # codes: (B*H*W, R) -> (B,H,W,R)
        codes = out["codes"].reshape(B, H, W, self.rvq.R)

        return {
            "codes": codes,      # [B,H,W,R]
            "z_q": z_q_tok,      # [B,C,H,W]
            "recon_loss": F.mse_loss(z_q_tok, z_map),
            "usage": out["usage"],  # [R,K] from last batch tokens
        }
        