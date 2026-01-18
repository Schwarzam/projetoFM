import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralPatchRVQ(nn.Module):
    """
    Patch-tokenize a 1D spectrum with features (flux, ivar) and apply ResidualVQ.

    Input:
      x: [B, L, C]  (you want C=2: [flux, ivar])

    Patchify:
      P = patch_size
      T = ceil(L / P)
      tokens: [B*T, P*C]

    Output:
      codes: [B, T, R]
      x_q : [B, L, C]  (reconstructed from quantized tokens, cropped to original L)
      recon_loss: MSE on valid positions (optional mask)
    """
    def __init__(
        self,
        rvq,                 # an instance of your ResidualVQ(dim=P*C, ...)
        patch_size: int = 8,
        channels: int = 2,
        pad_value: float = 0.0,
    ):
        super().__init__()
        self.rvq = rvq
        self.patch_size = int(patch_size)
        self.channels = int(channels)
        self.pad_value = float(pad_value)

    def _pad_to_patch_multiple(self, x: torch.Tensor):
        # x: [B, L, C]
        B, L, C = x.shape
        P = self.patch_size
        Lp = (L + P - 1) // P * P
        pad_len = Lp - L
        if pad_len == 0:
            return x, L, Lp
        pad = x.new_full((B, pad_len, C), self.pad_value)
        return torch.cat([x, pad], dim=1), L, Lp

    def patchify(self, x: torch.Tensor):
        # x: [B, L, C] -> tok: [B*T, P*C], plus (L, Lp, T)
        assert x.ndim == 3, f"Expected [B,L,C], got {tuple(x.shape)}"
        B, L, C = x.shape
        assert C == self.channels, f"Expected C={self.channels}, got {C}"

        x_pad, L_orig, Lp = self._pad_to_patch_multiple(x)
        P = self.patch_size
        T = Lp // P

        # [B, Lp, C] -> [B, T, P, C] -> [B*T, P*C]
        tok = x_pad.view(B, T, P, C).reshape(B * T, P * C)
        return tok, L_orig, Lp, T

    def unpatchify(self, tok_q: torch.Tensor, B: int, T: int, L_orig: int):
        # tok_q: [B*T, P*C] -> [B, L_orig, C]
        P = self.patch_size
        C = self.channels
        xq = tok_q.view(B, T, P, C).reshape(B, T * P, C)
        return xq[:, :L_orig, :].contiguous()

    @torch.no_grad()
    def encode(self, x: torch.Tensor, update_ema: bool = False):
        """
        Returns codes + z_q in token space + reconstructed x_q in sample space.
        """
        # disable ema updates automatically in eval unless user forces
        if not self.training:
            update_ema = False

        tok, L_orig, Lp, T = self.patchify(x)
        out = self.rvq(tok, update_ema=update_ema)  # your ResidualVQ returns dict
        codes = out["codes"].view(x.shape[0], T, self.rvq.R)          # [B,T,R]
        tok_q = out["z_q"]                                           # [B*T, P*C]
        x_q = self.unpatchify(tok_q, x.shape[0], T, L_orig)          # [B,L, C]
        return {
            "codes": codes,
            "tok_q": tok_q,
            "x_q": x_q,
            "T": T,
            "L": L_orig,
            "recon_loss": out["recon_loss"],
            "usage": out["usage"],
        }

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, L: int):
        """
        Decode from codes only.
        codes: [B, T, R] (ints)
        L: original length to crop to
        """
        assert codes.ndim == 3, f"Expected [B,T,R], got {tuple(codes.shape)}"
        B, T, R = codes.shape
        assert R == self.rvq.R, f"Expected R={self.rvq.R}, got {R}"

        codes_flat = codes.reshape(B * T, R).to(torch.long)          # [B*T,R]
        tok_q = self.rvq.decode_codes(codes_flat)                    # [B*T, P*C]
        x_q = self.unpatchify(tok_q, B, T, L)                        # [B,L,C]
        return x_q

    def compute_loss(self, x: torch.Tensor, update_ema: bool = True, mask: torch.Tensor | None = None):
        """
        x: [B,L,C]
        mask: optional [B,L] boolean (True = valid). If provided, loss uses only valid samples.
        """
        tok, L_orig, Lp, T = self.patchify(x)
        out = self.rvq(tok, update_ema=update_ema)
        tok_q = out["z_q"]
        x_q = self.unpatchify(tok_q, x.shape[0], T, L_orig)

        if mask is not None:
            # mask: [B,L] -> [B,L,1] broadcast over C
            m = mask.to(dtype=torch.bool, device=x.device).unsqueeze(-1)
            if m.any():
                loss = F.mse_loss(x_q[m], x[m])
            else:
                loss = x_q.sum() * 0.0
        else:
            loss = F.mse_loss(x_q, x)

        return {
            "loss": loss,
            "x_q": x_q,
            "codes": out["codes"].view(x.shape[0], T, self.rvq.R),
            "usage": out["usage"],
        }

    def train_epoch(self, dataloader, device=None, update_ema: bool = True):
        self.train()
        if device is not None:
            self.to(device)

        dev = self.rvq.embed.device  # works even if rvq has only buffers
        n = 0
        loss_sum = 0.0

        for batch in dataloader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(dev, non_blocking=True).float()

            out = self.compute_loss(x, update_ema=update_ema, mask=None)
            bs = x.shape[0]
            n += bs
            loss_sum += float(out["loss"].detach().cpu()) * bs

        return {"loss": loss_sum / max(n, 1)}