import torch
import torch.nn as nn
import torch.nn.functional as F
import logpool

from astromodal.tokenizers.rvq import ResidualVQ

def make_rvq(cfg: dict):
    """
    Rebuild ResidualVQ exactly as used during training.
    cfg comes from the checkpoint.
    """
    # dim is mandatory
    dim = cfg.get("dim", cfg.get("D", None))
    if dim is None:
        raise ValueError("RVQ config missing 'dim'")

    return ResidualVQ(
        dim=int(dim),
        num_stages=int(cfg.get("num_stages", 3)),
        codebook_size=int(cfg.get("codebook_size", 1024)),
        decay=float(cfg.get("decay", 0.99)),
    )
    
def _weighted_mse(x_q: torch.Tensor, x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x_q, x: [B, L, C]
    w:      [B, L] or [B, L, 1]
    Computes sum(w * (x_q-x)^2) / sum(w) over valid positions.
    """
    if w.ndim == 2:
        w = w.unsqueeze(-1)  # [B,L,1]
    w = w.to(dtype=x.dtype, device=x.device)

    diff2 = (x_q - x) ** 2  # [B,L,C]
    num = (diff2 * w).sum()
    den = w.sum().clamp_min(eps)
    return num / den

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

    def compute_loss(
        self,
        x: torch.Tensor,
        update_ema: bool = True,
        mask: torch.Tensor | None = None,
        w: torch.Tensor | None = None,
    ):
        """
        x:    [B,L,C]
        mask: optional [B,L] bool (True=valid)
        w:    optional [B,L] float weights (0 for invalid)
        """
        tok, L_orig, Lp, T = self.patchify(x)
        out = self.rvq(tok, update_ema=update_ema)

        tok_q = out["z_q"]
        x_q = self.unpatchify(tok_q, x.shape[0], T, L_orig)  # [B,L,C]

        # ----- build effective weights -----
        if w is None and mask is None:
            loss = F.mse_loss(x_q, x)
        else:
            # start with w if provided, else uniform
            if w is None:
                ww = torch.ones((x.shape[0], L_orig), device=x.device, dtype=x.dtype)
            else:
                ww = w.to(device=x.device, dtype=x.dtype)

            # apply mask if provided
            if mask is not None:
                mm = mask.to(device=x.device, dtype=torch.bool)
                ww = ww * mm.to(dtype=x.dtype)

            # if all weights are 0, return 0 safely
            if torch.any(ww > 0):
                loss = _weighted_mse(x_q, x, ww)
            else:
                loss = x_q.sum() * 0.0

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

        dev = self.rvq.embed.device
        n = 0
        loss_sum = 0.0

        for batch in dataloader:
            # retro-compatible parsing
            if isinstance(batch, (tuple, list)):
                x = batch[0]
                mask = batch[1] if len(batch) > 1 else None
                w = batch[2] if len(batch) > 2 else None
            else:
                x, mask, w = batch, None, None

            x = x.to(dev, non_blocking=True).float()
            if mask is not None:
                mask = mask.to(dev, non_blocking=True)
            if w is not None:
                w = w.to(dev, non_blocking=True).float()

            out = self.compute_loss(x, update_ema=update_ema, mask=mask, w=w)

            bs = x.shape[0]
            n += bs
            loss_sum += float(out["loss"].detach().cpu()) * bs

        return {"loss": loss_sum / max(n, 1)}
    

    def save(self, path: str, additional_info: dict | None = None) -> None:
        """
        Save tokenizer + RVQ config/state.

        Stores:
        - spectralpatchrvq_state_dict (includes rvq parameters if it is a submodule)
        - patch_size / channels / pad_value
        - rvq hyperparams when available (dim, R, codebook_size, etc.)
        - any extra metadata you pass in additional_info
        """
        ckpt = {
            "spectralpatchrvq_state_dict": self.state_dict(),
            "patch_size": self.patch_size,
            "channels": self.channels,
            "pad_value": self.pad_value,
        }

        # best-effort RVQ config capture (works for most ResidualVQ implementations)
        rvq = self.rvq
        rvq_cfg = {}
        for k in [
            "dim",
            "D",
            "R",
            "num_stages",
            "codebook_size",
            "K",
            "decay",
            "eps",
        ]:
            if hasattr(rvq, k):
                v = getattr(rvq, k)
                # tensors -> python
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu().tolist()
                rvq_cfg[k] = v
        ckpt["rvq_config"] = rvq_cfg

        if additional_info:
            ckpt.update(additional_info)

        torch.save(ckpt, path)
        try:
            logpool.info(f"Saved SpectralPatchRVQ to {path}")
        except Exception:
            pass

    @staticmethod
    def load_from_file(
        path: str,
        *,
        rvq_ctor=lambda cfg: make_rvq(cfg),
        map_location=None,
        strict: bool = True,
    ) -> "SpectralPatchRVQ":
        """
        Load from file.

        You must provide rvq_ctor: a callable that returns a ResidualVQ instance.
        Example:
        tok = SpectralPatchRVQ.load_from_file(
            "tok.pt",
            rvq_ctor=lambda cfg: ResidualVQ(
                dim=cfg.get("dim", D),
                num_stages=cfg.get("num_stages", 2),
                codebook_size=cfg.get("codebook_size", 2048),
                decay=cfg.get("decay", 0.99),
            ),
            map_location="cpu",
        )

        Why this design:
        ResidualVQ class signatures vary; this avoids hard-coding constructor args.
        """
        ckpt = torch.load(path, map_location=map_location)

        patch_size = int(ckpt.get("patch_size", 8))
        channels = int(ckpt.get("channels", 2))
        pad_value = float(ckpt.get("pad_value", 0.0))

        rvq_cfg = ckpt.get("rvq_config", {})
        rvq = rvq_ctor(rvq_cfg)

        tok = SpectralPatchRVQ(
            rvq=rvq,
            patch_size=patch_size,
            channels=channels,
            pad_value=pad_value,
        )

        tok.load_state_dict(ckpt["spectralpatchrvq_state_dict"], strict=strict)

        try:
            logpool.info(f"Loaded SpectralPatchRVQ from {path}")
        except Exception:
            pass

        return tok