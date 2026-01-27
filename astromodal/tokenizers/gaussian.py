#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaussianized Scalar Tokenizer (empirical CDF -> Normal quantiles -> equal-mass bins)

Implements the exact idea:
  z = Phi^{-1}( F_x(x) )
  token = quantize(z) into K fixed Gaussian quantiles
  x_hat ≈ F_x^{-1}( Phi(z_token_center) )

This is meant for scalar/tokenizing features like:
- Gaia BP/RP coefficients (per coefficient independently)
- Fluxes, mags (optionally with log10/asinh pre-transform)
- Any 1D scalar distributions (long tails, sharp peaks)

Dependencies:
  pip install numpy scipy

Usage quickstart:
  tok = GaussianScalarTokenizer1D.fit(x_train, K=1024, transform="none")
  ids = tok.encode(x_values)
  x_hat = tok.decode(ids)

Save/load:
  tok.save("tok_coeff_003.npz")
  tok2 = GaussianScalarTokenizer1D.load("tok_coeff_003.npz")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import norm


ArrayLike = Union[np.ndarray, Sequence[float]]


# -----------------------------
# Utilities
# -----------------------------
def _as_float64_1d(x: ArrayLike) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return a.ravel()


def _safe_clip_u(u: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(u, eps, 1.0 - eps)


def _gaussian_bin_centers(K: int) -> np.ndarray:
    """
    Fixed Gaussian bin centers at equally spaced quantiles:
      c_k = Phi^{-1}((k + 0.5) / K), k=0..K-1
    """
    k = np.arange(K, dtype=np.float64)
    u = (k + 0.5) / float(K)
    return norm.ppf(u)


# -----------------------------
# 1D Tokenizer
# -----------------------------
@dataclass
class GaussianScalarTokenizer1D:
    """
    One tokenizer for ONE scalar dimension.

    Stores a sorted training sample (after optional pre-transform).
    Encoding:
      x -> pre(x) -> u = empirical_CDF(x) -> z = Phi^{-1}(u) -> token by nearest Gaussian center
    Decoding:
      token -> z_center -> u = Phi(z_center) -> approx inverse CDF -> inv_pre(...)
    """

    sorted_x: np.ndarray  # sorted training samples after pre-transform
    K: int = 1024

    # optional pre-transform
    transform: str = "none"   # "none" | "log10" | "asinh"
    asinh_scale: float = 1.0

    # small eps used for stability
    eps_u: float = 1e-6

    def __post_init__(self):
        self.sorted_x = np.asarray(self.sorted_x, dtype=np.float64).ravel()
        if self.sorted_x.size == 0:
            raise ValueError("sorted_x is empty; fit on non-empty training data.")
        if not np.all(np.isfinite(self.sorted_x)):
            raise ValueError("sorted_x contains non-finite values; clean training data before fitting.")
        if self.K <= 1:
            raise ValueError("K must be >= 2")
        if self.transform not in ("none", "log10", "asinh"):
            raise ValueError("transform must be one of: 'none', 'log10', 'asinh'")
        if self.transform == "asinh" and (not np.isfinite(self.asinh_scale) or self.asinh_scale <= 0):
            raise ValueError("asinh_scale must be > 0 for transform='asinh'")

    # ---------- pre-transform ----------
    def _pre(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.transform == "log10":
            # assumes x>0 (flux-like). Clamp tiny to avoid -inf
            return np.log10(np.clip(x, 1e-30, None))
        if self.transform == "asinh":
            s0 = float(self.asinh_scale)
            return np.arcsinh(x / s0)
        return x

    def _inv_pre(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if self.transform == "log10":
            return 10.0 ** y
        if self.transform == "asinh":
            s0 = float(self.asinh_scale)
            return s0 * np.sinh(y)
        return y

    # ---------- empirical CDF & inverse ----------
    def _Fx(self, x_pre: np.ndarray) -> np.ndarray:
        """
        Empirical CDF approximation via rank in sorted_x.
        Uses (rank + 0.5)/N to avoid exactly 0 or 1.
        """
        ranks = np.searchsorted(self.sorted_x, x_pre, side="left").astype(np.float64)
        u = (ranks + 0.5) / float(self.sorted_x.size)
        return _safe_clip_u(u, self.eps_u)

    def _Fx_inv(self, u: np.ndarray) -> np.ndarray:
        """
        Approximate inverse empirical CDF by indexing into sorted_x.
        """
        u = _safe_clip_u(np.asarray(u, dtype=np.float64), self.eps_u)
        idx = np.floor(u * self.sorted_x.size).astype(np.int64)
        idx = np.clip(idx, 0, self.sorted_x.size - 1)
        return self.sorted_x[idx]

    # ---------- main API ----------
    def encode(self, x: ArrayLike) -> np.ndarray:
        """
        Encode scalar(s) to tokens in [0, K-1].
        Returns int64 array with same shape as input.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        x_pre = self._pre(x_arr)

        u = self._Fx(x_pre)
        z = norm.ppf(u)

        centers = _gaussian_bin_centers(self.K)  # [K]
        # nearest center in L2
        flat = z.reshape(-1)
        # vectorized nearest: argmin |z - centers|
        # (K=1024 is OK; if you want faster, use np.searchsorted since centers are sorted)
        # We'll do searchsorted-based nearest for speed:
        j = np.searchsorted(centers, flat, side="left")
        j0 = np.clip(j - 1, 0, self.K - 1)
        j1 = np.clip(j, 0, self.K - 1)
        d0 = np.abs(flat - centers[j0])
        d1 = np.abs(flat - centers[j1])
        tok_flat = np.where(d1 < d0, j1, j0).astype(np.int64)

        return tok_flat.reshape(z.shape)

    def decode(self, tokens: ArrayLike) -> np.ndarray:
        """
        Decode token(s) back to approximate original scalar(s).
        Returns float64 array with same shape as tokens.
        """
        t = np.asarray(tokens, dtype=np.int64)
        centers = _gaussian_bin_centers(self.K)
        z = centers[np.clip(t, 0, self.K - 1)]

        u = norm.cdf(z)
        x_pre = self._Fx_inv(u)
        x_hat = self._inv_pre(x_pre)
        return x_hat

    # ---------- persistence ----------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            sorted_x=self.sorted_x.astype(np.float64),
            K=np.int64(self.K),
            transform=np.array(self.transform),
            asinh_scale=np.float64(self.asinh_scale),
            eps_u=np.float64(self.eps_u),
        )

    @staticmethod
    def load(path: str | Path) -> "GaussianScalarTokenizer1D":
        z = np.load(str(path), allow_pickle=True)
        transform = str(z["transform"].tolist())
        return GaussianScalarTokenizer1D(
            sorted_x=z["sorted_x"].astype(np.float64),
            K=int(z["K"]),
            transform=transform,
            asinh_scale=float(z.get("asinh_scale", 1.0)),
            eps_u=float(z.get("eps_u", 1e-6)),
        )

    # ---------- fitting ----------
    @staticmethod
    def fit(
        x_train: ArrayLike,
        *,
        K: int = 1024,
        transform: str = "none",     # "none" | "log10" | "asinh"
        asinh_scale: float = 1.0,
        eps_u: float = 1e-6,
        max_samples: Optional[int] = None,  # optional subsample for huge sets
        seed: int = 0,
    ) -> "GaussianScalarTokenizer1D":
        x = _as_float64_1d(x_train)
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError("No finite training samples to fit tokenizer.")

        # optional subsample
        if max_samples is not None and x.size > max_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(x.size, size=int(max_samples), replace=False)
            x = x[idx]

        # pre-transform
        if transform == "log10":
            x = np.log10(np.clip(x, 1e-30, None))
        elif transform == "asinh":
            x = np.arcsinh(x / float(asinh_scale))
        elif transform != "none":
            raise ValueError("transform must be one of: 'none', 'log10', 'asinh'")

        x = np.sort(x.astype(np.float64))

        return GaussianScalarTokenizer1D(
            sorted_x=x,
            K=int(K),
            transform=transform,
            asinh_scale=float(asinh_scale),
            eps_u=float(eps_u),
        )


# -----------------------------
# Multi-dim convenience wrapper
# -----------------------------
@dataclass
class GaussianScalarTokenizerND:
    """
    Tokenize a vector of D scalars by applying independent 1D tokenizers.

    Useful for:
      - GaiaXP coefficients: D=110
      - any fixed-length scalar vector

    Encode:
      X: [N, D] -> tokens: [N, D] (int64)

    Decode:
      tokens: [N, D] -> X_hat: [N, D]
    """

    tokenizers: List[GaussianScalarTokenizer1D]

    def __post_init__(self):
        if len(self.tokenizers) == 0:
            raise ValueError("tokenizers list is empty")

    @property
    def D(self) -> int:
        return len(self.tokenizers)

    def encode(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            if X.size != self.D:
                raise ValueError(f"Expected shape [D]={self.D}, got {X.shape}")
            return np.array([self.tokenizers[d].encode(X[d]) for d in range(self.D)], dtype=np.int64)

        if X.ndim != 2 or X.shape[1] != self.D:
            raise ValueError(f"Expected shape [N,D] with D={self.D}, got {X.shape}")

        N = X.shape[0]
        out = np.empty((N, self.D), dtype=np.int64)
        for d in range(self.D):
            out[:, d] = self.tokenizers[d].encode(X[:, d])
        return out

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        t = np.asarray(tokens, dtype=np.int64)
        if t.ndim == 1:
            if t.size != self.D:
                raise ValueError(f"Expected shape [D]={self.D}, got {t.shape}")
            return np.array([self.tokenizers[d].decode(t[d]) for d in range(self.D)], dtype=np.float64)

        if t.ndim != 2 or t.shape[1] != self.D:
            raise ValueError(f"Expected shape [N,D] with D={self.D}, got {t.shape}")

        N = t.shape[0]
        out = np.empty((N, self.D), dtype=np.float64)
        for d in range(self.D):
            out[:, d] = self.tokenizers[d].decode(t[:, d])
        return out

    def save_folder(self, folder: str | Path, prefix: str = "tok") -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        meta = {"D": self.D, "prefix": prefix}
        np.savez_compressed(folder / "meta.npz", **meta)
        for d, tok in enumerate(self.tokenizers):
            tok.save(folder / f"{prefix}_{d:03d}.npz")

    @staticmethod
    def load_folder(folder: str | Path, prefix: str = "tok") -> "GaussianScalarTokenizerND":
        folder = Path(folder)
        meta_path = folder / "meta.npz"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {meta_path}")
        meta = np.load(str(meta_path), allow_pickle=True)
        D = int(meta["D"])
        toks = []
        for d in range(D):
            p = folder / f"{prefix}_{d:03d}.npz"
            toks.append(GaussianScalarTokenizer1D.load(p))
        return GaussianScalarTokenizerND(tokenizers=toks)


# -----------------------------
# Example main (optional)
# -----------------------------
if __name__ == "__main__":
    # Example: fit + encode + decode on synthetic data
    rng = np.random.default_rng(0)
    x_train = rng.lognormal(mean=0.0, sigma=1.0, size=200_000)

    tok = GaussianScalarTokenizer1D.fit(
        x_train,
        K=1024,
        transform="log10",   # good for positive heavy-tailed flux-like
        max_samples=200_000,
    )

    x_test = rng.lognormal(mean=0.0, sigma=1.0, size=10_000)
    ids = tok.encode(x_test)
    x_hat = tok.decode(ids)

    mae = np.median(np.abs(x_hat - x_test))
    print(f"K={tok.K} transform={tok.transform} median|err|={mae:.3e}")

    # Save/load
    tok.save("gauss_tok_1d.npz")
    tok2 = GaussianScalarTokenizer1D.load("gauss_tok_1d.npz")
    assert tok2.K == tok.K
    print("Saved/loaded OK")