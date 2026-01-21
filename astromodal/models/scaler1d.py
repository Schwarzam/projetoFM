from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

@dataclass
class StandardScaler1D:
    mean: float
    std: float
    # optional pre-transform for heavy tails
    transform: str = "none"   # "none" | "asinh"
    asinh_scale: float = 1.0  # used if transform == "asinh"

    def _pre(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.transform == "asinh":
            s0 = float(self.asinh_scale) if self.asinh_scale > 0 else 1.0
            return np.arcsinh(x / s0)
        return x

    def _inv_pre(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if self.transform == "asinh":
            s0 = float(self.asinh_scale) if self.asinh_scale > 0 else 1.0
            return s0 * np.sinh(y)
        return y

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        y = self._pre(x)
        return (y - self.mean) / (self.std + 1e-12)

    def inverse_transform_x(self, x_norm: np.ndarray) -> np.ndarray:
        y = x_norm * self.std + self.mean
        return self._inv_pre(y)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=np.float64(self.mean),
            std=np.float64(self.std),
            transform=np.array(self.transform),
            asinh_scale=np.float64(self.asinh_scale),
        )

    @staticmethod
    def load(path: str | Path) -> "StandardScaler1D":
        z = np.load(str(path), allow_pickle=True)
        transform = str(z["transform"].tolist())
        return StandardScaler1D(
            mean=float(z["mean"]),
            std=float(z["std"]),
            transform=transform,
            asinh_scale=float(z.get("asinh_scale", 1.0)),
        )