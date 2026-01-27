from dataclasses import dataclass
import numpy as np
from pathlib import Path

@dataclass
class StandardScalerVec:
    mean: np.ndarray          # [D]
    std: np.ndarray           # [D]
    transform: str = "none"   # "none" | "asinh"
    asinh_scale: float = 1.0

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
        np.savez(path, mean=self.mean.astype(np.float64), std=self.std.astype(np.float64),
                 transform=np.array(self.transform), asinh_scale=np.float64(self.asinh_scale))

    @staticmethod
    def load(path: str | Path) -> "StandardScalerVec":
        z = np.load(str(path), allow_pickle=True)
        return StandardScalerVec(
            mean=np.asarray(z["mean"], dtype=np.float64),
            std=np.asarray(z["std"], dtype=np.float64),
            transform=str(z["transform"].tolist()),
            asinh_scale=float(z.get("asinh_scale", 1.0)),
        )