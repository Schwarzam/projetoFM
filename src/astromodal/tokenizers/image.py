"""
Image tokenizer for astronomical cutout images.

Implements autoencoder-based tokenization using VQ-VAE (Vector Quantized
Variational Autoencoder) for multi-band astronomical image cutouts.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import warnings

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from .base import BaseTokenizer


class ImageTokenizer(BaseTokenizer):
    """
    Tokenizer for astronomical image cutouts using autoencoder.

    This tokenizer learns a discrete representation of multi-band image
    cutouts using an autoencoder architecture. The encoder maps images to
    a spatial latent representation, which is then quantized using a learned
    codebook to produce discrete tokens.

    The autoencoder should be trained separately using the image training
    script, and then loaded here for encoding.

    Parameters
    ----------
    cutout_size : int, default=96
        Size of image cutouts (assumed square)
    bands : List[str], optional
        List of band names (e.g., ['F378', 'F395', ...])
    latent_dim : int, default=2
        Dimension of latent representation per spatial location
    codebook_size : int, default=1024
        Size of learned codebook (vocabulary size)
    autoencoder_path : str, optional
        Path to trained autoencoder weights
    codebook_path : str, optional
        Path to learned codebook (if separate from autoencoder)
    device : str, default='cpu'
        Device for inference ('cpu' or 'cuda')

    Attributes
    ----------
    autoencoder_ : nn.Module
        Loaded autoencoder model
    codebook_ : np.ndarray
        Learned codebook vectors (codebook_size, latent_dim)
    is_fitted_ : bool
        Whether the tokenizer has been loaded/trained

    Examples
    --------
    >>> from astromodal.tokenizers import ImageTokenizer
    >>> tokenizer = ImageTokenizer(
    ...     cutout_size=96,
    ...     bands=['F378', 'F395', 'U', 'G', 'R', 'I', 'Z'],
    ...     autoencoder_path='autoencoder.pt'
    ... )
    >>> tokenizer.load('autoencoder.pt')
    >>> tokens = tokenizer.encode(dataframe)
    >>> tokenizer.save('image_tokenizer_config.npz')

    Notes
    -----
    This tokenizer requires a pre-trained autoencoder. Training should be
    done using dedicated training scripts that handle:
    - Multi-band image preprocessing
    - Elliptical masking for object regions
    - Robust normalization (arcsinh)
    - Autoencoder architecture (encoder-decoder)
    - Optional vector quantization
    """

    def __init__(
        self,
        cutout_size: int = 96,
        bands: Optional[List[str]] = None,
        latent_dim: int = 2,
        codebook_size: int = 1024,
        autoencoder_path: Optional[str] = None,
        codebook_path: Optional[str] = None,
        device: str = 'cpu',
    ):
        self.cutout_size = cutout_size
        self.bands = bands or []
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.autoencoder_path = autoencoder_path
        self.codebook_path = codebook_path
        self.device = device

        # Learned components
        self.autoencoder_: Optional[nn.Module] = None
        self.codebook_: Optional[np.ndarray] = None
        self.is_fitted_ = False

    def train(
        self,
        parquet_files: Union[str, Path, List[Union[str, Path]]],
        bands: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """
        Train the image tokenizer (autoencoder).

        Note: This is a placeholder. Actual training should be done using
        dedicated training scripts with proper GPU support, data augmentation,
        and training loops. See image_tokenizers/train_image.py for reference.

        Parameters
        ----------
        parquet_files : Union[str, Path, List[Union[str, Path]]]
            Path(s) to parquet files containing image cutouts
        bands : List[str], optional
            Band names to use
        **kwargs
            Additional training parameters

        Raises
        ------
        NotImplementedError
            Training must be done using dedicated scripts
        """
        raise NotImplementedError(
            "Image tokenizer training requires dedicated training scripts with GPU support. "
            "Please use the training script in image_tokenizers/train_image.py, "
            "then load the trained model using ImageTokenizer.load()."
        )

    def encode(
        self,
        data: Union[pl.DataFrame, str, Path],
        batch_size: int = 32,
    ) -> Dict[str, np.ndarray]:
        """
        Encode image cutouts into tokens.

        Parameters
        ----------
        data : Union[pl.DataFrame, str, Path]
            Data containing image cutouts (DataFrame or path to parquet)
        batch_size : int, default=32
            Batch size for encoding

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing:
            - 'image_tokens': (n_rows, latent_H, latent_W) spatial token map
            - 'image_latents': (n_rows, latent_dim, latent_H, latent_W) continuous latents
            - 'ids': row IDs if available
            - 'id_col': ID column name

        Raises
        ------
        RuntimeError
            If tokenizer has not been loaded
        """
        if not self.is_fitted_:
            raise RuntimeError("Tokenizer must be loaded before encoding")

        if self.autoencoder_ is None:
            raise RuntimeError("Autoencoder model not loaded")

        # Load data
        if isinstance(data, (str, Path)):
            df = pl.read_parquet(data)
        else:
            df = data

        n_rows = df.height
        if n_rows == 0:
            return {
                'image_tokens': np.zeros((0, 0, 0), dtype=np.uint16),
                'image_latents': np.zeros((0, self.latent_dim, 0, 0), dtype=np.float32),
                'ids': np.array([], dtype=np.int64),
                'id_col': '',
            }

        print(f"Encoding {n_rows:,} image cutouts")

        # Check required columns
        cutout_cols = [f"splus_cut_{b}" for b in self.bands]
        missing = [c for c in cutout_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing image cutout columns: {missing}")

        # Extract IDs
        id_col = self._choose_id_column(df.columns)
        if id_col is not None:
            ids = df[id_col].cast(pl.Int64).to_numpy()
        else:
            ids = np.array([], dtype=np.int64)

        # Build image batch
        images = []
        for i in range(n_rows):
            img = self._build_image(df, i)
            images.append(img)

        images_tensor = torch.stack(images, dim=0).to(self.device)  # (N, C, H, W)

        # Encode in batches
        all_latents = []
        all_tokens = []

        self.autoencoder_.eval()
        with torch.no_grad():
            for start in range(0, n_rows, batch_size):
                end = min(start + batch_size, n_rows)
                batch = images_tensor[start:end]

                # Encode to latent
                latents = self.autoencoder_.encode(batch)  # (B, latent_dim, H', W')

                # Quantize if codebook available
                if self.codebook_ is not None:
                    tokens = self._quantize_latents(latents)
                else:
                    # Simple spatial argmax if no codebook
                    tokens = torch.zeros(
                        latents.shape[0],
                        latents.shape[2],
                        latents.shape[3],
                        dtype=torch.long,
                        device=latents.device
                    )

                all_latents.append(latents.cpu().numpy())
                all_tokens.append(tokens.cpu().numpy())

        # Concatenate results
        latents_array = np.concatenate(all_latents, axis=0)  # (N, D, H', W')
        tokens_array = np.concatenate(all_tokens, axis=0)  # (N, H', W')

        return {
            'image_tokens': tokens_array.astype(np.uint16),
            'image_latents': latents_array.astype(np.float32),
            'ids': ids,
            'id_col': id_col or '',
        }

    def save(self, path: Union[str, Path]) -> None:
        """
        Save tokenizer configuration to disk.

        Note: This saves the configuration, not the full autoencoder weights.
        The autoencoder weights should be saved separately using torch.save().

        Parameters
        ----------
        path : Union[str, Path]
            Output path for config (typically .npz file)

        Raises
        ------
        RuntimeError
            If tokenizer has not been loaded
        """
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unloaded tokenizer")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'cutout_size': self.cutout_size,
            'bands': np.array(self.bands, dtype=object),
            'latent_dim': self.latent_dim,
            'codebook_size': self.codebook_size,
            'autoencoder_path': str(self.autoencoder_path) if self.autoencoder_path else '',
            'codebook_path': str(self.codebook_path) if self.codebook_path else '',
        }

        if self.codebook_ is not None:
            save_dict['codebook'] = self.codebook_

        np.savez_compressed(path, **save_dict)
        print(f"Saved image tokenizer config to {path}")

    @classmethod
    def load(cls, path: Union[str, Path], device: str = 'cpu') -> "ImageTokenizer":
        """
        Load tokenizer from saved configuration and weights.

        Parameters
        ----------
        path : Union[str, Path]
            Path to config file (.npz) or autoencoder weights (.pt)
        device : str, default='cpu'
            Device for inference

        Returns
        -------
        ImageTokenizer
            Loaded tokenizer instance

        Notes
        -----
        If path is a .pt file, it's assumed to be autoencoder weights.
        If path is a .npz file, it should contain the config and reference
        to the autoencoder weights file.
        """
        path = Path(path)

        if path.suffix == '.pt':
            # Load autoencoder weights directly
            warnings.warn(
                "Loading from .pt file. Using default configuration. "
                "For full config, save and load from .npz file."
            )
            tokenizer = cls(device=device)
            tokenizer._load_autoencoder(path, device)
            tokenizer.is_fitted_ = True
            return tokenizer

        elif path.suffix == '.npz':
            # Load config
            data = np.load(path, allow_pickle=True)

            tokenizer = cls(
                cutout_size=int(data['cutout_size']),
                bands=list(data['bands']),
                latent_dim=int(data['latent_dim']),
                codebook_size=int(data['codebook_size']),
                autoencoder_path=str(data.get('autoencoder_path', '')),
                codebook_path=str(data.get('codebook_path', '')),
                device=device,
            )

            # Load codebook if present
            if 'codebook' in data.files:
                tokenizer.codebook_ = data['codebook']

            # Load autoencoder if path specified
            if tokenizer.autoencoder_path:
                autoencoder_path = Path(tokenizer.autoencoder_path)
                if autoencoder_path.exists():
                    tokenizer._load_autoencoder(autoencoder_path, device)
                else:
                    warnings.warn(f"Autoencoder not found at {autoencoder_path}")

            tokenizer.is_fitted_ = True
            print(f"Loaded image tokenizer from {path}")
            return tokenizer

        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    # ==================== Private Methods ====================

    def _load_autoencoder(self, path: Path, device: str) -> None:
        """Load autoencoder model from weights file."""
        try:
            # Import AutoEncoder from models.autoencoder
            # This will be created in Phase 5
            from ..models.autoencoder import AutoEncoder

            # Infer number of channels from bands
            n_channels = len(self.bands) if self.bands else 12

            # Create model architecture
            model = AutoEncoder(
                in_channels=n_channels,
                latent_dim=self.latent_dim,
            )

            # Load weights
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            self.autoencoder_ = model
            print(f"Loaded autoencoder from {path}")

        except ImportError:
            warnings.warn(
                "AutoEncoder class not found in models.autoencoder. "
                "Image tokenization will not work until Phase 5 is complete. "
                "You can still save/load tokenizer config."
            )
            self.autoencoder_ = None

    def _build_image(self, df: pl.DataFrame, idx: int) -> torch.Tensor:
        """
        Build multi-band image tensor from DataFrame row.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame containing cutout columns
        idx : int
            Row index

        Returns
        -------
        torch.Tensor
            Image tensor (C, H, W)
        """
        images = []
        for band in self.bands:
            col = f"splus_cut_{band}"
            if col not in df.columns:
                # Fill with zeros if band missing
                img = torch.zeros((self.cutout_size, self.cutout_size), dtype=torch.float32)
            else:
                flat = df[col][idx]
                img = self._to_image_tensor(flat)

                # Handle non-finite values
                img[~torch.isfinite(img)] = 0.0

            images.append(img)

        # Stack to (C, H, W)
        img_tensor = torch.stack(images, dim=0)

        # Apply normalization (simplified arcsinh)
        img_tensor = torch.arcsinh(img_tensor)

        return img_tensor

    def _to_image_tensor(self, flat) -> torch.Tensor:
        """Convert flattened array to square image tensor."""
        if flat is None:
            return torch.zeros((self.cutout_size, self.cutout_size), dtype=torch.float32)

        arr = torch.tensor(flat, dtype=torch.float32)

        if arr.ndim == 2:
            return arr

        if arr.ndim == 1:
            n = arr.numel()
            side = int(np.sqrt(n))
            if side * side != n:
                # Fallback: pad or crop to cutout_size
                return torch.zeros((self.cutout_size, self.cutout_size), dtype=torch.float32)
            return arr.view(side, side)

        return torch.zeros((self.cutout_size, self.cutout_size), dtype=torch.float32)

    def _quantize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Quantize continuous latents using codebook.

        Parameters
        ----------
        latents : torch.Tensor
            Latent representations (B, D, H, W)

        Returns
        -------
        torch.Tensor
            Token IDs (B, H, W)
        """
        if self.codebook_ is None:
            raise RuntimeError("No codebook available for quantization")

        B, D, H, W = latents.shape
        codebook = torch.from_numpy(self.codebook_).to(latents.device)  # (K, D)

        # Reshape latents to (B*H*W, D)
        latents_flat = latents.permute(0, 2, 3, 1).reshape(-1, D)

        # Compute distances to codebook
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x.c
        x2 = (latents_flat ** 2).sum(dim=1, keepdim=True)  # (B*H*W, 1)
        c2 = (codebook ** 2).sum(dim=1, keepdim=True).T  # (1, K)
        xc = latents_flat @ codebook.T  # (B*H*W, K)

        dist2 = x2 + c2 - 2 * xc  # (B*H*W, K)

        # Find nearest codebook entry
        tokens = torch.argmin(dist2, dim=1)  # (B*H*W,)

        # Reshape back to (B, H, W)
        tokens = tokens.view(B, H, W)

        return tokens

    @staticmethod
    def _choose_id_column(columns: List[str]) -> Optional[str]:
        """Choose ID column from available columns."""
        preferred = ["id", "gaia_source_id", "source_id"]
        for c in preferred:
            if c in columns:
                return c
        return None
