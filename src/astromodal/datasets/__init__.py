"""
Dataset implementations for astromodal package.

This module provides PyTorch Dataset classes for loading and preprocessing
multi-modal astronomical data including images, scalars, and spectra.
"""

from .token_dataset import FieldIndex, MultiFieldTokenDataset, collate_sequences
from .image_dataset import SplusDataset

__all__ = [
    "FieldIndex",
    "MultiFieldTokenDataset",
    "SplusDataset",
    "collate_sequences",
]