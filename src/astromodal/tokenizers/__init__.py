"""
Tokenizers for astronomical multimodal data.

This module provides tokenizers for converting continuous astronomical data
into discrete tokens for transformer-based models:

- ScalarTokenizer: Quantile binning for catalog scalar features
- SpectrumTokenizer: K-means clustering for spectroscopic time-series
- ImageTokenizer: Autoencoder-based tokenization for image cutouts
"""

from .base import BaseTokenizer
from .scalar import ScalarTokenizer
from .spectrum import SpectrumTokenizer
from .image import ImageTokenizer

__all__ = [
    "BaseTokenizer",
    "ScalarTokenizer",
    "SpectrumTokenizer",
    "ImageTokenizer",
]
