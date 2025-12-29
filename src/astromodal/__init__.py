"""
Astromodal: Foundation Model for Astronomy

A multi-modal transformer-based foundation model for astronomical data,
handling images, scalars, and spectra across multiple surveys.
"""

__version__ = "0.1.0"

from . import config
from . import core
from . import tokenizers
from . import models
from . import datasets
from . import training
from . import data

__all__ = [
    "config",
    "core",
    "tokenizers",
    "models",
    "datasets",
    "training",
    "data",
]
