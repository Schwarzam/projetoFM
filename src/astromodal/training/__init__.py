"""
Training infrastructure for astromodal.

This module provides everything needed for training astromodal models:
- Loss functions (meaningful reconstruction + CE)
- Training utilities (LR scheduling, parameter counting, etc.)
- Trainer class (complete training loop)
"""

from .losses import (
    sample_positions,
    image_reconstruction_loss,
    scalar_reconstruction_loss,
    spectrum_reconstruction_loss,
    meaningful_loss,
    cross_entropy_loss,
    combined_loss,
)

from .utils import (
    set_seed,
    count_parameters,
    cosine_schedule_with_warmup,
    configure_cuda_performance,
    format_time,
    log_gradient_stats,
)

from .trainer import Trainer

__all__ = [
    # Loss functions
    "sample_positions",
    "image_reconstruction_loss",
    "scalar_reconstruction_loss",
    "spectrum_reconstruction_loss",
    "meaningful_loss",
    "cross_entropy_loss",
    "combined_loss",
    # Utilities
    "set_seed",
    "count_parameters",
    "cosine_schedule_with_warmup",
    "configure_cuda_performance",
    "format_time",
    "log_gradient_stats",
    # Trainer
    "Trainer",
]
