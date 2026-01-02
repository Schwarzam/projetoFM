"""
Inference utilities for AION model.

This module provides utilities for loading trained models and performing inference:
- Model loading from checkpoints
- Image reconstruction/inpainting
- Parameter prediction
- Iterative sampling (ROAR-style)
"""

from .model_loader import load_aion_model, load_checkpoint
from .inference_utils import (
    prepare_observation,
    decode_tokens,
    reconstruct_images,
    predict_parameters,
)

__all__ = [
    "load_aion_model",
    "load_checkpoint",
    "prepare_observation",
    "decode_tokens",
    "reconstruct_images",
    "predict_parameters",
]
