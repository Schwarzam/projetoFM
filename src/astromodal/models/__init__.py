"""
Neural network models for astromodal.

This module provides model implementations for multimodal astronomical data:
- AutoEncoder: Spatial autoencoder for image tokenization
- CausalTransformerLM: Transformer language model for sequence generation

Models
------
AutoEncoder
    AION-inspired autoencoder with spatial latents for astronomical images
ResBlock
    Residual block used in autoencoder
Encoder
    Encoder component of autoencoder
Decoder
    Decoder component of autoencoder
CausalTransformerLM
    Causal transformer for autoregressive sequence modeling

Examples
--------
>>> from astromodal.models import AutoEncoder, CausalTransformerLM
>>>
>>> # Create autoencoder for 12-band images
>>> autoencoder = AutoEncoder(in_channels=12, latent_dim=2)
>>>
>>> # Create transformer model
>>> transformer = CausalTransformerLM(
...     vocab_size=10000,
...     n_types=15,
...     d_model=768,
...     n_layers=12,
...     n_heads=12,
...     d_ff=3072,
...     dropout=0.1,
...     max_len=2048
... )
"""

from .autoencoder import AutoEncoder, ResBlock, Encoder, Decoder
from .transformer import CausalTransformerLM

__all__ = [
    "AutoEncoder",
    "ResBlock",
    "Encoder",
    "Decoder",
    "CausalTransformerLM",
]
