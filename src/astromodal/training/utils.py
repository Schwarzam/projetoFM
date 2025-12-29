"""
Training utilities for astromodal.

This module provides utility functions for training, including:
- Parameter counting
- Learning rate scheduling
- Training state management
"""

from __future__ import annotations

import math
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value

    Examples
    --------
    >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.

    Parameters
    ----------
    model : nn.Module
        PyTorch model

    Returns
    -------
    Tuple[int, int]
        (total_parameters, trainable_parameters)

    Examples
    --------
    >>> model = nn.Linear(10, 5)
    >>> total, trainable = count_parameters(model)
    >>> print(f"Total: {total:,}, Trainable: {trainable:,}")
    Total: 55, Trainable: 55
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def cosine_schedule_with_warmup(
    step: int,
    base_lr: float,
    warmup_steps: int,
    total_steps: int = 200_000,
    min_lr_ratio: float = 0.1,
) -> float:
    """
    Compute learning rate with linear warmup and cosine decay.

    Schedule:
    - Linear warmup from 0 to base_lr over warmup_steps
    - Cosine decay from base_lr to min_lr_ratio * base_lr

    Parameters
    ----------
    step : int
        Current training step (0-indexed)
    base_lr : float
        Base learning rate (peak after warmup)
    warmup_steps : int
        Number of warmup steps
    total_steps : int, default=200_000
        Total training steps for cosine decay
    min_lr_ratio : float, default=0.1
        Minimum learning rate as ratio of base_lr

    Returns
    -------
    float
        Learning rate for current step

    Examples
    --------
    >>> lr = cosine_schedule_with_warmup(100, base_lr=1e-3, warmup_steps=1000)
    >>> print(f"LR at step 100: {lr:.6f}")
    LR at step 100: 0.000100

    >>> lr = cosine_schedule_with_warmup(5000, base_lr=1e-3, warmup_steps=1000)
    >>> print(f"LR at step 5000: {lr:.6f}")
    LR at step 5000: 0.000998
    """
    if step < warmup_steps:
        # Linear warmup
        return base_lr * (step + 1) / max(1, warmup_steps)

    # Cosine decay
    t = step - warmup_steps
    T = total_steps
    frac = min(1.0, t / max(1, T))
    cos = 0.5 * (1 + math.cos(math.pi * frac))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cos)


def configure_cuda_performance() -> None:
    """
    Configure CUDA performance settings for H100/A100 GPUs.

    Enables:
    - TF32 for matmul operations
    - Flash attention and memory-efficient attention
    - High precision for float32 matmul

    Notes
    -----
    This function should be called before model creation.
    Failures are silently ignored for compatibility.

    Examples
    --------
    >>> configure_cuda_performance()
    >>> # Now create and train your model
    """
    # Enable TF32 for faster matmul on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Enable optimized attention backends
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    except Exception:
        pass


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Parameters
    ----------
    seconds : float
        Time in seconds

    Returns
    -------
    str
        Formatted time string (e.g., "1h 23m 45s")

    Examples
    --------
    >>> print(format_time(3665))
    1h 1m 5s
    >>> print(format_time(125))
    2m 5s
    >>> print(format_time(45))
    45s
    """
    if seconds < 60:
        return f"{int(seconds)}s"

    minutes = int(seconds // 60)
    secs = int(seconds % 60)

    if minutes < 60:
        return f"{minutes}m {secs}s"

    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs}s"


def log_gradient_stats(model: nn.Module) -> dict:
    """
    Compute gradient statistics for debugging.

    Parameters
    ----------
    model : nn.Module
        PyTorch model with computed gradients

    Returns
    -------
    dict
        Dictionary with gradient statistics:
        - grad_norm: Global gradient norm
        - grad_mean: Mean gradient magnitude
        - grad_max: Maximum gradient magnitude
        - grad_min: Minimum gradient magnitude

    Examples
    --------
    >>> model = nn.Linear(10, 5)
    >>> loss = model(torch.randn(2, 10)).sum()
    >>> loss.backward()
    >>> stats = log_gradient_stats(model)
    >>> print(f"Gradient norm: {stats['grad_norm']:.4f}")
    """
    total_norm = 0.0
    grads = []

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            grads.append(p.grad.data.abs().mean().item())

    total_norm = total_norm ** 0.5

    stats = {
        "grad_norm": total_norm,
    }

    if grads:
        stats["grad_mean"] = np.mean(grads)
        stats["grad_max"] = np.max(grads)
        stats["grad_min"] = np.min(grads)

    return stats
