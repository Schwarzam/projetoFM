"""
Model loading utilities for AION inference.
"""

from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn

from ..models import AIONTransformer


def load_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cuda",
) -> Dict:
    """
    Load a model checkpoint.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to checkpoint file (.pt)
    device : str
        Device to load model on ("cuda" or "cpu")

    Returns
    -------
    dict
        Checkpoint dictionary with keys:
        - model: model state dict
        - config: model configuration
        - epoch: training epoch
        - global_step: training step
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    return checkpoint


def load_aion_model(
    checkpoint_path: str | Path,
    device: str = "cuda",
    eval_mode: bool = True,
) -> tuple[AIONTransformer, Dict]:
    """
    Load a trained AION model from checkpoint.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to checkpoint file
    device : str
        Device to load model on
    eval_mode : bool
        If True, set model to eval mode

    Returns
    -------
    model : AIONTransformer
        Loaded model
    config : dict
        Model configuration

    Examples
    --------
    >>> model, config = load_aion_model("runs_aion_masked/ckpt_step_00005000.pt")
    >>> print(f"Loaded model with {config['vocab_size']} vocabulary size")
    """
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, device)

    # Extract config
    if "config" not in checkpoint:
        raise ValueError("Checkpoint missing 'config' key")

    config = checkpoint["config"]

    # Create model
    print("Creating model from config...")
    model = AIONTransformer(
        vocab_size=config["vocab_size"],
        n_types=config["n_types"],
        d_model=config["d_model"],
        n_enc_layers=config["n_enc_layers"],
        n_dec_layers=config["n_dec_layers"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        dropout=config.get("dropout", 0.1),
        max_len=config.get("max_len", 2048),
    )

    # Load weights
    print("Loading model weights...")
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)

    if eval_mode:
        model.eval()

    # Print info
    from ..models import count_parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Model loaded successfully:")
    print(f"  - Parameters: {total_params/1e6:.1f}M")
    print(f"  - Training step: {checkpoint.get('global_step', 'unknown')}")
    print(f"  - Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  - Device: {device}")

    return model, config
