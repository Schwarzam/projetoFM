"""
Inference utilities for AION model.

Provides functions for:
- Preparing observation tokens
- Decoding predictions
- Image reconstruction
- Parameter prediction
- Iterative sampling
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np

from ..core import VocabSpec


def prepare_observation(
    tokens: torch.Tensor,
    types: torch.Tensor,
    mask_positions: Optional[List[int]] = None,
    pad_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Prepare observation tokens for inference.

    Parameters
    ----------
    tokens : torch.Tensor
        Token IDs, shape (L,) or (B, L)
    types : torch.Tensor
        Token type IDs, shape (L,) or (B, L)
    mask_positions : List[int], optional
        Positions to mask (predict). If None, all positions are observed.
    pad_id : int
        Padding token ID

    Returns
    -------
    dict
        Dictionary with keys:
        - obs_tokens: observed token IDs
        - obs_types: observed token types
        - obs_positions: observed positions
        - obs_mask: observed validity mask
        - tgt_positions: target positions to predict
        - tgt_types: target types to predict
        - tgt_mask: target validity mask
    """
    # Ensure batch dimension
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
        types = types.unsqueeze(0)

    B, L = tokens.shape
    device = tokens.device

    # Get valid positions (non-padding)
    valid_mask = tokens != pad_id

    if mask_positions is None:
        # No masking - use all tokens as observations
        obs_indices = torch.arange(L, device=device)[valid_mask[0]]
        tgt_indices = torch.empty(0, dtype=torch.long, device=device)
    else:
        # Mask specified positions
        mask_positions = torch.tensor(mask_positions, device=device)
        all_positions = torch.arange(L, device=device)

        # Observed = valid & not masked
        obs_mask_bool = valid_mask[0].clone()
        obs_mask_bool[mask_positions] = False
        obs_indices = all_positions[obs_mask_bool]

        # Targets = masked & valid
        tgt_mask_bool = torch.zeros(L, dtype=torch.bool, device=device)
        tgt_mask_bool[mask_positions] = True
        tgt_mask_bool &= valid_mask[0]
        tgt_indices = all_positions[tgt_mask_bool]

    # Extract observed tokens
    obs_tokens = tokens[0, obs_indices].unsqueeze(0)
    obs_types = types[0, obs_indices].unsqueeze(0)
    obs_positions = obs_indices.unsqueeze(0)
    obs_mask = torch.ones(1, obs_indices.numel(), dtype=torch.bool, device=device)

    # Extract target info (no token values!)
    if tgt_indices.numel() > 0:
        tgt_types = types[0, tgt_indices].unsqueeze(0)
        tgt_positions = tgt_indices.unsqueeze(0)
        tgt_mask = torch.ones(1, tgt_indices.numel(), dtype=torch.bool, device=device)
    else:
        tgt_types = torch.empty(1, 0, dtype=torch.long, device=device)
        tgt_positions = torch.empty(1, 0, dtype=torch.long, device=device)
        tgt_mask = torch.empty(1, 0, dtype=torch.bool, device=device)

    return {
        "obs_tokens": obs_tokens,
        "obs_types": obs_types,
        "obs_positions": obs_positions,
        "obs_mask": obs_mask,
        "tgt_positions": tgt_positions,
        "tgt_types": tgt_types,
        "tgt_mask": tgt_mask,
    }


def decode_tokens(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Decode token predictions from logits.

    Parameters
    ----------
    logits : torch.Tensor
        Logits, shape (B, L, vocab_size) or (L, vocab_size)
    temperature : float
        Sampling temperature
    top_k : int, optional
        Top-k sampling
    top_p : float, optional
        Nucleus sampling threshold

    Returns
    -------
    torch.Tensor
        Predicted token IDs
    """
    # Apply temperature
    logits = logits / temperature

    # Top-k filtering
    if top_k is not None and top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")

    # Top-p (nucleus) filtering
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 0] = False

        # Scatter back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")

    # Sample
    probs = F.softmax(logits, dim=-1)
    tokens = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(probs.shape[:-1])

    return tokens


@torch.no_grad()
def reconstruct_images(
    model,
    obs_tokens: torch.Tensor,
    obs_types: torch.Tensor,
    obs_positions: torch.Tensor,
    obs_mask: torch.Tensor,
    tgt_positions: torch.Tensor,
    tgt_types: torch.Tensor,
    tgt_mask: torch.Tensor,
    vocab: VocabSpec,
    temperature: float = 1.0,
    num_iterations: int = 8,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Reconstruct masked image tokens using iterative refinement (ROAR-style).

    Parameters
    ----------
    model : AIONTransformer
        Trained model
    obs_tokens, obs_types, obs_positions, obs_mask : torch.Tensor
        Observed tokens and metadata
    tgt_positions, tgt_types, tgt_mask : torch.Tensor
        Target positions and types to predict
    vocab : VocabSpec
        Vocabulary specification
    temperature : float
        Sampling temperature
    num_iterations : int
        Number of iterative refinement steps

    Returns
    -------
    final_tokens : torch.Tensor
        Final predicted tokens at target positions
    history : List[torch.Tensor]
        Intermediate predictions at each iteration
    """
    device = obs_tokens.device
    history = []

    # Encode observed tokens once
    memory = model.encode(obs_tokens, obs_types, obs_positions, obs_mask)

    # Initialize with random tokens (or mask tokens)
    current_tokens = torch.full(
        (tgt_positions.size(0), tgt_positions.size(1)),
        vocab.miss_id,
        dtype=torch.long,
        device=device,
    )

    # Iterative refinement
    for iteration in range(num_iterations):
        # Decode current state
        logits = model.decode_step(
            memory=memory,
            memory_mask=obs_mask,
            tgt_types=tgt_types,
            tgt_positions=tgt_positions,
            tgt_mask=tgt_mask,
        )

        # Sample tokens
        predicted_tokens = decode_tokens(logits, temperature=temperature)

        # For ROAR-style: resample low-confidence tokens in later iterations
        # For simplicity, we update all tokens here
        current_tokens = predicted_tokens

        history.append(current_tokens.clone())

    return current_tokens, history


@torch.no_grad()
def predict_parameters(
    model,
    obs_tokens: torch.Tensor,
    obs_types: torch.Tensor,
    obs_positions: torch.Tensor,
    obs_mask: torch.Tensor,
    tgt_positions: torch.Tensor,
    tgt_types: torch.Tensor,
    tgt_mask: torch.Tensor,
    vocab: VocabSpec,
    scalar_bin_centers: Optional[np.ndarray] = None,
) -> Dict[str, torch.Tensor]:
    """
    Predict scalar parameter tokens.

    Parameters
    ----------
    model : AIONTransformer
        Trained model
    obs_tokens, obs_types, obs_positions, obs_mask : torch.Tensor
        Observed tokens and metadata
    tgt_positions, tgt_types, tgt_mask : torch.Tensor
        Target positions and types to predict
    vocab : VocabSpec
        Vocabulary specification
    scalar_bin_centers : np.ndarray, optional
        Bin centers for converting tokens back to values

    Returns
    -------
    dict
        Dictionary with:
        - predicted_tokens: predicted token IDs
        - probabilities: softmax probabilities
        - predicted_values: predicted parameter values (if bin_centers provided)
    """
    device = obs_tokens.device

    # Encode and decode
    memory = model.encode(obs_tokens, obs_types, obs_positions, obs_mask)
    logits = model.decode_step(
        memory=memory,
        memory_mask=obs_mask,
        tgt_types=tgt_types,
        tgt_positions=tgt_positions,
        tgt_mask=tgt_mask,
    )

    # Get probabilities
    probs = F.softmax(logits, dim=-1)

    # Get most likely tokens
    predicted_tokens = torch.argmax(logits, dim=-1)

    result = {
        "predicted_tokens": predicted_tokens,
        "probabilities": probs,
    }

    # Convert to parameter values if bin centers provided
    if scalar_bin_centers is not None:
        # Compute expected value from probability distribution
        scalar_bin_centers = torch.from_numpy(scalar_bin_centers).to(device)

        # Get scalar token range
        scalar_start = vocab.scalar_offset
        scalar_end = scalar_start + vocab.scalar_size

        # For each position, compute expected value
        # This is more robust than just using argmax
        scalar_logits = logits[..., scalar_start:scalar_end]
        scalar_probs = F.softmax(scalar_logits, dim=-1)

        # Expected value = sum(prob * bin_center)
        predicted_values = torch.sum(
            scalar_probs * scalar_bin_centers[None, None, :],
            dim=-1
        )

        result["predicted_values"] = predicted_values

    return result
