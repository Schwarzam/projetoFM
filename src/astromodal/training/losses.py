"""
Loss functions for astromodal training.

This module implements the "meaningful loss" approach that reconstructs
real-valued data instead of using standard next-token prediction.

The meaningful losses compute differentiable expected values:
- Image: Expected centroid (softmax over image vocab) vs true centroid
- Scalar: Expected bin center vs true bin center
- Spectra: Expected centroid vs true centroid

This allows the model to learn continuous representations while maintaining
a discrete tokenized architecture.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ..core.vocab import VocabSpec


def sample_positions(mask: torch.Tensor, k: int) -> torch.Tensor:
    """
    Sample up to k positions from a boolean mask.

    This function efficiently samples positions for auxiliary loss computation,
    which dramatically speeds up training by not computing loss on all positions.

    Parameters
    ----------
    mask : torch.Tensor
        Boolean mask of shape (B, L) indicating valid positions
    k : int
        Maximum number of positions to sample

    Returns
    -------
    torch.Tensor
        1D tensor of indices into flattened (B*L,) positions

    Examples
    --------
    >>> mask = torch.tensor([[True, False, True], [False, True, True]])
    >>> indices = sample_positions(mask, k=3)
    >>> print(indices.shape)
    torch.Size([3])
    """
    idx = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)
    if idx.numel() == 0:
        return idx
    if idx.numel() <= k:
        return idx
    # Uniform sample without replacement
    perm = torch.randperm(idx.numel(), device=idx.device)[:k]
    return idx[perm]


def image_reconstruction_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab: VocabSpec,
    codebook: torch.Tensor,
    sample_k: int,
) -> Tuple[torch.Tensor, int]:
    """
    Compute differentiable image reconstruction loss.

    Uses expected centroid from softmax distribution over image vocabulary.

    Loss = MSE(softmax(logits) @ codebook, true_centroid)

    Parameters
    ----------
    logits : torch.Tensor
        Model logits of shape (B, L, V)
    targets : torch.Tensor
        Target token IDs of shape (B, L)
    vocab : VocabSpec
        Vocabulary specification with token offsets
    codebook : torch.Tensor
        Image codebook of shape (V_IMAGE, D_image)
    sample_k : int
        Maximum number of positions to sample for loss

    Returns
    -------
    Tuple[torch.Tensor, int]
        (loss, num_positions_used)

    Notes
    -----
    Special tokens (PAD, BOS, EOS, SEP, MISS) are excluded from loss.
    """
    device = logits.device
    B, L, V = logits.shape

    # Filter out special tokens
    specials = torch.tensor(
        [vocab.pad_id, vocab.bos_id, vocab.eos_id, vocab.sep_id, vocab.miss_id],
        device=device,
        dtype=torch.long,
    )
    is_special = (targets[..., None] == specials[None, None, :]).any(dim=-1)

    # Identify image token positions
    img_lo = vocab.base_image
    v_image = codebook.shape[0]
    img_hi = vocab.base_image + v_image
    mask_img = (~is_special) & (targets >= img_lo) & (targets < img_hi)

    # Sample positions
    idx = sample_positions(mask_img, sample_k)
    if idx.numel() == 0:
        return torch.tensor(0.0, device=device), 0

    # Extract target tokens and corresponding logits
    targets_flat = targets.reshape(-1)[idx]
    logits_img = logits.reshape(-1, V)[:, img_lo:img_hi][idx]  # (N, V_IMAGE)

    # Compute expected centroid
    probs = torch.softmax(logits_img.float(), dim=-1).to(codebook.dtype)  # (N, V_IMAGE)
    pred = probs @ codebook  # (N, D_img)

    # Get true centroid
    true_idx = (targets_flat - img_lo).clamp(0, v_image - 1)
    true = codebook[true_idx]  # (N, D_img)

    # MSE loss
    loss = F.mse_loss(pred, true)
    return loss, idx.numel()


def scalar_reconstruction_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab: VocabSpec,
    bin_centers: torch.Tensor,
    sample_k: int,
) -> Tuple[torch.Tensor, int]:
    """
    Compute differentiable scalar reconstruction loss.

    Uses expected bin center from softmax distribution over scalar vocabulary.

    Loss = MSE(sum(softmax(logits) * bin_centers), true_bin_center)

    Parameters
    ----------
    logits : torch.Tensor
        Model logits of shape (B, L, V)
    targets : torch.Tensor
        Target token IDs of shape (B, L)
    vocab : VocabSpec
        Vocabulary specification with token offsets
    bin_centers : torch.Tensor
        Scalar bin centers of shape (V_SCALAR,)
    sample_k : int
        Maximum number of positions to sample for loss

    Returns
    -------
    Tuple[torch.Tensor, int]
        (loss, num_positions_used)

    Notes
    -----
    Special tokens (PAD, BOS, EOS, SEP, MISS) are excluded from loss.
    """
    device = logits.device
    B, L, V = logits.shape

    # Filter out special tokens
    specials = torch.tensor(
        [vocab.pad_id, vocab.bos_id, vocab.eos_id, vocab.sep_id, vocab.miss_id],
        device=device,
        dtype=torch.long,
    )
    is_special = (targets[..., None] == specials[None, None, :]).any(dim=-1)

    # Identify scalar token positions
    sc_lo = vocab.base_scalar
    v_scalar = bin_centers.shape[0]
    sc_hi = vocab.base_scalar + v_scalar
    mask_sca = (~is_special) & (targets >= sc_lo) & (targets < sc_hi)

    # Sample positions
    idx = sample_positions(mask_sca, sample_k)
    if idx.numel() == 0:
        return torch.tensor(0.0, device=device), 0

    # Extract target tokens and corresponding logits
    targets_flat = targets.reshape(-1)[idx]
    logits_sca = logits.reshape(-1, V)[:, sc_lo:sc_hi][idx]  # (N, V_SCALAR)

    # Compute expected bin center
    probs = torch.softmax(logits_sca.float(), dim=-1).to(bin_centers.dtype)  # (N, V_SCALAR)
    pred = (probs * bin_centers.unsqueeze(0)).sum(dim=-1)  # (N,)

    # Get true bin center
    true_idx = (targets_flat - sc_lo).clamp(0, v_scalar - 1)
    true = bin_centers[true_idx]  # (N,)

    # MSE loss
    loss = F.mse_loss(pred, true)
    return loss, idx.numel()


def spectrum_reconstruction_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab: VocabSpec,
    codebook: torch.Tensor,
    sample_k: int,
) -> Tuple[torch.Tensor, int]:
    """
    Compute differentiable spectrum reconstruction loss.

    Uses expected centroid from softmax distribution over spectrum vocabulary.
    Handles multiple spectrum groups (e.g., gaiaxp_bp, gaiaxp_rp, desi_b, desi_r, desi_z).

    Loss = MSE(softmax(logits) @ codebook, true_centroid)

    Parameters
    ----------
    logits : torch.Tensor
        Model logits of shape (B, L, V)
    targets : torch.Tensor
        Target token IDs of shape (B, L)
    vocab : VocabSpec
        Vocabulary specification with token offsets
    codebook : torch.Tensor
        Spectrum codebook of shape (V_SPECTRA, D_spec)
    sample_k : int
        Maximum number of positions to sample for loss (split across groups)

    Returns
    -------
    Tuple[torch.Tensor, int]
        (loss, num_positions_used)

    Notes
    -----
    - Special tokens (PAD, BOS, EOS, SEP, MISS) are excluded from loss
    - Sample budget is split evenly across spectrum groups
    - Loss is averaged across groups
    """
    device = logits.device
    B, L, V = logits.shape

    # Filter out special tokens
    specials = torch.tensor(
        [vocab.pad_id, vocab.bos_id, vocab.eos_id, vocab.sep_id, vocab.miss_id],
        device=device,
        dtype=torch.long,
    )
    is_special = (targets[..., None] == specials[None, None, :]).any(dim=-1)

    # Process each spectrum group
    loss_total = torch.tensor(0.0, device=device)
    n_used = 0
    n_positions = 0
    v_spectrum = codebook.shape[0]

    k_per_group = max(1, sample_k // max(1, len(vocab.base_spectrum)))

    for group_name, base in vocab.base_spectrum.items():
        sp_lo = base
        sp_hi = base + v_spectrum
        mask_sp = (~is_special) & (targets >= sp_lo) & (targets < sp_hi)

        # Sample positions for this group
        idx = sample_positions(mask_sp, k_per_group)
        if idx.numel() == 0:
            continue

        # Extract target tokens and corresponding logits
        targets_flat = targets.reshape(-1)[idx]
        logits_sp = logits.reshape(-1, V)[:, sp_lo:sp_hi][idx]  # (N, V_SPECTRA)

        # Compute expected centroid
        probs = torch.softmax(logits_sp.float(), dim=-1).to(codebook.dtype)  # (N, V_SPECTRA)
        pred = probs @ codebook  # (N, D_spec)

        # Get true centroid
        true_idx = (targets_flat - sp_lo).clamp(0, v_spectrum - 1)
        true = codebook[true_idx]  # (N, D_spec)

        # Accumulate MSE loss
        loss_g = F.mse_loss(pred, true)
        loss_total = loss_total + loss_g
        n_used += 1
        n_positions += idx.numel()

    # Average across groups
    if n_used > 0:
        loss_total = loss_total / n_used

    return loss_total, n_positions


def meaningful_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab: VocabSpec,
    image_codebook: torch.Tensor,
    spectrum_codebook: torch.Tensor,
    scalar_centers: torch.Tensor,
    config: dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute combined meaningful reconstruction loss.

    This is the main loss function that combines image, scalar, and spectrum
    reconstruction losses with configurable weights.

    Total loss = w_img * L_img + w_sca * L_sca + w_spec * L_spec

    Parameters
    ----------
    logits : torch.Tensor
        Model logits of shape (B, L, V)
    targets : torch.Tensor
        Target token IDs of shape (B, L)
    vocab : VocabSpec
        Vocabulary specification with token offsets
    image_codebook : torch.Tensor
        Image codebook of shape (V_IMAGE, D_img)
    spectrum_codebook : torch.Tensor
        Spectrum codebook of shape (V_SPECTRA, D_spec)
    scalar_centers : torch.Tensor
        Scalar bin centers of shape (V_SCALAR,)
    config : dict
        Configuration dict with keys:
        - sample_k: Max positions to sample per modality
        - weight_image: Weight for image loss
        - weight_scalar: Weight for scalar loss
        - weight_spectrum: Weight for spectrum loss

    Returns
    -------
    Tuple[torch.Tensor, Dict[str, float]]
        (total_loss, stats_dict)

    Examples
    --------
    >>> config = {
    ...     'sample_k': 4096,
    ...     'weight_image': 0.1,
    ...     'weight_scalar': 0.1,
    ...     'weight_spectrum': 0.1,
    ... }
    >>> loss, stats = meaningful_loss(logits, targets, vocab, img_cb, spec_cb, sca_centers, config)
    >>> print(f"Total loss: {loss.item():.4f}")
    >>> print(f"Image loss: {stats['loss_img']:.4f}")
    """
    sample_k = config.get("sample_k", 4096)
    w_img = config.get("weight_image", 0.1)
    w_sca = config.get("weight_scalar", 0.1)
    w_spec = config.get("weight_spectrum", 0.1)

    # Compute individual losses
    loss_img, n_img = image_reconstruction_loss(logits, targets, vocab, image_codebook, sample_k)
    loss_sca, n_sca = scalar_reconstruction_loss(logits, targets, vocab, scalar_centers, sample_k)
    loss_spec, n_spec = spectrum_reconstruction_loss(logits, targets, vocab, spectrum_codebook, sample_k)

    # Combine with weights
    total_loss = w_img * loss_img + w_sca * loss_sca + w_spec * loss_spec

    # Statistics for logging
    stats = {
        "loss_img": float(loss_img.detach().cpu().item()),
        "loss_sca": float(loss_sca.detach().cpu().item()),
        "loss_spec": float(loss_spec.detach().cpu().item()),
        "n_img": n_img,
        "n_sca": n_sca,
        "n_spec": n_spec,
    }

    return total_loss, stats


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """
    Compute standard cross-entropy loss for next-token prediction.

    Parameters
    ----------
    logits : torch.Tensor
        Model logits of shape (B, L, V)
    targets : torch.Tensor
        Target token IDs of shape (B, L)
    pad_id : int
        Padding token ID to ignore in loss

    Returns
    -------
    torch.Tensor
        Cross-entropy loss (scalar)

    Examples
    --------
    >>> logits = torch.randn(2, 10, 100)
    >>> targets = torch.randint(0, 100, (2, 10))
    >>> loss = cross_entropy_loss(logits, targets, pad_id=0)
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_id,
    )


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab: VocabSpec,
    image_codebook: torch.Tensor,
    spectrum_codebook: torch.Tensor,
    scalar_centers: torch.Tensor,
    config: dict,
    use_ce: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute combined loss with both CE and meaningful reconstruction.

    Total loss = CE + meaningful_loss

    This combines the discrete token prediction objective with the continuous
    reconstruction objectives for best of both worlds.

    Parameters
    ----------
    logits : torch.Tensor
        Model logits of shape (B, L, V)
    targets : torch.Tensor
        Target token IDs of shape (B, L)
    vocab : VocabSpec
        Vocabulary specification
    image_codebook : torch.Tensor
        Image codebook
    spectrum_codebook : torch.Tensor
        Spectrum codebook
    scalar_centers : torch.Tensor
        Scalar bin centers
    config : dict
        Loss configuration
    use_ce : bool, default=True
        Whether to include cross-entropy loss

    Returns
    -------
    Tuple[torch.Tensor, Dict[str, float]]
        (total_loss, stats_dict)
    """
    stats = {}

    # Cross-entropy loss
    if use_ce:
        loss_ce = cross_entropy_loss(logits, targets, vocab.pad_id)
        stats["loss_ce"] = float(loss_ce.detach().cpu().item())
        total_loss = loss_ce
    else:
        total_loss = torch.tensor(0.0, device=logits.device)

    # Meaningful reconstruction loss
    if config.get("enabled", True):
        loss_meaningful, meaningful_stats = meaningful_loss(
            logits, targets, vocab, image_codebook, spectrum_codebook, scalar_centers, config
        )
        total_loss = total_loss + loss_meaningful
        stats.update(meaningful_stats)

    return total_loss, stats
