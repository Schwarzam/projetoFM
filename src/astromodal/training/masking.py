"""
4M-Style Masking Strategy for AION

Implements the masking strategy from AION paper Section 5.2:
- Input token budget: randomly sample tokens to observe
- Output token budget: randomly sample tokens to predict
- Beta distribution sampling for output tokens (skewed toward zero)

References:
    AION-1 paper Section 5.2: Modality Masking Strategy
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MaskedBatch:
    """
    Container for a masked batch following AION/4M strategy.

    Attributes
    ----------
    obs_tokens : torch.Tensor
        Observed token IDs, shape (B, L_obs)
    obs_types : torch.Tensor
        Observed token types, shape (B, L_obs)
    obs_positions : torch.Tensor
        Observed token positions in original sequence, shape (B, L_obs)
    obs_mask : torch.Tensor
        Observed token validity mask, shape (B, L_obs)
    tgt_tokens : torch.Tensor
        Target token IDs (ground truth), shape (B, L_tgt)
    tgt_types : torch.Tensor
        Target token types, shape (B, L_tgt)
    tgt_positions : torch.Tensor
        Target token positions in original sequence, shape (B, L_tgt)
    tgt_mask : torch.Tensor
        Target token validity mask, shape (B, L_tgt)
    """
    obs_tokens: torch.Tensor
    obs_types: torch.Tensor
    obs_positions: torch.Tensor
    obs_mask: torch.Tensor
    tgt_tokens: torch.Tensor
    tgt_types: torch.Tensor
    tgt_positions: torch.Tensor
    tgt_mask: torch.Tensor


class FourMMaskingStrategy:
    """
    4M-style masking strategy for multimodal masked modeling.

    From AION paper Section 5.2:
    1. Input Token Budget: Randomly sample B tokens as observed
    2. Output Token Budget: Sample remaining tokens as targets with Beta distribution

    The Beta distribution is skewed toward zero, which aligns with iterative
    sampling patterns (like MaskGIT) used during inference.

    Parameters
    ----------
    input_budget : int
        Number of tokens to use as observed (encoder input)
    output_budget : int
        Maximum number of tokens to predict (decoder output)
    beta_alpha : float
        Alpha parameter for Beta distribution (controls skew toward 0)
    beta_beta : float
        Beta parameter for Beta distribution
    pad_id : int
        Padding token ID to ignore during masking
    """

    def __init__(
        self,
        input_budget: int = 256,
        output_budget: int = 128,
        beta_alpha: float = 0.5,
        beta_beta: float = 1.0,
        pad_id: int = 0,
    ):
        self.input_budget = input_budget
        self.output_budget = output_budget
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.pad_id = pad_id

    def _sample_beta(self, size: int) -> float:
        """
        Sample from Beta distribution, skewed toward zero.

        Returns a value in [0, 1] controlling what fraction of remaining
        tokens to use as targets.
        """
        return np.random.beta(self.beta_alpha, self.beta_beta)

    def _get_valid_mask(self, tokens: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Get mask of valid (non-padding) positions.

        Parameters
        ----------
        tokens : torch.Tensor
            Token IDs, shape (B, L)
        attn_mask : torch.Tensor
            Attention mask, shape (B, L)

        Returns
        -------
        torch.Tensor
            Boolean mask, shape (B, L), True = valid position
        """
        valid = (tokens != self.pad_id) & attn_mask
        return valid

    def _sample_indices(
        self,
        valid_mask: torch.Tensor,
        n_sample: int,
        exclude_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample n_sample random indices from valid positions.

        Parameters
        ----------
        valid_mask : torch.Tensor
            Boolean mask of valid positions, shape (L,)
        n_sample : int
            Number of indices to sample
        exclude_indices : Optional[torch.Tensor]
            Indices to exclude from sampling

        Returns
        -------
        torch.Tensor
            Sampled indices, shape (n_sample,)
        """
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)

        # Exclude already-selected indices
        if exclude_indices is not None and exclude_indices.numel() > 0:
            mask = torch.ones(valid_indices.numel(), dtype=torch.bool, device=valid_indices.device)
            for idx in exclude_indices:
                mask &= (valid_indices != idx)
            valid_indices = valid_indices[mask]

        # Sample randomly
        if valid_indices.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=valid_mask.device)

        n_sample = min(n_sample, valid_indices.numel())
        perm = torch.randperm(valid_indices.numel(), device=valid_indices.device)[:n_sample]
        return valid_indices[perm]

    def apply(
        self,
        tokens: torch.Tensor,
        types: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> MaskedBatch:
        """
        Apply 4M-style masking to a batch.

        Parameters
        ----------
        tokens : torch.Tensor
            Token IDs, shape (B, L)
        types : torch.Tensor
            Token type IDs, shape (B, L)
        attn_mask : torch.Tensor
            Attention mask, shape (B, L)

        Returns
        -------
        MaskedBatch
            Masked batch with observed and target tokens
        """
        B, L = tokens.shape
        device = tokens.device

        # Storage for batch
        obs_tokens_list = []
        obs_types_list = []
        obs_positions_list = []
        obs_mask_list = []
        tgt_tokens_list = []
        tgt_types_list = []
        tgt_positions_list = []
        tgt_mask_list = []

        max_obs_len = 0
        max_tgt_len = 0

        # Process each sample in batch
        for i in range(B):
            # Get valid positions (non-padding)
            valid = self._get_valid_mask(tokens[i:i+1], attn_mask[i:i+1]).squeeze(0)
            n_valid = valid.sum().item()

            if n_valid == 0:
                # Empty sequence - create dummy entries
                obs_tokens_list.append(torch.tensor([self.pad_id], device=device))
                obs_types_list.append(torch.tensor([0], device=device))
                obs_positions_list.append(torch.tensor([0], device=device))
                obs_mask_list.append(torch.tensor([False], device=device))
                tgt_tokens_list.append(torch.tensor([self.pad_id], device=device))
                tgt_types_list.append(torch.tensor([0], device=device))
                tgt_positions_list.append(torch.tensor([0], device=device))
                tgt_mask_list.append(torch.tensor([False], device=device))
                max_obs_len = max(max_obs_len, 1)
                max_tgt_len = max(max_tgt_len, 1)
                continue

            # 1. Sample input budget (observed tokens)
            n_obs = min(self.input_budget, n_valid)
            obs_indices = self._sample_indices(valid, n_obs)

            # 2. Sample output budget (target tokens) from remaining
            n_remaining = n_valid - n_obs
            if n_remaining > 0:
                # Sample fraction from Beta distribution
                beta_sample = self._sample_beta(1)
                n_tgt = int(beta_sample * min(self.output_budget, n_remaining))
                n_tgt = max(1, n_tgt)  # At least 1 target token
                n_tgt = min(n_tgt, n_remaining)

                tgt_indices = self._sample_indices(valid, n_tgt, exclude_indices=obs_indices)
            else:
                # No remaining tokens - use some observed as targets (fallback)
                n_tgt = min(1, n_obs)
                tgt_indices = obs_indices[:n_tgt]

            # Extract observed tokens
            obs_tok = tokens[i, obs_indices]
            obs_typ = types[i, obs_indices]
            obs_pos = obs_indices
            obs_msk = torch.ones(obs_indices.numel(), dtype=torch.bool, device=device)

            # Extract target tokens
            tgt_tok = tokens[i, tgt_indices]
            tgt_typ = types[i, tgt_indices]
            tgt_pos = tgt_indices
            tgt_msk = torch.ones(tgt_indices.numel(), dtype=torch.bool, device=device)

            obs_tokens_list.append(obs_tok)
            obs_types_list.append(obs_typ)
            obs_positions_list.append(obs_pos)
            obs_mask_list.append(obs_msk)
            tgt_tokens_list.append(tgt_tok)
            tgt_types_list.append(tgt_typ)
            tgt_positions_list.append(tgt_pos)
            tgt_mask_list.append(tgt_msk)

            max_obs_len = max(max_obs_len, obs_indices.numel())
            max_tgt_len = max(max_tgt_len, tgt_indices.numel())

        # Pad to maximum lengths
        obs_tokens = torch.full((B, max_obs_len), self.pad_id, dtype=torch.long, device=device)
        obs_types = torch.zeros((B, max_obs_len), dtype=torch.long, device=device)
        obs_positions = torch.zeros((B, max_obs_len), dtype=torch.long, device=device)
        obs_mask = torch.zeros((B, max_obs_len), dtype=torch.bool, device=device)

        tgt_tokens = torch.full((B, max_tgt_len), self.pad_id, dtype=torch.long, device=device)
        tgt_types = torch.zeros((B, max_tgt_len), dtype=torch.long, device=device)
        tgt_positions = torch.zeros((B, max_tgt_len), dtype=torch.long, device=device)
        tgt_mask = torch.zeros((B, max_tgt_len), dtype=torch.bool, device=device)

        for i in range(B):
            n_obs = obs_tokens_list[i].numel()
            n_tgt = tgt_tokens_list[i].numel()

            obs_tokens[i, :n_obs] = obs_tokens_list[i]
            obs_types[i, :n_obs] = obs_types_list[i]
            obs_positions[i, :n_obs] = obs_positions_list[i]
            obs_mask[i, :n_obs] = obs_mask_list[i]

            tgt_tokens[i, :n_tgt] = tgt_tokens_list[i]
            tgt_types[i, :n_tgt] = tgt_types_list[i]
            tgt_positions[i, :n_tgt] = tgt_positions_list[i]
            tgt_mask[i, :n_tgt] = tgt_mask_list[i]

        return MaskedBatch(
            obs_tokens=obs_tokens,
            obs_types=obs_types,
            obs_positions=obs_positions,
            obs_mask=obs_mask,
            tgt_tokens=tgt_tokens,
            tgt_types=tgt_types,
            tgt_positions=tgt_positions,
            tgt_mask=tgt_mask,
        )
