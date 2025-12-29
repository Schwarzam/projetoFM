"""
Vocabulary specification and token offset management.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class VocabSpec:
    """
    Vocabulary specification for multi-modal tokenization.

    Attributes
    ----------
    pad_id : int
        Padding token ID
    bos_id : int
        Beginning of sequence token ID
    eos_id : int
        End of sequence token ID
    sep_id : int
        Separator token ID
    miss_id : int
        Missing value token ID
    base_image : int
        Base offset for image tokens
    base_scalar : int
        Base offset for scalar tokens
    base_spectrum : Dict[str, int]
        Base offsets for each spectrum group
    vocab_size : int
        Total vocabulary size
    """
    pad_id: int
    bos_id: int
    eos_id: int
    sep_id: int
    miss_id: int
    base_image: int
    base_scalar: int
    base_spectrum: Dict[str, int]
    vocab_size: int

    def image_token_to_id(self, token: int) -> int:
        """Convert image token to vocabulary ID."""
        return self.base_image + token

    def scalar_token_to_id(self, token: int) -> int:
        """Convert scalar token to vocabulary ID."""
        return self.base_scalar + token

    def spectrum_token_to_id(self, group: str, token: int) -> int:
        """Convert spectrum token to vocabulary ID."""
        return self.base_spectrum[group] + token


def build_vocab_spec(
    v_image: int = 2048,
    v_scalar: int = 1024,
    v_spectra: int = 2048,
    spectrum_groups: Optional[List[str]] = None,
) -> VocabSpec:
    """
    Build vocabulary specification with token offsets.

    Parameters
    ----------
    v_image : int, default=2048
        Image vocabulary size
    v_scalar : int, default=1024
        Scalar vocabulary size
    v_spectra : int, default=2048
        Spectrum vocabulary size (per group)
    spectrum_groups : Optional[List[str]], default=None
        List of spectrum group names.
        If None, uses default: ["gaiaxp_bp", "gaiaxp_rp", "desi_b", "desi_r", "desi_z"]

    Returns
    -------
    VocabSpec
        Vocabulary specification with computed offsets
    """
    if spectrum_groups is None:
        spectrum_groups = ["gaiaxp_bp", "gaiaxp_rp", "desi_b", "desi_r", "desi_z"]

    # Special tokens
    PAD, BOS, EOS, SEP, MISS = 0, 1, 2, 3, 4
    next_id = 5

    # Image tokens
    base_image = next_id
    next_id += v_image

    # Scalar tokens
    base_scalar = next_id
    next_id += v_scalar

    # Spectrum tokens (per group)
    base_spectrum = {}
    for g in spectrum_groups:
        base_spectrum[g] = next_id
        next_id += v_spectra

    return VocabSpec(
        pad_id=PAD,
        bos_id=BOS,
        eos_id=EOS,
        sep_id=SEP,
        miss_id=MISS,
        base_image=base_image,
        base_scalar=base_scalar,
        base_spectrum=base_spectrum,
        vocab_size=next_id,
    )
