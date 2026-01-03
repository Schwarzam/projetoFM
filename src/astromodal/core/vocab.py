"""
Vocabulary specification and token offset management.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


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
    scalar_type_map : Dict[str, int]
        Mapping from scalar column name to type ID
    vocab_size : int
        Total vocabulary size
    n_types : int
        Total number of token types
    """
    pad_id: int
    bos_id: int
    eos_id: int
    sep_id: int
    miss_id: int
    base_image: int
    base_scalar: int
    base_spectrum: Dict[str, int]
    scalar_type_map: Dict[str, int]
    vocab_size: int
    n_types: int

    def image_token_to_id(self, token: int) -> int:
        """Convert image token to vocabulary ID."""
        return self.base_image + token

    def scalar_token_to_id(self, token: int) -> int:
        """Convert scalar token to vocabulary ID."""
        return self.base_scalar + token

    def spectrum_token_to_id(self, group: str, token: int) -> int:
        """Convert spectrum token to vocabulary ID."""
        return self.base_spectrum[group] + token

    def get_scalar_type_id(self, col_name: str) -> int:
        """Get type ID for a scalar column."""
        return self.scalar_type_map.get(col_name, 6)  # Default to 6 if not found


def build_vocab_spec(
    v_image: int = 2048,
    v_scalar: int = 1024,
    v_spectra: int = 2048,
    spectrum_groups: Optional[List[str]] = None,
    scalar_columns: Optional[List[str]] = None,
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
    scalar_columns : Optional[List[str]], default=None
        List of scalar column names for per-column type IDs.
        If None, loads from a standard scalar tokenizer file.

    Returns
    -------
    VocabSpec
        Vocabulary specification with computed offsets
    """
    if spectrum_groups is None:
        spectrum_groups = ["gaiaxp_bp", "gaiaxp_rp", "desi_b", "desi_r", "desi_z"]

    # Load scalar columns if not provided
    if scalar_columns is None:
        # Try to load from standard location
        try:
            from pathlib import Path
            scalar_tokenizer_path = Path("scalar_tokenizers/scalar_tokenizer_config.npz")
            if scalar_tokenizer_path.exists():
                data = np.load(scalar_tokenizer_path, allow_pickle=True)
                bin_edges_dict = data['bin_edges'].item()
                scalar_columns = sorted(bin_edges_dict.keys())
                print(f"[vocab] Loaded {len(scalar_columns)} scalar columns from {scalar_tokenizer_path}")
            else:
                print(f"[vocab] Warning: {scalar_tokenizer_path} not found, using empty scalar list")
                scalar_columns = []
        except Exception as e:
            print(f"[vocab] Warning: Could not load scalar columns: {e}")
            scalar_columns = []

    # Special tokens
    PAD, BOS, EOS, SEP, MISS = 0, 1, 2, 3, 4
    type_id = 5

    # Type IDs
    # 0: PAD, 1: BOS, 2: EOS, 3: SEP, 4: MISS
    # 5: IMAGE
    # 6 to 6+n_scalar_cols-1: SCALAR columns (one per column)
    # 6+n_scalar_cols to 6+n_scalar_cols+n_spectrum_groups-1: SPECTRUM groups

    IMAGE_TYPE = type_id
    type_id += 1

    # Scalar type IDs (one per column)
    scalar_type_map = {}
    for col_name in scalar_columns:
        scalar_type_map[col_name] = type_id
        type_id += 1

    # Spectrum type IDs (one per group)
    spectrum_type_map = {}
    for group_name in spectrum_groups:
        spectrum_type_map[group_name] = type_id
        type_id += 1

    n_types = type_id

    # Token IDs (vocabulary)
    next_id = 5  # After special tokens

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

    print(f"[vocab] Type IDs: IMAGE=5, SCALAR=6-{5+len(scalar_columns)}, "
          f"SPECTRUM={6+len(scalar_columns)}-{5+len(scalar_columns)+len(spectrum_groups)}")
    print(f"[vocab] Total types: {n_types}, Total vocab size: {next_id}")

    return VocabSpec(
        pad_id=PAD,
        bos_id=BOS,
        eos_id=EOS,
        sep_id=SEP,
        miss_id=MISS,
        base_image=base_image,
        base_scalar=base_scalar,
        base_spectrum=base_spectrum,
        scalar_type_map=scalar_type_map,
        vocab_size=next_id,
        n_types=n_types,
    )
