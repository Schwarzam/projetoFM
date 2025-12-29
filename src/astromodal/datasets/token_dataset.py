"""
Token-based dataset implementations for multi-field astronomical data.

This module provides efficient loading and indexing of tokenized astronomical data
across multiple fields, with support for images, scalars, and spectra.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from astromodal.core import VocabSpec, norm_splus_id, norm_gaia_id


class FieldIndex:
    """
    Per-field data indexing and lookup for multi-modal astronomical data.

    This class manages access to tokenized data for a single field, handling:
    - Image tokens (indexed by Gaia source ID)
    - Scalar tokens (indexed by S-PLUS object ID)
    - Spectrum tokens in multiple groups (indexed by S-PLUS object ID)

    Parameters
    ----------
    field : str
        Field identifier (e.g., "HYDRA-0011")
    datacube_template : str
        Path template for datacube parquet files with {field} placeholder
    image_tokens_template : str
        Path template for image token files with {field} placeholder
    scalar_tokens_template : str
        Path template for scalar token files with {field} placeholder
    spectrum_tokens_template : str
        Path template for spectrum token files with {field} placeholder
    spectrum_groups : Dict[str, Tuple[str, str]]
        Mapping of spectrum group names to (flat_key, indptr_key) pairs
    mag_col : str, default="mag_pstotal_r"
        Magnitude column for filtering
    magerr_col : str, default="err_mag_pstotal_r"
        Magnitude error column for filtering
    mag_min : float, default=14.0
        Minimum magnitude filter
    mag_max : float, default=22.0
        Maximum magnitude filter
    magerr_max : float, default=2.0
        Maximum magnitude error filter

    Attributes
    ----------
    field : str
        Field identifier
    df : Optional[pd.DataFrame]
        Datacube DataFrame with filtered objects
    img_tokens_flat : Optional[np.ndarray]
        Flattened image tokens array
    img_id_to_row : Dict[int, int]
        Mapping from Gaia ID to image token row
    scalar_tokens : Optional[np.ndarray]
        Scalar tokens array
    splus_id_to_scalar_row : Dict[str, int]
        Mapping from S-PLUS ID to scalar token row
    spec_npz : Optional[np.lib.npyio.NpzFile]
        Opened spectrum tokens NPZ file
    splus_id_to_spec_row : Dict[str, int]
        Mapping from S-PLUS ID to spectrum token row
    spec_groups_present : Dict[str, Tuple[str, str]]
        Spectrum groups present in this field

    Examples
    --------
    >>> field_idx = FieldIndex(
    ...     field="HYDRA-0011",
    ...     datacube_template="/path/datacube_{field}.parquet",
    ...     image_tokens_template="/path/image_{field}_tokens.npz",
    ...     scalar_tokens_template="/path/scalar_{field}_tokens.npz",
    ...     spectrum_tokens_template="/path/spectrum_{field}_tokens.npz",
    ...     spectrum_groups={"gaiaxp_bp": ("tokens_gaiaxp_bp_flat", "tokens_gaiaxp_bp_indptr")},
    ... )
    >>> len(field_idx)
    12345
    >>> sample = field_idx.get_row(0)
    >>> sample.keys()
    dict_keys(['image_tokens', 'scalar_tokens', 'spectrum_tokens'])
    """

    def __init__(
        self,
        field: str,
        datacube_template: str,
        image_tokens_template: str,
        scalar_tokens_template: str,
        spectrum_tokens_template: str,
        spectrum_groups: Dict[str, Tuple[str, str]],
        mag_col: str = "mag_pstotal_r",
        magerr_col: str = "err_mag_pstotal_r",
        mag_min: float = 14.0,
        mag_max: float = 22.0,
        magerr_max: float = 2.0,
    ):
        self.field = field
        self.datacube_template = datacube_template
        self.image_tokens_template = image_tokens_template
        self.scalar_tokens_template = scalar_tokens_template
        self.spectrum_tokens_template = spectrum_tokens_template
        self.spectrum_groups = spectrum_groups
        self.mag_col = mag_col
        self.magerr_col = magerr_col
        self.mag_min = mag_min
        self.mag_max = mag_max
        self.magerr_max = magerr_max

        self.df: Optional[pd.DataFrame] = None

        # Image data
        self.img_tokens_flat: Optional[np.ndarray] = None
        self.img_id_to_row: Dict[int, int] = {}

        # Scalar data
        self.scalar_tokens: Optional[np.ndarray] = None
        self.scalar_cols: Optional[List[str]] = None
        self.scalar_n_bins: int = 1024
        self.splus_id_to_scalar_row: Dict[str, int] = {}

        # Spectrum data (keep NPZ open for efficient access)
        self.spec_npz: Optional[np.lib.npyio.NpzFile] = None
        self.splus_id_to_spec_row: Dict[str, int] = {}
        self.spec_groups_present: Dict[str, Tuple[str, str]] = {}

        self._load()

    def _load(self):
        """Load and index all data for this field."""
        # Load datacube and apply filters
        datacube_path = self.datacube_template.format(field=self.field)
        df = pd.read_parquet(
            datacube_path,
            columns=["id", "ra", "dec", "gaia_source_id", self.mag_col, self.magerr_col],
        )
        mask = (
            (df[self.mag_col] > self.mag_min) &
            (df[self.mag_col] < self.mag_max) &
            (df[self.magerr_col] < self.magerr_max)
        )
        df = df.loc[mask].copy()
        df["id"] = df["id"].map(norm_splus_id)
        df["gaia_source_id"] = df["gaia_source_id"].map(norm_gaia_id)
        self.df = df

        # Load image tokens
        img_path = self.image_tokens_template.format(field=self.field)
        if Path(img_path).exists():
            z = np.load(img_path, allow_pickle=True)
            try:
                if "ids" in z.files and "tokens_flat" in z.files:
                    ids = [norm_gaia_id(x) for x in np.asarray(z["ids"], dtype=object)]
                    self.img_tokens_flat = z["tokens_flat"]
                    self.img_id_to_row = {
                        gid: i for i, gid in enumerate(ids) if gid is not None
                    }
            finally:
                z.close()

        # Load scalar tokens
        scalar_path = self.scalar_tokens_template.format(field=self.field)
        if Path(scalar_path).exists():
            z = np.load(scalar_path, allow_pickle=True)
            try:
                if "ids" in z.files and "scalar_tokens" in z.files:
                    ids = [norm_splus_id(x) for x in np.asarray(z["ids"], dtype=object)]
                    self.scalar_tokens = z["scalar_tokens"]
                    self.scalar_cols = list(z["scalar_cols"]) if "scalar_cols" in z.files else None
                    self.scalar_n_bins = int(z["N_BINS"]) if "N_BINS" in z.files else 1024
                    self.splus_id_to_scalar_row = {
                        sid: i for i, sid in enumerate(ids) if sid
                    }
            finally:
                z.close()

        # Load spectrum tokens (keep NPZ open)
        spectrum_path = self.spectrum_tokens_template.format(field=self.field)
        if Path(spectrum_path).exists():
            z = np.load(spectrum_path, allow_pickle=True)
            if "ids" in z.files:
                ids = [norm_splus_id(x) for x in np.asarray(z["ids"], dtype=object)]
                self.splus_id_to_spec_row = {sid: i for i, sid in enumerate(ids) if sid}

                # Check which spectrum groups are present
                for g, (fk, ik) in self.spectrum_groups.items():
                    if fk in z.files and ik in z.files:
                        self.spec_groups_present[g] = (fk, ik)

                self.spec_npz = z
            else:
                z.close()

    def __del__(self):
        """Clean up open file handles."""
        try:
            if self.spec_npz is not None:
                self.spec_npz.close()
        except Exception:
            pass

    def __len__(self) -> int:
        """Return number of objects in this field."""
        return int(len(self.df)) if self.df is not None else 0

    def get_row(self, idx: int) -> Dict[str, Any]:
        """
        Get all tokenized data for a single object.

        Parameters
        ----------
        idx : int
            Row index in the filtered datacube DataFrame

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys:
            - 'image_tokens': Optional[np.ndarray] or None
            - 'scalar_tokens': Optional[np.ndarray] or None
            - 'spectrum_tokens': Dict[str, np.ndarray] (group -> tokens)
        """
        r = self.df.loc[idx]
        sid: str = r["id"]
        gid: Optional[int] = r["gaia_source_id"]

        # Get image tokens
        img_tokens = None
        if gid is not None and self.img_tokens_flat is not None:
            j = self.img_id_to_row.get(gid)
            if j is not None:
                img_tokens = self.img_tokens_flat[j]

        # Get scalar tokens
        scalar_tokens = None
        scalar_cols = None
        scalar_n_bins = self.scalar_n_bins
        if self.scalar_tokens is not None:
            j = self.splus_id_to_scalar_row.get(sid)
            if j is not None:
                scalar_tokens = self.scalar_tokens[j]
                scalar_cols = self.scalar_cols

        # Get spectrum tokens
        spec = {}
        if self.spec_npz is not None:
            j = self.splus_id_to_spec_row.get(sid)
            if j is not None:
                for g, (fk, ik) in self.spec_groups_present.items():
                    flat = self.spec_npz[fk]
                    indp = self.spec_npz[ik]
                    a = int(indp[j])
                    b = int(indp[j + 1])
                    spec[g] = flat[a:b]

        return {
            "image_tokens": img_tokens,
            "scalar_tokens": scalar_tokens,
            "scalar_cols": scalar_cols,
            "scalar_n_bins": scalar_n_bins,
            "spectrum_tokens": spec,
        }


class MultiFieldTokenDataset(Dataset):
    """
    PyTorch Dataset for multi-field token loading with LRU cache.

    This dataset provides efficient access to tokenized astronomical data across
    multiple fields. It addresses the startup issue of loading all fields at once
    by using a cached index and lazy field loading with LRU eviction.

    Features:
    - Builds or loads a cached index of (field, row) pairs
    - Lazy FieldIndex instantiation per field
    - LRU cache for FieldIndex objects to control memory usage
    - Fast startup with pre-shuffled index

    Parameters
    ----------
    fields : List[str]
        List of field identifiers to include
    vocab : VocabSpec
        Vocabulary specification for token offsets
    datacube_template : str
        Path template for datacube parquet files with {field} placeholder
    image_tokens_template : str
        Path template for image token files with {field} placeholder
    scalar_tokens_template : str
        Path template for scalar token files with {field} placeholder
    spectrum_tokens_template : str
        Path template for spectrum token files with {field} placeholder
    spectrum_groups : Dict[str, Tuple[str, str]]
        Mapping of spectrum group names to (flat_key, indptr_key) pairs
    mag_col : str, default="mag_pstotal_r"
        Magnitude column for filtering
    magerr_col : str, default="err_mag_pstotal_r"
        Magnitude error column for filtering
    mag_min : float, default=14.0
        Minimum magnitude filter
    mag_max : float, default=22.0
        Maximum magnitude filter
    magerr_max : float, default=2.0
        Maximum magnitude error filter
    index_cache : str, default="cache_token_dataset_index.npz"
        Path to index cache file
    lru_fields : int, default=8
        Maximum number of FieldIndex objects to keep in memory

    Attributes
    ----------
    vocab : VocabSpec
        Vocabulary specification
    fields : List[str]
        List of field identifiers
    index_field : List[str]
        List of field identifiers per sample (parallel to index_row)
    index_row : np.ndarray
        Array of row indices per sample (parallel to index_field)

    Examples
    --------
    >>> from astromodal.core import build_vocab_spec
    >>> vocab = build_vocab_spec()
    >>> dataset = MultiFieldTokenDataset(
    ...     fields=["HYDRA-0011", "HYDRA-0012"],
    ...     vocab=vocab,
    ...     datacube_template="/path/datacube_{field}.parquet",
    ...     image_tokens_template="/path/image_{field}_tokens.npz",
    ...     scalar_tokens_template="/path/scalar_{field}_tokens.npz",
    ...     spectrum_tokens_template="/path/spectrum_{field}_tokens.npz",
    ...     spectrum_groups={"gaiaxp_bp": ("tokens_gaiaxp_bp_flat", "tokens_gaiaxp_bp_indptr")},
    ... )
    >>> len(dataset)
    24690
    >>> sample = dataset[0]
    >>> sample.keys()
    dict_keys(['token_ids', 'type_ids'])
    """

    def __init__(
        self,
        fields: List[str],
        vocab: VocabSpec,
        datacube_template: str,
        image_tokens_template: str,
        scalar_tokens_template: str,
        spectrum_tokens_template: str,
        spectrum_groups: Dict[str, Tuple[str, str]],
        mag_col: str = "mag_pstotal_r",
        magerr_col: str = "err_mag_pstotal_r",
        mag_min: float = 14.0,
        mag_max: float = 22.0,
        magerr_max: float = 2.0,
        index_cache: str = "cache_token_dataset_index.npz",
        lru_fields: int = 8,
    ):
        self.vocab = vocab
        self.fields = fields
        self.datacube_template = datacube_template
        self.image_tokens_template = image_tokens_template
        self.scalar_tokens_template = scalar_tokens_template
        self.spectrum_tokens_template = spectrum_tokens_template
        self.spectrum_groups = spectrum_groups
        self.mag_col = mag_col
        self.magerr_col = magerr_col
        self.mag_min = mag_min
        self.mag_max = mag_max
        self.magerr_max = magerr_max
        self.index_cache = index_cache
        self.lru_fields = lru_fields

        self.index_field: List[str] = []
        self.index_row: np.ndarray = np.empty((0,), dtype=np.int32)

        self._field_cache: Dict[str, FieldIndex] = {}
        self._field_cache_order: List[str] = []

        self._load_or_build_index()

    def _load_or_build_index(self):
        """Load cached index or build it from scratch."""
        p = Path(self.index_cache)
        if p.exists():
            z = np.load(str(p), allow_pickle=True)
            self.index_field = list(z["field"].astype(str))
            self.index_row = z["row"].astype(np.int32)
            print(f"[index] loaded {len(self.index_row)} samples from {p}")
            return

        print("[index] building index (first time; will be cached)...")
        t0 = time.time()
        fields_out = []
        rows_out = []

        for f in tqdm(self.fields, desc="[index-scan]", dynamic_ncols=True):
            datacube_path = self.datacube_template.format(field=f)
            if not Path(datacube_path).exists():
                continue

            # Read just the needed columns to filter
            df = pd.read_parquet(datacube_path, columns=[self.mag_col, self.magerr_col])
            mask = (
                (df[self.mag_col] > self.mag_min) &
                (df[self.mag_col] < self.mag_max) &
                (df[self.magerr_col] < self.magerr_max)
            ).values
            idxs = np.nonzero(mask)[0]
            if idxs.size == 0:
                continue

            fields_out.extend([f] * int(idxs.size))
            rows_out.append(idxs.astype(np.int32))

        if rows_out:
            row_all = np.concatenate(rows_out, axis=0)
        else:
            row_all = np.empty((0,), dtype=np.int32)

        self.index_field = fields_out
        self.index_row = row_all

        # Shuffle once here (and DataLoader shuffle=True can stay off if you want)
        perm = np.random.permutation(len(self.index_row))
        self.index_row = self.index_row[perm]
        self.index_field = [self.index_field[i] for i in perm.tolist()]

        np.savez_compressed(
            str(p),
            field=np.array(self.index_field, dtype=object),
            row=self.index_row
        )
        elapsed = (time.time() - t0) / 60
        print(f"[index] built {len(self.index_row)} samples in {elapsed:.1f} min; cached to {p}")

    def __len__(self) -> int:
        """Return total number of samples across all fields."""
        return int(len(self.index_row))

    def _get_field_index(self, field: str) -> FieldIndex:
        """
        Get FieldIndex for a field, using LRU cache.

        Parameters
        ----------
        field : str
            Field identifier

        Returns
        -------
        FieldIndex
            Field index object (cached or newly created)
        """
        if field in self._field_cache:
            # Refresh LRU position
            if field in self._field_cache_order:
                self._field_cache_order.remove(field)
            self._field_cache_order.append(field)
            return self._field_cache[field]

        # Create new FieldIndex
        fi = FieldIndex(
            field=field,
            datacube_template=self.datacube_template,
            image_tokens_template=self.image_tokens_template,
            scalar_tokens_template=self.scalar_tokens_template,
            spectrum_tokens_template=self.spectrum_tokens_template,
            spectrum_groups=self.spectrum_groups,
            mag_col=self.mag_col,
            magerr_col=self.magerr_col,
            mag_min=self.mag_min,
            mag_max=self.mag_max,
            magerr_max=self.magerr_max,
        )
        self._field_cache[field] = fi
        self._field_cache_order.append(field)

        # Evict LRU if needed
        while len(self._field_cache_order) > self.lru_fields:
            old = self._field_cache_order.pop(0)
            try:
                del self._field_cache[old]
            except Exception:
                pass

        return fi

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample by global index.

        Parameters
        ----------
        idx : int
            Global sample index

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys:
            - 'token_ids': torch.Tensor of shape (seq_len,)
            - 'type_ids': torch.Tensor of shape (seq_len,)

        Notes
        -----
        Builds a complete sequence: [BOS] [image] [SEP] [scalar] [SEP] [spectra] [EOS]
        Missing modalities are skipped. Type IDs indicate the modality of each token.
        """
        field = self.index_field[idx]
        row_i = int(self.index_row[idx])
        fi = self._get_field_index(field)
        sample = fi.get_row(row_i)

        # Build sequence from available modalities
        token_ids = [self.vocab.bos_id]
        type_ids = [1]  # BOS type

        # Add image tokens if available
        if sample["image_tokens"] is not None:
            img_toks = sample["image_tokens"].flatten()[:576]  # Max 576 tokens (24x24)
            img_toks_shifted = (img_toks + self.vocab.base_image).astype(np.int64)
            token_ids.extend(img_toks_shifted.tolist())
            type_ids.extend([5] * len(img_toks_shifted))  # IMAGE type
            token_ids.append(self.vocab.sep_id)
            type_ids.append(3)  # SEP type

        # Add scalar tokens if available
        if sample["scalar_tokens"] is not None and sample["scalar_cols"] is not None:
            scal_toks = sample["scalar_tokens"]  # (n_cols,) array of token values
            n_bins = sample["scalar_n_bins"]

            # Shift each token by: base_scalar + (col_idx * n_bins) + token_value
            scal_toks_shifted = []
            for col_idx, tok_val in enumerate(scal_toks):
                if col_idx >= 512:  # Max 512 columns
                    break
                shifted = self.vocab.base_scalar + (col_idx * n_bins) + int(tok_val)
                scal_toks_shifted.append(shifted)

            token_ids.extend(scal_toks_shifted)
            type_ids.extend([6] * len(scal_toks_shifted))  # SCALAR type
            token_ids.append(self.vocab.sep_id)
            type_ids.append(3)  # SEP type

        # Add spectrum tokens if available
        if sample["spectrum_tokens"]:
            for group_idx, (group_name, group_toks) in enumerate(sample["spectrum_tokens"].items()):
                if group_toks is not None and len(group_toks) > 0:
                    spec_toks = group_toks[:512]  # Max 512 per group
                    # Get base offset for this spectrum group
                    spec_base = self.vocab.base_spectrum.get(group_name)
                    if spec_base is None:
                        # Skip if group not in vocab (shouldn't happen)
                        continue
                    spec_toks_shifted = (spec_toks + spec_base).astype(np.int64)
                    token_ids.extend(spec_toks_shifted.tolist())
                    type_ids.extend([7 + group_idx] * len(spec_toks_shifted))  # SPECTRUM_GROUP type
                    token_ids.append(self.vocab.sep_id)
                    type_ids.append(3)  # SEP type

        # Add EOS
        token_ids.append(self.vocab.eos_id)
        type_ids.append(2)  # EOS type

        # Convert to tensors
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        type_ids = torch.tensor(type_ids, dtype=torch.long)

        return {
            "token_ids": token_ids,
            "type_ids": type_ids,
        }


def collate_sequences(batch: List[Dict[str, torch.Tensor]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching variable-length sequences.

    Parameters
    ----------
    batch : List[Dict[str, torch.Tensor]]
        List of samples from dataset
    pad_id : int
        Padding token ID (default: 0)

    Returns
    -------
    Dict[str, torch.Tensor]
        Batched tensors:
        - 'tokens': (batch_size, max_seq_len)
        - 'types': (batch_size, max_seq_len)
        - 'attn_mask': (batch_size, max_seq_len)
    """
    # Find max sequence length in batch
    max_len = max(sample["token_ids"].size(0) for sample in batch)
    batch_size = len(batch)

    # Initialize padded tensors
    tokens = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    types = torch.full((batch_size, max_len), 0, dtype=torch.long)
    attn_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    # Fill in the actual sequences
    for i, sample in enumerate(batch):
        seq_len = sample["token_ids"].size(0)
        tokens[i, :seq_len] = sample["token_ids"]
        types[i, :seq_len] = sample["type_ids"]
        attn_mask[i, :seq_len] = True

    return {
        "tokens": tokens,
        "types": types,
        "attn_mask": attn_mask,
    }
