"""
Scalar tokenizer for astronomy catalog data.

Implements quantile-based binning tokenization for scalar features like
magnitudes, proper motions, and other numerical catalog columns.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl

from .base import BaseTokenizer


class ScalarTokenizer(BaseTokenizer):
    """
    Tokenizer for scalar astronomical features using quantile binning.

    This tokenizer learns per-column quantile bins from training data and
    encodes continuous scalar values into discrete tokens. It handles:
    - Magnitude and error columns with specific value filters
    - Missing/non-finite values (filled with column mean)
    - Row-level filtering (e.g., magnitude range)
    - Value-level filtering (e.g., magnitude error clipping)

    Parameters
    ----------
    n_bins : int, default=1024
        Number of quantile bins per column
    scalar_columns : List[str], optional
        List of column names to tokenize. If None, must be set before training.
    mag_filter_col : str, default="mag_pstotal_r"
        Column name for row-level magnitude filtering
    mag_min : float, default=14.0
        Minimum magnitude for row filtering
    mag_max : float, default=22.0
        Maximum magnitude for row filtering
    apply_row_filter : bool, default=True
        Whether to apply row-level magnitude filtering
    magerr_max : float, default=2.0
        Maximum magnitude error (for clipping)
    apply_magerr_filter : bool, default=True
        Whether to clip magnitude errors
    max_values_per_col : int, default=2_000_000
        Maximum training samples per column
    max_values_per_file_per_col : int, default=100_000
        Maximum samples per file per column
    rng_seed : int, default=0
        Random seed for reproducibility
    allow_bool : bool, default=False
        Whether to allow boolean columns
    min_values_to_fit : int, default=1000
        Minimum values required to fit a column
    edge_eps : float, default=1e-8
        Small epsilon for ensuring strictly increasing bin edges

    Attributes
    ----------
    bin_edges_ : Dict[str, np.ndarray]
        Learned bin edges per column (N_BINS+1 edges)
    col_means_ : Dict[str, float]
        Column means for imputing missing values
    col_stds_ : Dict[str, float]
        Column standard deviations
    col_dtypes_ : Dict[str, str]
        Original column data types
    is_fitted_ : bool
        Whether the tokenizer has been trained

    Examples
    --------
    >>> from astromodal.tokenizers import ScalarTokenizer
    >>> tokenizer = ScalarTokenizer(n_bins=1024)
    >>> tokenizer.train(parquet_files, scalar_columns=columns)
    >>> tokens = tokenizer.encode(dataframe)
    >>> tokenizer.save("scalar_tokenizer.npz")
    >>> loaded = ScalarTokenizer.load("scalar_tokenizer.npz")
    """

    def __init__(
        self,
        n_bins: int = 1024,
        scalar_columns: Optional[List[str]] = None,
        mag_filter_col: str = "mag_pstotal_r",
        mag_min: float = 14.0,
        mag_max: float = 22.0,
        apply_row_filter: bool = True,
        magerr_max: float = 2.0,
        apply_magerr_filter: bool = True,
        max_values_per_col: int = 2_000_000,
        max_values_per_file_per_col: int = 100_000,
        rng_seed: int = 0,
        allow_bool: bool = False,
        min_values_to_fit: int = 1000,
        edge_eps: float = 1e-8,
    ):
        self.n_bins = n_bins
        self.scalar_columns = scalar_columns or []
        self.mag_filter_col = mag_filter_col
        self.mag_min = mag_min
        self.mag_max = mag_max
        self.apply_row_filter = apply_row_filter
        self.magerr_max = magerr_max
        self.apply_magerr_filter = apply_magerr_filter
        self.max_values_per_col = max_values_per_col
        self.max_values_per_file_per_col = max_values_per_file_per_col
        self.rng_seed = rng_seed
        self.allow_bool = allow_bool
        self.min_values_to_fit = min_values_to_fit
        self.edge_eps = edge_eps

        # Learned parameters
        self.bin_edges_: Dict[str, np.ndarray] = {}
        self.col_means_: Dict[str, float] = {}
        self.col_stds_: Dict[str, float] = {}
        self.col_dtypes_: Dict[str, str] = {}
        self.is_fitted_ = False

    def train(
        self,
        parquet_files: Union[str, Path, List[Union[str, Path]]],
        scalar_columns: Optional[List[str]] = None,
    ) -> None:
        """
        Train the scalar tokenizer on parquet files.

        Parameters
        ----------
        parquet_files : Union[str, Path, List[Union[str, Path]]]
            Path(s) to parquet file(s) containing training data
        scalar_columns : List[str], optional
            Column names to tokenize. If provided, overrides instance attribute.

        Raises
        ------
        ValueError
            If no scalar columns are specified or found
        RuntimeError
            If insufficient data for training
        """
        if scalar_columns is not None:
            self.scalar_columns = scalar_columns

        if not self.scalar_columns:
            raise ValueError("No scalar columns specified for training")

        # Normalize to list of Path objects
        if isinstance(parquet_files, (str, Path)):
            files = [Path(parquet_files)]
        else:
            files = [Path(f) for f in parquet_files]

        print(f"Training scalar tokenizer on {len(files)} file(s)")
        print(f"Columns to tokenize: {len(self.scalar_columns)}")

        rng = np.random.default_rng(self.rng_seed)

        # Check columns exist somewhere
        present_cols, missing_cols = self._check_columns_exist(files, self.scalar_columns)
        if missing_cols:
            print(f"Warning: {len(missing_cols)} columns not found in any file")

        if not present_cols:
            raise ValueError("No requested columns found in input files")

        # Pre-compute quantile positions
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)

        # Fit each column
        for col in present_cols:
            print(f"\nFitting column: {col}")
            vals = self._collect_values_for_column(col, files, rng)
            n = vals.size
            print(f"  Collected {n:,} values")

            if n < self.min_values_to_fit:
                print(f"  Skipping (< {self.min_values_to_fit} values)")
                continue

            vals_sorted = np.sort(vals)

            # Compute statistics
            mean = float(np.mean(vals_sorted))
            std = float(np.std(vals_sorted) + 1e-8)
            self.col_means_[col] = mean
            self.col_stds_[col] = std

            # Compute quantile edges
            edges = np.quantile(vals_sorted, quantiles)
            edges = self._make_strictly_increasing(edges, self.edge_eps)
            self.bin_edges_[col] = edges.astype(np.float64)

            # Record dtype
            dtype_str = self._infer_dtype(col, files)
            self.col_dtypes_[col] = dtype_str

            print(f"  Bin edges: [{edges[0]:.4g}, {edges[-1]:.4g}]")
            print(f"  Mean: {mean:.4g}, Std: {std:.4g}")

        if not self.bin_edges_:
            raise RuntimeError("No columns successfully fitted")

        self.is_fitted_ = True
        print(f"\nTraining complete: {len(self.bin_edges_)} columns fitted")

    def encode(
        self,
        data: Union[pl.DataFrame, str, Path],
        drop_rows_on_err: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Encode scalar data into tokens.

        Parameters
        ----------
        data : Union[pl.DataFrame, str, Path]
            Data to encode (DataFrame or path to parquet file)
        drop_rows_on_err : bool, default=False
            Whether to drop rows with errors exceeding threshold

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing:
            - 'scalar_tokens': (n_rows, n_cols) array of token IDs
            - 'scalar_cols': (n_cols,) array of column names
            - 'ids': (n_rows,) array of row IDs (if available)
            - 'id_col': column name used for IDs
            - 'N_BINS': number of bins

        Raises
        ------
        RuntimeError
            If tokenizer has not been trained
        """
        if not self.is_fitted_:
            raise RuntimeError("Tokenizer must be trained before encoding")

        # Load data
        if isinstance(data, (str, Path)):
            df = pl.read_parquet(data)
        else:
            df = data

        # Apply row filters
        df = self._apply_row_filters(df, drop_rows_on_err)
        n_rows = df.height

        if n_rows == 0:
            return {
                'scalar_tokens': np.zeros((0, 0), dtype=np.uint16),
                'scalar_cols': np.array([], dtype=object),
                'ids': np.array([], dtype=np.int64),
                'id_col': '',
                'N_BINS': self.n_bins,
            }

        # Determine which columns are present
        present_cols = [c for c in self.bin_edges_.keys() if c in df.columns]
        print(f"Encoding {n_rows:,} rows x {len(present_cols)} columns")

        # Allocate output array
        tokens = np.zeros((n_rows, len(present_cols)), dtype=np.uint16)

        # Encode each column
        for j, col in enumerate(present_cols):
            series = df[col]
            arr = self._series_to_float64(series)

            # Fill non-finite with mean
            default_val = self.col_means_.get(col, 0.0)
            bad_mask = ~np.isfinite(arr)
            if np.any(bad_mask):
                arr[bad_mask] = default_val

            # Apply magnitude error clipping if needed
            if self.apply_magerr_filter and col.startswith("err_mag_pstotal_"):
                arr = np.clip(arr, 0.0, self.magerr_max)

            # Encode to tokens
            edges = self.bin_edges_[col]
            tokens[:, j] = self._encode_values(arr, edges)

        # Extract IDs if available
        id_col = self._choose_id_column(df.columns)
        if id_col is not None:
            ids = df[id_col].cast(pl.Int64).to_numpy()
        else:
            ids = np.array([], dtype=np.int64)

        return {
            'scalar_tokens': tokens,
            'scalar_cols': np.array(present_cols, dtype=object),
            'ids': ids,
            'id_col': id_col or '',
            'N_BINS': self.n_bins,
        }

    def save(self, path: Union[str, Path]) -> None:
        """
        Save tokenizer configuration to disk.

        Parameters
        ----------
        path : Union[str, Path]
            Output path (typically .npz file)

        Raises
        ------
        RuntimeError
            If tokenizer has not been trained
        """
        if not self.is_fitted_:
            raise RuntimeError("Cannot save untrained tokenizer")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            bin_edges=self.bin_edges_,
            col_means=self.col_means_,
            col_stds=self.col_stds_,
            col_dtypes=self.col_dtypes_,
            N_BINS=self.n_bins,
            mag_filter_col=self.mag_filter_col,
            mag_min=self.mag_min,
            mag_max=self.mag_max,
            apply_row_filter=self.apply_row_filter,
            apply_magerr_filter=self.apply_magerr_filter,
            magerr_max=self.magerr_max,
            rng_seed=self.rng_seed,
            max_values_per_col=self.max_values_per_col,
            max_values_per_file_per_col=self.max_values_per_file_per_col,
            allow_bool=self.allow_bool,
            min_values_to_fit=self.min_values_to_fit,
            edge_eps=self.edge_eps,
        )
        print(f"Saved scalar tokenizer to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ScalarTokenizer":
        """
        Load tokenizer configuration from disk.

        Parameters
        ----------
        path : Union[str, Path]
            Path to tokenizer config file

        Returns
        -------
        ScalarTokenizer
            Loaded tokenizer instance
        """
        data = np.load(path, allow_pickle=True)

        # Extract config parameters
        tokenizer = cls(
            n_bins=int(data['N_BINS']),
            mag_filter_col=str(data.get('mag_filter_col', 'mag_pstotal_r')),
            mag_min=float(data.get('mag_min', 14.0)),
            mag_max=float(data.get('mag_max', 22.0)),
            apply_row_filter=bool(data.get('apply_row_filter', True)),
            magerr_max=float(data.get('magerr_max', 2.0)),
            apply_magerr_filter=bool(data.get('apply_magerr_filter', True)),
            max_values_per_col=int(data.get('max_values_per_col', 2_000_000)),
            max_values_per_file_per_col=int(data.get('max_values_per_file_per_col', 100_000)),
            rng_seed=int(data.get('rng_seed', 0)),
            allow_bool=bool(data.get('allow_bool', False)),
            min_values_to_fit=int(data.get('min_values_to_fit', 1000)),
            edge_eps=float(data.get('edge_eps', 1e-8)),
        )

        # Load learned parameters
        tokenizer.bin_edges_ = cls._normalize_dict_keys(data['bin_edges'].item())
        tokenizer.col_means_ = cls._normalize_dict_keys(data['col_means'].item())
        tokenizer.col_stds_ = cls._normalize_dict_keys(data['col_stds'].item())
        tokenizer.col_dtypes_ = cls._normalize_dict_keys(data.get('col_dtypes', {}).item() if 'col_dtypes' in data else {})

        # Ensure numpy arrays
        for k, v in tokenizer.bin_edges_.items():
            tokenizer.bin_edges_[k] = np.asarray(v, dtype=np.float64)

        tokenizer.is_fitted_ = True
        print(f"Loaded scalar tokenizer from {path} ({len(tokenizer.bin_edges_)} columns)")
        return tokenizer

    # ==================== Private Methods ====================

    @staticmethod
    def _normalize_dict_keys(d: Dict) -> Dict[str, any]:
        """Convert dict keys to strings (handles bytes from npz)."""
        return {
            (k.decode() if isinstance(k, bytes) else str(k)): v
            for k, v in d.items()
        }

    def _check_columns_exist(
        self,
        files: List[Path],
        requested: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Check which columns exist somewhere in the files."""
        missing = set(requested)
        present = set()

        for f in files:
            if not missing:
                break
            try:
                df0 = pl.read_parquet(f, n_rows=0)
                have = set(df0.columns)
                hit = missing.intersection(have)
                if hit:
                    present.update(hit)
                    missing.difference_update(hit)
            except Exception:
                continue

        present_list = [c for c in requested if c in present]
        missing_list = [c for c in requested if c in missing]
        return present_list, missing_list

    def _collect_values_for_column(
        self,
        col: str,
        files: List[Path],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Collect training values for a single column across files."""
        chunks = []
        total = 0

        for path in files:
            if total >= self.max_values_per_col:
                break

            # Read column plus filter column if needed
            cols_to_read = [col]
            if self.apply_row_filter and self.mag_filter_col not in cols_to_read:
                cols_to_read.append(self.mag_filter_col)

            try:
                df = pl.read_parquet(path, columns=cols_to_read)
            except Exception:
                continue

            # Apply row filter
            if self.apply_row_filter:
                if self.mag_filter_col not in df.columns:
                    continue
                df = df.filter(
                    pl.col(self.mag_filter_col).is_finite()
                    & (pl.col(self.mag_filter_col) > self.mag_min)
                    & (pl.col(self.mag_filter_col) < self.mag_max)
                )
                if df.height == 0:
                    continue

            if col not in df.columns:
                continue

            series = df[col]
            dtype = series.dtype

            # Check dtype
            if not self._dtype_is_allowed(dtype):
                continue

            # Convert to numpy
            try:
                arr = self._series_to_float64(series)
                arr = arr[np.isfinite(arr)]
            except Exception:
                continue

            if arr.size == 0:
                continue

            # Value-level filters
            if col.startswith("mag_pstotal_"):
                arr = arr[(arr > self.mag_min) & (arr < self.mag_max)]
            if self.apply_magerr_filter and col.startswith("err_mag_pstotal_"):
                arr = arr[(arr > 0.0) & (arr < self.magerr_max)]

            if arr.size == 0:
                continue

            # Subsample per file
            if arr.size > self.max_values_per_file_per_col:
                idx = rng.choice(arr.size, size=self.max_values_per_file_per_col, replace=False)
                arr = arr[idx]

            # Respect global cap
            remaining = self.max_values_per_col - total
            if remaining <= 0:
                break
            if arr.size > remaining:
                idx = rng.choice(arr.size, size=remaining, replace=False)
                arr = arr[idx]

            chunks.append(arr)
            total += arr.size

        if not chunks:
            return np.array([], dtype=np.float64)

        return np.concatenate(chunks, axis=0)

    def _dtype_is_allowed(self, dtype: pl.DataType) -> bool:
        """Check if dtype is allowed for tokenization."""
        if dtype.is_numeric():
            return True
        if self.allow_bool and dtype == pl.Boolean:
            return True
        return False

    @staticmethod
    def _series_to_float64(series: pl.Series) -> np.ndarray:
        """Convert Polars series to float64 numpy array."""
        if series.dtype == pl.Boolean:
            return series.cast(pl.UInt8).to_numpy().astype(np.float64)
        elif series.dtype.is_float():
            return series.to_numpy().astype(np.float64)
        elif series.dtype.is_integer():
            return series.to_numpy().astype(np.float64)
        else:
            return series.cast(pl.Float64).to_numpy()

    @staticmethod
    def _make_strictly_increasing(edges: np.ndarray, eps: float) -> np.ndarray:
        """Ensure bin edges are strictly increasing."""
        edges = np.asarray(edges, dtype=np.float64)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + eps
        return edges

    def _infer_dtype(self, col: str, files: List[Path]) -> str:
        """Infer dtype from first file where column exists."""
        for f in files:
            try:
                df0 = pl.read_parquet(f, n_rows=0)
                if col in df0.columns:
                    return str(df0.schema[col])
            except Exception:
                continue
        return "unknown"

    def _apply_row_filters(self, df: pl.DataFrame, drop_rows_on_err: bool) -> pl.DataFrame:
        """Apply row-level filters."""
        if self.apply_row_filter:
            if self.mag_filter_col not in df.columns:
                return df.head(0)
            df = df.filter(
                (pl.col(self.mag_filter_col) > self.mag_min) &
                (pl.col(self.mag_filter_col) < self.mag_max)
            )

        if drop_rows_on_err:
            err_cols = [c for c in df.columns if c.startswith("err_mag_pstotal_")]
            if err_cols:
                cond = None
                for c in err_cols:
                    c_cond = pl.col(c).is_finite() & (pl.col(c) <= self.magerr_max) & (pl.col(c) >= 0)
                    cond = c_cond if cond is None else (cond & c_cond)
                df = df.filter(cond)

        return df

    @staticmethod
    def _choose_id_column(columns: List[str]) -> Optional[str]:
        """Choose ID column from available columns."""
        preferred = ["id", "gaia_source_id", "source_id"]
        for c in preferred:
            if c in columns:
                return c
        return None

    @staticmethod
    def _encode_values(vals: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Encode values into bin indices.

        Parameters
        ----------
        vals : np.ndarray
            Values to encode (N,)
        edges : np.ndarray
            Bin edges (N_BINS+1,)

        Returns
        -------
        np.ndarray
            Token IDs in [0, N_BINS-1]
        """
        idx = np.searchsorted(edges, vals, side="right") - 1
        idx = np.clip(idx, 0, len(edges) - 2)
        return idx.astype(np.uint16)
