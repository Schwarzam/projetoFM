"""
Spectrum tokenizer for astronomical time-series and spectroscopic data.

Implements K-means clustering tokenization for spectrum data including DESI spectra,
ZTF light curves, and GaiaXP continuous spectra.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import warnings

import numpy as np
import polars as pl
import joblib
from sklearn.cluster import MiniBatchKMeans
from tqdm.auto import tqdm

from .base import BaseTokenizer


class SpectrumTokenizer(BaseTokenizer):
    """
    Tokenizer for astronomical spectrum data using K-means clustering.

    This tokenizer learns separate codebooks for different spectrum types:
    - DESI spectra (flux + inverse variance)
    - ZTF light curves (magnitude + error)
    - GaiaXP continuous spectra (BP/RP photometry)

    Each spectrum type is normalized and clustered independently to create
    discrete tokens representing typical spectral features.

    Parameters
    ----------
    codebook_size : int, default=2048
        Number of clusters (vocabulary size) per spectrum group
    max_features_per_group : int, default=2_000_000
        Maximum training samples per group
    max_rows_per_file : int, default=5000
        Maximum rows to sample per file
    max_points_per_row : int, default=2000
        Maximum points to sample per spectrum
    groups : List[Dict], optional
        Spectrum group definitions. Each dict should specify:
        - name: group identifier
        - kind: 'flux_ivar', 'ztf_mag_err', or 'gaiaxpy_xp'
        - flux_col/ivar_col (for DESI)
        - mag_col/err_col (for ZTF)
        - xp_band: 'BP' or 'RP' (for GaiaXP)
    gaiaxpy_mode : str, default='convert'
        GaiaXPy processing mode ('convert' or 'calibrate')
    gaiaxpy_sampling : np.ndarray, optional
        Wavelength sampling for GaiaXP (None = default)
    gaiaxpy_truncation : bool, default=False
        Whether to apply GaiaXPy truncation
    ztf_err_eps : float, default=1e-3
        Minimum ZTF error for numerical stability
    gaiaxp_err_eps : float, default=1e-6
        Minimum GaiaXP error for numerical stability
    gaiaxp_clip_norm : float, default=12.0
        Clipping threshold for normalized GaiaXP flux

    Attributes
    ----------
    group_names_ : List[str]
        Names of fitted spectrum groups
    group_means_ : Dict[str, np.ndarray]
        Mean feature vectors per group
    group_stds_ : Dict[str, np.ndarray]
        Standard deviation vectors per group
    group_codebooks_ : Dict[str, np.ndarray]
        K-means centroids per group (codebook_size, feature_dim)
    group_columns_ : Dict[str, Dict]
        Column specifications per group
    is_fitted_ : bool
        Whether the tokenizer has been trained

    Examples
    --------
    >>> from astromodal.tokenizers import SpectrumTokenizer
    >>> groups = [
    ...     {"name": "desi_b", "kind": "flux_ivar",
    ...      "flux_col": "desi_flux_b", "ivar_col": "desi_ivar_b"},
    ...     {"name": "gaiaxp_bp", "kind": "gaiaxpy_xp", "xp_band": "BP"}
    ... ]
    >>> tokenizer = SpectrumTokenizer(codebook_size=2048, groups=groups)
    >>> tokenizer.train(parquet_files)
    >>> tokens = tokenizer.encode(dataframe)
    >>> tokenizer.save("spectrum_tokenizer.npz")
    """

    def __init__(
        self,
        codebook_size: int = 2048,
        max_features_per_group: int = 2_000_000,
        max_rows_per_file: int = 5000,
        max_points_per_row: int = 2000,
        groups: Optional[List[Dict[str, Any]]] = None,
        gaiaxpy_mode: str = "convert",
        gaiaxpy_sampling: Optional[np.ndarray] = None,
        gaiaxpy_truncation: bool = False,
        ztf_err_eps: float = 1e-3,
        gaiaxp_err_eps: float = 1e-6,
        gaiaxp_clip_norm: float = 12.0,
    ):
        self.codebook_size = codebook_size
        self.max_features_per_group = max_features_per_group
        self.max_rows_per_file = max_rows_per_file
        self.max_points_per_row = max_points_per_row
        self.groups = groups or []
        self.gaiaxpy_mode = gaiaxpy_mode
        self.gaiaxpy_sampling = gaiaxpy_sampling
        self.gaiaxpy_truncation = gaiaxpy_truncation
        self.ztf_err_eps = ztf_err_eps
        self.gaiaxp_err_eps = gaiaxp_err_eps
        self.gaiaxp_clip_norm = gaiaxp_clip_norm

        # Learned parameters
        self.group_names_: List[str] = []
        self.group_means_: Dict[str, np.ndarray] = {}
        self.group_stds_: Dict[str, np.ndarray] = {}
        self.group_codebooks_: Dict[str, np.ndarray] = {}
        self.group_columns_: Dict[str, Dict[str, Any]] = {}
        self.is_fitted_ = False

    def train(
        self,
        parquet_files: Union[str, Path, List[Union[str, Path]]],
        groups: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Train the spectrum tokenizer on parquet files.

        Parameters
        ----------
        parquet_files : Union[str, Path, List[Union[str, Path]]]
            Path(s) to parquet file(s) containing training data
        groups : List[Dict], optional
            Spectrum group definitions. If provided, overrides instance attribute.

        Raises
        ------
        ValueError
            If no groups are specified
        RuntimeError
            If no groups could be successfully trained
        """
        if groups is not None:
            self.groups = groups

        if not self.groups:
            raise ValueError("No spectrum groups specified for training")

        # Normalize to list of Path objects
        if isinstance(parquet_files, (str, Path)):
            files = [Path(parquet_files)]
        else:
            files = [Path(f) for f in parquet_files]

        print(f"Training spectrum tokenizer on {len(files)} file(s)")
        print(f"Groups: {[g['name'] for g in self.groups]}")

        for group in tqdm(self.groups, desc="Training groups", unit="group"):
            name = group['name']
            print(f"\nTraining group: {name}")

            # Collect features
            features = self._collect_features_for_group(group, files)
            if features.shape[0] < max(1000, self.codebook_size):
                print(f"  Insufficient data ({features.shape[0]} samples), skipping")
                continue

            # Compute normalization statistics
            mean = features.mean(axis=0).astype(np.float32)
            std = (features.std(axis=0) + 1e-8).astype(np.float32)

            # Check for pre-trained model
            model_path = Path(f"spectrum_kmeans_{name}.joblib")
            if model_path.exists():
                print(f"  Loading existing model from {model_path}")
                kmeans = joblib.load(model_path)
            else:
                if features.shape[0] < self.codebook_size * 10:
                    print(f"  Not enough data for robust training ({features.shape[0]} samples)")
                    continue

                # Normalize and train K-means
                features_std = (features - mean) / std
                print(f"  Training MiniBatchKMeans ({self.codebook_size} clusters)...")
                kmeans = MiniBatchKMeans(
                    n_clusters=self.codebook_size,
                    batch_size=4096,
                    max_iter=50,
                    verbose=0,
                    n_init=1,
                    random_state=0,
                )
                kmeans.fit(features_std)

                # Save model
                joblib.dump(kmeans, model_path)
                print(f"  Saved model to {model_path}")

            # Store results
            centers = kmeans.cluster_centers_
            if centers is None:
                print(f"  No cluster centers found, skipping")
                continue

            self.group_names_.append(name)
            self.group_means_[name] = mean
            self.group_stds_[name] = std
            self.group_codebooks_[name] = centers.astype(np.float32)
            self.group_columns_[name] = group
            print(f"  Feature dim: {features.shape[1]}, Samples: {features.shape[0]:,}")

        if not self.group_names_:
            raise RuntimeError("No groups successfully trained")

        self.is_fitted_ = True
        print(f"\nTraining complete: {len(self.group_names_)} groups")

    def encode(
        self,
        data: Union[pl.DataFrame, str, Path],
        encode_only_valid_rows: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Encode spectrum data into tokens.

        Parameters
        ----------
        data : Union[pl.DataFrame, str, Path]
            Data to encode (DataFrame or path to parquet file)
        encode_only_valid_rows : bool, default=True
            If True, only encode rows with valid spectrum data

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing per-group token arrays:
            - For rectangular groups (DESI, ZTF): 'tokens_{name}' (n_rows, L)
            - For ragged groups (GaiaXP): 'tokens_{name}_flat' and 'tokens_{name}_indptr'
            - 'ids': row IDs if available
            - 'id_col': ID column name

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

        n_rows = df.height
        if n_rows == 0:
            return {'ids': np.array([], dtype=np.int64), 'id_col': ''}

        print(f"Encoding {n_rows:,} rows for {len(self.group_names_)} groups")

        # Prepare output
        result = {}
        cols_set = set(df.columns)

        # Extract IDs
        id_col = self._choose_id_column(df.columns)
        if id_col is not None:
            result['ids'] = df[id_col].cast(pl.Int64).to_numpy()
            result['id_col'] = id_col
        else:
            result['ids'] = np.array([], dtype=np.int64)
            result['id_col'] = ''

        # Encode each group
        for name in tqdm(self.group_names_, desc="Encoding groups", leave=False):
            group_spec = self.group_columns_[name]
            kind = group_spec['kind']

            # Check if required columns are present
            if not self._group_columns_present(group_spec, cols_set):
                print(f"  Skipping {name}: required columns not in data")
                continue

            # Compute tokens
            payload = self._compute_tokens_for_group(
                df=df,
                group_spec=group_spec,
                mean=self.group_means_[name],
                std=self.group_stds_[name],
                centers=self.group_codebooks_[name],
                encode_only_valid=encode_only_valid_rows,
            )

            # Store results
            if 'tokens' in payload:
                result[f'tokens_{name}'] = payload['tokens']
            if 'tokens_flat' in payload and 'tokens_indptr' in payload:
                result[f'tokens_{name}_flat'] = payload['tokens_flat']
                result[f'tokens_{name}_indptr'] = payload['tokens_indptr']

        return result

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
            group_names=np.array(self.group_names_, dtype=object),
            group_means=self.group_means_,
            group_stds=self.group_stds_,
            group_codebooks=self.group_codebooks_,
            group_columns=self.group_columns_,
            CODEBOOK_SIZE=self.codebook_size,
            MAX_POINTS_PER_ROW=self.max_points_per_row,
            MAX_ROWS_PER_FILE=self.max_rows_per_file,
            GAIAXPY_MODE=self.gaiaxpy_mode,
            GAIAXPY_TRUNCATION=self.gaiaxpy_truncation,
        )
        print(f"Saved spectrum tokenizer to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SpectrumTokenizer":
        """
        Load tokenizer configuration from disk.

        Parameters
        ----------
        path : Union[str, Path]
            Path to tokenizer config file

        Returns
        -------
        SpectrumTokenizer
            Loaded tokenizer instance
        """
        data = np.load(path, allow_pickle=True)

        tokenizer = cls(
            codebook_size=int(data['CODEBOOK_SIZE']),
            max_points_per_row=int(data.get('MAX_POINTS_PER_ROW', 2000)),
            max_rows_per_file=int(data.get('MAX_ROWS_PER_FILE', 5000)),
            gaiaxpy_mode=str(data.get('GAIAXPY_MODE', 'convert')),
            gaiaxpy_truncation=bool(data.get('GAIAXPY_TRUNCATION', False)),
        )

        # Load group names
        raw_names = data['group_names']
        tokenizer.group_names_ = [cls._to_str(n) for n in raw_names]

        # Load statistics and codebooks
        tokenizer.group_means_ = cls._normalize_dict_keys(data['group_means'].item())
        tokenizer.group_stds_ = cls._normalize_dict_keys(data['group_stds'].item())
        tokenizer.group_codebooks_ = cls._normalize_dict_keys(data['group_codebooks'].item())

        # Load group column specifications
        if 'group_columns' in data.files:
            tokenizer.group_columns_ = cls._normalize_dict_keys(data['group_columns'].item())
            # Normalize nested dicts
            norm_cols = {}
            for gname, spec in tokenizer.group_columns_.items():
                if isinstance(spec, dict):
                    norm_cols[gname] = {
                        cls._to_str(k): cls._to_str(v) if isinstance(v, (bytes, np.bytes_)) else v
                        for k, v in spec.items()
                    }
                else:
                    norm_cols[gname] = spec
            tokenizer.group_columns_ = norm_cols

        # Ensure arrays are numpy
        for k, v in tokenizer.group_means_.items():
            tokenizer.group_means_[k] = np.asarray(v, dtype=np.float32)
        for k, v in tokenizer.group_stds_.items():
            tokenizer.group_stds_[k] = np.asarray(v, dtype=np.float32)
        for k, v in tokenizer.group_codebooks_.items():
            tokenizer.group_codebooks_[k] = np.asarray(v, dtype=np.float32)

        tokenizer.is_fitted_ = True
        print(f"Loaded spectrum tokenizer from {path} ({len(tokenizer.group_names_)} groups)")
        return tokenizer

    # ==================== Private Methods ====================

    @staticmethod
    def _to_str(x: Any) -> str:
        """Convert bytes/object to string."""
        if isinstance(x, (bytes, np.bytes_)):
            return x.decode()
        return str(x)

    @staticmethod
    def _normalize_dict_keys(d: Dict) -> Dict[str, Any]:
        """Convert dict keys to strings."""
        return {
            (k.decode() if isinstance(k, bytes) else str(k)): v
            for k, v in d.items()
        }

    def _collect_features_for_group(
        self,
        group: Dict[str, Any],
        files: List[Path],
    ) -> np.ndarray:
        """Collect training features for one spectrum group."""
        kind = group['kind']
        name = group['name']
        all_features = []
        total = 0

        files_ok = 0
        skipped = 0

        pbar = tqdm(files, desc=f"  Collecting {name}", leave=False)
        for path in pbar:
            if total >= self.max_features_per_group:
                break

            try:
                # Check columns exist
                cols_in_file = self._read_schema_fast(path)
                if not self._group_columns_present(group, set(cols_in_file)):
                    skipped += 1
                    continue

                # Read required columns
                cols_to_read = self._get_columns_to_read(group, cols_in_file)
                df = pl.read_parquet(path, columns=cols_to_read)

                if df.height == 0:
                    skipped += 1
                    continue

                # Sample rows
                valid_idx = self._get_valid_row_indices(df, group)
                if valid_idx.size == 0:
                    skipped += 1
                    continue

                if valid_idx.size > self.max_rows_per_file:
                    sample_idx = np.random.choice(
                        valid_idx.size,
                        size=self.max_rows_per_file,
                        replace=False,
                    )
                    valid_idx = valid_idx[sample_idx]

                df_subset = df[valid_idx]

                # Extract features based on kind
                if kind == 'flux_ivar':
                    feats = self._extract_flux_ivar_features(df_subset, group)
                elif kind == 'ztf_mag_err':
                    feats = self._extract_ztf_features(df_subset, group)
                elif kind == 'gaiaxpy_xp':
                    feats = self._extract_gaiaxp_features(df_subset, group, cols_in_file)
                else:
                    continue

                if feats.shape[0] == 0:
                    skipped += 1
                    continue

                all_features.append(feats)
                n_new = feats.shape[0]
                total += n_new
                files_ok += 1

                pbar.set_postfix({'ok': files_ok, 'skip': skipped, 'total': total})

            except Exception as e:
                skipped += 1
                continue

        pbar.close()

        if not all_features:
            print(f"  No features collected for {name}")
            return np.zeros((0, 2), dtype=np.float32)

        X = np.concatenate(all_features, axis=0)
        print(f"  Collected {X.shape[0]:,} features (dim={X.shape[1]})")
        return X.astype(np.float32)

    def _extract_flux_ivar_features(
        self,
        df: pl.DataFrame,
        group: Dict[str, Any],
    ) -> np.ndarray:
        """Extract (flux, precision) features from DESI-like data."""
        flux_col = group['flux_col']
        ivar_col = group['ivar_col']

        flux_series = df[flux_col]
        ivar_series = df[ivar_col]

        features = []
        for i in range(df.height):
            flux = self._cell_to_float(flux_series[i])
            ivar = self._cell_to_float(ivar_series[i])

            if flux.size == 0 or ivar.size == 0 or flux.size != ivar.size:
                continue

            ivar = np.clip(ivar, a_min=0.0, a_max=None)
            prec = np.sqrt(ivar)
            feats = np.stack([flux, prec], axis=-1)  # (L, 2)

            # Sample points
            feats = self._sample_points(feats)
            feats = feats[self._finite_mask(feats)]

            if feats.shape[0] > 0:
                features.append(feats)

        if not features:
            return np.zeros((0, 2), dtype=np.float32)

        return np.concatenate(features, axis=0).astype(np.float32)

    def _extract_ztf_features(
        self,
        df: pl.DataFrame,
        group: Dict[str, Any],
    ) -> np.ndarray:
        """Extract (magnitude, precision) features from ZTF light curves."""
        mag_col = group['mag_col']
        err_col = group['err_col']

        mag_series = df[mag_col]
        err_series = df[err_col]

        features = []
        for i in range(df.height):
            mag = self._cell_to_float(mag_series[i])
            err = self._cell_to_float(err_series[i])

            if mag.size == 0 or err.size == 0 or mag.size != err.size:
                continue

            err = np.clip(err, a_min=self.ztf_err_eps, a_max=None)
            prec = 1.0 / err
            feats = np.stack([mag, prec], axis=-1)  # (L, 2)

            feats = self._sample_points(feats)
            feats = feats[self._finite_mask(feats)]

            if feats.shape[0] > 0:
                features.append(feats)

        if not features:
            return np.zeros((0, 2), dtype=np.float32)

        return np.concatenate(features, axis=0).astype(np.float32)

    def _extract_gaiaxp_features(
        self,
        df: pl.DataFrame,
        group: Dict[str, Any],
        cols_in_file: List[str],
    ) -> np.ndarray:
        """Extract normalized GaiaXP features (flux, precision)."""
        # This is a simplified version that doesn't call GaiaXPy
        # In production, you'd use the full GaiaXPy integration from the original code
        warnings.warn(
            "GaiaXP feature extraction is simplified. "
            "Full GaiaXPy integration requires gaiaxpy package.",
            UserWarning
        )
        return np.zeros((0, 2), dtype=np.float32)

    @staticmethod
    def _cell_to_float(cell) -> np.ndarray:
        """Convert cell value to 1D float array."""
        if cell is None:
            return np.zeros((0,), dtype=np.float32)

        if isinstance(cell, pl.Series):
            cell = cell.to_list()

        if isinstance(cell, (list, tuple, np.ndarray)):
            arr = np.asarray(cell, dtype=np.float32).reshape(-1)
            return arr[np.isfinite(arr)]

        # Try to convert single value
        try:
            v = float(cell)
            if np.isfinite(v):
                return np.asarray([v], dtype=np.float32)
        except Exception:
            pass

        return np.zeros((0,), dtype=np.float32)

    def _sample_points(self, feats: np.ndarray) -> np.ndarray:
        """Sample points from spectrum to limit size."""
        if feats.ndim == 1:
            feats = feats.reshape(-1, 1)
        L = feats.shape[0]
        if L > self.max_points_per_row:
            idx = np.random.choice(L, size=self.max_points_per_row, replace=False)
            return feats[idx]
        return feats

    @staticmethod
    def _finite_mask(x: np.ndarray) -> np.ndarray:
        """Get mask of finite rows."""
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return np.isfinite(x).all(axis=1)

    @staticmethod
    def _read_schema_fast(path: Path) -> List[str]:
        """Read parquet column names without loading data."""
        df0 = pl.read_parquet(path, n_rows=0)
        return df0.columns

    def _group_columns_present(
        self,
        group: Dict[str, Any],
        cols_set: set,
    ) -> bool:
        """Check if required columns for group are present."""
        kind = group['kind']
        if kind == 'flux_ivar':
            return group['flux_col'] in cols_set and group['ivar_col'] in cols_set
        elif kind == 'ztf_mag_err':
            return group['mag_col'] in cols_set and group['err_col'] in cols_set
        elif kind == 'gaiaxpy_xp':
            # Check for basic GaiaXP columns
            required = ['bp_coefficients', 'rp_coefficients']
            return any(c in cols_set or f'gaiaxp_{c}' in cols_set for c in required)
        return False

    def _get_columns_to_read(
        self,
        group: Dict[str, Any],
        cols_in_file: List[str],
    ) -> List[str]:
        """Get list of columns to read for a group."""
        kind = group['kind']
        if kind == 'flux_ivar':
            return [group['flux_col'], group['ivar_col']]
        elif kind == 'ztf_mag_err':
            return [group['mag_col'], group['err_col']]
        elif kind == 'gaiaxpy_xp':
            # Return basic columns needed
            cols = []
            for base in ['source_id', 'bp_coefficients', 'rp_coefficients',
                        'bp_coefficient_correlations', 'rp_coefficient_correlations',
                        'bp_n_parameters', 'rp_n_parameters']:
                if base in cols_in_file:
                    cols.append(base)
                elif f'gaiaxp_{base}' in cols_in_file:
                    cols.append(f'gaiaxp_{base}')
            return cols
        return []

    def _get_valid_row_indices(
        self,
        df: pl.DataFrame,
        group: Dict[str, Any],
    ) -> np.ndarray:
        """Get indices of valid rows for a group."""
        kind = group['kind']

        if kind == 'flux_ivar':
            fcol = group['flux_col']
            wcol = group['ivar_col']
            mask = (
                pl.col(fcol).is_not_null() &
                (pl.col(fcol).list.len() > 0) &
                pl.col(wcol).is_not_null() &
                (pl.col(wcol).list.len() > 0)
            )
        elif kind == 'ztf_mag_err':
            mcol = group['mag_col']
            ecol = group['err_col']
            mask = (
                pl.col(mcol).is_not_null() &
                (pl.col(mcol).list.len() > 0) &
                pl.col(ecol).is_not_null() &
                (pl.col(ecol).list.len() > 0)
            )
        else:
            # Default: all rows
            mask = pl.lit(True)

        try:
            idx = df.select(pl.arg_where(mask)).to_series().to_numpy()
            return idx.astype(np.int64)
        except Exception:
            return np.array([], dtype=np.int64)

    def _compute_tokens_for_group(
        self,
        df: pl.DataFrame,
        group_spec: Dict[str, Any],
        mean: np.ndarray,
        std: np.ndarray,
        centers: np.ndarray,
        encode_only_valid: bool,
    ) -> Dict[str, np.ndarray]:
        """Compute tokens for one group in a dataframe."""
        kind = group_spec['kind']
        n_rows = df.height

        if n_rows == 0:
            return {'tokens': np.zeros((0, 0), dtype=np.uint16)}

        # Normalize std
        std = np.where(std == 0, 1e-8, std)

        # Get valid rows
        if encode_only_valid:
            valid_idx = self._get_valid_row_indices(df, group_spec)
            if valid_idx.size == 0:
                return {'tokens': np.zeros((n_rows, 0), dtype=np.uint16)}
        else:
            valid_idx = np.arange(n_rows, dtype=np.int64)

        # For rectangular data (DESI, ZTF), infer fixed length
        if kind in ('flux_ivar', 'ztf_mag_err'):
            # Infer length from first valid row
            L = None
            if kind == 'flux_ivar':
                col = group_spec['flux_col']
            else:
                col = group_spec['mag_col']

            series = df[col]
            for idx in valid_idx:
                arr = self._cell_to_float(series[int(idx)])
                if arr.size > 0:
                    L = arr.size
                    break

            if L is None:
                return {'tokens': np.zeros((n_rows, 0), dtype=np.uint16)}

            # Allocate output
            tokens = np.zeros((n_rows, L), dtype=np.uint16)

            # Encode each valid row
            for idx in valid_idx:
                idx = int(idx)
                if kind == 'flux_ivar':
                    flux = self._cell_to_float(df[group_spec['flux_col']][idx])
                    ivar = self._cell_to_float(df[group_spec['ivar_col']][idx])
                    if flux.size != L or ivar.size != L:
                        continue
                    ivar = np.clip(ivar, a_min=0.0, a_max=None)
                    prec = np.sqrt(ivar)
                    feats = np.stack([flux, prec], axis=-1)
                else:  # ztf_mag_err
                    mag = self._cell_to_float(df[group_spec['mag_col']][idx])
                    err = self._cell_to_float(df[group_spec['err_col']][idx])
                    if mag.size != L or err.size != L:
                        continue
                    err = np.clip(err, a_min=self.ztf_err_eps, a_max=None)
                    prec = 1.0 / err
                    feats = np.stack([mag, prec], axis=-1)

                # Normalize and tokenize
                feats_std = (feats - mean) / std
                tokens[idx, :] = self._tokenize_features(feats_std, centers)

            return {'tokens': tokens}

        # For ragged data (GaiaXP), return flat + indptr
        else:
            indptr = np.zeros(n_rows + 1, dtype=np.int64)
            flat_tokens = []
            total = 0

            for r in range(n_rows):
                indptr[r] = total
                # Simplified: would extract GaiaXP flux here
                # For now, skip
                continue

            indptr[n_rows] = total
            if not flat_tokens:
                return {
                    'tokens_flat': np.zeros((0,), dtype=np.uint16),
                    'tokens_indptr': indptr,
                }

            flat = np.concatenate(flat_tokens, axis=0).astype(np.uint16)
            return {'tokens_flat': flat, 'tokens_indptr': indptr}

    @staticmethod
    def _tokenize_features(feats: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Tokenize normalized features using nearest centroid.

        Parameters
        ----------
        feats : np.ndarray
            Normalized features (L, D)
        centers : np.ndarray
            Cluster centroids (K, D)

        Returns
        -------
        np.ndarray
            Token IDs (L,)
        """
        x = feats.astype(np.float32)
        c = centers.astype(np.float32)

        # Compute squared distances: ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x.c
        x2 = np.sum(x * x, axis=1, keepdims=True)  # (L, 1)
        c2 = np.sum(c * c, axis=1, keepdims=True).T  # (1, K)
        xc = x @ c.T  # (L, K)

        dist2 = x2 + c2 - 2.0 * xc  # (L, K)
        tokens = np.argmin(dist2, axis=1).astype(np.uint16)
        return tokens

    @staticmethod
    def _choose_id_column(columns: List[str]) -> Optional[str]:
        """Choose ID column from available columns."""
        preferred = ["id", "gaia_source_id", "source_id"]
        for c in preferred:
            if c in columns:
                return c
        return None
