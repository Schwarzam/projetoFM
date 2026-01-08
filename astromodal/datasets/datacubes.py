from pathlib import Path
from typing import List, Tuple, Optional
import random

import logpool


def load_datacube_files(
    datacubes_paths: str,
    train_val_split: float = 0.9,
    nfiles_subsample: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[List[Path], List[Path]]:
    """
    Load datacube parquet files and split into train/validation sets.

    Parameters
    ----------
    datacubes_paths : str
        Path pattern to datacubes (e.g. "/data/datacube_*.parquet")
    train_val_split : float, default=0.9
        Fraction of files used for training
    nfiles_subsample : int or None, default=None
        If set, randomly subsample this many files from all matches
    seed : int or None, default=None
        Random seed for reproducible subsampling

    Returns
    -------
    (train_files, val_files) : Tuple[List[Path], List[Path]]
        Lists of parquet file paths
    """
    datacube_dir = Path(datacubes_paths).parent
    pattern = Path(datacubes_paths).name

    parquet_files = sorted(datacube_dir.glob(pattern))
    logpool.info(f"Found {len(parquet_files)} datacube files")

    if not parquet_files:
        raise FileNotFoundError(f"No files matched pattern: {datacubes_paths}")

    # Optional subsampling
    if nfiles_subsample is not None and nfiles_subsample < len(parquet_files):
        if seed is not None:
            random.seed(seed)

        parquet_files = random.sample(parquet_files, nfiles_subsample)
        parquet_files = sorted(parquet_files)
        logpool.info(f"Subsampled to {len(parquet_files)} files")

    # Train / validation split
    n_train = int(train_val_split * len(parquet_files))
    train_files = parquet_files[:n_train]
    val_files = parquet_files[n_train:]

    logpool.info(f"Training files: {len(train_files)}")
    logpool.info(f"Validation files: {len(val_files)}")

    return train_files, val_files