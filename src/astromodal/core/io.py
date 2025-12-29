"""
I/O utilities for loading codebooks, centroids, and tokenizer configs.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pickle

import numpy as np
import joblib


def load_codebook(
    path: str,
    key_candidates: Tuple[str, ...] = ("centroids", "codebook", "centers", "cluster_centers_", "clusters")
) -> np.ndarray:
    """
    Load a codebook/centroids array from various formats.

    Supports:
      - .npz files (looks for keys like centroids/codebook/centers)
      - .joblib files (sklearn KMeans or dict-like objects)
      - .pkl/.pickle files (sklearn KMeans or dict-like objects)

    Parameters
    ----------
    path : str
        Path to codebook file
    key_candidates : Tuple[str, ...], default=("centroids", "codebook", "centers", "cluster_centers_", "clusters")
        Keys to try when loading from dict-like or npz formats

    Returns
    -------
    np.ndarray
        Codebook array with shape (K, D) as float32

    Raises
    ------
    FileNotFoundError
        If path doesn't exist
    KeyError
        If no valid key found in npz/dict
    ValueError
        If unsupported format or wrong array shape
    TypeError
        If joblib/pickle object type not supported
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing: {path}")

    suffix = p.suffix.lower()

    # ---- NPZ ----
    if suffix == ".npz":
        z = np.load(str(p), allow_pickle=True)
        try:
            for k in key_candidates:
                if k in z.files:
                    arr = np.asarray(z[k])
                    if arr.ndim != 2:
                        raise ValueError(f"Key {k} in {path} has shape {arr.shape}, expected (K,D)")
                    return arr.astype(np.float32)
            raise KeyError(f"No key in {path} among {key_candidates}. Keys: {z.files}")
        finally:
            z.close()

    # ---- JOBLIB ----
    if suffix == ".joblib":
        obj = joblib.load(str(p))

        # sklearn KMeans-like
        if hasattr(obj, "cluster_centers_"):
            arr = np.asarray(obj.cluster_centers_)
            if arr.ndim != 2:
                raise ValueError(f"cluster_centers_ in {path} has shape {arr.shape}, expected (K,D)")
            return arr.astype(np.float32)

        # dict-like
        if isinstance(obj, dict):
            for k in key_candidates:
                if k in obj:
                    arr = np.asarray(obj[k])
                    if arr.ndim != 2:
                        raise ValueError(f"Key {k} in {path} has shape {arr.shape}, expected (K,D)")
                    return arr.astype(np.float32)

        raise TypeError(f"Don't know how to extract centroids from joblib object type={type(obj)} in {path}")

    # ---- PKL ----
    if suffix in (".pkl", ".pickle"):
        with open(str(p), "rb") as f:
            obj = pickle.load(f)

        if hasattr(obj, "cluster_centers_"):
            arr = np.asarray(obj.cluster_centers_)
            if arr.ndim != 2:
                raise ValueError(f"cluster_centers_ in {path} has shape {arr.shape}, expected (K,D)")
            return arr.astype(np.float32)

        if isinstance(obj, dict):
            for k in key_candidates:
                if k in obj:
                    arr = np.asarray(obj[k])
                    if arr.ndim != 2:
                        raise ValueError(f"Key {k} in {path} has shape {arr.shape}, expected (K,D)")
                    return arr.astype(np.float32)

        raise TypeError(f"Don't know how to extract centroids from pickle object type={type(obj)} in {path}")

    raise ValueError(f"Unsupported codebook format: {path} (expected .npz/.joblib/.pkl)")


def load_npz_key(npz_path: str, key_candidates: Tuple[str, ...]) -> np.ndarray:
    """
    Load a specific key from an NPZ file.

    Parameters
    ----------
    npz_path : str
        Path to .npz file
    key_candidates : Tuple[str, ...]
        Keys to try in order

    Returns
    -------
    np.ndarray
        Array from the first matching key

    Raises
    ------
    FileNotFoundError
        If npz_path doesn't exist
    KeyError
        If no key in key_candidates found
    """
    if not Path(npz_path).exists():
        raise FileNotFoundError(f"Missing: {npz_path}")
    z = np.load(npz_path, allow_pickle=True)
    try:
        for k in key_candidates:
            if k in z.files:
                arr = z[k]
                return np.asarray(arr)
    finally:
        z.close()
    raise KeyError(f"No key in {npz_path} among {key_candidates}. Keys are: {z.files}")


def load_scalar_centers(
    npz_path: str,
    key_candidates: Tuple[str, ...] = ("bin_centers", "centers", "quantile_centers", "centroids")
) -> np.ndarray:
    """
    Load scalar bin centers from tokenizer config.

    Parameters
    ----------
    npz_path : str
        Path to scalar tokenizer config .npz file
    key_candidates : Tuple[str, ...], default=("bin_centers", "centers", "quantile_centers", "centroids")
        Keys to try for scalar centers

    Returns
    -------
    np.ndarray
        1D float32 array of scalar bin centers

    Raises
    ------
    FileNotFoundError
        If npz_path doesn't exist
    KeyError
        If no key in key_candidates found
    """
    if not Path(npz_path).exists():
        raise FileNotFoundError(f"Missing: {npz_path}")
    z = np.load(npz_path, allow_pickle=True)
    try:
        for k in key_candidates:
            if k in z.files:
                c = np.asarray(z[k]).astype(np.float32)
                c = c.reshape(-1)
                return c
    finally:
        z.close()
    raise KeyError(f"No scalar centers key found in {npz_path}. Tried {key_candidates}.")
