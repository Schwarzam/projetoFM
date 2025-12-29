"""
ID normalization utilities for astronomy catalogs.
"""

from __future__ import annotations
from typing import Optional
import numpy as np


def norm_splus_id(x) -> str:
    """
    Normalize S-PLUS object IDs to strings.

    Parameters
    ----------
    x : Any
        Raw S-PLUS ID (can be bytes, string, float, None)

    Returns
    -------
    str
        Normalized string ID, or empty string if None/NaN
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def norm_gaia_id(x) -> Optional[int]:
    """
    Normalize Gaia source IDs to integers.

    Parameters
    ----------
    x : Any
        Raw Gaia source ID (can be int, float, string, None)

    Returns
    -------
    Optional[int]
        Normalized integer ID, or None if invalid/None/NaN
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        return int(x)
    except Exception:
        return None
