"""
File discovery and path utilities.
"""

from __future__ import annotations
from pathlib import Path
from typing import List


def discover_fields_from_datacubes(tmpl: str) -> List[str]:
    """
    Discover field names from datacube file template.

    Parameters
    ----------
    tmpl : str
        Template path with {field} placeholder, e.g.
        "/path/to/datacube_{field}.parquet"

    Returns
    -------
    List[str]
        Sorted list of discovered field names

    Examples
    --------
    >>> discover_fields_from_datacubes("/data/datacube_{field}.parquet")
    ['HYDRA-0011', 'HYDRA-0012', 'SPLUS-n18s07', ...]
    """
    pat = tmpl.replace("{field}", "*")
    files = sorted(Path(pat).parent.glob(Path(pat).name))
    out = []
    for f in files:
        name = f.name
        # Extract field name between "datacube_" and ".parquet"
        if name.startswith("datacube_") and name.endswith(".parquet"):
            out.append(name[len("datacube_"):-len(".parquet")])
    return sorted(out)


def discover_fields_from_template(tmpl: str, prefix: str, suffix: str) -> List[str]:
    """
    Generic field discovery from any file template.

    Parameters
    ----------
    tmpl : str
        Template path with {field} placeholder
    prefix : str
        Filename prefix before field name
    suffix : str
        Filename suffix after field name (including extension)

    Returns
    -------
    List[str]
        Sorted list of discovered field names
    """
    pat = tmpl.replace("{field}", "*")
    files = sorted(Path(pat).parent.glob(Path(pat).name))
    out = []
    for f in files:
        name = f.name
        if name.startswith(prefix) and name.endswith(suffix):
            out.append(name[len(prefix):-len(suffix)])
    return sorted(out)
