#!/usr/bin/env python3
"""
Standalone VISTA cone-search script (no external project imports).

Usage:
    python download.py \
        --ra 266.41683 \
        --dec -29.00781 \
        --radius 1.0 \
        --table VVVDR2.vvvSource \
        --out vvv_cone.csv
"""

import argparse
import io
import requests
import pandas as pd


# ---------------------------------------------------------
# TAP SYNC HELPER
# ---------------------------------------------------------
def tap_sync_query(
    base_url: str,
    adql: str,
    response_format: str = "csv",
) -> bytes:
    """
    Perform a synchronous TAP query.

    Parameters
    ----------
    base_url : str
        Base TAP service URL (without /sync)
    adql : str
        ADQL SQL query
    response_format : str
        TAP output format (csv recommended)

    Returns
    -------
    bytes
        Raw response content
    """
    sync_url = base_url.rstrip("/") + "/sync"

    try:
        r = requests.post(
            sync_url,
            data={
                "REQUEST": "doQuery",
                "LANG": "ADQL",
                "FORMAT": response_format,
                "QUERY": adql,
            },
            timeout=1800,
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error contacting TAP service at {sync_url}: {e}") from e

    if r.status_code != 200:
        raise RuntimeError(
            f"TAP sync query failed (status={r.status_code}):\n{r.text[:1000]}"
        )

    return r.content


# ---------------------------------------------------------
# ADQL BUILDER
# ---------------------------------------------------------
def build_cone_adql(
    table: str,
    ra: float,
    dec: float,
    radius_deg: float,
    ra_col: str = "ra",
    dec_col: str = "dec",
) -> str:
    """
    Build an ADQL cone-search query.

    Parameters
    ----------
    table : str
        Fully qualified table name (e.g. 'VVVDR2.vvvSource')
    ra, dec : float
        Cone center (deg, ICRS)
    radius_deg : float
        Cone radius in degrees
    ra_col, dec_col : str
        Column names for RA/Dec in the chosen table

    Returns
    -------
    str
        ADQL query string
    """
    
    return f"""
    SELECT
    sourceid,
    ra2000,
    dec2000,

    -- Y band
    yapermag3,
    yapermag3err,
    yapermag4,
    yapermag4err,
    yapermag6,
    yapermag6err,

    -- J band
    japermag3,
    japermag3err,
    japermag4,
    japermag4err,
    japermag6,
    japermag6err,

    -- H band
    hapermag3,
    hapermag3err,
    hapermag4,
    hapermag4err,
    hapermag6,
    hapermag6err,

    -- Ks band
    ksapermag3,
    ksapermag3err,
    ksapermag4,
    ksapermag4err,
    ksapermag6,
    ksapermag6err

    FROM {table}

    WHERE ra2000 > {ra} - {radius_deg}
    AND ra2000 < {ra} + {radius_deg}
    AND dec2000 > {dec} - {radius_deg}
    AND dec2000 < {dec} + {radius_deg};
    """


# ---------------------------------------------------------
# MAIN CONE-SEARCH FUNCTION
# ---------------------------------------------------------
def vista_cone_search(
    ra: float,
    dec: float,
    radius_deg: float = 1.0,
    table: str = "vhs_dr5.vhs_cat_v3",
    out: str | None = None,
    ra_col: str = "ra",
    dec_col: str = "dec",
) -> pd.DataFrame:
    """
    Performs a cone search on the VISTA Science Archive TAP.

    Parameters
    ----------
    ra, dec : float
        Cone center in degrees
    radius_deg : float
        Cone radius in degrees (default 1.0)
    table : str
        Fully qualified VISTA table
        (default 'VVVDR2.vvvSource' to avoid ambiguity)
    out : str or None
        If provided, saves result to CSV
    ra_col, dec_col : str
        Column names for RA/Dec in table

    Returns
    -------
    DataFrame
        Pandas DataFrame with the results
    """
    # IMPORTANT: VSA TAP is HTTP, not HTTPS
    VSA_TAP_BASE_URL = "https://datalab.noirlab.edu/tap"

    adql = build_cone_adql(
        table=table,
        ra=ra,
        dec=dec,
        radius_deg=radius_deg,
        ra_col=ra_col,
        dec_col=dec_col,
    )

    print(adql)
    raw = tap_sync_query(
        base_url=VSA_TAP_BASE_URL,
        adql=adql,
        response_format="csv",
    )

    df = pd.read_csv(io.BytesIO(raw))

    if out and len(df) > 0:
        df.to_parquet(out)

    return df


def main():
    import pandas as pd
    df = pd.read_csv("../dr6_list.csv")
    outfolder = "/home/astrodados4/downloads/vista"
    
    import os
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    
    for _, row in df.iterrows():
        
        if "SPLUS-b" in row['field'] or "SPLUS-d" in row['field'] or "MC" in row['field']:
            continue
        
        print(row['field'], row['ra'], row['dec'])
        df = vista_cone_search(
            ra=row['ra'],
            dec=row['dec'],
            radius_deg=0.8,
            out=os.path.join(outfolder, f"{row['field']}.parquet"),
            ra_col="ra",
            dec_col="dec",
        )
        print(f"Downloaded {len(df)} sources.")


if __name__ == "__main__":
    main()