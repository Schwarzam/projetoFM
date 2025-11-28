# projetoFM

# S-PLUS Hypercube Overview

Each hypercube row corresponds to **one S-PLUS source** in a given field from `dr6_list.csv`.

## Surveys and Data Included

- **S-PLUS catalog (IDR6)**
  - Full dual-catalog row for each detection
  - Includes `ra`, `dec`, magnitudes, morphology, flags, etc.

- **S-PLUS image cutouts**
  - 96×96 pixel cutouts in 12 bands:
    - `R, I, G, U, Z, F378, F395, F410, F430, F515, F660, F861`
  - Stored as flattened `float32` arrays:
    - `splus_cut_<BAND>`

- **Gaia DR3**
  - Cross-match within ~1 arcsec
  - Full Gaia row for each match, prefixed with:
    - `gaia_<column>`
  - J2000-propagated coordinates:
    - `gaia_ra_j2000`, `gaia_dec_j2000`

- **Gaia XP spectra (BP/RP)**
  - XP metadata for matched Gaia sources
  - All columns prefixed with:
    - `gaiaxp_<column>`

- **ZTF HiPS light curves**
  - Time-series photometry for filters g, r, i:
    - `ztf_mag_g`, `ztf_magerr_g`, `ztf_hmjd_g`, `ztf_nepochs_g`
    - `ztf_mag_r`, `ztf_magerr_r`, `ztf_hmjd_r`, `ztf_nepochs_r`
    - `ztf_mag_i`, `ztf_magerr_i`, `ztf_hmjd_i`, `ztf_nepochs_i`
  - Representative object id:
    - `ztf_objectid`

- **DESI coadded spectra**
  - Fiber metadata from FIBERMAP, prefixed with:
    - `desi_<column>`
  - Full 1D spectra per arm (if present):
    - `desi_wave_b`, `desi_flux_b`, `desi_ivar_b`
    - `desi_wave_r`, `desi_flux_r`, `desi_ivar_r`
    - `desi_wave_z`, `desi_flux_z`, `desi_ivar_z`
  - Origin file path:
    - `desi_desi_file`

- **APOGEE DR17**
  - Matched stellar parameters and abundances:
    - `apogee_<column>`

- **External spectroscopic redshifts (spec-z compilation)**
  - Matched spec-z and metadata:
    - `specz_<column>`

- **VISTA NIR photometry**
  - Per-field VISTA catalog, cross-matched in RA/DEC:
    - `vista_<column>`

## Output Format

For each S-PLUS field in `dr6_list.csv` the script writes:

```text
/home/astrodados4/downloads/hypercube/datacube_<FIELD>.parquet
```

Each `datacube_<FIELD>.parquet` contains all columns above, one row per S-PLUS source in that field.
