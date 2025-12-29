#!/usr/bin/env python3
"""
Train scalar tokenizer on datacube parquet files.
"""

from pathlib import Path
from astromodal.config import load_config
from astromodal.tokenizers import ScalarTokenizer

def main():
    # Load configuration
    config = load_config()

    print("=" * 60)
    print("Training Scalar Tokenizer")
    print("=" * 60)

    # Discover datacube parquet files
    print(f"\nDiscovering datacube files from: {config.paths.datacube_template}")
    datacube_dir = Path(config.paths.datacube_template).parent
    pattern = Path(config.paths.datacube_template).name.replace("{field}", "*")
    parquet_files = sorted(datacube_dir.glob(pattern))

    if not parquet_files:
        raise RuntimeError(f"No parquet files found matching: {config.paths.datacube_template}")

    print(f"Found {len(parquet_files)} datacube files")

    # Define scalar columns to tokenize
    splus_bands = config.data.splus_bands
    scalar_columns = [
        "ellipticity_det",
        "elongation_det",
        "a_pixel_det",
        "b_pixel_det",
        "theta_det",
        "fwhm_n_det",
        *[f"mag_pstotal_{b}" for b in splus_bands],
        *[f"err_mag_pstotal_{b}" for b in splus_bands],
        "gaia_parallax",
        "gaia_parallax_error",
        "gaia_pmra",
        "gaia_pmdec",
        "gaia_pmra_error",
        "gaia_pmdec_error",
        "gaia_phot_bp_mean_flux",
        "gaia_phot_rp_mean_flux",
        "gaia_phot_g_mean_flux",
        "gaia_phot_bp_mean_flux_error",
        "gaia_phot_rp_mean_flux_error",
        "gaia_phot_g_mean_flux_error",
        "gaia_teff_gspphot",
        "gaia_logg_gspphot",
        "gaia_mh_gspphot",
        "specz_z",
        "specz_e_z",
        "vista_yapermag6",
        "vista_yapermag6err",
        "vista_japermag6",
        "vista_japermag6err",
        "vista_hapermag6",
        "vista_hapermag6err",
        "vista_ksapermag6",
        "vista_ksapermag6err",
    ]

    print(f"Tokenizing {len(scalar_columns)} scalar columns")

    # Initialize tokenizer with config parameters
    tokenizer = ScalarTokenizer(
        n_bins=config.tokenizers.scalar.n_bins,
        mag_filter_col=config.data.mag_filter['column'],
        mag_min=config.data.mag_filter['min'],
        mag_max=config.data.mag_filter['max'],
        magerr_max=config.data.magerr_max,
        max_values_per_col=config.tokenizers.scalar.max_values_per_col,
        max_values_per_file_per_col=config.tokenizers.scalar.max_values_per_file_per_col,
    )

    # Train on datacubes
    print("\nTraining tokenizer (this may take 10-30 minutes)...")
    tokenizer.train(parquet_files, scalar_columns=scalar_columns)

    # Save configuration
    output_path = config.paths.scalar_tokenizer_config
    print(f"\nSaving tokenizer config to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_path)

    print("\n✓ Scalar tokenizer training complete!")
    print(f"   Config saved to: {output_path}")

if __name__ == "__main__":
    main()
