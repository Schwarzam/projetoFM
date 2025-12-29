#!/usr/bin/env python3
"""
Train spectrum tokenizers on datacube parquet files.
"""

from pathlib import Path
from astromodal.config import load_config
from astromodal.tokenizers import SpectrumTokenizer

def main():
    # Load configuration
    config = load_config()

    print("=" * 60)
    print("Training Spectrum Tokenizer")
    print("=" * 60)

    # Discover datacube parquet files
    print(f"\nDiscovering datacube files from: {config.paths.datacube_template}")
    datacube_dir = Path(config.paths.datacube_template).parent
    pattern = Path(config.paths.datacube_template).name.replace("{field}", "*")
    parquet_files = sorted(datacube_dir.glob(pattern))

    if not parquet_files:
        raise RuntimeError(f"No parquet files found matching: {config.paths.datacube_template}")

    print(f"Found {len(parquet_files)} datacube files")
    print(f"Spectrum groups: {config.tokenizers.spectrum.groups}")

    # Define spectrum groups (DESI + Gaia XP)
    # Valid kinds: 'flux_ivar', 'ztf_mag_err', 'gaiaxpy_xp'
    groups = []
    for group_name in config.tokenizers.spectrum.groups:
        if group_name.startswith("desi"):
            arm = group_name.split("_")[1]  # e.g., "desi_b" -> "b"
            groups.append({
                "name": group_name,
                "kind": "flux_ivar",  # DESI uses flux and inverse variance
                "flux_col": f"desi_flux_{arm}",
                "ivar_col": f"desi_ivar_{arm}",
            })
        elif group_name.startswith("gaiaxp"):
            groups.append({
                "name": group_name,
                "kind": "gaiaxpy_xp",  # Gaia XP uses coefficients
            })

    # Initialize tokenizer
    tokenizer = SpectrumTokenizer(
        codebook_size=config.tokenizers.spectrum.codebook_size,
    )

    # Train on datacubes (creates separate codebooks for each spectrum group)
    print("\nTraining tokenizer (this may take 1-3 hours)...")
    tokenizer.train(parquet_files, groups=groups)

    # Save configuration
    output_path = config.paths.spectrum_tokenizer_config
    print(f"\nSaving tokenizer config to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_path)

    print("\n✓ Spectrum tokenizer training complete!")
    print(f"   Config saved to: {output_path}")

if __name__ == "__main__":
    main()
