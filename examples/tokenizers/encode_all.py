#!/usr/bin/env python3
"""
Encode all datacubes to tokens for all modalities.

This script encodes:
- Image cutouts → image tokens + latents
- Scalar features → scalar tokens
- Spectra → spectrum tokens
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set GPU

from astromodal.config import load_config
from astromodal.tokenizers import ScalarTokenizer, SpectrumTokenizer, ImageTokenizer
from astromodal.core import discover_fields_from_datacubes

def main():
    # Load configuration
    config = load_config()

    print("=" * 60)
    print("Encoding All Datacubes to Tokens")
    print("=" * 60)

    # Discover fields
    fields = discover_fields_from_datacubes(config.paths.datacube_template)
    print(f"\nFound {len(fields)} fields to encode")

    # Load tokenizers
    print("\n[1/3] Loading scalar tokenizer...")
    scalar_tokenizer = ScalarTokenizer(config)
    scalar_tokenizer.load(config.paths.scalar_tokenizer_config)

    print("[2/3] Loading spectrum tokenizer...")
    spectrum_tokenizer = SpectrumTokenizer(config)
    spectrum_tokenizer.load(config.paths.spectrum_tokenizer_config)

    print("[3/3] Loading image tokenizer...")
    image_tokenizer = ImageTokenizer(config)
    image_tokenizer.load(config.paths.image_autoencoder)

    # Encode each field
    for i, field in enumerate(fields, 1):
        print(f"\n{'=' * 60}")
        print(f"Encoding field {i}/{len(fields)}: {field}")
        print(f"{'=' * 60}")

        datacube_path = config.paths.datacube_template.format(field=field)

        # Encode scalars
        print("  → Encoding scalars...")
        scalar_output = config.paths.scalar_tokens_template.format(field=field)
        scalar_tokenizer.encode(datacube_path, scalar_output)

        # Encode spectra
        print("  → Encoding spectra...")
        spectrum_output = config.paths.spectrum_tokens_template.format(field=field)
        spectrum_tokenizer.encode(datacube_path, spectrum_output)

        # Encode images
        print("  → Encoding images...")
        image_tokens_output = config.paths.image_tokens_template.format(field=field)
        image_latents_output = config.paths.image_latents_template.format(field=field)
        image_tokenizer.encode(datacube_path, image_tokens_output, image_latents_output)

    print("\n" + "=" * 60)
    print("✓ All fields encoded successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
