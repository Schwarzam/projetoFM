#!/usr/bin/env python3
"""
Train image tokenizer (autoencoder + codebook) on S-PLUS cutouts.

NOTE: Image tokenizer training uses the existing training scripts.

For actual training, use:
  1. Train autoencoder: python image_tokenizers/train_image.py
  2. Train codebook: python image_tokenizers/run_codebook.py
  3. Encode images: python image_tokenizers/run_image_inference.py

This script checks if the models are already trained.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set GPU

from pathlib import Path
from astromodal.config import load_config
from astromodal.tokenizers import ImageTokenizer

def main():
    # Load configuration
    config = load_config()

    print("=" * 60)
    print("Image Tokenizer - Check Status")
    print("=" * 60)
    print()
    print("Checking if image tokenizer models are trained...")
    print()

    # Check if models exist
    autoencoder_path = Path(config.paths.image_autoencoder)
    codebook_path = Path(config.paths.image_codebook)

    if not autoencoder_path.exists():
        print(f"❌ Autoencoder not found: {autoencoder_path}")
        print()
        print("Train the autoencoder using:")
        print("    export CUDA_VISIBLE_DEVICES=1")
        print("    python image_tokenizers/train_image.py")
        print()
        return

    print(f"✓ Autoencoder found: {autoencoder_path}")

    if not codebook_path.exists():
        print(f"❌ Codebook not found: {codebook_path}")
        print()
        print("Train the codebook using:")
        print("    export CUDA_VISIBLE_DEVICES=1")
        print("    python image_tokenizers/run_codebook.py")
        print()
        return

    print(f"✓ Codebook found: {codebook_path}")
    print()
    print("=" * 60)
    print("✓ Image tokenizer is ready!")
    print("=" * 60)
    print()
    print("To encode images, use:")
    print("    python examples/tokenizers/encode_all.py")
    print()
    print("Or manually:")
    print("    python image_tokenizers/run_image_inference.py")

if __name__ == "__main__":
    main()
