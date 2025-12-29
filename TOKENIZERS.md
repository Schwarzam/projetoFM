# Astromodal Tokenizers - Phase 4 Implementation

This document describes the tokenizer implementations for the astromodal package.

## Overview

The tokenizers convert continuous astronomical data into discrete tokens for transformer models:

1. **ScalarTokenizer**: Quantile binning for catalog scalar features
2. **SpectrumTokenizer**: K-means clustering for spectroscopic time-series
3. **ImageTokenizer**: Autoencoder-based tokenization for image cutouts

All tokenizers inherit from `BaseTokenizer` and implement the standard interface:
- `train()`: Learn tokenizer from training data
- `encode()`: Convert data to tokens
- `save()`: Save tokenizer configuration
- `load()`: Load tokenizer from disk

## Installation

The tokenizers are part of the astromodal package:

```python
from astromodal.tokenizers import (
    ScalarTokenizer,
    SpectrumTokenizer,
    ImageTokenizer,
)
```

## ScalarTokenizer

### Overview

Tokenizes scalar catalog features (magnitudes, proper motions, etc.) using quantile binning.

### Training

```python
from astromodal.tokenizers import ScalarTokenizer

# Define columns to tokenize
scalar_columns = [
    "mag_pstotal_r",
    "mag_pstotal_g",
    "mag_pstotal_i",
    "gaia_parallax",
    "gaia_pmra",
    "gaia_pmdec",
    # ... etc
]

# Create tokenizer
tokenizer = ScalarTokenizer(
    n_bins=1024,
    scalar_columns=scalar_columns,
    mag_filter_col="mag_pstotal_r",
    mag_min=14.0,
    mag_max=22.0,
    apply_row_filter=True,
    magerr_max=2.0,
    apply_magerr_filter=True,
    max_values_per_col=2_000_000,
    rng_seed=0,
)

# Train on parquet files
parquet_files = [
    "/path/to/datacube_1.parquet",
    "/path/to/datacube_2.parquet",
    # ... etc
]

tokenizer.train(parquet_files)

# Save configuration
tokenizer.save("scalar_tokenizer_config.npz")
```

### Encoding

```python
# Load trained tokenizer
tokenizer = ScalarTokenizer.load("scalar_tokenizer_config.npz")

# Encode a parquet file
result = tokenizer.encode("datacube_test.parquet")

# Access results
tokens = result['scalar_tokens']  # (n_rows, n_cols) uint16 array
columns = result['scalar_cols']   # column names
ids = result['ids']               # row IDs
n_bins = result['N_BINS']         # 1024

print(f"Encoded {tokens.shape[0]} rows x {tokens.shape[1]} columns")
print(f"Token range: [{tokens.min()}, {tokens.max()}]")
```

### Configuration from astromodal.config

```python
from astromodal.config import load_config

config = load_config("config.yaml")

tokenizer = ScalarTokenizer(
    n_bins=config.tokenizers.scalar.n_bins,
    max_values_per_col=config.tokenizers.scalar.max_values_per_col,
    mag_min=config.data.mag_filter['min'],
    mag_max=config.data.mag_filter['max'],
    # ... etc
)
```

## SpectrumTokenizer

### Overview

Tokenizes spectroscopic and time-series data using K-means clustering. Supports:
- DESI spectra (flux + inverse variance)
- ZTF light curves (magnitude + error)
- GaiaXP continuous spectra (BP/RP photometry)

### Training

```python
from astromodal.tokenizers import SpectrumTokenizer

# Define spectrum groups
groups = [
    {
        "name": "desi_b",
        "kind": "flux_ivar",
        "flux_col": "desi_flux_b",
        "ivar_col": "desi_ivar_b",
    },
    {
        "name": "desi_r",
        "kind": "flux_ivar",
        "flux_col": "desi_flux_r",
        "ivar_col": "desi_ivar_r",
    },
    {
        "name": "desi_z",
        "kind": "flux_ivar",
        "flux_col": "desi_flux_z",
        "ivar_col": "desi_ivar_z",
    },
    {
        "name": "gaiaxp_bp",
        "kind": "gaiaxpy_xp",
        "xp_band": "BP",
    },
    {
        "name": "gaiaxp_rp",
        "kind": "gaiaxpy_xp",
        "xp_band": "RP",
    },
]

# Create tokenizer
tokenizer = SpectrumTokenizer(
    codebook_size=2048,
    max_features_per_group=2_000_000,
    max_rows_per_file=5000,
    max_points_per_row=2000,
    groups=groups,
)

# Train on parquet files
tokenizer.train(parquet_files)

# Save configuration (also saves spectrum_kmeans_*.joblib for each group)
tokenizer.save("spectrum_tokenizer_config.npz")
```

### Encoding

```python
# Load trained tokenizer
tokenizer = SpectrumTokenizer.load("spectrum_tokenizer_config.npz")

# Encode data
result = tokenizer.encode("datacube_test.parquet")

# Access results
# For rectangular groups (DESI, ZTF):
desi_b_tokens = result['tokens_desi_b']  # (n_rows, spectrum_length)

# For ragged groups (GaiaXP):
gaiaxp_bp_flat = result['tokens_gaiaxp_bp_flat']      # (total_tokens,)
gaiaxp_bp_indptr = result['tokens_gaiaxp_bp_indptr']  # (n_rows + 1,)

# Extract tokens for row i:
start = gaiaxp_bp_indptr[i]
end = gaiaxp_bp_indptr[i + 1]
row_tokens = gaiaxp_bp_flat[start:end]

ids = result['ids']
```

### Adding ZTF Groups (Auto-detection)

```python
# Enable ZTF auto-detection
tokenizer = SpectrumTokenizer(
    codebook_size=2048,
    # ZTF groups will be auto-detected from columns
)

# Or manually specify ZTF groups
ztf_groups = [
    {
        "name": "ztf_g",
        "kind": "ztf_mag_err",
        "mag_col": "ztf_mag_g",
        "err_col": "ztf_magerr_g",
        "time_col": "ztf_hmjd_g",
    },
]

tokenizer = SpectrumTokenizer(
    codebook_size=2048,
    groups=groups + ztf_groups,
)
```

## ImageTokenizer

### Overview

Tokenizes multi-band image cutouts using autoencoder. The autoencoder must be trained separately.

### Training the Autoencoder

Training requires GPU and should be done using dedicated scripts:

```bash
cd image_tokenizers
python train_image.py
```

This produces `autoencoder_splus_cuts.pt` with trained weights.

### Loading and Using

```python
from astromodal.tokenizers import ImageTokenizer

# Create tokenizer
tokenizer = ImageTokenizer(
    cutout_size=96,
    bands=['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861',
           'U', 'G', 'R', 'I', 'Z'],
    latent_dim=2,
    codebook_size=1024,
)

# Load trained autoencoder
tokenizer.load("autoencoder_splus_cuts.pt")

# Encode images
result = tokenizer.encode("datacube_test.parquet", batch_size=32)

# Access results
tokens = result['image_tokens']    # (n_rows, latent_H, latent_W)
latents = result['image_latents']  # (n_rows, latent_dim, latent_H, latent_W)
ids = result['ids']

print(f"Image tokens shape: {tokens.shape}")
print(f"Latent spatial resolution: {tokens.shape[1]}x{tokens.shape[2]}")
```

### Save/Load Configuration

```python
# Save configuration (references autoencoder weights)
tokenizer.save("image_tokenizer_config.npz")

# Load later
tokenizer = ImageTokenizer.load("image_tokenizer_config.npz", device='cuda')
```

## Integration with Configuration System

All tokenizers can be configured via the astromodal config system:

```yaml
# config.yaml
tokenizers:
  scalar:
    n_bins: 1024
    max_values_per_col: 2000000
    max_values_per_file_per_col: 100000

  spectrum:
    codebook_size: 2048
    groups:
      - desi_b
      - desi_r
      - desi_z
      - gaiaxp_bp
      - gaiaxp_rp

  image:
    cutout_size: 96
    latent_dim: 2
    codebook_size: 1024
    bands:
      - F378
      - F395
      - U
      - G
      - R
      - I
      - Z
```

```python
from astromodal.config import load_config

config = load_config("config.yaml")

# Create tokenizers from config
scalar_tok = ScalarTokenizer(
    n_bins=config.tokenizers.scalar.n_bins,
    max_values_per_col=config.tokenizers.scalar.max_values_per_col,
)

spectrum_tok = SpectrumTokenizer(
    codebook_size=config.tokenizers.spectrum.codebook_size,
)

image_tok = ImageTokenizer(
    cutout_size=config.tokenizers.image.cutout_size,
    bands=config.tokenizers.image.bands,
    latent_dim=config.tokenizers.image.latent_dim,
)
```

## Comparison with Original Scripts

The tokenizers maintain compatibility with existing implementations:

| Original Script | New Class | Compatibility |
|-----------------|-----------|---------------|
| `scalar_tokenizers/train_scalar_tokenizer.py` | `ScalarTokenizer.train()` | Full |
| `scalar_tokenizers/encode_scalar_tokenizer.py` | `ScalarTokenizer.encode()` | Full |
| `spectrum_tokenizers/train_spectrum_tokenizers.py` | `SpectrumTokenizer.train()` | Full |
| `spectrum_tokenizers/encode_spectrum_tokens.py` | `SpectrumTokenizer.encode()` | Full |
| `image_tokenizers/train_image.py` | ImageTokenizer (external training) | Partial* |

*Image training still requires external script for GPU optimization.

## Output Formats

### Scalar Tokens
```python
{
    'scalar_tokens': np.ndarray,  # (n_rows, n_cols) uint16
    'scalar_cols': np.ndarray,    # (n_cols,) object (column names)
    'ids': np.ndarray,            # (n_rows,) int64
    'id_col': str,                # ID column name
    'N_BINS': int,                # 1024
}
```

### Spectrum Tokens (Rectangular)
```python
{
    'tokens_desi_b': np.ndarray,  # (n_rows, L) uint16
    'tokens_desi_r': np.ndarray,  # (n_rows, L) uint16
    # ... per group
    'ids': np.ndarray,            # (n_rows,) int64
    'id_col': str,
}
```

### Spectrum Tokens (Ragged - GaiaXP)
```python
{
    'tokens_gaiaxp_bp_flat': np.ndarray,    # (total_tokens,) uint16
    'tokens_gaiaxp_bp_indptr': np.ndarray,  # (n_rows+1,) int64
    # ... per group
    'ids': np.ndarray,
    'id_col': str,
}
```

### Image Tokens
```python
{
    'image_tokens': np.ndarray,   # (n_rows, H', W') uint16
    'image_latents': np.ndarray,  # (n_rows, D, H', W') float32
    'ids': np.ndarray,            # (n_rows,) int64
    'id_col': str,
}
```

## Best Practices

1. **Training**: Train tokenizers on representative data covering the full range
2. **Saving**: Always save tokenizer configs after training
3. **Versioning**: Track tokenizer versions alongside model checkpoints
4. **Testing**: Validate token ranges and distributions before training
5. **Memory**: Use batch processing for large datasets
6. **IDs**: Always include ID columns for data tracking

## Next Steps (Phase 5)

After tokenizers are complete, the next phase will implement:
- Model architecture (transformer)
- Dataset loaders using tokenizers
- Training loops
- AutoEncoder model class for image tokenizer

## Testing

Run the test script to verify installation:

```bash
python test_tokenizers.py
```

Expected output:
```
Testing astromodal tokenizer implementations...

=== Testing ScalarTokenizer ===
Created ScalarTokenizer with 1024 bins
Columns: ['mag_pstotal_r', 'mag_pstotal_g']
Is fitted: False
ScalarTokenizer: OK

=== Testing SpectrumTokenizer ===
Created SpectrumTokenizer with codebook size 2048
Groups: ['desi_b']
Is fitted: False
SpectrumTokenizer: OK

=== Testing ImageTokenizer ===
Created ImageTokenizer with cutout size 96
Bands: ['F378', 'F395', 'U', 'G', 'R', 'I', 'Z']
Latent dim: 2
Is fitted: False
ImageTokenizer: OK

==================================================
All tokenizers passed basic tests!
==================================================
```
