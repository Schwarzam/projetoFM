# Astromodal Usage Guide

## Complete Pipeline: From Raw Data to Trained Transformer

This guide explains how to run the complete astromodal pipeline with **reconstruction loss** (not next-token prediction).

---

## Overview

The astromodal package has been completely reorganized into a clean Python package with:

- ✅ **No code duplication** - All utilities consolidated in `core/`
- ✅ **YAML configuration** - No hardcoded paths
- ✅ **Reconstruction loss** - Predicts actual data values, not just tokens
- ✅ **Easy-to-use classes** - Clean APIs for each component
- ✅ **Fully documented** - Comprehensive docstrings

---

## Installation

```bash
cd /home/schwarz/projetoFM
pip install -e .
```

Verify installation:
```bash
python -c "import astromodal; print(f'astromodal v{astromodal.__version__} installed')"
```

---

## Step 1: Train Tokenizers

Tokenizers convert raw astronomical data into discrete tokens.

### 1.1 Train Scalar Tokenizer

Quantile-based binning for catalog features (photometry, astrometry, stellar parameters):

```bash
python examples/tokenizers/train_scalar.py
```

**What it does:**
- Scans all datacube parquet files
- Creates quantile bins (1024 bins per column) for scalar features
- Handles: S-PLUS photometry, Gaia parameters, VISTA, spec-z
- Applies magnitude filters (mag 14-22)
- Saves config to: `scalar_tokenizers/scalar_tokenizer_config.npz`

**Duration:** ~10-30 minutes depending on data volume

---

### 1.2 Train Spectrum Tokenizer

KMeans clustering for spectral data:

```bash
python examples/tokenizers/train_spectrum.py
```

**What it does:**
- Trains separate KMeans models (2048 clusters each) for:
  - DESI spectra (b, r, z arms)
  - Gaia XP spectra (BP, RP)
- Features: (flux_norm, precision) pairs
- Saves config to: `spectrum_tokenizers/spectrum_tokenizer_config.npz`
- Saves individual codebooks: `spectrum_tokenizers/spectrum_kmeans_{name}.joblib`

**Duration:** ~1-3 hours (CPU-intensive, uses MiniBatchKMeans)

---

### 1.3 Train Image Tokenizer

Autoencoder + KMeans for image cutouts:

```bash
# Set GPU first
export CUDA_VISIBLE_DEVICES=1

python examples/tokenizers/train_image.py
```

**What it does:**
- **Part 1**: Trains autoencoder (96×96×12 → 2×24×24 latents)
  - AION-style architecture with spatial latents
  - Elliptical masking based on object morphology
  - Per-band arcsinh normalization
- **Part 2**: Trains KMeans codebook (2048 clusters) on latents
- Saves autoencoder to: `codecs/autoencoder_splus_cuts.pt`
- Saves codebook to: `codecs/codebook_kmeans_2048.joblib`

**Duration:** ~4-12 hours (GPU-intensive, depends on GPU)

---

## Step 2: Encode Data to Tokens

After training all tokenizers, encode your datacubes:

```bash
export CUDA_VISIBLE_DEVICES=1  # Set GPU for image encoding

python examples/tokenizers/encode_all.py
```

**What it does:**
- Discovers all fields from datacube template
- For each field:
  - **Scalars**: Encodes catalog features → uint16 tokens
  - **Spectra**: Encodes spectral data → uint16 tokens (per group)
  - **Images**: Encodes cutouts → uint16 tokens + float32 latents

**Outputs** (per field):
```
outputs/
├── image_tokens/datacube_{field}_tokens.npz
├── image_latents/datacube_{field}_latents.npz
├── scalar_tokens/datacube_{field}_scalar_tokens.npz
└── spectrum_tokens/datacube_{field}_spectrum_tokens.npz
```

**Duration:** ~2-8 hours depending on number of fields and GPU speed

---

## Step 3: Train Transformer with Reconstruction Loss

Now train the transformer using **meaningful reconstruction loss**:

```bash
export CUDA_VISIBLE_DEVICES=1  # Set GPU

python examples/training/train_transformer.py
```

### What Makes This Different

**Traditional Approach** (next-token prediction):
- Loss: Cross-entropy on discrete token IDs
- Model learns: "Token 1523 follows token 892"
- Problem: No connection to actual data values

**Astromodal Approach** (reconstruction loss):
- Loss: MSE between expected value and true value
- Model learns to reconstruct actual physical values:
  - **Images**: Expected latent centroid vs true centroid
  - **Scalars**: Expected bin center vs true bin center
  - **Spectra**: Expected spectrum centroid vs true centroid
- Benefit: Meaningful gradients that reconstruct real data

### How Reconstruction Loss Works

For each modality:

**Images:**
```python
# Get logits for image tokens
logits_image = model_output[image_positions]  # Shape: (K, 2048)

# Compute expected centroid using softmax
probs = softmax(logits_image)  # Shape: (K, 2048)
expected_centroid = probs @ image_codebook  # Shape: (K, latent_dim)

# Compare with true centroid
loss_image = MSE(expected_centroid, true_centroids)
```

**Scalars:**
```python
# Get logits for scalar tokens
logits_scalar = model_output[scalar_positions]  # Shape: (K, 1024)

# Compute expected bin center
probs = softmax(logits_scalar)  # Shape: (K, 1024)
expected_center = sum(probs * bin_centers)  # Shape: (K,)

# Compare with true value
loss_scalar = MSE(expected_center, true_values)
```

**Spectra:**
```python
# Get logits for spectrum tokens
logits_spectrum = model_output[spectrum_positions]  # Shape: (K, 2048)

# Compute expected spectrum centroid
probs = softmax(logits_spectrum)  # Shape: (K, 2048)
expected_spectrum = probs @ spectrum_codebook  # Shape: (K, spectrum_dim)

# Compare with true spectrum
loss_spectrum = MSE(expected_spectrum, true_spectra)
```

**Combined Loss:**
```python
total_loss = (
    cross_entropy_loss +  # Standard token prediction
    λ_image * loss_image +  # Image reconstruction
    λ_scalar * loss_scalar +  # Scalar reconstruction
    λ_spectrum * loss_spectrum  # Spectrum reconstruction
)
```

### Training Configuration

Default settings (in `src/astromodal/config/defaults.yaml`):

```yaml
model:
  d_model: 768
  n_layers: 12
  n_heads: 12
  d_ff: 3072
  dropout: 0.1

training:
  batch_size: 16
  epochs: 5
  learning_rate: 0.0002
  weight_decay: 0.05
  warmup_steps: 2000
  grad_clip: 1.0

  meaningful_loss:
    enabled: true
    sample_k_per_modality: 64  # Sample K positions per modality for speed
    weight_image: 0.1
    weight_scalar: 0.1
    weight_spectrum: 0.1
```

To customize, create a custom config file:

```yaml
# my_config.yaml
training:
  batch_size: 32
  learning_rate: 0.0001
  meaningful_loss:
    weight_image: 0.2  # Increase image reconstruction weight
```

Then run:
```bash
python examples/training/train_transformer.py --config my_config.yaml
```

### Outputs

Training saves:
```
runs_tokens_transformer/
├── config.json                    # Full configuration
├── vocab_spec.json                # Vocabulary specification
├── checkpoint_step_5000.pt        # Periodic checkpoints
├── checkpoint_step_10000.pt
├── ...
└── final_model.pt                 # Final trained model
```

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Training step and epoch
- Loss history
- Config and vocab spec

**Duration:** ~1-3 days depending on data size, GPU, and number of epochs

---

## Step 4: Using the Trained Model

### Load and Use for Inference

```python
import torch
from astromodal.config import load_config
from astromodal.models import CausalTransformerLM
from astromodal.core import build_vocab_spec

# Load config
config = load_config()

# Build vocab
vocab_spec = build_vocab_spec(
    v_image=config.vocab.image,
    v_scalar=config.vocab.scalar,
    v_spectra=config.vocab.spectrum,
    spectrum_groups=config.tokenizers.spectrum.groups,
)

# Create model
model = CausalTransformerLM(
    vocab_size=vocab_spec.vocab_size,
    d_model=config.model.d_model,
    n_layers=config.model.n_layers,
    n_heads=config.model.n_heads,
    d_ff=config.model.d_ff,
    dropout=config.model.dropout,
    max_seq_len=config.sequences.max_seq_len,
)

# Load checkpoint
checkpoint = torch.load("runs_tokens_transformer/final_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run inference
with torch.no_grad():
    logits = model(token_ids, token_types, key_padding_mask)

    # For reconstruction, use the meaningful loss functions
    from astromodal.training import image_reconstruction_loss, scalar_reconstruction_loss

    # Reconstruct image latents
    reconstructed_latents = image_reconstruction_loss(
        logits, target_tokens, image_codebook,
        return_reconstruction=True
    )
```

---

## Summary of Commands

### Complete Pipeline

```bash
# 1. Install
pip install -e .

# 2. Train tokenizers (can run in parallel on different machines)
python examples/tokenizers/train_scalar.py      # ~30 min
python examples/tokenizers/train_spectrum.py    # ~2 hours
python examples/tokenizers/train_image.py       # ~8 hours (GPU)

# 3. Encode data
python examples/tokenizers/encode_all.py        # ~4 hours (GPU)

# 4. Train transformer
python examples/training/train_transformer.py   # ~2 days (GPU)
```

### Time Estimates (Approximate)

| Step | Duration | Hardware |
|------|----------|----------|
| Scalar tokenizer | 30 min | CPU |
| Spectrum tokenizer | 2 hours | CPU |
| Image tokenizer | 8 hours | GPU (H100) |
| Encoding all fields | 4 hours | GPU (H100) |
| Transformer training | 2 days | GPU (H100) |
| **Total** | **~3 days** | GPU + CPU |

---

## Configuration

All settings can be customized in `src/astromodal/config/defaults.yaml`:

### Paths
```yaml
paths:
  datacube_template: "/home/astrodados4/downloads/hypercube/datacube_{field}.parquet"
  output_dir: "runs_tokens_transformer"
```

### Tokenizers
```yaml
tokenizers:
  scalar:
    n_bins: 1024
  spectrum:
    codebook_size: 2048
  image:
    latent_dim: 2
```

### Model
```yaml
model:
  d_model: 768
  n_layers: 12
  n_heads: 12
```

### Training
```yaml
training:
  batch_size: 16
  learning_rate: 0.0002
  meaningful_loss:
    enabled: true
    weight_image: 0.1
    weight_scalar: 0.1
    weight_spectrum: 0.1
```

---

## Key Benefits of the Reorganization

### Before (Old Structure)
- ❌ Code duplication (ID normalization repeated 5x)
- ❌ Hardcoded paths everywhere
- ❌ Difficult to modify or extend
- ❌ No clear API
- ❌ Hard to test individual components

### After (Astromodal Package)
- ✅ Zero code duplication (DRY principle)
- ✅ YAML configuration system
- ✅ Clean class-based APIs
- ✅ Easy to test and extend
- ✅ Comprehensive documentation
- ✅ Modular and maintainable

---

## Troubleshooting

### Import Errors
```bash
# Reinstall in editable mode
pip install -e .
```

### GPU Memory Issues
Reduce batch size in config:
```yaml
training:
  batch_size: 8  # Reduce from 16
```

### Slow Training
Enable mixed precision (should be on by default):
```yaml
training:
  use_amp: true
```

Check GPU utilization:
```bash
watch -n 1 nvidia-smi
```

### Out of Memory During Encoding
Process fields in batches instead of all at once. Edit `encode_all.py` to process fewer fields:
```python
fields = fields[:10]  # Process first 10 fields only
```

---

## Next Steps

1. **Experiment with hyperparameters**: Adjust learning rate, batch size, model size
2. **Analyze reconstruction quality**: Evaluate how well the model reconstructs actual values
3. **Fine-tune for specific tasks**: Use the pretrained model for downstream tasks
4. **Scale up**: Train on more data with larger models

---

## Support

For issues or questions:
- Check the documentation in `docs/`
- Review example scripts in `examples/`
- See detailed implementation notes in `PHASE*_SUMMARY.md` files

---

**The reconstruction loss approach provides meaningful supervision that goes beyond discrete token prediction, learning to reconstruct actual astronomical data values!**
