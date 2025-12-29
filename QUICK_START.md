# Astromodal Quick Start

## Installation
```bash
pip install -e .
```

## Run Complete Pipeline

### Step 1: Train Tokenizers (run these in sequence or parallel)

```bash
# Scalar tokenizer (~30 min, CPU)
python examples/tokenizers/train_scalar.py

# Spectrum tokenizer (~2 hours, CPU)
python examples/tokenizers/train_spectrum.py

# Image tokenizer (~8 hours, GPU)
export CUDA_VISIBLE_DEVICES=1
# Step 1: Train autoencoder (~4-6 hours)
python examples/tokenizers/train_image_autoencoder.py
# Step 2: Train KMeans codebook (~30 min)
python examples/tokenizers/train_image_codebook.py
```

### Step 2: Encode All Data

```bash
# Encode all fields to tokens (~4 hours, GPU)
export CUDA_VISIBLE_DEVICES=1
python examples/tokenizers/encode_all.py
```

**Output locations:**
- `outputs/image_tokens/datacube_{field}_tokens.npz`
- `outputs/image_latents/datacube_{field}_latents.npz`
- `outputs/scalar_tokens/datacube_{field}_scalar_tokens.npz`
- `outputs/spectrum_tokens/datacube_{field}_spectrum_tokens.npz`

### Step 3: Train Transformer

```bash
# Train with RECONSTRUCTION LOSS (~2 days, GPU)
export CUDA_VISIBLE_DEVICES=1
python examples/training/train_transformer.py
```

**What's different:** Uses meaningful reconstruction loss instead of next-token prediction:
- Images → Reconstructs actual latent centroids
- Scalars → Reconstructs actual bin centers
- Spectra → Reconstructs actual spectrum centroids

**Output:** `runs_tokens_transformer/`

---

## Key Reconstruction Loss Feature

The model learns to predict **actual data values**, not just discrete tokens:

```python
# Instead of: "predict token 1523"
# The model learns: "reconstruct centroid [0.23, -0.45, 0.67, ...]"

loss = (
    cross_entropy(predicted_tokens, true_tokens) +  # Discrete tokens
    λ * MSE(reconstructed_values, true_values)      # Actual values ✓
)
```

This provides much better gradients for learning meaningful representations!

---

## Configuration

Edit `src/astromodal/config/defaults.yaml` to customize:
- Paths
- Model architecture (d_model, n_layers, n_heads)
- Training hyperparameters (batch_size, learning_rate)
- Reconstruction loss weights

---

## Total Time: ~3 days
- Tokenizer training: ~10 hours
- Data encoding: ~4 hours
- Transformer training: ~2 days

---

See `USAGE_GUIDE.md` for detailed explanations.
