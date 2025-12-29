# Training Infrastructure

This document describes the training infrastructure for astromodal, focusing on the "meaningful loss" approach that reconstructs real-valued data instead of standard next-token prediction.

## Overview

The training infrastructure consists of three main components:

1. **Loss Functions** (`src/astromodal/training/losses.py`) - Differentiable reconstruction losses
2. **Training Utilities** (`src/astromodal/training/utils.py`) - Helper functions for training
3. **Trainer Class** (`src/astromodal/training/trainer.py`) - Complete training loop

## Meaningful Loss Approach

Instead of using standard next-token prediction (cross-entropy on discrete tokens), we compute **differentiable expected values** that reconstruct the original continuous data:

### Image Reconstruction Loss

```python
# Compute expected centroid from softmax over image vocabulary
probs = softmax(logits_image)  # (N, V_IMAGE)
predicted_centroid = probs @ image_codebook  # (N, D_img)

# Compare to true centroid
loss_image = MSE(predicted_centroid, true_centroid)
```

**Key idea**: Instead of predicting the discrete token ID, we predict the expected value of the image patch in latent space.

### Scalar Reconstruction Loss

```python
# Compute expected bin center from softmax over scalar vocabulary
probs = softmax(logits_scalar)  # (N, V_SCALAR)
predicted_value = sum(probs * bin_centers)  # (N,)

# Compare to true bin center
loss_scalar = MSE(predicted_value, true_bin_center)
```

**Key idea**: Instead of predicting which bin, we predict the expected numerical value.

### Spectrum Reconstruction Loss

```python
# Compute expected centroid from softmax over spectrum vocabulary
probs = softmax(logits_spectrum)  # (N, V_SPECTRA)
predicted_centroid = probs @ spectrum_codebook  # (N, D_spec)

# Compare to true centroid
loss_spectrum = MSE(predicted_centroid, true_centroid)
```

**Key idea**: Similar to images, we predict the expected spectral embedding.

### Combined Loss

```python
total_loss = loss_ce + α_img * loss_img + α_sca * loss_sca + α_spec * loss_spec
```

Where:
- `loss_ce`: Standard cross-entropy loss (optional)
- `α_img`, `α_sca`, `α_spec`: Weights for each modality (default: 0.1)

## Usage

### Basic Training with Trainer Class

```python
from src.astromodal.config import load_config
from src.astromodal.core import build_vocab_spec
from src.astromodal.training import Trainer, set_seed, configure_cuda_performance

# Load configuration
config = load_config()

# Set random seed
set_seed(config.training.seed)

# Configure CUDA performance
configure_cuda_performance()

# Build vocabulary
vocab = build_vocab_spec(
    v_image=config.vocab.image,
    v_scalar=config.vocab.scalar,
    v_spectra=config.vocab.spectrum,
)

# Load codebooks (image, spectrum, scalar)
img_codebook = ...  # Load from disk
spec_codebook = ...
sca_centers = ...

# Create model and data loader
model = ...
train_loader = ...

# Create trainer
trainer = Trainer(
    model=model,
    vocab=vocab,
    train_loader=train_loader,
    config=config.training,
    image_codebook=img_codebook,
    spectrum_codebook=spec_codebook,
    scalar_centers=sca_centers,
    device="cuda",
    output_dir="runs/experiment_001",
    use_amp=True,
    use_ce=True,
    compile_model=False,
)

# Train
trainer.train(num_epochs=10)
```

### Loading Codebooks

The meaningful loss requires codebooks that map token IDs to their continuous representations:

```python
from src.astromodal.core.io import load_codebook, load_npz_key

# Image codebook: (V_IMAGE, D_img)
img_codebook = load_codebook(
    "path/to/codebook_kmeans_2048.joblib",
    key_candidates=("centroids", "codebook", "centers"),
)

# Spectrum codebook: (V_SPECTRA, D_spec)
spec_codebook = load_npz_key(
    "path/to/spectrum_tokenizer_config.npz",
    key_candidates=("centroids", "codebook", "centers"),
)

# Scalar bin centers: (V_SCALAR,)
sca_centers = load_npz_key(
    "path/to/scalar_tokenizer_config.npz",
    key_candidates=("bin_centers", "centers", "quantile_centers"),
)
```

### Custom Training Loop

If you need more control, you can use the loss functions directly:

```python
from src.astromodal.training.losses import combined_loss

# Training loop
for batch in train_loader:
    tokens = batch["tokens"].to(device)
    types = batch["types"].to(device)
    attn_mask = batch["attn_mask"].to(device)

    # Next-token setup
    x = tokens[:, :-1]
    x_types = types[:, :-1]
    x_mask = attn_mask[:, :-1]
    y = tokens[:, 1:]

    # Forward pass
    logits = model(x, x_types, x_mask)

    # Compute loss
    loss, stats = combined_loss(
        logits=logits,
        targets=y,
        vocab=vocab,
        image_codebook=img_codebook,
        spectrum_codebook=spec_codebook,
        scalar_centers=sca_centers,
        config={
            "enabled": True,
            "sample_k": 4096,
            "weight_image": 0.1,
            "weight_scalar": 0.1,
            "weight_spectrum": 0.1,
        },
        use_ce=True,
    )

    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Log statistics
    print(f"Loss: {stats['loss_ce']:.4f}, "
          f"Img: {stats['loss_img']:.4f}, "
          f"Sca: {stats['loss_sca']:.4f}, "
          f"Spec: {stats['loss_spec']:.4f}")
```

## Configuration

The training behavior is controlled through the configuration system:

```yaml
training:
  seed: 0
  batch_size: 16
  epochs: 5
  learning_rate: 0.0002
  weight_decay: 0.05
  warmup_steps: 2000
  grad_clip: 1.0
  grad_accum_steps: 1
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
  log_every: 50
  save_every_steps: 5000

  meaningful_loss:
    enabled: true
    sample_k_per_modality: 4096
    weight_image: 0.1
    weight_scalar: 0.1
    weight_spectrum: 0.1
```

### Key Parameters

- **sample_k_per_modality**: Number of positions to sample per modality for loss computation. This dramatically speeds up training by not computing loss on all positions.
- **weight_image/scalar/spectrum**: Weights for each modality's reconstruction loss.
- **enabled**: Whether to use meaningful loss. If false, only CE loss is used.

## Learning Rate Schedule

The default learning rate schedule uses linear warmup followed by cosine decay:

```python
from src.astromodal.training.utils import cosine_schedule_with_warmup

lr = cosine_schedule_with_warmup(
    step=global_step,
    base_lr=2e-4,
    warmup_steps=2000,
    total_steps=200_000,
    min_lr_ratio=0.1,
)
```

Schedule:
1. **Warmup (0 to warmup_steps)**: Linear increase from 0 to base_lr
2. **Cosine decay (warmup_steps to total_steps)**: Cosine decay from base_lr to min_lr_ratio * base_lr

## Performance Optimizations

The training infrastructure includes several performance optimizations:

### 1. Mixed Precision Training (AMP)

Automatically enabled on CUDA devices:

```python
with torch.amp.autocast("cuda", enabled=use_amp):
    logits = model(x, x_types, x_mask)
    loss = compute_loss(logits, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. Gradient Accumulation

Accumulate gradients over multiple batches:

```yaml
grad_accum_steps: 4  # Effective batch size = batch_size * grad_accum_steps
```

### 3. Sampled Loss Computation

Sample positions for meaningful loss instead of computing on all:

```yaml
sample_k_per_modality: 4096  # Sample up to 4096 positions per modality
```

This can provide 10-100x speedup on the auxiliary loss computation.

### 4. CUDA Performance Settings

```python
from src.astromodal.training.utils import configure_cuda_performance

configure_cuda_performance()
# Enables:
# - TF32 for matmul
# - Flash attention
# - Memory-efficient attention
```

### 5. Torch Compile (PyTorch 2.0+)

```python
trainer = Trainer(
    ...,
    compile_model=True,  # Use torch.compile for extra speed
)
```

## Checkpointing

The Trainer automatically saves checkpoints:

1. **Step checkpoints**: Saved every `save_every_steps`
2. **Epoch checkpoints**: Saved at the end of each epoch

Checkpoint format:

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scaler": scaler.state_dict(),
    "global_step": global_step,
    "epoch": epoch,
    "config": {
        "vocab_size": vocab.vocab_size,
        "base_image": vocab.base_image,
        "base_scalar": vocab.base_scalar,
        "base_spectrum": vocab.base_spectrum,
        "loss_config": loss_config,
    },
}
```

### Resuming from Checkpoint

```python
trainer = Trainer(...)
trainer.load_checkpoint("runs/experiment_001/ckpt_step_00010000.pt")
trainer.train()  # Continues from loaded step
```

## Evaluation

Evaluate on a validation set:

```python
val_stats = trainer.evaluate(val_loader)
print(f"Validation loss: {val_stats['loss_total']:.4f}")
print(f"Validation CE: {val_stats['loss_ce']:.4f}")
print(f"Validation img: {val_stats['loss_img']:.4f}")
```

## Architecture Compatibility

The training infrastructure works with any model that:

1. Takes inputs `(tokens, types, attn_mask)`
2. Returns logits of shape `(B, L, V)`

Current supported model:

```python
from src.astromodal.models import CausalTransformerLM

model = CausalTransformerLM(
    vocab_size=vocab.vocab_size,
    n_types=20,
    d_model=768,
    n_layers=12,
    n_heads=12,
    d_ff=3072,
    dropout=0.1,
    max_len=2048,
)
```

## Examples

See `examples/train_with_meaningful_loss.py` for a complete training example.

## Implementation Details

### Why Sampling?

Computing the meaningful loss on all positions is expensive:

- **Without sampling**: ~1000 positions × 3 modalities = 3000 loss computations per batch
- **With sampling (k=4096)**: ~1365 positions per modality (sampled) = much faster

The sampling strategy:
1. Identify all valid positions for each modality (exclude PAD, MISS, etc.)
2. Randomly sample up to k positions
3. Compute loss only on sampled positions

This provides similar gradient signal with much better computational efficiency.

### Why Expected Values?

Standard next-token prediction:
- Loss: CE(predicted_distribution, true_token_id)
- Gradient only flows to the true token
- Ignores similarity between tokens

Meaningful loss:
- Loss: MSE(expected_value, true_value)
- Gradient flows to all tokens proportional to their probability
- Naturally captures similarity in continuous space
- Encourages smoother, more meaningful representations

### Why Include CE Loss?

The combined approach (CE + meaningful) offers advantages of both:

- **CE**: Ensures discrete token prediction remains accurate
- **Meaningful**: Provides continuous reconstruction signal
- **Together**: Best of both worlds - discrete + continuous

You can disable CE loss and use only meaningful loss:

```python
trainer = Trainer(
    ...,
    use_ce=False,  # Only use meaningful reconstruction loss
)
```

## Troubleshooting

### Out of Memory

1. Reduce batch size
2. Reduce `sample_k_per_modality`
3. Enable gradient accumulation
4. Use gradient checkpointing (not yet implemented)

### Loss Not Decreasing

1. Check learning rate (may need warmup)
2. Check gradient clipping value
3. Verify codebooks are loaded correctly
4. Check for NaN values in codebooks/data

### Slow Training

1. Enable `use_amp=True`
2. Enable `compile_model=True` (PyTorch 2.0+)
3. Increase `sample_k_per_modality` to smaller value
4. Increase number of workers
5. Use persistent workers

## References

The meaningful loss approach is inspired by:

1. Vector Quantized Variational Autoencoders (VQ-VAE)
2. Continuous relaxations of discrete variables
3. Straight-through estimators

Key difference: We compute expected values using softmax probabilities rather than using straight-through gradients.
