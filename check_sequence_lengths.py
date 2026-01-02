#!/usr/bin/env python3
"""
Check actual sequence lengths to diagnose low GPU utilization.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.astromodal.core import build_vocab_spec, discover_fields_from_datacubes
from src.astromodal.datasets import MultiFieldTokenDataset
import numpy as np

# Build vocab
vocab = build_vocab_spec(
    v_image=2048,
    v_scalar=1024,
    v_spectra=2048,
    spectrum_groups={
        'gaiaxp_bp': ('tokens_gaiaxp_bp_flat', 'tokens_gaiaxp_bp_indptr'),
        'gaiaxp_rp': ('tokens_gaiaxp_rp_flat', 'tokens_gaiaxp_rp_indptr'),
        'desi_b': ('tokens_desi_b_flat', 'tokens_desi_b_indptr'),
        'desi_r': ('tokens_desi_r_flat', 'tokens_desi_r_indptr'),
        'desi_z': ('tokens_desi_z_flat', 'tokens_desi_z_indptr'),
    }
)

# Create dataset
print("Loading dataset...")
fields = discover_fields_from_datacubes('/home/astrodados4/downloads/hypercube/datacube_{field}.parquet')
dataset = MultiFieldTokenDataset(
    fields=fields,
    vocab=vocab,
    datacube_template='/home/astrodados4/downloads/hypercube/datacube_{field}.parquet',
    image_tokens_template='/home/schwarz/projetoFM/codecs/image_tokens/datacube_{field}_tokens.npz',
    scalar_tokens_template='/home/schwarz/projetoFM/scalar_tokenizers/scalar_tokens/datacube_{field}_scalar_tokens.npz',
    spectrum_tokens_template='/home/schwarz/projetoFM/spectrum_tokenizers/spectrum_tokens/datacube_{field}_spectrum_tokens.npz',
    spectrum_groups={
        'gaiaxp_bp': ('tokens_gaiaxp_bp_flat', 'tokens_gaiaxp_bp_indptr'),
        'gaiaxp_rp': ('tokens_gaiaxp_rp_flat', 'tokens_gaiaxp_rp_indptr'),
        'desi_b': ('tokens_desi_b_flat', 'tokens_desi_b_indptr'),
        'desi_r': ('tokens_desi_r_flat', 'tokens_desi_r_indptr'),
        'desi_z': ('tokens_desi_z_flat', 'tokens_desi_z_indptr'),
    },
    mag_col='mag_pstotal_r',
    mag_min=14.0,
    mag_max=22.0,
    magerr_max=2.0,
)

print(f'Dataset size: {len(dataset):,}')
print()
print('Checking 20 random samples...')
print()
print('Sample | Total Tokens | Non-Padding | Image | Scalar | Spectrum | Padding %')
print('-' * 85)

lengths = []
for i in range(20):
    idx = np.random.randint(0, len(dataset))
    sample = dataset[idx]
    tokens = sample['tokens']
    types = sample['types']

    total = tokens.numel()
    non_padding = (tokens != vocab.pad_id).sum().item()
    image_tokens = (types == 5).sum().item()
    scalar_tokens = (types == 6).sum().item()
    spectrum_tokens = (types >= 7).sum().item()
    padding_pct = (1 - non_padding / total) * 100

    lengths.append(non_padding)

    print(f'{i+1:6d} | {total:12d} | {non_padding:11d} | {image_tokens:5d} | {scalar_tokens:6d} | {spectrum_tokens:8d} | {padding_pct:8.1f}%')

print()
print(f'Average non-padding tokens: {np.mean(lengths):.1f}')
print(f'Median non-padding tokens: {np.median(lengths):.1f}')
print(f'Min non-padding tokens: {np.min(lengths)}')
print(f'Max non-padding tokens: {np.max(lengths)}')
print()

if np.mean(lengths) < 100:
    print('⚠️  WARNING: Sequences are very short!')
    print('   This explains the low GPU utilization.')
    print('   Most of each batch is padding, so the GPU has little work to do.')
elif np.mean(lengths) < 500:
    print('⚠️  Sequences are somewhat short.')
    print('   GPU could be better utilized with longer sequences.')
else:
    print('✓ Sequence lengths look reasonable.')
    print('  The low GPU utilization must be from another cause (DataLoader, I/O, etc.)')
