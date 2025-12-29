"""
Configuration loading and management system.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml


@dataclass
class PathsConfig:
    """File paths and templates configuration."""
    datacube_template: str
    image_tokens_template: str
    scalar_tokens_template: str
    spectrum_tokens_template: str
    image_latents_template: str
    scalar_tokenizer_config: str
    spectrum_tokenizer_config: str
    image_autoencoder: str
    image_codebook: str
    output_dir: str


@dataclass
class DataConfig:
    """Data filtering and preprocessing configuration."""
    mag_filter: Dict[str, Union[str, float]]
    magerr_max: float
    splus_bands: List[str]


@dataclass
class ScalarTokenizerConfig:
    """Scalar tokenizer configuration."""
    n_bins: int
    max_values_per_col: int
    max_values_per_file_per_col: int


@dataclass
class SpectrumTokenizerConfig:
    """Spectrum tokenizer configuration."""
    codebook_size: int
    groups: List[str]


@dataclass
class ImageTokenizerConfig:
    """Image tokenizer configuration."""
    cutout_size: int
    latent_dim: int
    codebook_size: int
    bands: List[str]


@dataclass
class TokenizersConfig:
    """All tokenizers configuration."""
    scalar: ScalarTokenizerConfig
    spectrum: SpectrumTokenizerConfig
    image: ImageTokenizerConfig


@dataclass
class VocabConfig:
    """Vocabulary sizes configuration."""
    image: int
    scalar: int
    spectrum: int


@dataclass
class SequencesConfig:
    """Sequence limits configuration."""
    max_image_tokens: int
    max_scalar_tokens: int
    max_spec_tokens_per_group: int
    max_seq_len: int


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float


@dataclass
class MeaningfulLossConfig:
    """Meaningful loss configuration."""
    enabled: bool
    sample_k_per_modality: int
    weight_image: float
    weight_scalar: float
    weight_spectrum: float


@dataclass
class TrainingConfig:
    """Training configuration."""
    seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    grad_clip: float
    grad_accum_steps: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    log_every: int
    save_every_steps: int
    meaningful_loss: MeaningfulLossConfig


@dataclass
class AstromodalConfig:
    """Complete astromodal configuration."""
    paths: PathsConfig
    data: DataConfig
    tokenizers: TokenizersConfig
    vocab: VocabConfig
    sequences: SequencesConfig
    model: ModelConfig
    training: TrainingConfig


def _dict_to_dataclass(cls, data: dict):
    """Convert nested dict to dataclass instances."""
    if not isinstance(data, dict):
        return data

    if not hasattr(cls, '__dataclass_fields__'):
        return data

    field_info = {f.name: f for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for key, value in data.items():
        if key in field_info:
            field_type = field_info[key].type
            # Handle nested dataclasses
            if isinstance(value, dict) and hasattr(field_type, '__dataclass_fields__'):
                kwargs[key] = _dict_to_dataclass(field_type, value)
            else:
                kwargs[key] = value

    return cls(**kwargs)


def load_config(config_path: Optional[Union[str, Path]] = None) -> AstromodalConfig:
    """
    Load configuration from YAML file, with defaults fallback.

    Parameters
    ----------
    config_path : Optional[Union[str, Path]], default=None
        Path to user configuration YAML file.
        If None, uses defaults.yaml from package.

    Returns
    -------
    AstromodalConfig
        Complete configuration object

    Examples
    --------
    >>> config = load_config()  # Use defaults
    >>> config = load_config("my_config.yaml")  # Use custom config
    >>> print(config.model.d_model)
    768
    """
    # Load defaults
    defaults_path = Path(__file__).parent / "defaults.yaml"
    with open(defaults_path, 'r') as f:
        defaults = yaml.safe_load(f)

    # Merge with user config if provided
    if config_path is not None:
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        defaults = _merge_dicts(defaults, user_config)

    # Convert to dataclasses manually to ensure proper nesting
    config = AstromodalConfig(
        paths=_dict_to_dataclass(PathsConfig, defaults['paths']),
        data=_dict_to_dataclass(DataConfig, defaults['data']),
        tokenizers=TokenizersConfig(
            scalar=_dict_to_dataclass(ScalarTokenizerConfig, defaults['tokenizers']['scalar']),
            spectrum=_dict_to_dataclass(SpectrumTokenizerConfig, defaults['tokenizers']['spectrum']),
            image=_dict_to_dataclass(ImageTokenizerConfig, defaults['tokenizers']['image']),
        ),
        vocab=_dict_to_dataclass(VocabConfig, defaults['vocab']),
        sequences=_dict_to_dataclass(SequencesConfig, defaults['sequences']),
        model=_dict_to_dataclass(ModelConfig, defaults['model']),
        training=TrainingConfig(
            **{k: v for k, v in defaults['training'].items() if k != 'meaningful_loss'},
            meaningful_loss=_dict_to_dataclass(MeaningfulLossConfig, defaults['training']['meaningful_loss']),
        ),
    )
    return config


def _merge_dicts(base: dict, override: dict) -> dict:
    """
    Recursively merge override dict into base dict.

    Parameters
    ----------
    base : dict
        Base dictionary
    override : dict
        Override dictionary

    Returns
    -------
    dict
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def save_config(config: AstromodalConfig, path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Parameters
    ----------
    config : AstromodalConfig
        Configuration object to save
    path : Union[str, Path]
        Output YAML file path
    """
    def _dataclass_to_dict(obj):
        """Convert dataclass to dict recursively."""
        if hasattr(obj, '__dataclass_fields__'):
            return {
                key: _dataclass_to_dict(value)
                for key, value in obj.__dict__.items()
            }
        elif isinstance(obj, list):
            return [_dataclass_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: _dataclass_to_dict(v) for k, v in obj.items()}
        else:
            return obj

    config_dict = _dataclass_to_dict(config)
    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
