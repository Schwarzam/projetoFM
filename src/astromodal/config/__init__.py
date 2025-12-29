"""
Configuration system for astromodal.
"""

from .loader import (
    AstromodalConfig,
    PathsConfig,
    DataConfig,
    ScalarTokenizerConfig,
    SpectrumTokenizerConfig,
    ImageTokenizerConfig,
    TokenizersConfig,
    VocabConfig,
    SequencesConfig,
    ModelConfig,
    MeaningfulLossConfig,
    TrainingConfig,
    load_config,
    save_config,
)

__all__ = [
    "AstromodalConfig",
    "PathsConfig",
    "DataConfig",
    "ScalarTokenizerConfig",
    "SpectrumTokenizerConfig",
    "ImageTokenizerConfig",
    "TokenizersConfig",
    "VocabConfig",
    "SequencesConfig",
    "ModelConfig",
    "MeaningfulLossConfig",
    "TrainingConfig",
    "load_config",
    "save_config",
]
