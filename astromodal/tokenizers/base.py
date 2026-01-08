from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union


class BaseTokenizer(ABC):
    """
    Abstract base class for all tokenizers.

    Tokenizers must define:
    - train   : learn parameters from data
    - encode  : data -> tokens
    - decode  : tokens -> reconstructed data
    - save    : persist tokenizer state
    - load    : restore tokenizer state
    """

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def encode(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def decode(self, *args, **kwargs) -> Any:
        """
        Decode tokens back into data.

        Returns
        -------
        Any
            Reconstructed data (format tokenizer-dependent)
        """
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        pass