"""
Base tokenizer abstract class.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union


class BaseTokenizer(ABC):
    """
    Abstract base class for all tokenizers.

    All tokenizers must implement train, encode, save, and load methods.
    """

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        """
        Train the tokenizer on data.

        Parameters
        ----------
        *args, **kwargs
            Training data and parameters (tokenizer-specific)
        """
        pass

    @abstractmethod
    def encode(self, *args, **kwargs) -> Any:
        """
        Encode data into tokens.

        Parameters
        ----------
        *args, **kwargs
            Data to encode (tokenizer-specific)

        Returns
        -------
        Any
            Encoded tokens (format depends on tokenizer)
        """
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """
        Save tokenizer configuration to disk.

        Parameters
        ----------
        path : Union[str, Path]
            Output path for tokenizer config
        """
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """
        Load tokenizer configuration from disk.

        Parameters
        ----------
        path : Union[str, Path]
            Path to tokenizer config
        """
        pass
