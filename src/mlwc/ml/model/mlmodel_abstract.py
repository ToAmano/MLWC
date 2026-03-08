from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

import __version__


# https://stackoverflow.com/questions/72077539/should-i-inherit-from-both-nn-module-and-abc
class AbstractModel(nn.Module, ABC):

    def __init__(self):
        # use version info for backward compatibility
        self.version = __version__.__version__
        super().__init__()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def get_rcut(self) -> float:
        """Get cutoff radius of the model."""

    @abstractmethod
    def get_modelname(self) -> str:
        """Get the model name."""

    @abstractmethod
    def save_torchscript_py(self) -> None:
        """save the model as torchscript (python)"""

    @abstractmethod
    def save_torchscript_cpp(self) -> None:
        """save the model as torchscript (C++)"""

    @abstractmethod
    def save_weight(self) -> None:
        """save the model weight"""


class BaseModelWrapper(ABC):
    @abstractmethod
    def preprocess(self, raw_input: Any) -> torch.Tensor:
        """generate torch.Tensor input"""

    @abstractmethod
    def predict(self, tensor_input: torch.Tensor) -> Any:
        """wrapper for forward"""

    @abstractmethod
    def save(self, path: str) -> None:
        """wrapper for save model"""

    @classmethod
    @abstractmethod
    def load_from_file(cls, path: str) -> "BaseModelWrapper":
        """load a file and generate BaseModelWrapper"""
