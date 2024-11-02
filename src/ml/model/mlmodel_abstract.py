from abc import (
    ABC,
    abstractmethod,
)

import torch       # ライブラリ「PyTorch」のtorchパッケージをインポート
import torch.nn as nn  # 「ニューラルネットワーク」モジュールの別名定義
import __version__
# https://stackoverflow.com/questions/72077539/should-i-inherit-from-both-nn-module-and-abc
class Model_abstract(nn.Module,ABC):

    def __init__(self):
        self.version = __version__.__version__  # use version info for backward compatibility
        super().__init__() 

    
    @abstractmethod
    def forward(self, x: torch.Tensor):
        raise NotImplementedError
    
    @abstractmethod
    def get_rcut(self) -> float:
        """Get cutoff radius of the model."""
        pass

    @abstractmethod
    def get_modelname(self) -> str:
        """Get the model name."""
        pass
    
    @abstractmethod
    def save_torchscript_py(self)-> None:
        """save the model as torchscript (python)"""
        pass

    @abstractmethod
    def save_torchscript_cpp(self)-> None:
        """save the model as torchscript (C++)"""
        pass
    
    @abstractmethod
    def save_weight(self)-> None:
        """save the model weight"""
        pass

