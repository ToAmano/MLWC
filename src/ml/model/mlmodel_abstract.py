from abc import (
    ABC,
    abstractmethod,
)

import torch       # ライブラリ「PyTorch」のtorchパッケージをインポート
import torch.nn as nn  # 「ニューラルネットワーク」モジュールの別名定義

# https://stackoverflow.com/questions/72077539/should-i-inherit-from-both-nn-module-and-abc
class Model_abstract(nn.Module,ABC):

    def __init__(self):
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


