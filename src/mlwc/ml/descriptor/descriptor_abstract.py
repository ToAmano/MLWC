from abc import ABC, abstractmethod

import numpy as np
import torch.nn as nn

import __version__


class DescriptorAbstract(nn.Module, ABC):
    """abstract method for dataset

    Args:
        ABC (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(self):
        # use version info for backward compatibility
        self.version = __version__.__version__
        super().__init__()

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def calc_descriptor(self):
        pass


class Descriptor:
    """
    ConcreteStrategy をインスタンス変数として持つクラス
    """

    def __init__(self, strategy: type[DescriptorAbstract]):
        self._strategy = strategy

    def calc_descriptor(self, **kwargs) -> np.ndarray:
        """ConcreteStrategy のメソッドを呼ぶことで、一部の処理を委託する"""
        return self._strategy.calc_descriptor(**kwargs)

    def forward(self, **kwargs) -> np.ndarray:
        """ConcreteStrategy のメソッドを呼ぶことで、一部の処理を委託する"""
        return self._strategy.forward(**kwargs)

    @property
    def strategy(self) -> DescriptorAbstract:
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: DescriptorAbstract) -> None:
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """
        self._strategy = strategy
