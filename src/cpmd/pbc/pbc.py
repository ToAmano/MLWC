"""calculate atomic distances
"""

import numpy as np
import abc
import torch

class pbc_abstract(abc.ABC):
    """
    アルゴリズム（ConcreteStrategy）が実装する共通のインターフェイス
    """
    @classmethod
    @abc.abstractmethod
    def compute_pbc(cls):
        pass
    
class pbc():
    """
    ConcreteStrategy をインスタンス変数として持つクラス
    """
    def __init__(self, strategy: pbc_abstract):
        self.strategy = strategy

    def compute_pbc(self,**kwargs):
        # ConcreteStrategy のメソッドを呼ぶことで、一部の処理を委託する
        return self.strategy.compute_pbc(**kwargs)




