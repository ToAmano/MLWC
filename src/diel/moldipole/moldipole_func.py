import sys
import ase.units
import ase.io
import numpy as np
import pandas as pd
import abc
from functools import singledispatch
import cpmd.read_core
import cpmd.read_traj
from diel.moldipole.moldipole import moldipole
from diel.moldipole.moldipole_io import read_file
from include.mlwc_logger import root_logger
logger = root_logger("MLWC."+__name__)


class moldipole_abstract(abc.ABC):
    """
    アルゴリズム（ConcreteStrategy）が実装する共通のインターフェイス
    """
    @classmethod
    @abc.abstractmethod
    def execute(cls):
        pass
    
class strategy_cls(moldipole_abstract):
    """
    アルゴリズム（ConcreteStrategy）の具体的な実装
    """
    @classmethod
    def execute(cls,method_name: str,moldipole:moldipole):
        method = getattr(moldipole, method_name)
        return method()
    
class strategy_param(moldipole_abstract):
    """
    アルゴリズム（ConcreteStrategy）の具体的な実装
    """
    @classmethod
    def execute(cls,method_name: str,data:np.ndarray,unitcell:np.ndarray,timestep:float,temperature:float):
        moldipole_instance = moldipole()
        moldipole_instance.set_params(data,unitcell,timestep,temperature)
        method = getattr(moldipole_instance, method_name)
        return method()

class strategy_file(moldipole_abstract):
    """
    アルゴリズム（ConcreteStrategy）の具体的な実装
    """
    @classmethod
    def execute(cls,method_name: str,filename:str):
        moldipole_instance = read_file(filename)
        method = getattr(moldipole_instance, method_name)
        return method()

class Context:
    def __init__(self, strategy: moldipole_abstract):
        self._strategy = strategy

    def set_strategy(self, strategy: moldipole_abstract):
        self._strategy = strategy

    def execute_strategy(self, *args):
        return self._strategy.execute(*args)


# singleton dispatch decorator
@singledispatch
def calc_gfactor(arg):
    raise NotImplementedError(f"No strategy for {type(arg)}")

@calc_gfactor.register
def _(moldipole: moldipole):
    strategy = strategy_cls()
    return strategy.execute("calc_gfactor",moldipole)

@calc_gfactor.register
def _(data:np.ndarray,unitcell:np.ndarray,timestep:float,temperature:float):
    strategy = strategy_param()
    return strategy.execute("calc_gfactor",data,unitcell,timestep,temperature)

@calc_gfactor.register
def _(filename: str):
    strategy = strategy_file()
    return strategy.execute("calc_gfactor",filename)

# 引数がmoldipoleのみ -> calc_gfactorを呼びだす
# 引数がdataなどの場合 -> moldipoleを作成してcalc_gfactorを呼び出す
# これは，すべての関数に共通する処理として実装できる．
# あとは，それを自動で判定してcalc_gfactorで呼び出すようにすれば良い．

