import numpy as np
import torch
import logging
import os
import ase
import numpy as np
import __version__
from typing import Callable, Optional, Union, Tuple, List
from cpmd.class_atoms_wan import atoms_wan
from abc import (
    ABC,
    abstractmethod,
)

class DataSet_abstract(ABC):
    """abstract method for dataset

    Args:
        ABC (_type_): _description_

    Returns:
        _type_: _description_
    """
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def __len__(self):
        pass
        
    @abstractmethod
    def __getitem__(self, index):
        pass

    @property
    def logger(self):
        # return logging.getLogger(self.logfile)
        return logging.getLogger("DataSet")

class Factory_dataset(ABC):
    # create dataset 
    @abstractmethod
    def create_dataset(self):
        pass


class DataSetContext:
    # context for stratefy pattern  
    def __init__(self, strategy: Factory_dataset):
        self._strategy = strategy

    def create_dataset(self, input_atoms_wan_list:list[atoms_wan], bond_index, desctype:str="allinone", Rcs:float=4, Rc:float=6, MaxAt:int=24, bondtype:str="bond"):
        return self._strategy.create_dataset(
                                            input_atoms_wan_list, bond_index, 
                                            desctype, Rcs, Rc, 
                                            MaxAt, bondtype)
