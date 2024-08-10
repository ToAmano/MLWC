import numpy as np
import torch
import logging
import os
import ase
import numpy as np
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

