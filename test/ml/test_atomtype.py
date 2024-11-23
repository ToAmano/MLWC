# Test functions for src/cpmd/pbc/pbc_mol.py
import pytest
import numpy as np
import os
from ase import Atoms
from ml.atomtype import read_mol


def test_read_eth():
    eth = read_mol(os.path.dirname(__file__)+"/../../src/dataset/ethanol/ethanol.mol")
    print(eth.bonds_list)
    
    

