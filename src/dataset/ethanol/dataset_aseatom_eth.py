# example methanol structure for test
# 32 molecule system

import pytest
import numpy as np
import ase
import ase.io
import os

@pytest.fixture
def ethanol_bond():
    # C, C, C, O, H, H, H, H, H
    bondlist= [[0, 4], [0, 1], [1, 6], [1, 7], [2, 8], [2, 1], [3, 0], [5, 0]]
    NUM_MOL_PAR_MOL=9
    ref_atom_index=1
    return bondlist, NUM_MOL_PAR_MOL, ref_atom_index

@pytest.fixture
def ethanol_atoms():
    return ase.io.read(os.path.dirname(__file__)+"/ethanol.xyz")

@pytest.fixture
def ethanol_pbc_atoms():
    return np.loadtxt(os.path.dirname(__file__)+"/list_mol_coords.txt")

@pytest.fixture
def ethanol_bc():
    return np.loadtxt(os.path.dirname(__file__)+"/list_bond_centers.txt")

@pytest.fixture
def ethanol_ch_truey():
    return np.zeros(3)

@pytest.fixture
def ethanol_co_truey():
    return np.zeros(3)

@pytest.fixture
def ethanol_oh_truey():
    return np.zeros(3)

@pytest.fixture
def ethanol_cc_truey():
    return np.zeros(3)

@pytest.fixture
def ethanol_olp_truey():
    return np.zeros(3)

@pytest.fixture
def ethanol_ch_descriptor():
    descs = np.loadtxt(os.path.dirname(__file__)+"/descs_ch.txt")
    return descs

@pytest.fixture
def ethanol_co_descriptor():
    return np.loadtxt(os.path.dirname(__file__)+"/descs_co.txt")

@pytest.fixture
def ethanol_oh_descriptor():
    return np.loadtxt(os.path.dirname(__file__)+"/descs_oh.txt")

@pytest.fixture
def ethanol_cc_descriptor():
    return np.loadtxt(os.path.dirname(__file__)+"/descs_cc.txt")

@pytest.fixture
def ethanol_olp_descriptor():
    return np.loadtxt(os.path.dirname(__file__)+"/descs_olp.txt")