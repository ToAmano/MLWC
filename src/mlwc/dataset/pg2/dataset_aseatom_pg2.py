# example pg2 structure for test
# 32 molecule system

import pytest
import numpy as np
import ase
import ase.io
import os

@pytest.fixture
def pg2_bond():
    # C, C, C, O, H, H, H, H, H
    bondlist= [[0, 9], [1, 10], [1, 2], [1, 0], [2, 12], [2, 4], [3, 14], [3, 2], [3, 13], [4, 5], [5, 6], [6, 18], [6, 7], [6, 8], [7, 20], [11, 1], [15, 3], [16, 5], [17, 5], [19, 7], [21, 7], [22, 8]]
    NUM_MOL_PAR_MOL=23
    ref_atom_index=4
    return bondlist, NUM_MOL_PAR_MOL, ref_atom_index

@pytest.fixture
def pg2_atoms():
    return ase.io.read(os.path.dirname(__file__)+"/pg2.xyz")

@pytest.fixture
def pg2_pbc_atoms():
    return np.loadtxt(os.path.dirname(__file__)+"/list_mol_coords.txt")

@pytest.fixture
def pg2_bc():
    return np.loadtxt(os.path.dirname(__file__)+"/list_bond_centers.txt")

@pytest.fixture
def pg2_ch_truey():
    return np.zeros(3)

@pytest.fixture
def pg2_co_truey():
    return np.zeros(3)

@pytest.fixture
def pg2_oh_truey():
    return np.zeros(3)

@pytest.fixture
def pg2_cc_truey():
    return np.zeros(3)

@pytest.fixture
def pg2_olp_truey():
    return np.zeros(3)

@pytest.fixture
def pg2_ch_descriptor():
    descs = np.loadtxt(os.path.dirname(__file__)+"/descs_ch.txt")
    return descs

@pytest.fixture
def pg2_co_descriptor():
    return np.loadtxt(os.path.dirname(__file__)+"/descs_co.txt")

@pytest.fixture
def pg2_oh_descriptor():
    return np.loadtxt(os.path.dirname(__file__)+"/descs_oh.txt")

@pytest.fixture
def pg2_cc_descriptor():
    return np.loadtxt(os.path.dirname(__file__)+"/descs_cc.txt")

@pytest.fixture
def pg2_olp_descriptor():
    return np.loadtxt(os.path.dirname(__file__)+"/descs_olp.txt")