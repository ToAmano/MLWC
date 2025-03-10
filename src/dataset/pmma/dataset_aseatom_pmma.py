# example pmma structure for test
# 32 molecule system

import pytest
import numpy as np
import ase
import ase.io
import os

@pytest.fixture
def pmma_bond():
    # C, C, C, O, H, H, H, H, H
    bondlist=  [[1, 0], [1, 24], [2, 1], [2, 19], [3, 2], [4, 2], [4, 5], [5, 15], [5, 6], [6, 32], [6, 31], [7, 8], [7, 5], [7, 34], [8, 14], [9, 10], [9, 8], [11, 35], [11, 10], [12, 9], [13, 39], [13, 8], [13, 38], [15, 16], [16, 17], [17, 41], [17, 43], [18, 15], [19, 22], [19, 20], [20, 21], [21, 46], [21, 45], [23, 1], [25, 3], [26, 3], [27, 3], [28, 4], [29, 4], [30, 6], [33, 7], [36, 11], [37, 11], [40, 13], [42, 17], [44, 21]]
    NUM_MOL_PAR_MOL=47
    ref_atom_index=4
    return bondlist, NUM_MOL_PAR_MOL, ref_atom_index

@pytest.fixture
def pmma_atoms():
    return ase.io.read(os.path.dirname(__file__)+"/pmma.xyz")

@pytest.fixture
def pmma_pbc_atoms():
    return np.loadtxt(os.path.dirname(__file__)+"/list_mol_coords.txt")

@pytest.fixture
def pmma_bc():
    return np.loadtxt(os.path.dirname(__file__)+"/list_bond_centers.txt")

@pytest.fixture
def pmma_ch_truey():
    return np.zeros(3)

@pytest.fixture
def pmma_co_truey():
    return np.zeros(3)

@pytest.fixture
def pmma_oh_truey():
    return np.zeros(3)

@pytest.fixture
def pmma_cc_truey():
    return np.zeros(3)

@pytest.fixture
def pmma_olp_truey():
    return np.zeros(3)

@pytest.fixture
def pmma_ch_descriptor():
    descs = np.loadtxt(os.path.dirname(__file__)+"/descs_ch.txt")
    return descs

@pytest.fixture
def pmma_co_descriptor():
    return np.loadtxt(os.path.dirname(__file__)+"/descs_co.txt")

@pytest.fixture
def pmma_oh_descriptor():
    return np.loadtxt(os.path.dirname(__file__)+"/descs_oh.txt")

@pytest.fixture
def pmma_cc_descriptor():
    return np.loadtxt(os.path.dirname(__file__)+"/descs_cc.txt")

@pytest.fixture
def pmma_olp_descriptor():
    return np.loadtxt(os.path.dirname(__file__)+"/descs_olp.txt")