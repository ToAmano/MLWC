# example pg structure for test
# 32 molecule system

import pytest
import numpy as np
import ase
import ase.io
import os

@pytest.fixture
def pg_bond():
    # C, C, C, O, H, H, H, H, H
    bondlist=  [[1, 7], [1, 0], [2, 3], [2, 8], [2, 1], [3, 9], [4, 2], [5, 0], [6, 1], [10, 4], [11, 4], [12, 4]]
    NUM_MOL_PAR_MOL=13
    ref_atom_index=2
    return bondlist, NUM_MOL_PAR_MOL, ref_atom_index

@pytest.fixture
def pg_atoms():
    return ase.io.read(os.path.dirname(__file__)+"/pg.xyz")

@pytest.fixture
def pg_pbc_atoms():
    return np.loadtxt(os.path.dirname(__file__)+"/list_mol_coords.txt")

@pytest.fixture
def pg_bc():
    return np.loadtxt(os.path.dirname(__file__)+"/list_bond_centers.txt")

@pytest.fixture
def pg_ch_truey():
    return np.zeros(3)

@pytest.fixture
def pg_co_truey():
    return np.zeros(3)

@pytest.fixture
def pg_oh_truey():
    return np.zeros(3)

@pytest.fixture
def pg_cc_truey():
    return np.zeros(3)

@pytest.fixture
def pg_olp_truey():
    return np.zeros(3)

@pytest.fixture
def pg_ch_descriptor():
    descs = np.loadtxt(os.path.dirname(__file__)+"/descs_ch.txt")
    return descs

@pytest.fixture
def pg_co_descriptor():
    return np.loadtxt(os.path.dirname(__file__)+"/descs_co.txt")

@pytest.fixture
def pg_oh_descriptor():
    return np.loadtxt(os.path.dirname(__file__)+"/descs_oh.txt")

@pytest.fixture
def pg_cc_descriptor():
    return np.loadtxt(os.path.dirname(__file__)+"/descs_cc.txt")

@pytest.fixture
def pg_olp_descriptor():
    return np.loadtxt(os.path.dirname(__file__)+"/descs_olp.txt")