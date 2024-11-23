# example pg12 structure for test
# 32 molecule system

import pytest
import numpy as np
import ase
import ase.io
import os

@pytest.fixture
def pg12_bond():
    # C, C, C, O, H, H, H, H, H
    bondlist= [[1, 2], [1, 52], [1, 0], [2, 53], [2, 4], [3, 55], [3, 54], 
               [3, 2], [4, 5], [5, 58], [5, 6], [5, 57], [6, 59], [6, 7], 
               [6, 8], [7, 61], [7, 60], [8, 9], [9, 64], [9, 63], [10, 65], 
               [10, 11], [10, 12], [10, 9], [11, 66], [11, 68], [13, 12],
               [13, 70], [13, 69], [14, 71], [14, 16], [14, 13], [15, 72], 
               [15, 74], [15, 14], [17, 16], [17, 18], [18, 77], [18, 19],
               [18, 20], [19, 78], [19, 79], [20, 21], [21, 22], [21, 82],
               [22, 83], [22, 24], [23, 85], [23, 22], [24, 25], [25, 87], 
               [25, 26], [26, 89], [26, 27], [26, 28], [27, 92], [27, 90],
               [29, 28], [29, 93], [30, 95], [30, 32], [30, 31], [30, 29],
               [31, 98], [33, 34], [33, 32], [34, 101], [34, 35], [35, 103],
               [35, 104], [35, 102], [36, 34], [37, 105], [37, 36], [38, 107], 
               [38, 39], [38, 37], [39, 109], [39, 110], [40, 38], [41, 112], 
               [41, 111], [41, 40], [42, 113], [42, 41], [43, 42], [43, 116], 
               [44, 42], [45, 44], [45, 117], [46, 119], [46, 45], [46, 47], 
               [47, 122], [47, 120], [48, 46], [49, 48], [50, 0], [51, 1], 
               [56, 3], [62, 7], [67, 11], [73, 15], [75, 17], [76, 17],
               [80, 19], [81, 21], [84, 23], [86, 23], [88, 25], [91, 27],
               [94, 29], [96, 31], [97, 31], [99, 33], [100, 33], [106, 37], 
               [108, 39], [114, 43], [115, 43], [118, 45], [121, 47]]
    NUM_MOL_PAR_MOL=123
    ref_atom_index=23
    return bondlist, NUM_MOL_PAR_MOL, ref_atom_index

@pytest.fixture
def pg12_atoms():
    return ase.io.read(os.path.dirname(__file__)+"/pg12.xyz")

@pytest.fixture
def pg12_pbc_atoms():
    return np.loadtxt(os.path.dirname(__file__)+"/list_mol_coords.txt")

@pytest.fixture
def pg12_1mol_atoms():
    return ase.io.read(os.path.dirname(__file__)+"/pg12_1mol.xyz")

@pytest.fixture
def pg12_bc():
    return np.loadtxt(os.path.dirname(__file__)+"/list_bond_centers.txt")

@pytest.fixture
def pg12_ch_truey():
    return np.zeros(3)

@pytest.fixture
def pg12_co_truey():
    return np.zeros(3)

@pytest.fixture
def pg12_oh_truey():
    return np.zeros(3)

@pytest.fixture
def pg12_cc_truey():
    return np.zeros(3)

@pytest.fixture
def pg12_olp_truey():
    return np.zeros(3)

@pytest.fixture
def pg12_ch_descriptor():
    descs = np.loadtxt(os.path.dirname(__file__)+"/descs_ch.txt")
    return descs

@pytest.fixture
def pg12_co_descriptor():
    return np.loadtxt(os.path.dirname(__file__)+"/descs_co.txt")

@pytest.fixture
def pg12_oh_descriptor():
    return np.loadtxt(os.path.dirname(__file__)+"/descs_oh.txt")

@pytest.fixture
def pg12_cc_descriptor():
    return np.loadtxt(os.path.dirname(__file__)+"/descs_cc.txt")

@pytest.fixture
def pg12_olp_descriptor():
    return np.loadtxt(os.path.dirname(__file__)+"/descs_olp.txt")