import os

import numpy as np
import pytest
from ase import Atoms

from mlwc.bond.extractor_rdkit import ReadMolFile, create_molecular_info


def test_read_met():
    """test using metanol.mol"""
    met = create_molecular_info(
        os.path.dirname(__file__) + "/../../src/mlwc/dataset/methanol/methanol.mol"
    )
    true_bond_list = [[0, 4], [0, 2], [0, 3], [1, 0], [5, 1]]
    true_atom_list = ["C", "O", "H", "H", "H", "H"]
    true_bonds_type = [1, 1, 1, 1, 1]
    true_ch_bond = [[0, 4], [0, 2], [0, 3]]
    true_co_bond = [[1, 0]]
    true_oh_bond = [[5, 1]]
    assert true_bond_list == met.bonds_list
    assert true_atom_list == met.atom_list
    assert true_bonds_type == met.bonds_type
    assert met.num_atoms_per_mol == 6
    assert true_ch_bond == met.bonds["CH_1_bond"]
    assert true_co_bond == met.bonds["CO_1_bond"]
    assert true_oh_bond == met.bonds["OH_1_bond"]


def test_read_eth():
    """test using eth.mol"""
    eth = create_molecular_info(
        os.path.dirname(__file__) + "/../../src/mlwc/dataset/ethanol/ethanol.mol"
    )
    print(eth.bonds_list)
    true_bond_list = [[0, 4], [0, 1], [1, 6], [1, 7], [2, 8], [2, 1], [3, 0], [5, 0]]
    true_atom_list = ["C", "C", "O", "H", "H", "H", "H", "H", "H"]
    true_bonds_type = [1, 1, 1, 1, 1, 1, 1, 1]
    true_ch_bond = [[0, 4], [1, 6], [1, 7], [3, 0], [5, 0]]
    true_co_bond = [[2, 1]]
    true_oh_bond = [[2, 8]]
    true_cc_bond = [[0, 1]]
    assert true_bond_list == eth.bonds_list
    assert true_atom_list == eth.atom_list
    assert true_bonds_type == eth.bonds_type
    assert eth.num_atoms_per_mol == 9
    assert true_ch_bond == eth.bonds["CH_1_bond"]
    assert true_co_bond == eth.bonds["CO_1_bond"]
    assert true_oh_bond == eth.bonds["OH_1_bond"]
    assert true_cc_bond == eth.bonds["CC_1_bond"]


def test_read_pg():
    """test using pg.mol"""
    pg = create_molecular_info(
        os.path.dirname(__file__) + "/../../src/mlwc/dataset/pg/pg.mol"
    )
    true_bond_list = [
        [1, 7],
        [1, 0],
        [2, 3],
        [2, 8],
        [2, 1],
        [3, 9],
        [4, 2],
        [5, 0],
        [6, 1],
        [10, 4],
        [11, 4],
        [12, 4],
    ]
    true_atom_list = ["O", "C", "C", "O", "C", "H", "H", "H", "H", "H", "H", "H", "H"]
    true_bonds_type = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    true_ch_bond = [[1, 7], [2, 8], [6, 1], [10, 4], [11, 4], [12, 4]]
    true_co_bond = [[1, 0], [2, 3]]
    true_oh_bond = [[3, 9], [5, 0]]
    true_cc_bond = [[2, 1], [4, 2]]
    assert true_bond_list == pg.bonds_list
    assert true_atom_list == pg.atom_list
    assert true_bonds_type == pg.bonds_type
    assert pg.num_atoms_per_mol == 13
    assert true_ch_bond == pg.bonds["CH_1_bond"]
    assert true_co_bond == pg.bonds["CO_1_bond"]
    assert true_oh_bond == pg.bonds["OH_1_bond"]
    assert true_cc_bond == pg.bonds["CC_1_bond"]
