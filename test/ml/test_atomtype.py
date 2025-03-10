# Test functions for src/cpmd/pbc/pbc_mol.py
import pytest
import numpy as np
import os
from ase import Atoms
from ml.atomtype import read_mol


def test_read_met(): 
    """test using metanol.mol"""
    met = read_mol(os.path.dirname(__file__)+"/../../src/dataset/methanol/methanol.mol")
    true_bond_list  = [[0, 4], [0, 2], [0, 3], [1, 0], [5, 1]]
    true_atom_list  = ['C', 'O', 'H', 'H', 'H', 'H']
    true_bonds_type = [1, 1, 1, 1, 1]
    true_ch_bond = [[0, 4], [0, 2], [0, 3]]
    true_co_bond = [[1, 0]]
    true_oh_bond = [[5, 1]]
    assert true_bond_list == met.bonds_list
    assert true_atom_list == met.atom_list
    assert true_bonds_type == met.bonds_type
    assert met.num_atoms_per_mol == 6
    assert true_ch_bond == met.bonds['ch_1_bond']
    assert true_co_bond == met.bonds['co_1_bond']
    assert true_oh_bond == met.bonds['oh_1_bond']
    
    

def test_read_eth():
    """test using eth.mol"""
    eth = read_mol(os.path.dirname(__file__)+"/../../src/dataset/ethanol/ethanol.mol")
    print(eth.bonds_list)
    true_bond_list  = [[0, 4], [0, 1], [1, 6], [1, 7], [2, 8], [2, 1], [3, 0], [5, 0]]
    true_atom_list  = ['C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H']
    true_bonds_type = [1, 1, 1, 1, 1, 1, 1, 1]
    true_ch_bond = [[0, 4], [1, 6], [1, 7], [3, 0], [5, 0]]
    true_co_bond = [[2, 1]]
    true_oh_bond = [[2, 8]]
    true_cc_bond = [[0, 1]]
    assert true_bond_list == eth.bonds_list
    assert true_atom_list == eth.atom_list
    assert true_bonds_type == eth.bonds_type
    assert eth.num_atoms_per_mol == 9
    assert true_ch_bond == eth.bonds['ch_1_bond']
    assert true_co_bond == eth.bonds['co_1_bond']
    assert true_oh_bond == eth.bonds['oh_1_bond']
    assert true_cc_bond == eth.bonds['cc_1_bond']
    
def test_read_pg():
    """test using pg.mol"""
    pg = read_mol(os.path.dirname(__file__)+"/../../src/dataset/pg/pg.mol")
    true_bond_list = [[1, 7], [1, 0], [2, 3], [2, 8], [2, 1], [3, 9], [4, 2], [5, 0], [6, 1], [10, 4], [11, 4], [12, 4]]
    true_atom_list = ['O', 'C', 'C', 'O', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H']
    true_bonds_type = [ 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1]
    true_ch_bond = [[1, 7], [2, 8], [6, 1], [10, 4], [11, 4], [12, 4]]
    true_co_bond = [[1, 0], [2, 3]]
    true_oh_bond = [[3, 9], [5, 0]]
    true_cc_bond = [[2, 1], [4, 2]]
    assert true_bond_list == pg.bonds_list
    assert true_atom_list == pg.atom_list
    assert true_bonds_type == pg.bonds_type
    assert pg.num_atoms_per_mol == 13
    assert true_ch_bond == pg.bonds['ch_1_bond']
    assert true_co_bond == pg.bonds['co_1_bond']
    assert true_oh_bond == pg.bonds['oh_1_bond']
    assert true_cc_bond == pg.bonds['cc_1_bond']

@DeprecationWarning
def _get_bonds(bonds_list,bonds_type,atom_list,mol_rdkit):
    ch_bond=[]
    co_bond=[]
    oh_bond=[]
    oo_bond=[]
    cc_bond=[]
    ring_bond=[] # これがベンゼン環
    co_double_bond=[]
    for bond,type in zip(bonds_list,bonds_type):
        # 原子番号に変換
        tmp=[atom_list[bond[0]],atom_list[bond[1]]]
        
        if (tmp == ["H","C"]) & (type == 1): # CH
            ch_bond.append(bond)
        if (tmp == ["C","H"]) & (type == 1): # CH
            ch_bond.append(bond)
        if (tmp == ["O","C"]) & (type == 1): # CO
            co_bond.append(bond)
        if (tmp == ["C","O"]) & (type == 1): # CO
            co_bond.append(bond)
        if (tmp == ["H","O"]) & (type == 1): # OH
            oh_bond.append(bond)
        if (tmp == ["O","H"]) & (type == 1): # OH
            oh_bond.append(bond)
        if (tmp == ["O","O"]) & (type == 1): # OO
            oo_bond.append(bond)
        if (tmp == ["C","C"]) & (type == 1): # CC
            if mol_rdkit.GetAtoms()[bond[0]].GetIsAromatic() == True & mol_rdkit.GetAtoms()[bond[1]].GetIsAromatic() == True: 
                ring_bond.append(bond) # ベンゼン環が複数ある場合には未対応
            else:
                cc_bond.append(bond) # 芳香族以外のみ検出
        if (tmp == ["C","O"]) & (type == 2): # CO
            co_double_bond.append(bond)
        if (tmp == ["O","C"]) & (type == 2): # CO
            co_double_bond.append(bond)
    

    
    # TODO :: ベンゼン環は複数のリングに分解する．
    # この時，ナフタレンのようなことを考えると，完全には繋がっていない部分で分割するのが良い．
    # divide_cc_ring(ring_bond)
    ch_bond=ch_bond
    co_bond=co_bond
    oh_bond=oh_bond
    oo_bond=oo_bond
    cc_bond=cc_bond
    ring_bond=ring_bond
    co_double_bond=co_double_bond
    
    if len(ch_bond)+len(co_bond)+len(oh_bond)+len(oo_bond)+len(cc_bond)+len(ring_bond)+\
        len(co_double_bond) != len(bonds_list):
        raise ValueError(" WARNING :: There are unkown bonds in bonds_list... ")
    
    print(" ===========  _get_bonds ========== ")
    print(f" CH bonds...        {ch_bond}")
    print(f" CO bonds...        {co_bond}")
    print(f" OH bonds...        {oh_bond}")
    print(f" OO bonds...        {oo_bond}")
    print(f" CC bonds...        {cc_bond}")
    print(f" CC ring bonds...   {ring_bond}")
    print(f" CO double bonds... {co_double_bond}")
    print(" ")
    
    return 0
