import numpy as np
import torch
import logging
import os
import ase
import numpy as np
from typing import Callable, Optional, Union, Tuple, List, Literal
from cpmd.class_atoms_wan import atoms_wan
import ml.dataset.mldataset_abstract
from ml.dataset.mldataset_abstract import Factory_dataset


class DataSet_xyz_2(ml.dataset.mldataset_abstract.DataSet_abstract):
    '''
    原案：xyzを受け取り，そこからdescriptorを計算してdatasetにする．
    ただし，これだとやっぱりワニエの割り当て計算が重いので，それは先にやっておいて，
    atoms_wanクラスのリストとして入力を受け取った方が良い．．．
    '''
    def __init__(self,
                input_atoms_wan_list:list[atoms_wan], 
                bond_index, 
                desctype:str="allinone", 
                Rcs:float=4, Rc:float=6, 
                MaxAt:int=24, 
                bondtype:Literal["bond","lonepair"]="bond"):
        self.bond_index  = bond_index #  itp_data.bond_index['CC_1_bond'] etc
        self.desctype    = desctype # allinone or old
        self.Rcs:float     = Rcs
        self.Rc:float      = Rc
        self.MaxAt:int     = MaxAt
        self.bondtype:str = bondtype # bond or lonepair
        # convert from numpy to torch
        # descs_x = torch.from_numpy(descs_x.astype(np.float32)).clone()
        # true_y  = torch.from_numpy(true_y.astype(np.float32)).clone()
        self.data = input_atoms_wan_list
        # self.x =  descs_x     # 入力
        # self.y =  true_y     # 出力
        if bondtype not in ["bond", "lonepair"]:
            raise ValueError("ERROR :: bondtype should be bond or lonepair")
        
    def __len__(self)->float:
        return len(self.data) # データ数を返す
        
    def __getitem__(self, index):
        """index番目の入出力ペアを返す"""
        # atomic_coordinate
        # atomic_number
        # bond_list
        # true_y
        atomic_positions = self.data[index].input_atoms.get_positions()
        atomic_numbers   = self.data[index].input_atoms.get_atomic_numbers()
        unitcell_vector  = self.data[index].input_atoms.get_cell()
        if self.bondtype == "bond":
            true_y  = self.data[index].DESC.calc_bondmu_descripter_at_frame(self.data[index].list_mu_bonds, 
                                                                            self.bond_index) # .reshape(-1,3)
            bc_positions     = self.data[index].list_bond_centers[self.bond_index] # extract specific bond center
        elif self.bondtype == "lonepair":
            # !! hard code :: 酸素ローンペアに限定
            true_y  = self.data[index].list_mu_lpO.reshape(-1,3)  
            # TODO extract oxygen positions
            bc_positions     = atomic_positions
        elif self.bondtype == "coc":
            raise ValueError("ERROR :: For bondtype coc or coh, please use DataSet_xyz_coc")
        else: 
            raise ValueError("ERROR :: bondtype is not bond or lonepair")

        dict = { # input for descriptor.forward 
                "atomic_coordinate": torch.from_numpy(atomic_positions.astype(np.float32)).clone(),
                "atomic_numbers":    torch.from_numpy(atomic_numbers.astype(np.int32)).clone(),
                "bond_centers":      torch.from_numpy(bc_positions.astype(np.float32)).clone(),
                "UNITCELL_VECTOR":   torch.from_numpy(unitcell_vector.astype(np.float32)).clone(),
                "device": "cuda"
        }
        return dict, torch.from_numpy(true_y.astype(np.float32)).clone()

class ConcreteFactory_xyz_descs(Factory_dataset):
    def create_dataset(self, input_atoms_wan_list:list[atoms_wan], 
                        bond_index, desctype, 
                        Rcs:float=4, Rc:float=6, 
                        MaxAt:int=24, bondtype:str="bond"):
        return DataSet_xyz_2(
            input_atoms_wan_list, 
            bond_index, 
            desctype, Rcs, Rc, 
            MaxAt, bondtype
        )
