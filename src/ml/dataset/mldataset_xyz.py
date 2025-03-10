import numpy as np
import torch
import logging
import os
import ase
import numpy as np
from typing import Callable, Optional, Union, Tuple, List
from cpmd.class_atoms_wan import atoms_wan
import ml.dataset.mldataset_abstract
from ml.dataset.mldataset_abstract import Factory_dataset


class DataSet_xyz(ml.dataset.mldataset_abstract.DataSet_abstract):
    '''
    原案：xyzを受け取り，そこからdescriptorを計算してdatasetにする．
    ただし，これだとやっぱりワニエの割り当て計算が重いので，それは先にやっておいて，
    atoms_wanクラスのリストとして入力を受け取った方が良い．．．
    '''
    def __init__(self,input_atoms_wan_list:list[atoms_wan], bond_index, desctype:str="allinone", Rcs:float=4, Rc:float=6, MaxAt:int=24, bondtype:str="bond"):
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
        # self.x[index], self.y[index]
        # index番目の入出力ペアを返す
        # tmp = self.data[index]
        if self.bondtype == "bond":
            descs_x = self.data[index].DESC.calc_bond_descripter_at_frame(self.data[index].atoms_nowan, 
                                                                        self.data[index].list_bond_centers, 
                                                                        self.bond_index, self.desctype, 
                                                                        self.Rcs, self.Rc, self.MaxAt) # .reshape(-1,288)
            true_y  = self.data[index].DESC.calc_bondmu_descripter_at_frame(self.data[index].list_mu_bonds, 
                                                                            self.bond_index) # .reshape(-1,3)
            # print(f" SHAPE of DESCS_X = {np.shape(descs_x)}  :: DESCS_Y = {np.shape(true_y)}")
            return torch.from_numpy(descs_x.astype(np.float32)).clone(), torch.from_numpy(true_y.astype(np.float32)).clone()
        elif self.bondtype == "lonepair":
            # !! hard code :: 酸素ローンペアに限定
            descs_x = self.data[index].DESC.calc_lonepair_descripter_at_frame_type2(self.data[index].atoms_nowan, 
                                                                                    self.data[index].list_mol_coords, 
                                                                                    self.bond_index, 
                                                                                    self.desctype, 
                                                                                    self.Rcs, self.Rc, self.MaxAt)
            true_y  = self.data[index].list_mu_lpO.reshape(-1,3)  
            # print(f" SHAPE of DESCS_X = {np.shape(descs_x)}  :: DESCS_Y = {np.shape(true_y)}")
            return torch.from_numpy(descs_x.astype(np.float32)).clone(), torch.from_numpy(true_y.astype(np.float32)).clone()
        elif self.bondtype == "coc":
            raise ValueError("ERROR :: For bondtype coc or coh, please use DataSet_xyz_coc")
        else: 
            raise ValueError("ERROR :: bondtype is not bond or lonepair")


class ConcreteFactory_xyz(Factory_dataset):
    def create_dataset(self, input_atoms_wan_list:list[atoms_wan], bond_index, desctype, Rcs:float=4, Rc:float=6, MaxAt:int=24, bondtype:str="bond"):
        return DataSet_xyz(
            input_atoms_wan_list, bond_index, 
            desctype, Rcs, Rc, 
            MaxAt, bondtype
        )


# ========================================================================================================
class DataSet_xyz_coc(ml.dataset.mldataset_abstract.DataSet_abstract):
    '''
    原案：xyzを受け取り，そこからdescriptorを計算してdatasetにする．
    ただし，これだとやっぱりワニエの割り当て計算が重いので，それは先にやっておいて，
    atoms_wanクラスのリストとして入力を受け取った方が良い．．．
    '''
    def __init__(self,input_atoms_wan_list:list[atoms_wan], 
                itp_data, 
                desctype, 
                Rcs:float=4, Rc:float=6, 
                MaxAt:int=24, bondtype:str="coc"):
        self.itp_data    = itp_data
        self.desctype    = desctype
        self.Rcs         = Rcs
        self.Rc          = Rc
        self.MaxAt       = MaxAt
        self.bondtype    = bondtype
        # convert from numpy to torch
        # descs_x = torch.from_numpy(descs_x.astype(np.float32)).clone()
        # true_y  = torch.from_numpy(true_y.astype(np.float32)).clone()
        self.data = input_atoms_wan_list
        # self.x =  descs_x     # 入力
        # self.y =  true_y     # 出力
        
    def __len__(self)->float:
        return len(self.data) # データ数を返す
        
    def __getitem__(self, index):
        # self.x[index], self.y[index]
        # index番目の入出力ペアを返す
        # tmp = self.data[index]
        if self.bondtype == "coc":
            # hard code :: 酸素ローンペアに限定
            descs_x = self.data[index].DESC.calc_coc_descripter_at_frame(self.data[index].atoms_nowan, self.data[index].list_mol_coords, self.itp_data.coc_index, self.desctype, self.Rcs, self.Rc, self.MaxAt)
            true_y  = self.data[index].DESC.calc_coc_bondmu_descripter_at_frame(self.data[index].list_mu_bonds, self.data[index].list_mu_lpO, self.itp_data.coc_index, self.itp_data.bond_index['CO_1_bond'])
            return torch.from_numpy(descs_x.astype(np.float32)).clone(), torch.from_numpy(true_y.astype(np.float32)).clone()
        elif self.bondtype == "coh":
            descs_x = self.data[index].DESC.calc_coc_descripter_at_frame(self.data[index].atoms_nowan, self.data[index].list_mol_coords, self.itp_data.coh_index, self.desctype, self.Rcs, self.Rc, self.MaxAt)
            true_y  = self.data[index].DESC.calc_coh_bondmu_descripter_at_frame(self.data[index].list_mu_bonds, self.data[index].list_mu_lpO, self.itp_data.coh_index, self.itp_data.bond_index['CO_1_bond'], self.itp_data.bond_index['OH_1_bond'])
            # print(f" SHAPE of DESCS_X = {np.shape(descs_x)}  :: DESCS_Y = {np.shape(true_y)}")
            return torch.from_numpy(descs_x.astype(np.float32)).clone(), torch.from_numpy(true_y.astype(np.float32)).clone()
        else:                        
            raise ValueError("ERROR :: bondtype shoud be COC or COH")
            

class ConcreteFactory_xyz_coc(Factory_dataset):
    def create_dataset(self, input_atoms_wan_list:list[atoms_wan], bond_index, desctype, Rcs:float=4, Rc:float=6, MaxAt:int=24, bondtype:str="bond"):
        return DataSet_xyz_coc(
            input_atoms_wan_list, bond_index, 
            desctype, Rcs, Rc, 
            MaxAt, bondtype
        )
