import numpy as np
import torch
import logging
import os
import ase
import numpy as np
from typing import Callable, Optional, Union, Tuple, List
from cpmd.class_atoms_wan import atoms_wan


class DataSet_custom():
    '''
    numpy.arrayを受け取り，そこからtensorにしてdatasetにする
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    入力としてnumpy arrayを受け取る想定で作成してみよう．
    '''
    def __init__(self,descs_x:np.ndarray,true_y:np.ndarray):
        # 記述子の形は，(フレーム数*ボンド数，記述子の次元数)となっている．これが前提なので注意
        self.logger.info(" ==  reading descs_x and true_y == ")
        self.logger.info(f"shape descs_x :: {np.shape(descs_x)}")
        self.logger.info(f"shape true_y  :: {np.shape(true_y)}" )
        self.logger.info(f"max descs_x   :: {np.max(descs_x)}"  )

        # convert from numpy to torch
        descs_x = torch.from_numpy(descs_x.astype(np.float32)).clone()
        true_y  = torch.from_numpy(true_y.astype(np.float32)).clone()
        self.x =  descs_x     # 入力
        self.y =  true_y     # 出力
        
    def __len__(self):
        return len(self.x) # データ数を返す
        
    def __getitem__(self, index):
        # index番目の入出力ペアを返す
        return self.x[index], self.y[index]
    @property
    def logger(self):
        # return logging.getLogger(self.logfile)
        return logging.getLogger("DataSet")

class DataSet_xyz():
    '''
    原案：xyzを受け取り，そこからdescriptorを計算してdatasetにする．
    ただし，これだとやっぱりワニエの割り当て計算が重いので，それは先にやっておいて，
    atoms_wanクラスのリストとして入力を受け取った方が良い．．．
    
    '''
    def __init__(self,input_atoms_wan_list:list[atoms_wan], bond_index, desctype, Rcs=4, Rc=6, MaxAt=24):
        self.bond_index = bond_index
        self.desctype   = desctype
        self.Rcs       = Rcs
        self.Rc       = Rc
        self.MaxAt       = MaxAt
        # convert from numpy to torch
        # descs_x = torch.from_numpy(descs_x.astype(np.float32)).clone()
        # true_y  = torch.from_numpy(true_y.astype(np.float32)).clone()
        self.data = input_atoms_wan_list
        # self.x =  descs_x     # 入力
        # self.y =  true_y     # 出力
        
    def __len__(self):
        return len(self.data) # データ数を返す
        
    def __getitem__(self, index):
        # self.x[index], self.y[index]
        # index番目の入出力ペアを返す
        # tmp = self.data[index]
        # TODO :: 288がhard codeなので修正する
        descs_x = self.data[index].DESC.calc_bond_descripter_at_frame(self.data[index].atoms_nowan, self.data[index].list_bond_centers, self.bond_index, self.desctype, self.Rcs, self.Rc, self.MaxAt) # .reshape(-1,288)
        true_y  = self.data[index].DESC.calc_bondmu_descripter_at_frame(self.data[index].list_mu_bonds, self.bond_index) # .reshape(-1,3)
        return torch.from_numpy(descs_x.astype(np.float32)).clone(), torch.from_numpy(true_y.astype(np.float32)).clone()

    
    @property
    def logger(self):
        # return logging.getLogger(self.logfile)
        return logging.getLogger("DataSet")