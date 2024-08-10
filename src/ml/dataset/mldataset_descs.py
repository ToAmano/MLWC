import numpy as np
import torch
import logging
import os
import ase
import numpy as np
from typing import Callable, Optional, Union, Tuple, List
from cpmd.class_atoms_wan import atoms_wan
import mldataset_abstract

class DataSet_descs(mldataset_abstract.DataSet_abstract):
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