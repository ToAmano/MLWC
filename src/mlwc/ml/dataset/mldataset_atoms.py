"""

- 事前処理として，割り当ては完了させ(atoms, dict_wcs) ておく．
- bondcentersを先に計算しておくか，あとから計算するか．
- 一旦dataset内で計算させるようにしよう．
- descriptorでは，bcs, atoms, unitcell, atomic_numberを入力する．
- C++側から利用する場合，先にbondcentersを計算する必要がある．

"""

import numpy as np
import torch
import logging
import os
import ase
import numpy as np
from typing import Callable, Optional, Union, Tuple, List, Literal
from mlwc.cpmd.class_atoms_wan import atoms_wan
import ml.dataset.mldataset_abstract
from mlwc.ml.dataset.mldataset_abstract import Factory_dataset


class DataSet_atoms(ml.dataset.mldataset_abstract.DataSet_abstract):
    '''
    原案：xyzを受け取り，そこからdescriptorを計算してdatasetにする．
    ただし，これだとやっぱりワニエの割り当て計算が重いので，それは先にやっておいて，
    atoms_wanクラスのリストとして入力を受け取った方が良い．．．
    '''

    def __init__(self,
                 input_atoms_wan_list: list[atoms_wan],
                 bond_key):
        self.data = input_atoms_wan_list
        self.key = bond_key

    def __len__(self) -> float:
        return len(self.data)  # データ数を返す

    def __getitem__(self, index):
        """index番目の入出力ペアを返す"""
        # atomicdata
        atomic_positions = self.data[index].atoms_nowan.get_positions()
        atomic_numbers = self.data[index].atoms_nowan.get_atomic_numbers()
        unitcell_vector = self.data[index].atoms_nowan.get_cell()

        # bcsdict_bcs
        bc_positions = self.data[index].dict_bcs[self.key].reshape(-1, 3)
        # true_y
        true_y = self.data[index].dict_mu[self.key].reshape(-1, 3)

        dict = {  # input for descriptor.forward
            "atomic_numbers":    torch.from_numpy(atomic_numbers.astype(np.int32)).clone(),
            "atomic_coordinate": torch.from_numpy(atomic_positions.astype(np.float32)).clone().requires_grad_(True),
            "UNITCELL_VECTOR":   torch.from_numpy(unitcell_vector.astype(np.float32)).clone().requires_grad_(True),
            "bond_centers":      torch.from_numpy(bc_positions.astype(np.float32)).clone().requires_grad_(True),
            "device": "cpu"
        }
        return dict, torch.from_numpy(true_y.astype(np.float32)).clone().requires_grad_(True)


class ConcreteFactory_atoms(Factory_dataset):
    def create_dataset(self, input_atoms_wan_list: list[atoms_wan],
                       bond_key: str):
        return DataSet_atoms(
            input_atoms_wan_list,
            bond_key)
