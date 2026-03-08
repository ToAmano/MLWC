"""

- 事前処理として，割り当ては完了させ(atoms, dict_wcs) ておく．
- bondcentersを先に計算しておくか，あとから計算するか．
- 一旦dataset内で計算させるようにしよう．
- descriptorでは，bcs, atoms, unitcell, atomic_numberを入力する．
- C++側から利用する場合，先にbondcentersを計算する必要がある．

"""

from typing import Dict, Tuple

import numpy as np
import torch

# from mlwc.cpmd.class_atoms_wan import atoms_wan
from mlwc.cpmd.assign_wcs.assign_wcs_torch import atoms_wan
from mlwc.ml.dataset.mldataset_abstract import AbstractDataset, Factory_dataset


def prepare_input_tensor(
    data: atoms_wan, bond_key: str, device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """prepare Tensor inputs for dataset from atoms_wan"""

    def to_tensor(x: np.ndarray, dtype=np.float32, requires_grad=True):
        return torch.from_numpy(x.astype(dtype)).clone().requires_grad_(requires_grad)

    return {
        "atomic_numbers": to_tensor(
            data.atoms_nowan.get_atomic_numbers(),
            dtype=np.int32,
            requires_grad=False,
        ),
        "atomic_coordinate": to_tensor(data.atoms_nowan.get_positions()),
        "unitcell_vector": to_tensor(data.atoms_nowan.get_cell()),
        "bond_centers": to_tensor(data.dict_bcs[bond_key].reshape(-1, 3)),
    }


class DatasetAtoms(AbstractDataset):
    """
    原案：xyzを受け取り，そこからdescriptorを計算してdatasetにする．
    ただし，これだとやっぱりワニエの割り当て計算が重いので，それは先にやっておいて，
    atoms_wanクラスのリストとして入力を受け取った方が良い．．．
    """

    def __init__(self, input_atoms_wan_list: list[atoms_wan], bond_key: str):
        self.data = input_atoms_wan_list
        self.key = bond_key

    def __len__(self) -> float:
        """return number of data"""
        return len(self.data)

    def __getitem__(self, index) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """return 'index'-th data (x,y)"""
        indexed_data = self.data[index]
        dict = prepare_input_tensor(indexed_data, self.key, "cpu")
        true_y = indexed_data.dict_mu[self.key].reshape(-1, 3)
        return dict, torch.from_numpy(true_y.astype(np.float32)).clone().requires_grad_(
            True
        )


class ConcreteFactory_atoms(Factory_dataset):
    def create_dataset(self, input_atoms_wan_list: list[atoms_wan], bond_key: str):
        return DatasetAtoms(input_atoms_wan_list, bond_key)


#
