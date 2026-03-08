from typing import Dict, Tuple

import numpy as np
import torch

from mlwc.cpmd.assign_wcs.assign_wcs_torch import atoms_wan
from mlwc.ml.transform.registry import register_transform


@register_transform("NET_withoutBN_descs")
class DeePMDModelTransform:
    def __init__(self, bondtype: str):
        self.bond_key = bondtype

    def __call__(self, data: atoms_wan) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        def to_tensor(x: np.ndarray, dtype=np.float32, requires_grad=True):
            return (
                torch.from_numpy(x.astype(dtype)).clone().requires_grad_(requires_grad)
            )

        input_dict = {
            "atomic_numbers": to_tensor(
                data.atoms_nowan.get_atomic_numbers(),
                dtype=np.int32,
                requires_grad=False,
            ),
            "atomic_coordinate": to_tensor(data.atoms_nowan.get_positions()),
            "unitcell_vector": to_tensor(data.atoms_nowan.get_cell()),
            "bond_centers": to_tensor(data.dict_bcs[self.bond_key].reshape(-1, 3)),
        }
        # --- 以前__getitem__にあったラベル作成ロジック ---
        true_value = (
            torch.from_numpy(
                data.dict_mu[self.bond_key].reshape(-1, 3).astype(np.float32)
            )
            .clone()
            .requires_grad_(True)
        )
        # -----------------------------------------
        return input_dict, true_value
