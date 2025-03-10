import numpy as np
from cpmd.pbc.pbc import pbc_abstract
import torch
from include.mlwc_logger import root_logger
logger = root_logger("MLWC."+__name__)


class pbc_2d_torch(pbc_abstract):
    """
    Strategy インターフェイスを実装するクラス
    """
    @classmethod
    def compute_pbc(cls,vectors_array:torch.tensor,cell:np.ndarray,device:str)->torch.tensor:
        """compute pbc

        Args:
            vectors_array (np.ndarray): vectors array
            cell (np.ndarray): cell

        Returns:
            np.ndarray: pbc vectors array
        """
        # vectors_arrayの形状を確認
        if vectors_array.ndim != 2 or vectors_array.shape[1] != 3:
            raise ValueError(f"Invalid shape for vectors_array. Expected shape [a, 3], but got {np.shape(vectors_array)}.")
        # vectors_array = torch.tensor(np.array(vectors_array, dtype="float32")).to(device)
        vectors_array = vectors_array.to(device)
        cell = torch.tensor(np.array(cell, dtype="float32")).to(device)
        # !! caution torch.dot only apply to 1D tensor. Instead, we use torch.mm for 2D*2D matrix product        
        pbc_vectors = torch.mm(vectors_array, torch.linalg.inv(cell.T))
        pbc_vectors -= torch.round(pbc_vectors)
        pbc_vectors = torch.mm(pbc_vectors, cell.T)
        return pbc_vectors #.to("cpu") # .detach().numpy() 

class pbc_3d_torch(pbc_abstract):
    """
    Strategy インターフェイスを実装するクラス
    """
    @classmethod
    def compute_pbc(cls,vectors_array:torch.tensor, cell:np.ndarray,device:str) -> torch.tensor:
        """
        3次元ベクトル配列に対して周期境界条件（PBC）を適用します。

        Args:
            vectors_array (torch.tensor): PBCを適用する3次元ベクトル配列。形状は(a, b, 3)である必要があります。
            cell (np.ndarray): 単位格子のセルパラメータ。形状は(3, 3)である必要があります。
            device (str): 計算に使用するデバイス（例: "cpu", "cuda"）。

        Returns:
            torch.tensor: PBCが適用されたベクトル配列。形状は(a, b, 3)です。

        詳細:
            この関数は、3次元ベクトル配列に対して周期境界条件を適用します。
            ベクトル配列は、一度2次元配列にreshapeされ、pbc_2d_torch.compute_pbc関数を用いてPBCが適用されます。
            その後、元の3次元形状に戻されます。
        """
        # vectors_arrayの形状を確認
        if vectors_array.ndim != 3 or vectors_array.shape[2] != 3:
            raise ValueError(f"Invalid shape for vectors_array. Expected shape [a, b, 3], but got {vectors_array.shape}.")

        # vectors_array全体に対して周期境界条件を適用
        a, b, _ = vectors_array.shape
        reshaped_vectors = vectors_array.reshape(-1, 3)  # [a*b, 3] に変換
        
        # compute pbc for 2d vectors
        pbc_vectors = pbc_2d_torch.compute_pbc(reshaped_vectors, cell, device)

        # 元の形 [a, b, 3] に戻す
        pbc_vectors = pbc_vectors.reshape(a, b, 3)
        return pbc_vectors # .to("cpu")



def compute_pbc(vectors_array:np.ndarray,cell:np.ndarray)->np.ndarray:
    """
    指定されたベクトル配列に対して、周期境界条件（PBC）を適用します。

    Args:
        vectors_array (np.ndarray): PBCを適用するベクトル配列。形状は(N, 3)である必要があります。
        cell (np.ndarray): 単位格子のセルパラメータ。形状は(3, 3)である必要があります。

    Returns:
        np.ndarray: PBCが適用されたベクトル配列。

    詳細:
        この関数は、与えられたベクトル配列と単位格子セルに基づいて、周期境界条件を適用します。
        ベクトルはまず、逆格子空間に変換され、最近傍の単位格子に折りたたまれ、その後、元の空間に戻されます。
        この操作により、ベクトルが単位格子内に収まるように調整されます。
    """
    # vectors_arrayの形状を確認
    if vectors_array.ndim != 2 or vectors_array.shape[1] != 3:
        raise ValueError(f"Invalid shape for vectors_array. Expected shape [a, 3], but got {vectors_array.shape}.")

    pbc_vectors = np.dot(vectors_array, np.linalg.inv(cell.T))
    pbc_vectors -= np.round(pbc_vectors)
    pbc_vectors = np.dot(pbc_vectors, cell.T)
    return pbc_vectors

