"""calculate atomic distances
"""

import numpy as np
from cpmd.pbc.pbc import pbc_abstract
from include.mlwc_logger import root_logger
logger = root_logger("MLWC."+__name__)


class pbc_1d(pbc_abstract):
    """
    Strategy インターフェイスを実装するクラス
    """
    @classmethod
    def compute_pbc(cls,vectors_array:np.ndarray,cell:np.ndarray)->np.ndarray:
        """1次元ベクトル配列に対する周期境界条件を計算します。

        Args:
            vectors_array (np.ndarray): 周期境界条件を適用するベクトル配列。形状は(3,)である必要があります。
            cell (np.ndarray): 単位格子を表す行列。

        Returns:
            np.ndarray: 周期境界条件が適用されたベクトル配列。

        詳細:
            1次元ベクトル配列に対して、指定された単位格子を用いて周期境界条件を適用します。
            内部では、2次元のpbc_2d.compute_pbcを呼び出して計算を行います。
        """
        # vectors_arrayの形状を確認
        if vectors_array.ndim != 1 or vectors_array.shape[0] != 3:
            raise ValueError(f"Invalid shape for vectors_array. Expected shape [3], but got {np.shape(vectors_array)}.")

        reshaped_vectors = vectors_array.reshape(1,3)
        # compute pbc for 2d vectors
        pbc_vectors = pbc_2d.compute_pbc(reshaped_vectors, cell)

        # 元の形 [3] に戻す
        pbc_vectors = pbc_vectors.reshape(3)
        
        #pbc_vectors = np.dot(vectors_array, np.linalg.inv(cell.T))
        #pbc_vectors -= np.round(pbc_vectors)
        #pbc_vectors = np.dot(pbc_vectors, cell.T)
        return pbc_vectors
    

class pbc_2d(pbc_abstract):
    """
    Strategy インターフェイスを実装するクラス
    """
    @classmethod
    def compute_pbc(cls,vectors_array:np.ndarray,cell:np.ndarray)->np.ndarray:
        """2次元ベクトル配列に対する周期境界条件を計算します。

        Args:
            vectors_array (np.ndarray): 周期境界条件を適用するベクトル配列。形状は(a, 3)である必要があります。
            cell (np.ndarray): 単位格子を表す行列。

        Returns:
            np.ndarray: 周期境界条件が適用されたベクトル配列。

        詳細:
            2次元ベクトル配列に対して、指定された単位格子を用いて周期境界条件を適用します。
            各ベクトルの成分を単位格子の逆行列で変換し、最近傍の格子点に丸めることで、周期境界条件を適用します。
        """
        # vectors_arrayの形状を確認
        if vectors_array.ndim != 2 or vectors_array.shape[1] != 3:
            raise ValueError(f"Invalid shape for vectors_array. Expected shape [a, 3], but got {np.shape(vectors_array)}.")

        # 1/aをかけて，0<x<1の範囲になるように修正する．
        pbc_vectors = np.dot(vectors_array, np.linalg.inv(cell.T))
        pbc_vectors -= np.round(pbc_vectors)
        pbc_vectors = np.dot(pbc_vectors, cell.T)
        return pbc_vectors

class pbc_3d(pbc_abstract):
    """
    Strategy インターフェイスを実装するクラス
    """
    @classmethod
    def compute_pbc(cls,vectors_array:np.ndarray, cell:np.ndarray) -> np.ndarray:
        """3次元ベクトル配列に対する周期境界条件を計算します。

        Args:
            vectors_array (np.ndarray): 周期境界条件を適用する3次元ベクトル配列。形状は(a, b, 3)である必要があります。
            cell (np.ndarray): 単位格子を表す行列。

        Returns:
            np.ndarray: 周期境界条件が適用されたベクトル配列。形状は(a, b, 3)です。

        詳細:
            3次元ベクトル配列に対して、指定された単位格子を用いて周期境界条件を適用します。
            内部では、2次元のpbc_2d.compute_pbcを呼び出して計算を行います。
        """
        # vectors_arrayの形状を確認
        if vectors_array.ndim != 3 or vectors_array.shape[2] != 3:
            raise ValueError(f"Invalid shape for vectors_array. Expected shape [a, b, 3], but got {vectors_array.shape}.")

        # vectors_array全体に対して周期境界条件を適用
        a, b, _ = vectors_array.shape
        reshaped_vectors = vectors_array.reshape(-1, 3)  # [a*b, 3] に変換
        # compute pbc for 2d vectors
        pbc_vectors = pbc_2d.compute_pbc(reshaped_vectors, cell)

        # 元の形 [a, b, 3] に戻す
        pbc_vectors = pbc_vectors.reshape(a, b, 3)
        
        return pbc_vectors