"""calculate atomic distances
"""

import numpy as np
import abc

class pbc_abstract(abc.ABC):
    """
    アルゴリズム（ConcreteStrategy）が実装する共通のインターフェイス
    """
    @classmethod
    @abc.abstractmethod
    def compute_pbc(cls):
        pass
    

class pbc_2d(pbc_abstract):
    """
    Strategy インターフェイスを実装するクラス
    """
    @classmethod
    def compute_pbc(cls,vectors_array:np.ndarray,cell:np.ndarray)->np.ndarray:
        """compute pbc

        Args:
            vectors_array (np.ndarray): vectors array
            cell (np.ndarray): cell

        Returns:
            np.ndarray: pbc vectors array
        """
        # vectors_arrayの形状を確認
        if vectors_array.ndim != 2 or vectors_array.shape[1] != 3:
            raise ValueError(f"Invalid shape for vectors_array. Expected shape [a, 3], but got {vectors_array.shape}.")

        pbc_vectors = np.dot(vectors_array, np.linalg.inv(cell.T))
        pbc_vectors -= np.round(pbc_vectors)
        pbc_vectors = np.dot(pbc_vectors, cell.T)
        return pbc_vectors

class pbc_3d(pbc_abstract):
    """
    Strategy インターフェイスを実装するクラス
    """
    @classmethod
    def compute_pbc_3d(cls,vectors_array: np.ndarray, cell: np.ndarray) -> np.ndarray:
        """Compute PBC for 3D vectors_array with shape [a, b, 3]

        Args:
            vectors_array (np.ndarray): 3D vectors array, expected to be [a, b, 3]
            cell (np.ndarray): Cell matrix representing the unit cell

        Returns:
            np.ndarray: PBC applied vectors array with shape [a, b, 3]
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


def compute_pbc(vectors_array:np.ndarray,cell:np.ndarray)->np.ndarray:
    """compute pbc

    Args:
        vectors_array (np.ndarray): vectors array
        cell (np.ndarray): cell

    Returns:
        np.ndarray: pbc vectors array
    """
    # vectors_arrayの形状を確認
    if vectors_array.ndim != 2 or vectors_array.shape[1] != 3:
        raise ValueError(f"Invalid shape for vectors_array. Expected shape [a, 3], but got {vectors_array.shape}.")

    pbc_vectors = np.dot(vectors_array, np.linalg.inv(cell.T))
    pbc_vectors -= np.round(pbc_vectors)
    pbc_vectors = np.dot(pbc_vectors, cell.T)
    return pbc_vectors


