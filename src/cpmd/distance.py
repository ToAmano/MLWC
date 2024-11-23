"""calculate atomic distances
"""

import numpy as np
import abc
from cpmd.pbc.pbc_numpy import pbc_1d
from cpmd.pbc.pbc_numpy import pbc_2d
from cpmd.pbc.pbc_numpy import pbc_3d

class distance_abstract(abc.ABC):
    """
    アルゴリズム（ConcreteStrategy）が実装する共通のインターフェイス
    """
    @classmethod
    @abc.abstractmethod
    def compute_distances(cls):
        pass


class distance_1d(distance_abstract):
    """
    Strategy インターフェイスを実装するクラス
    """
    @classmethod
    def compute_distances(cls,vector_array1:np.ndarray, vector_array2:np.ndarray,cell:np.ndarray, pbc:bool=True)->np.ndarray:
        """compute distances

        Args:
            vector_array1 (np.ndarray): vectors array 1. shape should be [a, 3]
            vector_array2 (np.ndarray): vectors array 2. shape should be [a, 3], the same as vectors_array1
            cell (np.ndarray): cell. shape should be [3, 3]
            pbc (bool): pbc. Default is True (apply pbc).

        Returns:
            np.ndarray: distances
        """
        if vector_array1.shape != vector_array2.shape:
            raise ValueError("The two arrays must have the same shape.")
        if vector_array1.ndim != 1 or vector_array1.shape[0] != 3:
            raise ValueError(f"Invalid shape for vectors_array. Expected shape [3], but got {vector_array1.shape}.")
        if vector_array2.ndim != 1 or vector_array2.shape[0] != 3:
            raise ValueError(f"Invalid shape for vectors_array. Expected shape [3], but got {vector_array2.shape}.")

        # Apply PBC using ASE's wrap_positions function for differences
        # Compute hydrogen minus oxygen bond vectors, accounting for PBC
        distance_vectors = vector_array2 - vector_array1
        if pbc:
            distance_vectors = pbc_1d.compute_pbc(distance_vectors, cell)
        # Normalize the bond vectors
        # norm_bond_vectors = bond_vectors / np.linalg.norm(bond_vectors, axis=1)[:, np.newaxis]
        return distance_vectors



class distance_2d(distance_abstract):
    """
    Strategy インターフェイスを実装するクラス
    """
    @classmethod
    def compute_distances(cls,vector_array1:np.ndarray, vector_array2:np.ndarray,cell:np.ndarray, pbc:bool=True)->np.ndarray:
        """compute distances

        Args:
            vector_array1 (np.ndarray): vectors array 1. shape should be [a, 3]
            vector_array2 (np.ndarray): vectors array 2. shape should be [a, 3], the same as vectors_array1
            cell (np.ndarray): cell. shape should be [3, 3]
            pbc (bool): pbc. Default is True (apply pbc).

        Returns:
            np.ndarray: distances
        """
        if vector_array1.shape != vector_array2.shape:
            raise ValueError("The two arrays must have the same shape.")
        if vector_array1.ndim != 2 or vector_array1.shape[1] != 3:
            raise ValueError(f"Invalid shape for vectors_array. Expected shape [a, 3], but got {vector_array1.shape}.")
        if vector_array2.ndim != 2 or vector_array2.shape[1] != 3:
            raise ValueError(f"Invalid shape for vectors_array. Expected shape [a, 3], but got {vector_array2.shape}.")

        # Apply PBC using ASE's wrap_positions function for differences
        # Compute hydrogen minus oxygen bond vectors, accounting for PBC
        distance_vectors = vector_array2 - vector_array1
        if pbc:
            distance_vectors = pbc_2d.compute_pbc(distance_vectors, cell)
        # Normalize the bond vectors
        # norm_bond_vectors = bond_vectors / np.linalg.norm(bond_vectors, axis=1)[:, np.newaxis]
        return distance_vectors

class distance_matrix(distance_abstract):
    """
    Strategy インターフェイスを実装するクラス
    """
    @classmethod
    def compute_distances(cls,vector_array1: np.ndarray, vector_array2: np.ndarray, cell: np.ndarray, pbc: bool = True) -> np.ndarray:
        """Compute PBC for 2D vectors_array with shape [a, 3]

        Args:
            vector_array1 (np.ndarray): 2D vectors array 1, expected to be [a, 3]
            vector_array2 (np.ndarray): 2D vectors array 2, expected to be [b, 3]
            cell (np.ndarray): Cell matrix representing the unit cell
            pbc (bool): pbc

        Returns:
            np.ndarray: PBC applied vectors array with shape [a, b, 3]
        """
        if vector_array1.ndim != 2 or vector_array1.shape[1] != 3:
            raise ValueError(f"Invalid shape for vectors_array. Expected shape [a, 3], but got {vector_array1.shape}.")
        if vector_array2.ndim != 2 or vector_array2.shape[1] != 3:
            raise ValueError(f"Invalid shape for vectors_array. Expected shape [b, 3], but got {vector_array2.shape}.")

        # 原子間の位置ベクトルを計算（行列の形で計算）
        distance_vectors = vector_array2[:, np.newaxis, :] - vector_array1[np.newaxis, :, :]
        # print(np.shape(distance_vectors))
        
        if pbc:
            distance_vectors = pbc_3d.compute_pbc(distance_vectors, cell)
        # Normalize the bond vectors
        # norm_bond_vectors = bond_vectors / np.linalg.norm(bond_vectors, axis=1)[:, np.newaxis]
        return distance_vectors


class distance_ase(distance_abstract):
    """
    Strategy インターフェイスを実装するクラス
    ase get_distances
    """
    @classmethod
    def compute_distances(cls,vector_array1: np.ndarray, vector_array2: np.ndarray, cell: np.ndarray, pbc: bool = True) -> np.ndarray:
        """Compute PBC for 2D vectors_array with shape [a, 3]

        Args:
            vector_array1 (np.ndarray): 2D vectors array 1, expected to be [3]
            vector_array2 (np.ndarray): 2D vectors array 2, expected to be [b, 3]
            cell (np.ndarray): Cell matrix representing the unit cell
            pbc (bool): pbc

        Returns:
            np.ndarray: PBC applied vectors array with shape [a, b, 3]
        """
        if vector_array1.ndim != 1 or vector_array1.shape[0] != 3:
            raise ValueError(f"Invalid shape for vectors_array. Expected shape [3], but got {vector_array1.shape}.")
        if vector_array2.ndim != 2 or vector_array2.shape[1] != 3:
            raise ValueError(f"Invalid shape for vectors_array. Expected shape [b, 3], but got {vector_array2.shape}.")

        # 原子間の位置ベクトルを計算（行列の形で計算）
        distance_vectors = vector_array2 - vector_array1[np.newaxis, :]
        # print(np.shape(distance_vectors))
        
        if pbc:
            distance_vectors = pbc_2d.compute_pbc(distance_vectors, cell)
        # Normalize the bond vectors
        # norm_bond_vectors = bond_vectors / np.linalg.norm(bond_vectors, axis=1)[:, np.newaxis]
        return distance_vectors


def compute_distances(vector_array1:np.ndarray, vector_array2:np.ndarray,cell, pbc:bool=True)->np.ndarray:
    if vector_array1.shape != vector_array2.shape:
        raise ValueError("The two arrays must have the same shape.")
    # Apply PBC using ASE's wrap_positions function for differences
    # Compute hydrogen minus oxygen bond vectors, accounting for PBC
    distance_vectors = vector_array2 - vector_array1
    if pbc:
        distance_vectors = pbc_2d.compute_pbc(distance_vectors, cell)
    # Normalize the bond vectors
    # norm_bond_vectors = bond_vectors / np.linalg.norm(bond_vectors, axis=1)[:, np.newaxis]
    return distance_vectors


