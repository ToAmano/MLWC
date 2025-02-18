"""calculate atomic distances
"""

import numpy as np
import abc
from cpmd.pbc.pbc_numpy import pbc_1d
from cpmd.pbc.pbc_numpy import pbc_2d
from cpmd.pbc.pbc_numpy import pbc_3d

class distance_abstract(abc.ABC):
    """
    Abstract base class for distance calculation strategies.
    This class defines the common interface that concrete strategy classes (e.g., distance_1d, distance_2d) will implement.
    """
    @classmethod
    @abc.abstractmethod
    def compute_distances(cls):
        pass


class distance_1d(distance_abstract):
    """
    Class implementing the distance calculation strategy for 1D vectors.
    This class provides a concrete implementation of the distance_abstract interface for 1D vectors, utilizing periodic boundary conditions (PBC) if specified.
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

        Example:
            >>> import numpy as np
            >>> cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
            >>> vector_array1 = np.array([1.0, 2.0, 3.0])
            >>> vector_array2 = np.array([4.0, 5.0, 6.0])
            >>> pbc = True
            >>> distances = distance_1d.compute_distances(vector_array1, vector_array2, cell, pbc)
            >>> print(distances)
            [3. 3. 3.]
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
    Class implementing the distance calculation strategy for 2D vectors.
    This class provides a concrete implementation of the distance_abstract interface for 2D vectors, utilizing periodic boundary conditions (PBC) if specified.
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

        Example:
            >>> import numpy as np
            >>> cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
            >>> vector_array1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> vector_array2 = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
            >>> pbc = True
            >>> distances = distance_2d.compute_distances(vector_array1, vector_array2, cell, pbc)
            >>> print(distances)
            [[3. 3. 3.]
             [3. 3. 3.]]
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
    Class implementing the distance calculation strategy using a distance matrix.
    This class provides a concrete implementation of the distance_abstract interface, calculating distances between two sets of vectors using a matrix representation and applying periodic boundary conditions (PBC) if specified.
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

        Example:
            >>> import numpy as np
            >>> cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
            >>> vector_array1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> vector_array2 = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
            >>> pbc = True
            >>> distances = distance_matrix.compute_distances(vector_array1, vector_array2, cell, pbc)
            >>> print(distances)
            [[[3. 3. 3.]
              [6. 6. 6.]]

             [[0. 0. 0.]
              [3. 3. 3.]]]
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
    Class implementing the distance calculation strategy using ASE's get_distances method.
    This class provides a concrete implementation of the distance_abstract interface, leveraging the Atomic Simulation Environment (ASE) library to compute distances between atoms, applying periodic boundary conditions (PBC) if specified.
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

        Example:
            >>> import numpy as np
            >>> cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
            >>> vector_array1 = np.array([1.0, 2.0, 3.0])
            >>> vector_array2 = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
            >>> pbc = True
            >>> distances = distance_ase.compute_distances(vector_array1, vector_array2, cell, pbc)
            >>> print(distances)
            [[3. 3. 3.]
             [6. 6. 6.]]
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
    """Compute distances between two sets of vectors.
    This function calculates the distances between two arrays of vectors, applying periodic boundary conditions (PBC) if specified.
    It serves as a general interface for distance computation, utilizing other distance calculation strategies internally.


    Parameters
    ----------
    vector_array1 : np.ndarray
        _description_
    vector_array2 : np.ndarray
        _description_
    cell : _type_
        _description_
    pbc : bool, optional
        _description_, by default True

    Returns
    -------
    np.ndarray
        _description_

    Raises
    ------
    ValueError
        _description_
        
    Examples
    ------        
        >>> import numpy as np
        >>> cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        >>> vector_array1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> vector_array2 = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        >>> pbc = True
        >>> distances = compute_distances(vector_array1, vector_array2, cell, pbc)
        >>> print(distances)
        [[3. 3. 3.]
            [3. 3. 3.]]
    """
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


