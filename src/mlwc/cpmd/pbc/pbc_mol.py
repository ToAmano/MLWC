"""
This module provides functions and a class for calculating atomic distances,
taking into account periodic boundary conditions (PBC).

It includes functions for performing breadth-first search (BFS) to recalculate
atomic distances when bond lengths exceed a certain threshold.

"""

from collections import deque  # 深さ優先探索用

import numpy as np
from ase.cell import Cell

from mlwc.bond.atomtype import make_bondgraph
from mlwc.cpmd.distance.distance import distance_1d, distance_ase
from mlwc.cpmd.pbc.pbc import PbcAbstract
from mlwc.include.mlwc_logger import setup_cmdline_logger

logger = setup_cmdline_logger("MLWC." + __name__)


# まずは，全ての分子で，representative atomからの距離を計算する．（numpyで早い）
# その後，ボンドリストを元に，ボンド間距離が3.0 Angstromより大きい分子を探索する．
# そのような分子に対しては，BFSを行い，原子間距離を再計算する．


def raw_bfs(vectors, nodes, cell, representative: int = 0):
    """
    Performs a breadth-first search (BFS) and calculates vectors accordingly.

    Parameters
    ----------
    vectors : np.ndarray
        Array of atomic coordinates.
    nodes : list[Node]
        List of nodes in the bond graph.
    cell : np.ndarray
        Cell of the system.
    representative : int, optional
        Index of the representative atom, by default 0.

    Returns
    -------
    np.ndarray
        Array of calculated atomic coordinates.

    Examples
    --------
    >>> vectors = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
    >>> nodes = [Node(0, [1,2]),Node(1, [0,3]),Node(2, [0,3]),Node(3, [1,2])]
    >>> cell = np.eye(3) * 10
    >>> representative = 0
    >>> raw_bfs(vectors, nodes, cell, representative)
    array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
    """
    # 探索キューを作成
    queue = deque([])
    # ノードreoresentativeからBFS開始
    queue.append(nodes[representative])
    # ノード0の親ノードを便宜上0とする
    nodes[representative].parent = 0

    # BFS 開始
    while queue:
        # キューから探索先を取り出す
        node = queue.popleft()
        # print("current node :: {}".format(node)) # ←現在地を出力
        # 現在地の隣接リストを取得
        nears = node.nears
        for near in nears:
            if nodes[near].parent == -1:  # 親ノードが-1なら未探索
                # 未探索ノードをキューに追加
                queue.append(nodes[near])
                # 親ノードを追加
                nodes[near].parent = node.index
                # node（親）からnodes[near]（子）へのmicをかけたベクトルを計算
                # revised_vector = aseatom.get_distances(node.index, nodes[near].index, mic=True, vector=True)
                # mol_inds[0]はmolの最初の原子のindexで，これを足して元のaseatomのindexに戻す．
                revised_vector = distance_1d.compute_distances(
                    vectors[node.index], vectors[nodes[near].index], cell=cell, pbc=True
                )
                # if np.linalg.norm(revised_vector) > 5:
                #     print(f"ERROR revised fail :: {revised_vector} :: {node.index} :: {nodes[near].index}")
                # revised_vector = raw_get_distances_mic(aseatom, node.index+mol_inds[representative], nodes[near].index+mol_inds[representative], mic=True, vector=True) # !! これ間違ってる？
                vectors[nodes[near].index] = (
                    vectors[node.index] + revised_vector
                )  # vectorsはrepresentativeからの距離
                # debug # print("node/parent/revised/vectors {}/{}/{}/{}".format(nodes[near].index,node.index,revised_vector,vectors[nodes[near].index]))
    return vectors


def check_if_bondlength_large(vectors: np.ndarray, bonds_list: list[list[int]]) -> bool:
    """
    Checks if any bond length is larger than 3.0 Angstrom.

    Parameters
    ----------
    vectors : np.ndarray
        Array of atomic coordinates.
    bonds_list : list[list[int]]
        List of bonds.

    Returns
    -------
    bool
        True if any bond length is larger than 3.0 Angstrom, False otherwise.

    Examples
    --------
    >>> vectors = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
    >>> bonds_list = [[0,1],[1,2],[2,3]]
    >>> check_if_bondlength_large(vectors, bonds_list)
    False
    """
    position1 = vectors[np.array(bonds_list)[:, 0]]
    position2 = vectors[np.array(bonds_list)[:, 1]]
    # ボンド長を計算
    bond_length = np.linalg.norm(position1 - position2, axis=1)
    indices = np.where(bond_length > 3.0)
    # ボンド長が3.0 Angstromより大きい分子を抽出
    if np.any(bond_length > 3.0):
        print(indices)
        return True
    else:
        return False


class pbc_mol(PbcAbstract):
    """
    This class implements the Strategy interface for computing atomic coordinates
    of molecules with periodic boundary conditions (PBC).
    """

    @classmethod
    def compute_pbc(
        cls,
        vectors_array: np.ndarray,
        cell: np.ndarray | Cell,
        bonds_list: list[list[int]],
        NUM_ATOM_PAR_MOL: int,
        ref_atom_index: int,
    ) -> np.ndarray:
        """
        Computes the atomic coordinates of molecules with periodic boundary conditions (PBC).

        Parameters
        ----------
        vectors_array : np.ndarray
            Array of atomic coordinates for all atoms. Shape is (number of atoms, 3).
        cell : np.ndarray
            Cell vectors of the unit cell. Shape is (3, 3).
        bonds_list : list[list[int]]
            List of bonds within the molecule.
        NUM_ATOM_PAR_MOL : int
            Number of atoms per molecule.
        ref_atom_index : int
            Index of the reference atom within the molecule.

        Returns
        -------
        np.ndarray
            Array of atomic coordinates of the molecule with PBC applied.
            Shape is (number of molecules, number of atoms per molecule, 3).

        Details
        -------
        Applies periodic boundary conditions to optimize the interatomic distances within the molecule.
        First, the distance from the reference atom is calculated. If the bond length is greater than 3.0 Angstroms,
        breadth-first search (BFS) is used to recalculate the interatomic distances.

        Examples
        --------
        >>> import numpy as np
        >>> vectors_array = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
        >>> cell = np.eye(3) * 10
        >>> bonds_list = [[0,1],[1,2],[2,3],[4,5],[5,6],[6,7]]
        >>> NUM_ATOM_PAR_MOL = 4
        >>> ref_atom_index = 0
        >>> result = pbc_mol.compute_pbc(vectors_array, cell, bonds_list, NUM_ATOM_PAR_MOL, ref_atom_index)
        >>> print(result)
        [[[ 0.  0.  0.]
          [ 1.  0.  0.]
          [ 0.  1.  0.]
          [ 1.  1.  0.]]

         [[ 0.  0.  0.]
          [ 1.  0.  0.]
          [ 0.  1.  0.]
          [ 1.  1.  0.]]]
        """
        # vectors_arrayの形状を確認
        if vectors_array.ndim != 2 or vectors_array.shape[1] != 3:
            raise ValueError(
                f"Invalid shape for vectors_array. Expected shape [a, 3], but got {np.shape(vectors_array)}."
            )
        if ref_atom_index >= NUM_ATOM_PAR_MOL:
            raise IndexError(
                f"Invalid ref_atom_index. ref_atom_index should be less than NUM_MOL_PAR_MOL, but got {ref_atom_index}."
            )
        # if len(vectors_array) is devided by NUM_ATOM_PAR_MOL?
        if len(vectors_array) % NUM_ATOM_PAR_MOL != 0:
            raise ValueError(
                f"len(vectors_array) {len(vectors_array)} is not devided by NUM_ATOM_PAR_MOL ({NUM_ATOM_PAR_MOL})"
            )

        # calculate number of molecules
        NUM_MOL = len(vectors_array) // NUM_ATOM_PAR_MOL

        # initialize pbc_vectors
        pbc_vectors = np.empty((NUM_MOL, NUM_ATOM_PAR_MOL, 3))

        for mol_id in range(NUM_MOL):
            # extract positions of a molecule
            mol_vectors = vectors_array[
                mol_id * NUM_ATOM_PAR_MOL : (mol_id + 1) * NUM_ATOM_PAR_MOL
            ]
            # 基準原子から全ての分子内原子へのベクトルを計算．
            base_position = mol_vectors[ref_atom_index]
            # distance from ref_atom_index to others (with PBC)
            mol_vectors = distance_ase.compute_distances(
                base_position, mol_vectors, cell, pbc=True
            )
            mol_vectors = mol_vectors + base_position
            #
            # もしもボンド長が3.0 Angstromより大きい分子がある場合は，再計算を行う．
            if check_if_bondlength_large(mol_vectors, bonds_list):
                # print(f"ERROR multipbc :: mol_id = {mol_id}")
                # ボンドリストを元に，ボンド間距離が3.0 Angstromより大きい分子を探索する．
                # そのような分子に対しては，BFSを行い，原子間距離を再計算する．
                # mol_nodes = make_bondgraph(bonds_list,NUM_ATOM_PAR_MOL) is graph of molecules
                # !! caution !! make_bondgraph must be initialized every time
                mol_vectors = raw_bfs(
                    mol_vectors,
                    make_bondgraph(bonds_list, NUM_ATOM_PAR_MOL),
                    cell,
                    ref_atom_index,
                )
            # check bond length again
            if check_if_bondlength_large(mol_vectors, bonds_list):
                logger.error(f"ERROR bond length too large :: mol_id = {mol_id}")

            # pbc_vectorsにmol_vectorsを格納
            pbc_vectors[mol_id] = mol_vectors
        return pbc_vectors  # .to("cpu") # .detach().numpy()
