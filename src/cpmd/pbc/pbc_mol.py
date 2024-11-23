"""calculate atomic distances
"""

import numpy as np
import ase
from collections import deque # 深さ優先探索用
from cpmd.distance import distance_ase, distance_1d
from cpmd.pbc.pbc import pbc_abstract
from ml.atomtype import make_bondgraph


# まずは，全ての分子で，representative atomからの距離を計算する．（numpyで早い）
# その後，ボンドリストを元に，ボンド間距離が3.0 Angstromより大きい分子を探索する．
# そのような分子に対しては，BFSを行い，原子間距離を再計算する．



def raw_bfs(vectors, nodes, cell, representative:int=0):
    '''
    幅優先探索を行い，それにそってraw_get_distances_micでベクトルを計算する
    '''
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
            if nodes[near].parent == -1: # 親ノードが-1なら未探索
                # 未探索ノードをキューに追加
                queue.append(nodes[near])
                # 親ノードを追加
                nodes[near].parent = node.index
                # node（親）からnodes[near]（子）へのmicをかけたベクトルを計算
                # revised_vector = aseatom.get_distances(node.index, nodes[near].index, mic=True, vector=True)
                # mol_inds[0]はmolの最初の原子のindexで，これを足して元のaseatomのindexに戻す．
                revised_vector = distance_1d.compute_distances(vectors[node.index], vectors[nodes[near].index], cell=cell, pbc=True)
                # if np.linalg.norm(revised_vector) > 5:
                #     print(f"ERROR revised fail :: {revised_vector} :: {node.index} :: {nodes[near].index}")
                # revised_vector = raw_get_distances_mic(aseatom, node.index+mol_inds[representative], nodes[near].index+mol_inds[representative], mic=True, vector=True) # !! これ間違ってる？
                vectors[nodes[near].index] = vectors[node.index]+revised_vector # vectorsはrepresentativeからの距離
                # debug # print("node/parent/revised/vectors {}/{}/{}/{}".format(nodes[near].index,node.index,revised_vector,vectors[nodes[near].index]))
    return vectors



def check_if_bondlength_large(vectors:np.ndarray, bonds_list:list[list[int]])->bool:
    # return True if bond length is longer than 3.0 Angstrom
    # ボンド長を計算して，3.0 Angstromより大きい分子を確認する．
    # ase atomsの中でボンドリストに含まれる原子のみを抽出
    position1 = vectors[np.array(bonds_list)[:,0]]
    position2 = vectors[np.array(bonds_list)[:,1]]
    # ボンド長を計算
    bond_length = np.linalg.norm(position1-position2, axis=1)
    indices = np.where(bond_length>3.0)
    # ボンド長が3.0 Angstromより大きい分子を抽出
    if np.any(bond_length > 3.0):
        print(indices)
        return True
    else:
        return False


class pbc_mol(pbc_abstract):
    """
    Strategy インターフェイスを実装するクラス
    """
    @classmethod
    def compute_pbc(cls,vectors_array:np.ndarray,
                    cell:np.ndarray,
                    bonds_list:list[list[int]], 
                    NUM_ATOM_PAR_MOL:int, 
                    ref_atom_index:int)->np.ndarray:
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
        if ref_atom_index >= NUM_ATOM_PAR_MOL:
            raise IndexError(f"Invalid ref_atom_index. ref_atom_index should be less than NUM_MOL_PAR_MOL, but got {ref_atom_index}.")
        # if len(vectors_array) is devided by NUM_ATOM_PAR_MOL?
        if len(vectors_array)%NUM_ATOM_PAR_MOL != 0:
            raise ValueError(f"len(vectors_array) {len(vectors_array)} is not devided by NUM_ATOM_PAR_MOL ({NUM_ATOM_PAR_MOL})")

        # calculate number of molecules
        NUM_MOL = len(vectors_array)//NUM_ATOM_PAR_MOL

        # initialize pbc_vectors
        pbc_vectors = np.empty((NUM_MOL,NUM_ATOM_PAR_MOL,3))

        for mol_id in range(NUM_MOL):
            # extract positions of a molecule
            mol_vectors = vectors_array[mol_id*NUM_ATOM_PAR_MOL:(mol_id+1)*NUM_ATOM_PAR_MOL]
            # 基準原子から全ての分子内原子へのベクトルを計算．
            base_position = mol_vectors[ref_atom_index]
            # distance from ref_atom_index to others (with PBC)
            mol_vectors = distance_ase.compute_distances(base_position,mol_vectors,cell, pbc=True)
            mol_vectors = mol_vectors+base_position 
            # 
            # もしもボンド長が3.0 Angstromより大きい分子がある場合は，再計算を行う．
            if check_if_bondlength_large(mol_vectors,bonds_list):
                # print(f"ERROR multipbc :: mol_id = {mol_id}")
                # ボンドリストを元に，ボンド間距離が3.0 Angstromより大きい分子を探索する．
                # そのような分子に対しては，BFSを行い，原子間距離を再計算する．
                # mol_nodes = make_bondgraph(bonds_list,NUM_ATOM_PAR_MOL) is graph of molecules
                # !! caution !! make_bondgraph must be initialized every time
                mol_vectors = raw_bfs(mol_vectors, make_bondgraph(bonds_list,NUM_ATOM_PAR_MOL), cell, ref_atom_index)
                # check bond length again 
                if check_if_bondlength_large(mol_vectors,bonds_list):
                    print(f"ERROR bond length too large :: mol_id = {mol_id}")
                    
            # pbc_vectorsにmol_vectorsを格納
            pbc_vectors[mol_id] = mol_vectors
        return pbc_vectors  #.to("cpu") # .detach().numpy()
