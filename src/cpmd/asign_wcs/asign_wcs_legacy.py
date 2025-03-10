# WCsのアサインのregacy implementation
# 速度が遅い代わりに，多分割り当てについてはより確実に行えると思う．

from typing_extensions import deprecated # https://qiita.com/junkmd/items/479a8bafa03c8e0428ac
from ase.io import read
import ase
import sys
import numpy as np
from ml.atomtype import Node # 深さ優先探索用
from ml.atomtype import raw_make_graph_from_itp # 深さ優先探索用
from collections import deque # 深さ優先探索用
# from types import NoneType
import torch
from cpmd.distance.distance import distance_matrix
from include.mlwc_logger import root_logger
logger = root_logger("MLWC."+__name__)


# * 
def apply_pbc_torch(coords:np.array, UNITCELL_VECTORS:np.array):
    """
    Apply periodic boundary conditions to bring coordinates inside the unit cell.

    Parameters:
    coords (torch.Tensor): The coordinates of the atoms, shape (N, 3).
    lattice_vectors (torch.Tensor): The lattice vectors, shape (3, 3).

    Returns:
    torch.Tensor: The coordinates wrapped inside the unit cell.
    """
    # UNITCELL_VECTORS = torch.from_numpy(UNITCELL_VECTORS.astype(np.float32))
    # coords = torch.from_numpy(coords.astype(np.float32))
    
    # Convert coordinates to fractional coordinates
    inv_lattice_vectors = torch.linalg.inv(UNITCELL_VECTORS)
    fractional_coords = torch.matmul(coords, inv_lattice_vectors.T)

    # Apply periodic boundary conditions: bring fractional coordinates into the range [0, 1)
    fractional_coords = fractional_coords - torch.floor(fractional_coords)

    # Convert back to cartesian coordinates
    wrapped_coords = torch.matmul(fractional_coords, UNITCELL_VECTORS)

    return wrapped_coords


def relative_vector_torch(atom1, atom2, UNITCELL_VECTORS:np.array):
    """
    !! atom1からatom2へのベクトル
    Calculate the relative vector between two atoms considering periodic boundary conditions.

    Parameters:
    atom1 (torch.Tensor): The coordinates of the first atom, shape (3,).
    atom2 (torch.Tensor): The coordinates of the second atom, shape (3,).
    lattice_vectors (torch.Tensor): The lattice vectors, shape (3, 3).

    Returns:
    torch.Tensor: The relative vector considering periodic boundary conditions, shape (3,).
    """
    # UNITCELL_VECTORS = torch.from_numpy(UNITCELL_VECTORS.astype(np.float32))
    
    # * atom1とatom2のサイズが違う
    relative_vec = atom2 - atom1
    # relative_vec = torch.from_numpy(relative_vec.astype(np.float32))
    
    # Convert the relative vector to fractional coordinates
    inv_lattice_vectors = torch.linalg.inv(UNITCELL_VECTORS)
    fractional_relative_vec = torch.matmul(relative_vec, inv_lattice_vectors.T)
    
    # Apply minimum image convention
    fractional_relative_vec = fractional_relative_vec - torch.round(fractional_relative_vec)
    
    # Convert back to cartesian coordinates
    relative_vec = torch.matmul(fractional_relative_vec, UNITCELL_VECTORS)
    
    return relative_vec

def relative_vectors_torch2(atom1, atom2, UNITCELL_VECTORS):
    """
    Calculate the relative vectors between two sets of atoms considering periodic boundary conditions.

    Parameters:
    atom1 (torch.Tensor): The coordinates of the first set of atoms, shape (N, 3).
    atom2 (torch.Tensor): The coordinates of the second set of atoms, shape (M, 3).
    lattice_vectors (torch.Tensor): The lattice vectors, shape (3, 3).

    Returns:
    torch.Tensor: The relative vectors considering periodic boundary conditions, shape (N, M, 3).
    """
    # Expand dimensions to calculate all pairwise differences
    # !! atom1とatom2が一つの場合，このコードだとうまくいかない．
    if (atom1.dim() == 1) and (atom2.dim() == 1):
        return relative_vector_torch(atom1, atom2, UNITCELL_VECTORS)

    atom1_expanded = atom1.unsqueeze(1)  # Shape (N, 1, 3)
    atom2_expanded = atom2.unsqueeze(0)  # Shape (1, M, 3)
    
    relative_vecs = atom2_expanded - atom1_expanded  # Shape (N, M, 3)
    print(" ============== relative_vecs ==============")
    print(atom1.shape)
    print(atom1.dim())
    print(atom2.shape)    
    print(atom2.dim())
    print(atom1_expanded.shape)
    print(atom2_expanded.shape)
    print(relative_vecs.shape)
    
    # Convert the relative vectors to fractional coordinates
    inv_lattice_vectors = torch.linalg.inv(UNITCELL_VECTORS)
    fractional_relative_vecs = torch.matmul(relative_vecs, inv_lattice_vectors.T)
    
    # Apply minimum image convention
    fractional_relative_vecs = fractional_relative_vecs - torch.round(fractional_relative_vecs)
    
    # Convert back to cartesian coordinates
    relative_vecs = torch.matmul(fractional_relative_vecs, UNITCELL_VECTORS)
    
    return relative_vecs


# !! >>>>>>>>>>>>>>  新しい実装 >>>>>>>>>>>>>> 
# !! 計算したいのは，各ボンドセンター，ローンペアに対してのWCの割り当て
# !! アルゴリズムとしては，まずボンドに対して最近接を全て攫う．
# !! 次に，残りの中からlpを探索する．


# !! まず，ある座標のリストとwfcリストの最も近いWCを計算する関数を作成する．
def find_nearest_wfc(bondcenters, wfc_list, UNITCELL_VECTORS, num_wcs:int):
    # 全てのボンドセンターに対してWCを割り当てる
        # bondcenters と wfc_list の各ペア間の距離を計算する
    # `bondcenters` の形状は (N, 3)、`wfc_list` の形状は (M, 3) であると仮定
    # diff = bondcenters[:, np.newaxis, :] - wfc_list[np.newaxis, :, :]
    # TODO :: apply pbc
    diff = distance_matrix.compute_distances(bondcenters,wfc_list,UNITCELL_VECTORS)    
    # calculate distance matrix
    distances = np.linalg.norm(diff, axis=2)
    
    # 各 `bondcenters` の座標に対する最も近い `wfc_list` のインデックスを取得
    nearest_indices = np.argmin(distances, axis=0)
    # 最も近い `wfc_list` の座標を抽出
    nearest_wfc = wfc_list[nearest_indices]
    
    # 各 `bondcenters` に対する最も近い2つの `wfc_list` のインデックスを取得
    nearest_indices = np.argsort(distances, axis=0)[:, :num_wcs]
    # TODO :: indexが重複指定ないかの確認
    
    # 最も近い2つの `wfc_list` の座標を抽出
    nearest_wfc = wfc_list[nearest_indices]
    # TODO :: indexを返した方が良い説ある．
    return nearest_wfc

# !! 次に，ボンド双極子を計算する．
def calc_bondmu():
    wfc_distances=np.linalg.norm(wfc_vectors,axis=1)
    if wcs_num == 1:
        wcs_indices = np.argsort(wfc_distances).reshape(-1)[:1] # 最も近いWCsのインデックスを一つ取り出す．
        mu_lp = (-2.0)*coef*wfc_vectors[wcs_indices[0]]
        if np.linalg.norm(wfc_vectors[wcs_indices[0]]) > 1.0: # 原子とwcsの距離があまりに遠い場合はWARNINGを出す
            print("WARNING :: The distance between atom {} and lonepair {} ({}) is too large !! :: {} ".format(atom_coord, wfc_list[wcs_indices[0]],wcs_indices[0], np.linalg.norm(wfc_vectors[wcs_indices[0]])))
        if wcs_indices[0] in picked_wfcs:
            print("WARNING SAME wcs is assined !! :: {}".format(wcs_indices[0]))
    if wcs_num == 2:
        wcs_indices = np.argsort(wfc_distances).reshape(-1)[:2] # 最も近いWCsのインデックスを二つ取り出す．
        # 二つのWannierCenterによる双極子モーメントを計算する．
        mu_lp = (-4.0)*coef*(wfc_vectors[wcs_indices[0]]+wfc_vectors[wcs_indices[1]])/2.0


# !! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class asign_wcs_torch:
    import ase
    '''
    関数をメソッドとしてこちらにうつしていく．
    その際，基本となる変数をinitで定義する
    '''
    def __init__(self, NUM_MOL:int, NUM_MOL_ATOMS:int, UNITCELL_VECTORS):
        self.NUM_MOL       = NUM_MOL # 分子数
        self.NUM_MOL_ATOMS = NUM_MOL_ATOMS # 1分子あたりの原子数
        self.UNITCELL_VECTORS = torch.from_numpy(UNITCELL_VECTORS.astype(np.float32)).clone() # 単位胞ベクトル
    
    def aseatom_to_mol_coord_bc(self, ase_atoms:ase.atoms, itp_data, bonds_list:list): # ase_atomsのボンドセンターを計算する
        return raw_aseatom_to_mol_coord_bc(ase_atoms, bonds_list, itp_data, self.NUM_MOL_ATOMS, self.NUM_MOL)
    
    def find_all_lonepairs(self, wfc_list,atO_list,list_mol_coords,picked_wfcs,wcs_num:int):
        return raw_find_all_lonepairs(wfc_list,atO_list,list_mol_coords,picked_wfcs,wcs_num,self.UNITCELL_VECTORS)
    # TODO :: ここはボンドごとに，例えばfind_all_chbondsのようにしたい
    def find_all_bonds(self, wfc_list,list_bond_centers,picked_wfcs,wcs_num:int): 
        return raw_find_all_bonds(wfc_list,list_bond_centers,picked_wfcs,self.UNITCELL_VECTORS,wcs_num)
    
    def find_all_pi(self, wfc_list,list_bond_centers,picked_wfcs,double_bonds):
        return raw_find_all_pi(wfc_list,list_bond_centers,picked_wfcs,double_bonds,self.UNITCELL_VECTORS)
    
    def calc_mu_bond_lonepair(self, wfc_list,ase_atoms,bonds_list,itp_data,double_bonds):
        # 実際のﾜﾆｴの割当
        return raw_calc_mu_bond_lonepair(wfc_list,ase_atoms,bonds_list,itp_data,double_bonds,self.NUM_MOL_ATOMS,self.NUM_MOL,self.UNITCELL_VECTORS)
    
        # 
    # * calc_mu_bond_lonepair用の部品関数たち
    def _find_all_lonepairs(self, wfc_list,atO_list,list_mol_coords,picked_wfcs,wcs_num:int,UNITCELL_VECTORS):
        '''
        最後の変数num_wcsでOローンペアとNローンペアに対応
        '''
        list_mu_lp   =  []
        list_lp_wfcs =  []
        
        if wcs_num != 1 and wcs_num != 2:
            print("ERROR :: wcs_num should be 1 or 2 !!")
            print("wfc_num = ", wcs_num)
            return -1
        
        # Oのローンペア
        for atOs,mol_coords in zip(atO_list,list_mol_coords) : #分子の数に関するループ
            mu_lpO_mol = []
            wcs_mol = [] # wcsの座標
            for atO in atOs : #ある分子内の原子に関するループ
                center_atom_coord  = mol_coords[atO]
                #wfc_list_exclude_pickedwfcs = [wfc_list[config][i] for i in range(len(wfc_list[config])) if i not in picked_wfcs]
                wcs_indices, mu_lp, wcs_lp = raw_find_lonepairs(center_atom_coord, wfc_list, wcs_num,UNITCELL_VECTORS,picked_wfcs)
                picked_wfcs        = picked_wfcs + list(wcs_indices)
                mu_lpO_mol.append(mu_lp)
                wcs_mol.append(wcs_lp)
            list_mu_lp.append(mu_lpO_mol)
            list_lp_wfcs.append(wcs_mol) 
        return np.array(list_mu_lp), np.array(list_lp_wfcs), picked_wfcs

    

    def _find_nearest_lonepairs(self,atom_coord:np.array,wfc_list,wcs_num:int,UNITCELL_VECTORS,picked_wfcs):
        """
        !! 複数のatom_coordに対応できるようにしたい．
        !! atom_coordとwcs_listは一つのセル内にあると家庭
        ローンペアまたはボンドセンターから最も近いwcsを探索する
        input
        -------------
        wfc_list   :: WCsの座標リスト
        atom_coord :: 原子の座標（np.array）：実際にはボンドセンターの座標
        wcs_num    :: 見つけるwcsの数．1（N lonepair）か2（O lonepair）
        picked_wcs :: これはoption
        
        output
        -------------
        wcs_indices :: 答えのワニエのindex
        mu_lp :: 計算された双極子
        """
        # 物理定数
        from include.constants import constant  
        # Debye   = 3.33564e-30
        # charge  = 1.602176634e-019
        # ang      = 1.0e-10 
        coef    = constant.Ang*constant.Charge/constant.Debye 
        
        if wcs_num != 1 and wcs_num != 2:
            print("ERROR :: wcs_num should be 1 or 2 !!")
            print("wfc_num = ", wcs_num)
            return -1

        #選択したボンドセンターを先頭において，残りはwcsで構成されるAtomsオブジェクトを作成する
        atom_wan=raw_make_aseatoms_from_wc(atom_coord,wfc_list,UNITCELL_VECTORS)

        # 先頭原子(atom_coord)とWCsの距離ベクトル（PBC，mirror imageを考慮）
        wfc_vectors = relative_vector_torch(atom_coord, wfc_list, UNITCELL_VECTORS)
        wfc_distances=torch.linalg.norm(wfc_vectors,axis=1)
        
        if wcs_num == 1: 
            wcs_indices = torch.argsort(wfc_distances).reshape(-1)[:1] # 最も近いWCsのインデックスを一つ取り出す．
            mu_lp = (-2.0)*coef*wfc_vectors[wcs_indices[0]]
            if np.linalg.norm(wfc_vectors[wcs_indices[0]]) > 1.0: # 原子とwcsの距離があまりに遠い場合はWARNINGを出す
                print("WARNING :: The distance between atom {} and lonepair {} ({}) is too large !! :: {} ".format(atom_coord, wfc_list[wcs_indices[0]],wcs_indices[0], np.linalg.norm(wfc_vectors[wcs_indices[0]])))
            if wcs_indices[0] in picked_wfcs:
                print("WARNING SAME wcs is assined !! :: {}".format(wcs_indices[0]))
        if wcs_num == 2:
            wcs_indices = np.argsort(wfc_distances).reshape(-1)[:2] # 最も近いWCsのインデックスを二つ取り出す．
            # 二つのWannierCenterによる双極子モーメントを計算する．
            mu_lp = (-4.0)*coef*(wfc_vectors[wcs_indices[0]]+wfc_vectors[wcs_indices[1]])/2.0
        # 最後にwcsの（micがかかった）座標を取得
        wcs_lp=[atom_wan.get_positions()[0]+wfc_vectors[i] for i in wcs_indices]
        return wcs_indices, mu_lp, wcs_lp
    
    
    
    
def raw_aseatom_to_mol_coord_bc(ase_atoms, bonds_list, itp_data, NUM_MOL_ATOMS:int, NUM_MOL:int) :
    '''
    ase_atomsから，
     - 1: ボンドセンターの計算
     - 2: micを考慮した原子座標の再計算
    を行う．基本的にはcalc_mol_coordのwrapper関数
    
    input
    ------------
    ase_atoms       :: ase.atoms
    mol_ats         ::
    bonds_list      :: itpdataに入っているボンドリスト

    output
    ------------
    list_mol_coords :: [mol_id,atom,coord(3)]
    list_bond_centers :: [mol_id,bond,coord(3)]
    
    NOTE
    ------------
    2023/4/16 :: inputとしていたunit_cell_bondsをより基本的な変数bond_listへ変更．
    bond_listは1分子内でのボンドの一覧であり，そこからunit_cell_bondsを関数の内部で生成する．
    '''

    list_mol_coords=[] #分子の各原子の座標の格納用
    list_bond_centers=[] #各分子の化学結合の中点の座標リストの格納用
    # 
    # 分子のindexは0,1,,,NUM_MOL_ATOMS
    mol_inds = np.arange(NUM_MOL_ATOMS)
    for j in range(NUM_MOL):
        # bonds_list_jは不要に！！
        # mol_coords,bond_centers = raw_calc_mol_coord_and_bc_mic_onemolecule(mol_inds,bonds_list_j,ase_atoms,itp_data) # 1つの分子のmic座標/bond center計算
        mol_coords,bond_centers = raw_calc_mol_coord_and_bc_mic_onemolecule_new(mol_inds,bonds_list,ase_atoms[NUM_MOL_ATOMS*j:NUM_MOL_ATOMS*(j+1)],itp_data) # 1つの分子のmic座標/bond center計算
        list_mol_coords.append(mol_coords)
        list_bond_centers.append(bond_centers)

    # 最後に，list_mol_coordsとlist_bond_centersにPBCを適用
    
    return  [list_mol_coords,list_bond_centers]


def raw_get_distances_mic(aseatom, a:int, indices, mic=False, vector=False):
    '''
    ase.atomのget_distances関数でmicをかけるとかなり計算に時間がかかる．
    そこで，get_distances(self, a, indices, mic=False, vector=False)の置き換え関数を作成する．
    a: 求める原子のaseatomでの順番（index）
    '''
    coordinate = aseatom.get_positions()
    position = coordinate[a]
    distances = coordinate[indices]-position
    if mic == True: # micの時だけdistancesを計算しなおす．
        # cell = aseatom.get_cell()
        cell = aseatom.get_cell()[0][0] # xの座標だけを利用する． # TODO :: 一般の格子に対応させる．
        distances = np.where(np.abs(distances) > cell/2, distances-cell*np.sign(distances),distances)
    if vector == True:
        return distances
    else:
        return np.linalg.norm(distances,axis=1)
    
def raw_get_distances_mic_multiPBC(aseatom, a:int, indices, mic=False, vector=False):
    '''
    raw_get_distances_micとの違いは，PBCを複数回かけられるということ．
    raw_get_distances_micでは隣のmirror imageだけを考えて座標の修正を行なっていたが，
    こちらのraw_get_distances_mic_multiPBCではさらに遠くのmirror imageのことも考慮して
    座標の修正を行う．
    '''
    # print("using multiPBC !!") # debug
    coordinate = aseatom.get_positions()
    position = coordinate[a]
    distances = coordinate[indices]-position
    if mic == True: # micの時だけdistancesを計算しなおす．
        # cell = aseatom.get_cell()
        cell = aseatom.get_cell()[0][0] # xの座標だけを利用する． # TODO :: 一般の格子に対応させる．
        count = 0
        while np.any(np.abs(distances) > cell/2): # 最短micに入るまでwhere演算を繰り返す
            distances = np.where(np.abs(distances) > cell/2, distances-cell*np.sign(distances),distances)
            count += 1
        if count > 1:
            print(" !! CAUTION !! multiPBC works !! :: {}".format(count))
    if vector == True:
        return distances
    else:
        return np.linalg.norm(distances,axis=1)
    
def raw_get_pbc_mol(aseatom,mol_inds,bonds_list_j,itp_data):
    '''
    aseatomの中でmol_indsに入っている原子のみを抽出し，それらの原子間距離をmicで計算する．
    '''
    # 基準原子から全ての分子内原子へのベクトルを計算．
    vectors = raw_get_distances_mic(aseatom,mol_inds[itp_data.representative_atom_index], mol_inds, mic=True, vector=True)
    vectors_old = vectors
    # print("vectors :: {}".format(np.shape(vectors)))
    # print("bonds_list_j :: {}".format(bonds_list_j))
    # bond_list_jを0スタートにする．（0スタートにするだけなのでmol_inds[0]で引くので正しい）
    bonds_list_from_zero=[[i[0]-mol_inds[0],i[1]-mol_inds[0]] for i in bonds_list_j] # TODO :: bonds_list_jから単に数字をシフトする場合にのみ対応
    # print(bonds_list_from_zero)
    CALC_FLAG = False # 通常は探索によるvectorsの更新は行わない
    for bond in bonds_list_from_zero: # 全てのボンドに対するloop（ボンド間距離が一つでも3 Angstromより大きかったら再計算）
        # ボンド間距離の計算
        bond_distance=np.linalg.norm(vectors[bond[0]]-vectors[bond[1]])
        # print(bond[0], bond[1], bond_distance) # !! debug
        if bond_distance > 3.0: # angstrom
            CALC_FLAG = True # ボンド間距離が大きいものがある場合は探索によりvectorsを計算しなおす．
            # print("Warning: bond distance is too long. bond distance = {0} Angstrom between {1}/{2} and {3}/{4}".format(bond_distance, bond[0],aseatom.get_positions()[bond[0]+mol_inds[0]], bond[1],aseatom.get_positions()[bond[1]+mol_inds[0]] )) # debug
    if CALC_FLAG == True: # 
        print("WARNING(raw_get_pbc_mol) :: mol_index {} :: recalculation of vectors is required.".format(mol_inds[0]))
        vectors = raw_bfs(aseatom, raw_make_graph_from_itp(itp_data), vectors, mol_inds, itp_data.representative_atom_index)
        # print(mol_inds[0],vectors-vectors_old) # !! debug
        # print(np.shape(vectors-vectors_old)) # !! debug
        for bond in bonds_list_from_zero: # 全てのボンドに対するloopでvectorsが正しく再計算されたかチェック
            # ボンド間距離の計算
            bond_distance=np.linalg.norm(vectors[bond[0]]-vectors[bond[1]])
            if bond_distance > 3.0: # angstroml
                print(" !!ERROR!!: bond distance is too long after BFS modification. bond distance = {0} Angstrom between atom {1}/{2}/{3} and atom {4}/{5}/{6}".format(bond_distance, bond[0], aseatom.get_positions()[bond[0]+mol_inds[0]],aseatom.get_chemical_symbols()[bond[0]+mol_inds[0]], bond[1], aseatom.get_positions()[bond[1]+mol_inds[0]], aseatom.get_chemical_symbols()[bond[1]+mol_inds[0]]))
                print(" vectors[bond[0]]-vectors[bond[1]] {} {}".format(vectors[bond[0]],vectors[bond[1]]))
                print(" ")
                # sys.exit(1)
    return vectors

def raw_bfs(aseatom, nodes, vectors, mol_inds, representative:int=0):
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
                revised_vector = raw_get_distances_mic(aseatom, node.index+mol_inds[0], nodes[near].index+mol_inds[0], mic=True, vector=True)
                # revised_vector = raw_get_distances_mic(aseatom, node.index+mol_inds[representative], nodes[near].index+mol_inds[representative], mic=True, vector=True) # !! これ間違ってる？
                vectors[nodes[near].index] = vectors[node.index]+revised_vector # vectorsはrepresentativeからの距離
                # debug # print("node/parent/revised/vectors {}/{}/{}/{}".format(nodes[near].index,node.index,revised_vector,vectors[nodes[near].index]))
    # # 親ノードを格納
    # ans = [nodes[i].parent for i in range(len(nodes))]
    # # -1が含まれていたらノード1に辿り着けないノードが存在する
    # if -1 in ans:
    #     print("No")
    # else:
    #     print("Yes")
    #     for node,a in zip(nodes,ans):
    #         print("node/parent/vector :: {}/{}/{}".format(node.index, a,vectors[node.index]))
    return vectors



def raw_calc_mol_coord_and_bc_mic_onemolecule_new(mol_inds,bonds_list,aseatoms,itp_data) :
    '''
        TODO :: itp_data.representative_atom_indexを使って書き直す
        # * 系内のあるひとつの分子に着目し，ボンドセンターと（micを考慮した）分子座標を計算する．
        # * 特にmicを考慮した分子座標計算では，gromacsの gmx trjconv -pbc molに相当する処理を行う．
        inputs
        ----------------
        mol_inds # list :: 分子を構成する原子のindexのリスト (先頭原子が分子の始点となる)
        bonds_list_j # list[list]] :: 分子中の各結合の原子indexのリスト。
        mol_inds[0] :: 分子の先頭の原子．これを基準とするコードになってる．

        output
        -----------------
        # mol_coords # numpy array 分子を構成する原子の座標（MIC考慮）
        # bond_centers # numpy array 各結合の中点座標のリスト
    '''
    # vectors = aseatoms.get_all_distances(mic=True,vector=True)  #GromacsでPBC=Molの場合は、mic=Falseとする
    # vectors = aseatoms.get_distances(mol_inds[0], mol_inds, mic=True, vector=True) # 必要なベクトルだけを求めることもできる．
    # TODO :: 単にこうやってmicに従って距離を求めただけだと分子が一つにならない場合が出てきた．
    # TODO :: そのような場合に対処するため，距離を求めた後にボンドを為すもう片方の原子との距離を計算してそれがあまりに大きい場合には警告を出すことから始める．
    # vectors = raw_get_distances_mic(aseatoms,mol_inds[0], mol_inds, mic=True, vector=True) # 上のコードと同じことをしている．
    vectors = raw_get_pbc_mol(aseatoms,mol_inds,bonds_list,itp_data)

    # 分子内の原子の座標をR0基準に再計算
    R0 = aseatoms.get_positions()[mol_inds[itp_data.representative_atom_index]] # 最初の原子の座標
    
    # 全ての原子（分子に含まれる）の座標を取得する．
    mol_coords=R0+vectors 
    
    # 全てのボンドセンターの座標を取得する．
    # 二つのdrがボンドの両端の原子への距離
    bond_centers = [R0+(vectors[l[0]]+vectors[l[1]])/2.0 for l in bonds_list] # R0にボンドセンターへの座標をたす．
    if np.any(np.linalg.norm(bond_centers)) > 2.0:
        print("WARNING :: bond length is too long !! ")
    
    return np.array(mol_coords), np.array(bond_centers)



def raw_calc_mol_coord_and_bc_mic_onemolecule(mol_inds,bonds_list_j,aseatoms,itp_data) :
    '''
        TODO :: itp_data.representative_atom_indexを使って書き直す
        # * 系内のあるひとつの分子に着目し，ボンドセンターと（micを考慮した）分子座標を計算する．
        # * 特にmicを考慮した分子座標計算では，gromacsの gmx trjconv -pbc molに相当する処理を行う．
        inputs
        ----------------
        mol_inds # list :: 分子を構成する原子のindexのリスト (先頭原子が分子の始点となる)
        bonds_list_j # list[list]] :: 分子中の各結合の原子indexのリスト。
        mol_inds[0] :: 分子の先頭の原子．これを基準とするコードになってる．

        output
        -----------------
        # mol_coords # numpy array 分子を構成する原子の座標（MIC考慮）
        # bond_centers # numpy array 各結合の中点座標のリスト
    '''
    # vectors = aseatoms.get_all_distances(mic=True,vector=True)  #GromacsでPBC=Molの場合は、mic=Falseとする
    # vectors = aseatoms.get_distances(mol_inds[0], mol_inds, mic=True, vector=True) # 必要なベクトルだけを求めることもできる．
    # TODO :: 単にこうやってmicに従って距離を求めただけだと分子が一つにならない場合が出てきた．
    # TODO :: そのような場合に対処するため，距離を求めた後にボンドを為すもう片方の原子との距離を計算してそれがあまりに大きい場合には警告を出すことから始める．
    # vectors = raw_get_distances_mic(aseatoms,mol_inds[0], mol_inds, mic=True, vector=True) # 上のコードと同じことをしている．
    vectors = raw_get_pbc_mol(aseatoms,mol_inds,bonds_list_j,itp_data)
    coords  = aseatoms.get_positions()
    
    # 分子内の原子の座標をR0基準に再計算
    R0 = coords[mol_inds[itp_data.representative_atom_index]] # 最初の原子の座標
    
    # mol_indsとbonds_listを0始まりのインデックスで書き直す．
    # TODO :: hard code :: とりあえずの処置として，全てのインデックスの番号をずらすやり方をとる．
    # TODO :: これだと将来的に原子番号が綺麗な順番になっていない場合に対応できない．
    mol_inds_from_zero=[i-mol_inds[0] for i in mol_inds]
    bonds_list_from_zero=[[i[0]-mol_inds[0],i[1]-mol_inds[0]] for i in bonds_list_j]
    
    # 全ての原子（分子に含まれる）の座標を取得する．
    mol_coords=[R0+vectors[k] for k in mol_inds_from_zero ]
    # for k in mol_inds_from_zero: # 古いコード
    #     mol_coords.append(R0+vectors[k])
    
    # 全てのボンドセンターの座標を取得する．
    bond_centers = []
    # bond_infos = []
    for l in bonds_list_from_zero :
        # 二つのdrがボンドの両端の原子への距離
        bc = R0+(vectors[l[0]]+vectors[l[1]])/2.0 # R0にボンドセンターへの座標をたす．
        if np.linalg.norm(vectors[l[0]]-vectors[l[1]]) > 2.0:
            print("WARNING :: bond length is too long !! ")
        bond_centers.append(bc) 
        # bond_infos.append(molecule.bondinfo(pair=l,bc=bc, wcs=[]))
    return np.array(mol_coords), np.array(bond_centers)


def raw_calc_bc_mic_onemolecule(mol_inds,mol_coords, bonds_list_j):
    '''
    TODO :: 未完成関数というか，まだ実際には使っていない．
    raw_calc_mol_coord_and_bc_mic_onemolecule
    で計算したmol_coordsを利用してbond_centersを計算する．
    関数をより細かく分割するための取り組み．
    '''
    bonds_list_from_zero=[[i[0]-mol_inds[0],i[1]-mol_inds[0]] for i in bonds_list_j]
    
    # 全てのボンドセンターの座標（l[0]とl[1]の中点の座標）を取得する．
    bond_centers = [ (mol_coords[l[0]]+mol_coords(l[1]))/2.0 for l in bonds_list_from_zero ]
    return np.array(bond_centers)


# * find_lonepair/find_bondwcs/find_piの補助関数として，
# * 原子一つとWCsたちのase.atomsを作るmake_aseatomsを定義した
def raw_make_aseatoms_from_wc(atom_coord:np.array,wfc_list,UNITCELL_VECTORS):
    import ase.atoms
    #原子座標(atom_coord)を先頭においたAtomsオブジェクトを作成する
    atom_wcs_coord=np.array(list([atom_coord,])+list(wfc_list))
    num_element=len(atom_wcs_coord) # WCsの数

    #ワニエ中心のラベルはAuとする
    elements = {"Au":79}
    atom_id= ["Au",]*num_element
    atom_id = [elements[i] for i in atom_id ]

    atom_wan = ase.Atoms(atom_id,
               positions=atom_wcs_coord,        
               cell= UNITCELL_VECTORS,   
               pbc=[1, 1, 1]) 
    return atom_wan

def raw_find_lonepairs(atom_coord:np.array,wfc_list,wcs_num:int,UNITCELL_VECTORS,picked_wfcs):
    '''
    ローンペアまたはボンドセンターから最も近いwcsを探索する
    input
    -------------
    wfc_list   :: WCsの座標リスト
    atom_coord :: 原子の座標（np.array）：実際にはボンドセンターの座標
    wcs_num    :: 見つけるwcsの数．1（N lonepair）か2（O lonepair）
    picked_wcs :: これはoption
    
    output
    -------------
    wcs_indices :: 答えのワニエのindex
    mu_lp :: 計算された双極子
    '''
    # 物理定数
    from include.constants import constant  
    # Debye   = 3.33564e-30
    # charge  = 1.602176634e-019
    # ang      = 1.0e-10 
    coef    = constant.Ang*constant.Charge/constant.Debye 
    
    if wcs_num != 1 and wcs_num != 2:
        print("ERROR :: wcs_num should be 1 or 2 !!")
        print("wfc_num = ", wcs_num)
        return -1

    #選択したボンドセンターを先頭において，残りはwcsで構成されるAtomsオブジェクトを作成する
    atom_wan=raw_make_aseatoms_from_wc(atom_coord,wfc_list,UNITCELL_VECTORS)
    num_element=len(wfc_list)+1 # 1はatom_coordの分

    # 先頭原子とWCsの距離ベクトル
    # wfc_vectors = atom_wan.get_distances(0,range(1,num_element),mic=True,vector=True) #ワニエ中心は一度CP.xを通しているから点ごとにPBCが適用されている。mic=Trueでよい。
    wfc_vectors = raw_get_distances_mic_multiPBC(atom_wan,0, range(1,num_element), mic=True, vector=True) # 上のコードと同じことをしている．
    
    wfc_distances=np.linalg.norm(wfc_vectors,axis=1)
    if wcs_num == 1:
        wcs_indices = np.argsort(wfc_distances).reshape(-1)[:1] # 最も近いWCsのインデックスを一つ取り出す．
        mu_lp = (-2.0)*coef*wfc_vectors[wcs_indices[0]]
        if np.linalg.norm(wfc_vectors[wcs_indices[0]]) > 1.0: # 原子とwcsの距離があまりに遠い場合はWARNINGを出す
            print("WARNING :: The distance between atom {} and lonepair {} ({}) is too large !! :: {} ".format(atom_coord, wfc_list[wcs_indices[0]],wcs_indices[0], np.linalg.norm(wfc_vectors[wcs_indices[0]])))
        if wcs_indices[0] in picked_wfcs:
            print("WARNING SAME wcs is assined !! :: {}".format(wcs_indices[0]))
    if wcs_num == 2:
        wcs_indices = np.argsort(wfc_distances).reshape(-1)[:2] # 最も近いWCsのインデックスを二つ取り出す．
        # 二つのWannierCenterによる双極子モーメントを計算する．
        mu_lp = (-4.0)*coef*(wfc_vectors[wcs_indices[0]]+wfc_vectors[wcs_indices[1]])/2.0
    # 最後にwcsの（micがかかった）座標を取得
    wcs_lp=[atom_wan.get_positions()[0]+wfc_vectors[i] for i in wcs_indices]
    return wcs_indices, mu_lp, wcs_lp


def raw_find_bondwcs(atom_coord:np.array,wfc_list,wcs_num:int,UNITCELL_VECTORS):
    '''
    1：ボンドセンターから最も近いwcsを探索する
    input
    -------------
    wfc_list :: WCsの座標リスト
    atom_coord :: 原子の座標（np.array）
    wcs_num  :: 見つけるwcsの数．1（N lonepair）か2（O lonepair）
    
    output
    -------------
    wcs_indices :: 答えのワニエのindex
    mu_lp :: 計算された双極子
    '''
    # 物理定数
    from include.constants import constant  
    # Debye   = 3.33564e-30
    # charge  = 1.602176634e-019
    # ang      = 1.0e-10 
    coef    = constant.Ang*constant.Charge/constant.Debye
    
    import ase
    if wcs_num != 1 and wcs_num != 2:
        print("ERROR :: wcs_num should be 1 or 2 !!")
        print("wfc_num = ", wcs_num)
        return -1
    
    #選択した原子座標を先頭においたAtomsオブジェクトを作成する
    atom_wan=raw_make_aseatoms_from_wc(atom_coord,wfc_list,UNITCELL_VECTORS)
    num_element=len(wfc_list)+1 # 1はatom_coordの分
    
    # 先頭原子とWCsの距離ベクトル
    # wfc_vectors = atom_wan.get_distances(0,range(1,num_element),mic=True,vector=True) #ワニエ中心は一度CP.xを通しているから点ごとにPBCが適用されている。mic=Trueでよい。
    wfc_vectors = raw_get_distances_mic_multiPBC(atom_wan,0, range(1,num_element), mic=True, vector=True) # 上のコードと同じことをしている．
    wfc_distances=np.linalg.norm(wfc_vectors,axis=1)
    if wcs_num == 1:
        wcs_indices = np.argsort(wfc_distances).reshape(-1)[:1]
        mu_lp = (-2.0)*coef*wfc_vectors[wcs_indices[0]]
        if np.linalg.norm(wfc_vectors[wcs_indices[0]]) > 1.0: # bcとwcsの距離があまりに遠い場合はWARNINGを出す
            print("WARNING :: The distance between bc {} and wc {} is too large !! :: {} Ang.".format(atom_coord, wcs_indices[0],np.linalg.norm(wfc_vectors[wcs_indices[0]])))
    if wcs_num == 2:
        wcs_indices = np.argsort(wfc_distances).reshape(-1)[:2] # 最も近いWCsのインデックスを二つ取り出す．
        # 二つのWannierCenterによる双極子モーメントを計算する．
        mu_lp = (-4.0)*coef*(wfc_vectors[wcs_indices[0]]+wfc_vectors[wcs_indices[1]])/2.0
    # 最後にwcsの（micがかかった）座標を取得
    wcs_bond=[atom_wan.get_positions()[0]+wfc_vectors[i] for i in wcs_indices]
    # for i in wcs_indices:
    #    wcs_bond.append(atom_wan.get_positions()[0]+wfc_vectors[i])
    return wcs_indices, mu_lp, wcs_bond

def raw_find_pi(atom_coord:np.array,wfc_list,r_threshold:float,picked_wcs,UNITCELL_VECTORS):
    '''
    find_lonepairから一行違うだけ！
    
    input
    -------------
    wfc_list :: WCsの座標リスト
    atom_coord :: 原子の座標（np.array）
    wcs_num  :: 見つけるwcsの数．1（N lonepair）か2（O lonepair）
    
    output
    -------------
    wcs_indices :: 答えのワニエのindex
    mu_lp :: 計算された双極子
    '''
    # 物理定数
    from include.constants import constant  
    # Debye   = 3.33564e-30
    # charge  = 1.602176634e-019
    # ang      = 1.0e-10 
    coef    = constant.Ang*constant.Charge/constant.Debye
    
    #選択した原子座標を先頭においたAtomsオブジェクトを作成する
    atom_wan=raw_make_aseatoms_from_wc(atom_coord,wfc_list,UNITCELL_VECTORS)
    num_element=len(wfc_list)+1 # 1はatom_coordの分
    
    # 先頭原子とWCsの距離ベクトル
    # wfc_vectors = atom_wan.get_distances(0,range(1,num_element),mic=True,vector=True) #ワニエ中心は一度CP.xを通しているから点ごとにPBCが適用されている。mic=Trueでよい。
    wfc_vectors =  raw_get_distances_mic_multiPBC(atom_wan,0, range(1,num_element), mic=True, vector=True) # 上のコードと同じことをしている．    
    wfc_distances=np.linalg.norm(wfc_vectors,axis=1)
    wcs_indices = np.argwhere(wfc_distances<r_threshold).reshape(-1) #まずは条件に合うのを抽出
    wcs_indices = [i for i in wcs_indices if i not in picked_wcs][:1]
    #[wfc_list[config][i] for i in range(len(wfc_list[config])) if i not in picked_wfcs]
    if len(wcs_indices) == 0:
        # print("WARNING :: ボンドに割り当てるWCがありません !! " )
        # print("WARNING :: この場合はreturnがNull Nullになります !!" )
        return None, None, None
    else:
        mu_lp = (-2.0)*coef*wfc_vectors[wcs_indices[0]]
        # 最後にwcsの（micがかかった）座標を取得
        wcs_dbond=[atom_wan.get_positions()[0]+wfc_vectors[i] for i in wcs_indices]
        # for i in wcs_indices:
        #    wcs_dbond.append(atom_wan.get_positions()[0]+wfc_vectors[i])
        return wcs_indices, mu_lp, wcs_dbond


# 
# * calc_mu_bond_lonepair用の部品関数たち
# 
def raw_find_all_lonepairs(wfc_list,atO_list,list_mol_coords,picked_wfcs,wcs_num:int,UNITCELL_VECTORS):
    '''
    最後の変数num_wcsでOローンペアとNローンペアに対応
    '''
    list_mu_lp   =  []
    list_lp_wfcs =  []
    
    if wcs_num != 1 and wcs_num != 2:
        print("ERROR :: wcs_num should be 1 or 2 !!")
        print("wfc_num = ", wcs_num)
        return -1
    
    # Oのローンペア
    for atOs,mol_coords in zip(atO_list,list_mol_coords) : #分子の数に関するループ
        mu_lpO_mol = []
        wcs_mol = [] # wcsの座標
        for atO in atOs : #ある分子内の原子に関するループ
            center_atom_coord  = mol_coords[atO]
            #wfc_list_exclude_pickedwfcs = [wfc_list[config][i] for i in range(len(wfc_list[config])) if i not in picked_wfcs]
            wcs_indices, mu_lp, wcs_lp = raw_find_lonepairs(center_atom_coord, wfc_list, wcs_num,UNITCELL_VECTORS,picked_wfcs)
            picked_wfcs        = picked_wfcs + list(wcs_indices)
            mu_lpO_mol.append(mu_lp)
            wcs_mol.append(wcs_lp)
        list_mu_lp.append(mu_lpO_mol)
        list_lp_wfcs.append(wcs_mol) 
    return np.array(list_mu_lp), np.array(list_lp_wfcs), picked_wfcs

def raw_find_all_bonds(wfc_list,list_bond_centers,picked_wfcs,UNITCELL_VECTORS,wfc_num:int=1):
    '''
    シングルボンドの場合のワニエのアサインを実施
    '''
    list_mu_bonds = [] # [NUM_MOL,NUM_BONDS,3]型
    list_bond_wfcs = [] # [NUM_MOL,NUM_BONDS,3]型
    
    for bcs in list_bond_centers :  # 分子数に関するループ
        mu_bonds_mol = []
        wcs_mol = []
        for bond_center_coord in bcs : # ある分子内のボンドセンターに関するループ
            # wfc_list_exclude_pickedwfcs = [wfc_list[config][i] for i in range(len(wfc_list[config])) if i not in picked_wfcs]
            wcs_indices, mu_bond,wcs_bond = raw_find_lonepairs(bond_center_coord, wfc_list, wfc_num,UNITCELL_VECTORS, picked_wfcs)
            picked_wfcs        = picked_wfcs + list(wcs_indices)
            wcs_mol.append(wcs_bond)
            mu_bonds_mol.append(mu_bond)
        list_mu_bonds.append(mu_bonds_mol)
        list_bond_wfcs.append(wcs_mol)
    return np.array(list_mu_bonds), np.array(list_bond_wfcs), picked_wfcs


def raw_find_all_pi(wfc_list,list_bond_centers,picked_wfcs,double_bonds,UNITCELL_VECTORS):
    # TODO :: hard code :: ここは改善の余地あり．
    list_mu_pai = [] 
    list_pi_wfcs = []

    for bcs in list_bond_centers : # 再度ボンドセンターを探索
        dbcs = bcs[double_bonds] # double_bondsに属するものを選択
        mu_bonds_mol = []
        wcs_mol = []
        for dbond_center_coord  in dbcs : # 二重結合的に探索する？
            r_threshold = 0.65 # angstrom
            # wfc_list_exclude_pickedwfcs = [wfc_list[config][i] for i in range(len(wfc_list[config])) if i not in picked_wfcs]
            wcs_indices, mu_bond, wcs_bond = raw_find_pi(dbond_center_coord,wfc_list,r_threshold,picked_wfcs,UNITCELL_VECTORS)
            if wcs_indices is not None and mu_bond is not None:
                picked_wfcs        = picked_wfcs + list(wcs_indices)
                mu_bonds_mol.append(mu_bond)
                wcs_mol.append(wcs_bond)
        list_mu_pai.append(mu_bonds_mol)
        list_pi_wfcs.append(wcs_mol)
    return np.array(list_mu_pai), np.array(list_pi_wfcs), picked_wfcs



#
# * 全てのwcsの割り当て
def raw_calc_mu_bond_lonepair(wfc_list,ase_atoms:ase.Atoms,bonds_list, itp_data, double_bonds,NUM_MOL_ATOMS:int,NUM_MOL:int,UNITCELL_VECTORS) :
    '''
    # * wfc_list：あるconfigでのワニエの座標リスト
    # * この時WCsの各ボンドへの割り当ても行われる．
    # * parsed_resultsはparse_cpmd_result関数の出力を入れる．(list_mol_coords,list_bond_centers)
    # * output
    ボンドの双極子: list_mu_bonds
    π結合の双極子：list_mu_pai
    Oのローンペアの双極子：list_mu_lpO
    Nのローンペアの双極子：list_mu_lpN
    
        # parsed_results : 関数parse_cpmd_resultを参照 
    '''
    # Calculate atomic & BC coordinates from ase_atoms
    results=raw_aseatom_to_mol_coord_bc(ase_atoms,bonds_list, itp_data, NUM_MOL_ATOMS, NUM_MOL ) 
    #result into mol_coords and bond_centers
    list_mol_coords,list_bond_centers = results

    #各結合上のワニエ中心の座標を取得する
    r_threshold =0.65 #[ang.] 結合中点位置からどの距離までをワニエ中心とみなすかのしきい値

    list_mu_bonds = []
    picked_wfcs = [] #すでにアサインされたwcsを入れる．
    
    list_bond_wfcs = []

    ##########O原子とN原子上のLonePairを探索する###########

    list_mu_lpO = []
    list_mu_lpN = []
    
    list_lpO_wfcs = []
    list_lpN_wfcs = []

    # O/N原子があるところのリスト (reshape to NUM_MOL * NUM_MOL_ATOMS)
    list_atomic_nums = list(np.array(ase_atoms.get_atomic_numbers()).reshape(NUM_MOL,-1))
    atO_list = [np.argwhere(js==8).reshape(-1) for js in list_atomic_nums] #原子番号8
    atN_list = [np.argwhere(js==7).reshape(-1) for js in list_atomic_nums] #原子番号7
    # for js in list_atomic_nums: # 古いコード
    #     atO = np.argwhere(js==8).reshape(-1) #原子番号8
    #     atN = np.argwhere(js==7).reshape(-1) #原子番号7
    #     atO_list.append(atO)
    #     atN_list.append(atN)

    # Oローンペア
    list_mu_lpO, list_lpO_wfcs, picked_wcs_O = raw_find_all_lonepairs(wfc_list,atO_list,list_mol_coords,picked_wfcs,2,UNITCELL_VECTORS)
    picked_wfcs = picked_wfcs + list(picked_wcs_O)

    # Nローンペア
    list_mu_lpN, list_lpN_wfcs, picked_wcs_N = raw_find_all_lonepairs(wfc_list,atN_list,list_mol_coords,picked_wfcs,1,UNITCELL_VECTORS)
    picked_wfcs = picked_wfcs + list(picked_wcs_N)

    # ボンドセンター    
    list_mu_bonds, list_bond_wfcs, picked_wcs_bond = raw_find_all_bonds(wfc_list,list_bond_centers,picked_wfcs,UNITCELL_VECTORS,1)
    picked_wfcs = picked_wfcs + list(picked_wcs_bond)
    

    ##########π電子を探索する###########    
    # TODO :: hard code :: ここは改善の余地あり．
    list_mu_pai, list_pi_wfcs, picked_wcs_pi = raw_find_all_pi(wfc_list,list_bond_centers,picked_wfcs,double_bonds,UNITCELL_VECTORS) 
    picked_wfcs = picked_wfcs + list(picked_wcs_pi)

    ###for Debug###
    #print(np.sort(np.array(picked_wfcs)))
    #print(len(picked_wfcs))

    return list_mu_bonds,list_mu_pai,list_mu_lpO,list_mu_lpN, list_bond_wfcs,list_pi_wfcs,list_lpO_wfcs,list_lpN_wfcs


def save_yaml(NUM_MOL:int, itpdata, list_bond_centers, list_mu_bonds):
    import cpmd.descripter
    molecule = {}
    # 分子ループ
    for mol_id in range(NUM_MOL):
        # ボンドループ
        test = []
        for bond_index, bond in enumerate(itpdata.bonds_list):
            # ボンドindexの座標を計算
            cent_mol   =  cpmd.descripter.find_specific_bondcenter(list_bond_centers, bond_index) 
            # ボンド双極子
            bond_dipole = list_mu_bonds[mol_id][bond_index]
            # ボンドリスト作成
            test[bond_index] = {"bc": cent_mol, "bd": bond_dipole, "index": bond}
            print("")
            molecule["molecule"][mol_id] = test
    
    # 各ボンドでの双極子とボンドセンターを保持．
    return 0
