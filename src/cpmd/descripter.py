'''
descripterを作成するためのコード
'''

from ase.io import read
import ase
import sys
import numpy as np
# from types import NoneType
from cpmd.asign_wcs import raw_get_distances_mic # get_distances(mic)の計算用
    
#Cutoff関数の定義
import numpy as np
def fs(Rij,Rcs,Rc) :
    '''
    #####Inputs####
    # Rij : float 原子間距離 [ang. unit] 
    # Rcs : float inner cut off [ang. unit]
    # Rc  : float outer cut off [ang. unit] 
    ####Outputs####
    # sij value 
    ###############
    '''
    
    if Rij < Rcs :
        s = 1/Rij 
    elif Rij < Rc :
        s = (1/Rij)*(0.5*np.cos(np.pi*(Rij-Rcs)/(Rc-Rcs))+0.5)
    else :
        s = 0 
    return s 

#ベクトルの回転
def rot_vec(vec,ths):
    thx,thy,thz = np.pi*ths
    Rx = np.array([ [1.0, 0.0, 0.0],[0.0,np.cos(thx),np.sin(thx)],[0.0,-np.sin(thx),np.cos(thx)] ])
    Ry = np.array([ [np.cos(thy),0.0,np.sin(thy)],[0.0, 1.0, 0.0],[-np.sin(thy),0.0,np.cos(thy)] ])
    Rz = np.array([ [np.cos(thz),np.sin(thz),0.0],[-np.sin(thz),np.cos(thz),0.0], [0.0, 0.0, 1.0] ])
    new_vec = np.dot(Rz,np.dot(Ry,np.dot(Rx,vec)))
    return new_vec


class descripter:
    import ase
    '''
    関数をメソッドとしてこちらにうつしていく．
    その際，基本となる変数をinitで定義する
    '''
    def __init__(self, NUM_MOL:int, NUM_MOL_ATOMS:int, UNITCELL_VECTORS):
        self.NUM_MOL       = NUM_MOL
        self.NUM_MOL_ATOMS = NUM_MOL_ATOMS
        self.UNITCELL_VECTORS = UNITCELL_VECTORS

    def get_desc_bondcent(self,atoms,bond_center,mol_id):
        return raw_get_desc_bondcent(atoms, bond_center, mol_id, self.UNITCELL_VECTORS, self.NUM_MOL_ATOMS)
        
    def get_desc_lonepair(self,atoms,bond_center,mol_id):
        return raw_get_desc_lonepair(atoms, bond_center, mol_id, self.UNITCELL_VECTORS, self.NUM_MOL_ATOMS)
    
    def calc_bond_descripter_at_frame(self,atoms_fr,list_bond_centers,bond_index, desctype):
        return raw_calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,bond_index, self.NUM_MOL,self.UNITCELL_VECTORS, self.NUM_MOL_ATOMS, desctype)

    def calc_lonepair_descripter_at_frame(self,atoms_fr,list_mol_coords, at_list, atomic_index:int, desctype):
        return raw_calc_lonepair_descripter_at_frame(atoms_fr,list_mol_coords, at_list, self.NUM_MOL, atomic_index, self.UNITCELL_VECTORS, self.NUM_MOL_ATOMS, desctype)

    def calc_bondmu_descripter_at_frame(self, list_mu_bonds, bond_index):
        return raw_calc_bondmu_descripter_at_frame(list_mu_bonds, bond_index)
    
    def calc_lonepairmu_descripter_at_frame(self,list_mu_lp, list_atomic_nums, at_list, atomic_index:int):
        return raw_calc_lonepairmu_descripter_at_frame(list_mu_lp, list_atomic_nums, at_list, atomic_index)

    
def raw_make_atoms(bond_center,atoms,UNITCELL_VECTORS) :
    '''
    ######INPUTS#######
    bond_center     # vector 記述子を求めたい結合中心の座標
    list_mol_coords # array  分子ごとの原子座標
    list_atomic_nums #array  分子ごとの原子座標
    '''
    from ase import Atoms
    list_mol_coords=atoms.get_positions()
    list_atomic_nums=atoms.get_atomic_numbers()
    
    #選択した結合中点の座標を先頭においたAtomsオブジェクトを作成する
    pos = np.array(list([bond_center,])+list(list_mol_coords))
    #結合中心のラベルはAuとする
    elements = {"Au":79}
    atom_id= list(["Au",])+list(list_atomic_nums)
    
    WBC = ase.Atoms(atom_id,
             positions=pos,        
             cell= UNITCELL_VECTORS,   
             pbc=[1, 1, 1]) 
    return WBC

def calc_descripter(dist_wVec, atoms_index,Rcs,Rc,MaxAt):
    ''' 
    ある原子種に対する記述子を作成する．
    input
    -----------
    dist_wVec :: ある原子種からの距離
    atoms :: 
    MaxAt :: 最大の原子数
    '''
    drs =np.array([v for l,v in enumerate(dist_wVec) if (l in atoms_index) and (l!=0)]) # 相対ベクトル(x,y,z)
    
    # もしdの中に0のものがあったらそれを排除したい．
    # そこでnp.sum(np.abs(drs[j])) = 0（要するに全ての要素が0）のものを排除する．
    drs_tmp = [] # 変更するための配列
    for j in range(len(drs)):
        if np.sum(np.abs(drs[j])) > 0.001: # 0.001は適当な閾値．現状これでうまくいっている
            drs_tmp.append(drs[j])
    drs = np.array(drs_tmp) #新しいもので置き換え
    # >>>> ここまでで不要な要素の削除 >>>>>>
    
    if np.shape(drs)[0] == 0: # 要素が0の時．dijは空とする（これをやらないと要素0時にエラーになる）
        dij = []    
    else:
        d = np.sqrt(np.sum(drs**2,axis=1)) # 距離r
        s = np.array([fs(Rij,Rcs,Rc) for Rij in d ]) # cutoff関数
        order_indx = np.argsort(s)[-1::-1]  # sの大きい順に並べる
        sorted_drs = drs[order_indx]
        sorted_s   = s[order_indx]
        sorted_d   = d[order_indx]
        dij  = [ [si,]+list(si*vi/di) for si,vi,di in zip(sorted_s,sorted_drs,sorted_d)]

    #原子数がMaxAtよりも少なかったら０埋めして固定長にする。1原子あたり4要素(1,x/r,y/r,z/r)
    if len(dij) < MaxAt :
        dij_desc = list(np.array(dij).reshape(-1)) + [0]*(MaxAt - len(dij))*4
    else :
        dij_desc = list(np.array(dij).reshape(-1))[:MaxAt*4]
    return dij_desc


def raw_get_desc_bondcent(atoms,bond_center,mol_id, UNITCELL_VECTORS, NUM_MOL_ATOMS:int) :
    
    
    from ase import Atoms
    '''
    ボンドセンター用の記述子を作成
    ######Inputs########
    atoms : ASE atom object 構造の入力
    Rcs : float inner cut off [ang. unit]
    Rc  : float outer cut off [ang. unit] 
    MaxAt : int 記述子に記載する原子数（これにより固定長の記述子となる）
    #bond_center : vector 記述子を計算したい結合の中心
    ######Outputs#######
    Desc : 原子番号,[List O原子のSij x MaxAt : H原子のSij x MaxAt] x 原子数 の二次元リストとなる.
    ####################
    '''
    ###INPUTS###
    # parsed_results : 関数parse_cpmd_resultを参照 
    ######parameter入力######
    Rcs = 4.0 #[ang. unit] TODO : hard code
    Rc  = 6.0 #[ang. unit] TODO : hard code
    MaxAt = 12 # とりあえずは12個の原子で良いはず．
    ##########################

    # ボンドセンターを追加したatoms
    atoms_w_bc = raw_make_atoms(bond_center,atoms, UNITCELL_VECTORS)
    
    atoms_in_molecule = [i for i in range(mol_id*NUM_MOL_ATOMS+1,(mol_id+1)*NUM_MOL_ATOMS+1)] #結合中心を先頭に入れたAtomsなので+1
    
    # 各原子の記述子を作成する．
    Catoms_all   =  [i for i,j in enumerate(atoms_w_bc.get_atomic_numbers()) if (j == 6) ]
    Hatoms_all   =  [i for i,j in enumerate(atoms_w_bc.get_atomic_numbers()) if (j == 1) ]
    Oatoms_all   =  [i for i,j in enumerate(atoms_w_bc.get_atomic_numbers()) if (j == 8) ]
    Catoms_intra =  [i for i in Catoms_all if i in atoms_in_molecule]
    Catoms_inter =  [i for i in Catoms_all if i not in atoms_in_molecule ]
    Hatoms_intra =  [i for i in Hatoms_all if i in atoms_in_molecule]
    Hatoms_inter =  [i for i in Hatoms_all if i not in atoms_in_molecule ]   
    Oatoms_intra =  [i for i in Oatoms_all if i in atoms_in_molecule]
    Oatoms_inter =  [i for i in Oatoms_all if i not in atoms_in_molecule ]   

    at_list = [i for i in range(len(atoms_w_bc))] # 全ての原子との距離を求める
    # dist_wVec = atoms_w_bc.get_distances(0,at_list,mic=True,vector=True)  #0-0間距離も含まれる
    dist_wVec = raw_get_distances_mic(atoms_w_bc,0, at_list, mic=True,vector=True) # 0-0間距離も含まれる
    # at_nums = atoms_w_bc.get_atomic_numbers()

    #for C atoms (intra) 
    dij_C_intra=calc_descripter(dist_wVec, Catoms_intra, Rcs,Rc,MaxAt)
    #for H atoms (intra)
    dij_H_intra=calc_descripter(dist_wVec, Hatoms_intra, Rcs,Rc,MaxAt)
    #for O  atoms (intra)
    dij_O_intra=calc_descripter(dist_wVec, Oatoms_intra, Rcs,Rc,MaxAt)
    #for C atoms (inter)
    dij_C_inter=calc_descripter(dist_wVec, Catoms_inter, Rcs,Rc,MaxAt)
    #for H atoms (inter)
    dij_H_inter=calc_descripter(dist_wVec, Hatoms_inter,Rcs,Rc,MaxAt)
    #for O atoms (inter)
    dij_O_inter=calc_descripter(dist_wVec, Oatoms_inter,Rcs,Rc,MaxAt)

    return(dij_C_intra+dij_H_intra+dij_O_intra+dij_C_inter+dij_H_inter+dij_O_inter)


def raw_get_desc_bondcent_allinone(atoms,bond_center,mol_id, UNITCELL_VECTORS, NUM_MOL_ATOMS:int) :
    
    
    from ase import Atoms
    '''
    ボンドセンター用の記述子を作成
    2023/6/27 :: 分子内と分子間を分けない．その代わりMaxAtを24まで増やす．
    ######Inputs########
    atoms : ASE atom object 構造の入力
    Rcs : float inner cut off [ang. unit]
    Rc  : float outer cut off [ang. unit] 
    MaxAt : int 記述子に記載する原子数（これにより固定長の記述子となる）
    #bond_center : vector 記述子を計算したい結合の中心
    ######Outputs#######
    Desc : 原子番号,[List O原子のSij x MaxAt : H原子のSij x MaxAt] x 原子数 の二次元リストとなる.
    ####################
    '''
    ###INPUTS###
    # parsed_results : 関数parse_cpmd_resultを参照 
    ######parameter入力######
    Rcs = 4.0 #[ang. unit] TODO : hard code
    Rc  = 6.0 #[ang. unit] TODO : hard code
    MaxAt = 24 # intraとinterを分けない分，元の12*2=24としている．
    ##########################

    # ボンドセンターを追加したatoms
    atoms_w_bc = raw_make_atoms(bond_center,atoms, UNITCELL_VECTORS)
    
    # atoms_in_molecule = [i for i in range(mol_id*NUM_MOL_ATOMS+1,(mol_id+1)*NUM_MOL_ATOMS+1)] #結合中心を先頭に入れたAtomsなので+1
    
    # 各原子の記述子を作成する．
    Catoms_all   =  [i for i,j in enumerate(atoms_w_bc.get_atomic_numbers()) if (j == 6) ]
    Hatoms_all   =  [i for i,j in enumerate(atoms_w_bc.get_atomic_numbers()) if (j == 1) ]
    Oatoms_all   =  [i for i,j in enumerate(atoms_w_bc.get_atomic_numbers()) if (j == 8) ]

    at_list = [i for i in range(len(atoms_w_bc))] # 全ての原子との距離を求める
    # dist_wVec = atoms_w_bc.get_distances(0,at_list,mic=True,vector=True)  #0-0間距離も含まれる
    dist_wVec = raw_get_distances_mic(atoms_w_bc,0, at_list, mic=True,vector=True) # 0-0間距離も含まれる
    # at_nums = atoms_w_bc.get_atomic_numbers()

    #for C atoms 
    dij_C_all=calc_descripter(dist_wVec, Catoms_all, Rcs,Rc,MaxAt)
    #for H atoms
    dij_H_all=calc_descripter(dist_wVec, Hatoms_all, Rcs,Rc,MaxAt)
    #for O  atoms
    dij_O_all=calc_descripter(dist_wVec, Oatoms_all, Rcs,Rc,MaxAt)

    return(dij_C_all+dij_H_all+dij_O_all)


def raw_get_desc_lonepair(atoms,lonepair_coord,mol_id, UNITCELL_VECTORS, NUM_MOL_ATOMS:int):
    
    from ase import Atoms
    '''
    ######Inputs########
    # atoms : ASE atom object 構造の入力
    # Rcs : float inner cut off [ang. unit]
    # Rc  : float outer cut off [ang. unit] 
    # MaxAt : int 記述子に記載する原子数（これにより固定長の記述子となる）
    #bond_center : vector 記述子を計算したい結合の中心
     mol_id : bond_centerが含まれる分子のid．
    ######Outputs#######
    # Desc : 原子番号,[List O原子のSij x MaxAt : H原子のSij x MaxAt] x 原子数 の二次元リストとなる.
    ####################
    
    ###INPUTS###
    # parsed_results : 関数parse_cpmd_resultを参照 
    '''
    ######parameter入力######
    Rcs = 4.0 #[ang. unit] TODO :: hard code 
    Rc  = 6.0 #[ang. unit] TODO :: hard code 
    MaxAt = 12 # とりあえずは12個の原子で良いはず．
    ##########################

    
    # ボンドセンターを追加したatoms
    atoms_w_bc = raw_make_atoms(lonepair_coord,atoms, UNITCELL_VECTORS)

    atoms_in_molecule = [i for i in range(mol_id*NUM_MOL_ATOMS+1,(mol_id+1)*NUM_MOL_ATOMS+1)] #結合中心を先頭に入れたAtomsなので+1

    # 各原子のインデックスを取得
    Catoms_all = [ i for i,j in enumerate(atoms_w_bc.get_atomic_numbers()) if (j == 6) ]
    Hatoms_all = [ i for i,j in enumerate(atoms_w_bc.get_atomic_numbers()) if (j == 1) ]
    Oatoms_all = [ i for i,j in enumerate(atoms_w_bc.get_atomic_numbers()) if (j == 8) ]
    Catoms_intra =  [i for i in Catoms_all if i in atoms_in_molecule]
    Catoms_inter =  [i for i in Catoms_all if i not in atoms_in_molecule ]
    Hatoms_intra =  [i for i in Hatoms_all if i in atoms_in_molecule]
    Hatoms_inter =  [i for i in Hatoms_all if i not in atoms_in_molecule ]   
    Oatoms_intra =  [i for i in Oatoms_all if i in atoms_in_molecule]
    Oatoms_inter =  [i for i in Oatoms_all if i not in atoms_in_molecule ]   

    at_list = [i for i in range(len(atoms_w_bc))]
    # dist_wVec = atoms_w_bc.get_distances(0,at_list,mic=True,vector=True)  #0-0間距離も含まれる
    dist_wVec = raw_get_distances_mic(atoms_w_bc,0,at_list,mic=True,vector=True)  #0-0間距離も含まれる
    # at_nums = atoms_w_bc.get_atomic_numbers()

    # dist_wVec：ボンドセンターから他の原子までの距離
    #for C atoms (intra) 
    dij_C_intra=calc_descripter(dist_wVec, Catoms_intra,Rcs,Rc,MaxAt)    
    #for H atoms (intra)
    dij_H_intra=calc_descripter(dist_wVec, Hatoms_intra,Rcs,Rc,MaxAt)  
    #for O  atoms (intra)
    dij_O_intra=calc_descripter(dist_wVec, Oatoms_intra,Rcs,Rc,MaxAt)  
    #for C atoms (inter)
    dij_C_inter=calc_descripter(dist_wVec, Catoms_inter,Rcs,Rc,MaxAt) 
    #for H atoms (inter)
    dij_H_inter=calc_descripter(dist_wVec, Hatoms_inter,Rcs,Rc,MaxAt) 
    #for O atoms (inter)
    dij_O_inter=calc_descripter(dist_wVec, Oatoms_inter,Rcs,Rc,MaxAt)     

    return(dij_C_intra+dij_H_intra+dij_O_intra+dij_C_inter+dij_H_inter+dij_O_inter)


def raw_get_desc_lonepair_allinone(atoms,lonepair_coord, UNITCELL_VECTORS, NUM_MOL_ATOMS:int):
    
    from ase import Atoms
    '''
    ######Inputs########
    # atoms : ASE atom object 構造の入力
    # Rcs : float inner cut off [ang. unit]
    # Rc  : float outer cut off [ang. unit] 
    # MaxAt : int 記述子に記載する原子数（これにより固定長の記述子となる）
    #bond_center : vector 記述子を計算したい結合の中心
     mol_id : bond_centerが含まれる分子のid．
    ######Outputs#######
    # Desc : 原子番号,[List O原子のSij x MaxAt : H原子のSij x MaxAt] x 原子数 の二次元リストとなる.
    ####################
    
    ###INPUTS###
    # parsed_results : 関数parse_cpmd_resultを参照 
    '''
    ######parameter入力######
    Rcs = 4.0 #[ang. unit] TODO :: hard code 
    Rc  = 6.0 #[ang. unit] TODO :: hard code 
    MaxAt = 24 # とりあえずは12個の原子で良いはず．
    ##########################

    
    # ボンドセンターを追加したatoms
    atoms_w_bc = raw_make_atoms(lonepair_coord,atoms, UNITCELL_VECTORS)

    atoms_in_molecule = [i for i in range(mol_id*NUM_MOL_ATOMS+1,(mol_id+1)*NUM_MOL_ATOMS+1)] #結合中心を先頭に入れたAtomsなので+1

    # 各原子のインデックスを取得
    Catoms_all = [ i for i,j in enumerate(atoms_w_bc.get_atomic_numbers()) if (j == 6) ]
    Hatoms_all = [ i for i,j in enumerate(atoms_w_bc.get_atomic_numbers()) if (j == 1) ]
    Oatoms_all = [ i for i,j in enumerate(atoms_w_bc.get_atomic_numbers()) if (j == 8) ]
 
    at_list = [i for i in range(len(atoms_w_bc))]
    # dist_wVec = atoms_w_bc.get_distances(0,at_list,mic=True,vector=True)  #0-0間距離も含まれる
    dist_wVec = raw_get_distances_mic(atoms_w_bc,0,at_list,mic=True,vector=True)  #0-0間距離も含まれる
    # at_nums = atoms_w_bc.get_atomic_numbers()

    # dist_wVec：ボンドセンターから他の原子までの距離
    #for C atoms
    dij_C_all=calc_descripter(dist_wVec, Catoms_all,Rcs,Rc,MaxAt)    
    #for H atoms 
    dij_H_all=calc_descripter(dist_wVec, Hatoms_all,Rcs,Rc,MaxAt)  
    #for O  atoms
    dij_O_all=calc_descripter(dist_wVec, Oatoms_all,Rcs,Rc,MaxAt)  
   

    return(dij_C_all+dij_H_all+dij_O_all)


#
# TODO :: ここは将来的にはボンドをあらかじめ分割するようにして無くしてしまいたい．

def find_specific_bondcenter(list_bond_centers, bond_index):
    '''
    list_bond_centersからbond_index情報をもとに特定のボンド（CHなど）だけ取り出す．
    '''
    cent_mol  = []
    # ボンドセンターの座標と双極子をappendする．
    for mol_bc in list_bond_centers: #UnitCellの分子ごとに分割 
        # chボンド部分（chボンドの重心と双極子をappend）
        cent_mol.append(mol_bc[bond_index])
    # reshape
    cent_mol = np.array(cent_mol).reshape((-1,3))
    return cent_mol

def find_specific_bondmu(list_mu_bonds, bond_index):
    '''
    list_bond_centersからbond_index情報をもとに特定のボンド（CHなど）だけ取り出す．
    '''
    mu_mol  = []
    # ボンドセンターの座標と双極子をappendする．
    for mol_mu_bond in list_mu_bonds: #UnitCellの分子ごとに分割 
        # chボンド部分（chボンドの重心と双極子をappend）
        mu_mol.append(mol_mu_bond[bond_index])
    return np.array(mu_mol)

def find_specific_ringcenter(list_bond_centers, ring_index):
    ring_cent_mol = []
    # ボンドセンターの座標と双極子をappendする．
    for mol_bc in list_bond_centers: #UnitCellの分子ごとに分割 
        # ring部分（リングの重心とリングの双極子を計算）
        ring_center = np.mean(mol_bc[ring_index],axis=0)
        ring_cent_mol.append([ring_center])
    # reshape
    ring_cent_mol = np.array(ring_cent_mol).reshape((-1,3))
    return ring_cent_mol

def find_specific_ringmu(list_mu_bonds,list_mu_pai,ring_index):
    ring_mu_mol = []
    # ボンドセンターの座標と双極子をappendする．
    for mol_mu_bond,mol_mu_pai in zip(list_mu_bonds,list_mu_pai) : #UnitCellの分子ごとに分割 
        ## ring_center = mol_bc[ring_bond_index][0] # 2023/3/31: 試しにここを変更してみる！（あまり意味なかった．．．）
        ring_mu     = np.sum(mol_mu_bond[ring_index],axis=0) + np.sum(mol_mu_pai,axis=0)
        ring_mu_mol.append([ring_mu])
    return ring_mu_mol

def find_specific_lonepair(list_mol_coords, aseatoms, atomic_index:int, NUM_MOL:int):
    '''
    与えられたaseatomとlist_mol_coordsの中から，atomic_indexに対応する原子の座標を抽出する
    '''
    
    # ローンペアのために，原子番号がatomic_indexの原子があるところのリストを取得
    at_list = raw_find_atomic_index(aseatoms, atomic_index, NUM_MOL)

    
    cent_mol=[]
    # 原子にまつわる（ローンペア系）座標と双極子をappendする．
    for atOs,mol_coords in zip(at_list,list_mol_coords):
        # oローンペア部分
        cent_mol.append(mol_coords[atOs]) #ここはatomic_indexに対応した原子（酸素なら8）の座標をappendする
    # reshape
    cent_mol = np.array(cent_mol).reshape((-1,3)) #最後フラットな形に変更
    return cent_mol


def find_specific_lonepairmu(list_mu_lp, list_atomic_nums, atomic_index:int):
    
    # ローンペアのために，原子があるところのリストを取得
    at_list = []
    for js in list_atomic_nums:
        at = np.argwhere(js==atomic_index).reshape(-1).tolist() #リストにしておく
        at_list.append(at)

    mu_mol = []
    # print(atO_list)
    # 原子にまつわる（ローンペア系）座標と双極子をappendする．
    for mol_mu_lone in list_mu_lp:
        # oローンペア部分
        mu_mol.append(mol_mu_lone)
    return mu_mol


def raw_calc_bond_descripter_at_frame(atoms_fr, list_bond_centers, bond_index, NUM_MOL:int, UNITCELL_VECTORS, NUM_MOL_ATOMS:int, desctype="allinone"):
    '''
    1つのframe中の一種のボンドの記述子を計算する
    '''
    Descs = []
    cent_mol   = find_specific_bondcenter(list_bond_centers, bond_index) #特定ボンドの座標だけ取得
    if len(bond_index) != 0: # 中身が0でなければ計算を実行
        i=0 
        for bond_center in cent_mol:
            mol_id = i % NUM_MOL // len(bond_index) # 対応する分子ID（mol_id）を出すように書き直す．ボンドが1分子内に複数ある場合，その数で割らないといけない．（メタノールならCH結合が3つあるので3でわる）
            # 2023/6/27 ここをallinoneへ変更
            if desctype == "allinone":
                Descs.append(raw_get_desc_bondcent_allinone(atoms_fr,bond_center,mol_id,UNITCELL_VECTORS,NUM_MOL_ATOMS))
            elif desctype == "old":
                Descs.append(raw_get_desc_bondcent(atoms_fr,bond_center,mol_id,UNITCELL_VECTORS,NUM_MOL_ATOMS))
            i += 1
    return np.array(Descs)





def raw_calc_bondmu_descripter_at_frame(list_mu_bonds, bond_index):
    '''
    各種ボンドの双極子の真値を計算するコード
    （元のコードでいうところのdata_y_chとか）
    まず，list_mu_bondsからbond_indexに対応するデータだけをmu_molに取り出す．
    '''
    data_y = []
    mu_mol = find_specific_bondmu(list_mu_bonds, bond_index)
    mu_mol = mu_mol.reshape((-1,3)) # !! descriptorと形を合わせる
    if len(bond_index) != 0: # 中身が0でなければ計算を実行
        for mu_b in mu_mol:
            data_y.append(mu_b)
    return np.array(data_y)


def raw_find_atomic_index(aseatoms, atomic_index:int, NUM_MOL:int):
    '''
    ase.atomsの中で特定の原子番号の部分のリストを作成する
    '''
    list_atomic_nums = list(np.array(aseatoms.get_atomic_numbers()).reshape(NUM_MOL,-1)) # atomic_numbersを分子ごとにreshape
    at_list = [ np.argwhere(js==atomic_index).reshape(-1).tolist() for js in list_atomic_nums] # atomic_indexに対応するindexを返す
    return at_list


def raw_calc_lonepair_descripter_at_frame(atoms_fr, list_mol_coords, at_list, NUM_MOL:int, atomic_index:int, UNITCELL_VECTORS, NUM_MOL_ATOMS:int, desctype = "allinone"):
    '''
    1つのframe中の一種のローンペアの記述子を計算する

    atomic__index : 原子量（原子のリストを取得するのと，原子座標の取得に使う）
    at_list      : 1分子内での原子のある場所のリスト
    TODO :: at_listは単に1分子内のO原子の数を数えるのに使っているだけなので，もっとよい方法を考える．
    TODO :: そもそもここではatomic_indexを入力としているが，よく考えると現在はitp_data.o_listがあるのだから，それを使ってcent_molの抽出ができるのでは？

    分子ID :: 分子1~分子NUM_MOLまで
    '''

    # ローンペアのために，原子があるところのリストを取得
    # !! こうやってatomic_indexからat_listを取得できるようになった．
    # !! したがって，入力のat_listはもういらん．
    at_list2 = raw_find_atomic_index(atoms_fr, atomic_index, NUM_MOL)
    # print(" at_list & at_list2  :: {}, {}".format(at_list,at_list2))  # !! debug

    list_lonepair_coords = find_specific_lonepair(list_mol_coords, atoms_fr, atomic_index, NUM_MOL) #atomic_indexに対応した原子の座標を抜き取る
    # >>> 古いコード．新しくat_listを入力に与えるようにしたので不要に >>>>>
    # get_atomic_numbersから与えられた原子種の数を取得
    # at_list = raw_find_atomic_index(atoms_fr,atomic_index, NUM_MOL)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # print("DEBUG :: cent_mol :: ", cent_mol)
    
    if len(at_list) != 0: # 中身が0でなければ計算を実行
        if desctype == "old":
            Descs = []
            i=0
            for bond_center in list_lonepair_coords:
                mol_id = i % NUM_MOL // len(at_list) # 対応する分子ID（mol_id）を出すように書き直す．（1分子内のO原子の数（len(at_list）でわって分子idを出す）
                Descs.append(raw_get_desc_lonepair(atoms_fr,bond_center,mol_id,UNITCELL_VECTORS,NUM_MOL_ATOMS))
                i += 1 
        elif desctype == "allinone":
            Descs = [raw_get_desc_lonepair_allinone(atoms_fr,bond_center,UNITCELL_VECTORS,NUM_MOL_ATOMS) for bond_center in list_lonepair_coords]
    return np.array(Descs)

def raw_calc_lonepair_descripter_at_frame2(atoms_fr, list_mol_coords, at_list, NUM_MOL:int, UNITCELL_VECTORS, NUM_MOL_ATOMS:int, desctype = "allinone"):
    '''
    TODO :: desctypeとしてはallaloneのみ対応．
    1つのframe中の一種のローンペアの記述子を計算する．version2
    入力を見直す．
    せっかくat_listがあるので，これを使ってlist_lonepair_coordsを作成する．
    
    atomic__index : 原子量（原子のリストを取得するのと，原子座標の取得に使う）
    at_list      : atoms_frの中で求めたい原子の場所のリスト（0~NUM_ATOMSの中から）
    
    最も使いやすいかたちとしては，at_list
    
    TODO :: at_listは単に1分子内のO原子の数を数えるのに使っているだけなので，もっとよい方法を考える．
    TODO :: そもそもここではatomic_indexを入力としているが，よく考えると現在はitp_data.o_listがあるのだから，それを使ってcent_molの抽出ができるのでは？

    分子ID :: 分子1~分子NUM_MOLまで
    '''

    if desctype == "old":
        print("ERROR :: desctype = old is not supported.")
        sys.exit(1)
    
    # 記述子を求めたい原子座標の取得
    list_lonepair_coords = [coord for coord in list_mol_coords.reshape(-1,3)[at_list]]
    # 実際の記述子の計算
    Descs = [raw_get_desc_lonepair_allinone(atoms_fr,bond_center,UNITCELL_VECTORS,NUM_MOL_ATOMS) for bond_center in list_lonepair_coords]

    return np.array(Descs)


def raw_calc_lonepairmu_descripter_at_frame(list_mu_lp, list_atomic_nums, at_list, atomic_index:int):
    '''
    各種ローンペアの双極子の真値を計算するコード
    （元のコードでいうところのdata_y_chとか）
    まず，list_mu_bondsからbond_indexに対応するデータだけをmu_molに取り出す．
    '''
    data_y = []
    mu_mol = find_specific_lonepairmu(list_mu_lp, list_atomic_nums, atomic_index)
    if len(at_list) != 0: # 中身が0でなければ計算を実行
        for mu_b in mu_mol:
            data_y.append(mu_b)
    return np.array(data_y)

