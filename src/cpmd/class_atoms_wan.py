"""_summary_

asign_wcs.pyを利用して，とりあえずframeのボンドやwcの情報を保持するクラスを作成する．
このクラスは後々グラフベースのものに書き換える予定なので，一時的なクラス．

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
"""


from ase.io import read
import ase
import sys
import logging
import numpy as np
from ml.atomtype import Node # 深さ優先探索用
from ml.atomtype import raw_make_graph_from_itp # 深さ優先探索用
from collections import deque # 深さ優先探索用
# from types import NoneType
# 全てのワニエ間の距離を確認し，あまりに小さい場合に警告を出す（CPMDのワニエ計算が失敗している可能性あり）
from scipy.spatial import distance
import cpmd.asign_wcs

class atoms_wan():
    '''
    どういう構成にするかはちょっと難しいところだ．
    1frameに対する定義にしたいので，入力としてxyzを受け取るのが良いのではないかと思うのだが．．．
    ということで，とりあえずはxyz一つを入力として受け取り，いくつかの変数を取得するようにする
    '''
    def __init__(self, input_atoms:ase.atoms, NUM_MOL_ATOMS, itp_data):
        
        import numpy as np
        import torch
        # instance variables
        self.input_atoms   = input_atoms
        self.NUM_MOL_ATOMS = NUM_MOL_ATOMS
        self.itp_data      = itp_data
        # load unitcell vector
        self._set_cell()
        # wannierとatoms_nowanに分割
        self._set_aseatoms_wannier_oneframe()
        self.logger.debug(f"DEBUG :: self.atoms_nowan is {self.atoms_nowan}")
        
        # 必要な定数の計算（atoms_nowanから計算しないとwannierが入っちゃう）
        self.NUM_ATOM:int    = len(self.atoms_nowan.get_atomic_numbers()) #原子数
        # self.NUM_CONFIG:int  = len(self.atoms_nowan) #フレーム数        
        self.NUM_MOL:int     = int(self.NUM_ATOM/NUM_MOL_ATOMS) #UnitCell中の総分子数
        self.logger.debug(f"DEBUG :: NUM_ATOM = {self.NUM_ATOM} : NUM_MOL = {self.NUM_MOL} ")
        
        # calculate atomic coordinates, bond centers, and wannier centers
        import cpmd.descripter
        import cpmd.asign_wcs
        # * wannierの割り当て部分のメソッド化
        self.ASIGN=cpmd.asign_wcs.asign_wcs(self.NUM_MOL, self.NUM_MOL_ATOMS,self.UNITCELL_VECTORS)
        self.DESC=cpmd.descripter.descripter(self.NUM_MOL,self.NUM_MOL_ATOMS,self.UNITCELL_VECTORS)


    def _set_cell(self):
        if type(self.input_atoms.get_cell()) != ase.cell.Cell:
            raise ValueError("input_atoms.get_cell() do not have cell data !!")
        self.UNITCELL_VECTORS = self.input_atoms.get_cell() 
        
    def _set_aseatoms_wannier_oneframe(self):    
        # ワニエの座標を廃棄する．
        # for debug
        # 配列の原子種&座標を取得
        atom_list=self.input_atoms.get_chemical_symbols()
        coord_list=self.input_atoms.get_positions()

        atom_list_tmp=[]
        coord_list_tmp=[]
        wan_list_tmp=[]
        for i,j in enumerate(atom_list):
            if j != "X": # 原子がXだったらappendしない
                atom_list_tmp.append(atom_list[i])
                coord_list_tmp.append(coord_list[i])
            else:
                wan_list_tmp.append(coord_list[i])
        # class として定義
        self.atoms_nowan = ase.Atoms(atom_list_tmp,
                    positions=coord_list_tmp,
                    cell= self.UNITCELL_VECTORS,
                    pbc=[1, 1, 1])
        self.wannier = wan_list_tmp
        
    def aseatom_to_mol_coord_bc(self): #ase_atoms, bonds_list, itp_data, NUM_MOL_ATOMS:int, NUM_MOL:int) :
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
        mol_at0 = [ i for i in range(self.NUM_MOL_ATOMS) ] # 0からNUM_MOL_ATOMSのリスト

        for j in range(self.NUM_MOL): # 全ての分子に対する繰り返し．
            mol_inds_j  = [ int(at+self.NUM_MOL_ATOMS*j) for at in mol_at0 ] # j番目の分子を構成する分子のindex
            bonds_list_j = [[int(b_pair[0]+self.NUM_MOL_ATOMS*j),int(b_pair[1]+self.NUM_MOL_ATOMS*j)] for b_pair in self.itp_data.bonds_list ] # j番目の分子に入っている全てのボンドのindex  
            mol_coords,bond_centers = cpmd.asign_wcs.raw_calc_mol_coord_and_bc_mic_onemolecule(mol_inds_j,bonds_list_j,self.atoms_nowan,self.itp_data) # 1つの分子のmic座標/bond center計算
            list_mol_coords.append(mol_coords)
            list_bond_centers.append(bond_centers)
        self.list_mol_coords = list_mol_coords
        self.list_bond_centers = list_bond_centers
        # return  [list_mol_coords,list_bond_centers]
        
    def _calc_wcs(self)->int:
        # * wanとatomsへの変換
        import cpmd.read_traj_cpmd
        self.logger.debug(f" DEBUG :: self.input_atoms[0] = {self.input_atoms.get_positions()[0]}")
        # * 原子座標とボンドセンターの計算
        # 原子座標,ボンドセンターを分子基準で再計算
        # TODO :: list_mol_coordsを使うのではなく，原子座標からatomsを作り直した方が良い．
        # TODO :: そうしておけば後ろでatomsを使う時にmicのことを気にしなくて良い（？）ので楽かも．
        self.logger.debug(f" DEBUG :: self.atoms_nowan is {self.atoms_nowan}")
        # calc BC and atomic coordinate
        self.aseatom_to_mol_coord_bc()
        # self.list_mol_coords, self.list_bond_centers = self.aseatom_to_mol_coord_bc()
        # self.list_mol_coords, self.list_bond_centers = self.aseatom_to_mol_coord_bc(self.atoms_nowan, self.itp_data, self.itp_data.bonds_list)
        
        # そもそものwcsがちゃんとしているかの確認
        test_wan_distances = distance.cdist(np.array(self.wannier),np.array(self.wannier), metric='euclidean')
        # print(test_wan_distances) 
        if test_wan_distances[test_wan_distances>0].any() < 0.2:
            raise ValueError("ERROR :: wcs are too small !! :: check CPMD calculation")
        
        # wcsをbondに割り当て，bondの双極子まで計算
        # !! 注意 :: calc_mu_bond_lonepairの中で，再度raw_aseatom_to_mol_coord_bcを呼び出して原子/BCのMIC座標を計算している．
        double_bonds = []
        self.list_mu_bonds,self.list_mu_pai,self.list_mu_lpO,self.list_mu_lpN, self.list_bond_wfcs,self.list_pi_wfcs,self.list_lpO_wfcs,self.list_lpN_wfcs = \
            self.ASIGN.calc_mu_bond_lonepair(self.wannier, self.atoms_nowan,self.itp_data.bonds_list,self.itp_data,double_bonds)
        return 0

    @property
    def logger(self):
        # return logging.getLogger(self.logfile)
        return logging.getLogger("atoms_wan")