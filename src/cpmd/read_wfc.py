# -*- coding: utf-8 -*-
"""
convert *.wfc to Atom xyz and Atom Trajectory.
!! To read the cell, We need CP.in or xyz file.
"""

# 座標：wfcファイル
# 格子定数：他のファイルが必要
# WCsの数．：wfcファイルから取得可能．
    
from ase.io import read
import ase
import sys
import numpy as np
from cpmd.read_traj import raw_read_unitcell_vector
import cpmd.read_traj 
import cpmd.read_core


def get_nbands(filename:str):
    '''
    cp.xの作る*.wfcファイルの最初のconfigurationを読み込んでWCsがいくつあるかをcount_lineで数える．
    configurationの判定はcheck_lineによって行われ，行の要素が4つあればcheck_lineと判定する．
    '''
    count_line=0
    check_line=0
    f = open(filename) 

    while True:
        count_line+=1
        if len(f.readline().split()) ==4:
            check_line+=1
        if check_line == 2: # 2回目の時にbreakする．
            break

    nbands = count_line-2 #2つ分引かないといけない
    if not __debug__:
        print(" -------------- ")
        print(" finish reading nbands :: nbands = ", nbands)
        print("")
    return nbands



def raw_read_wfc(filename:str, unitcell_vector=None):
    '''
    *.wfcファイルを読みこんでase.atomsのリストを返す.

    input
    ----------------
      - filename        :: str
            *.wfc filename
      - UNITCELL_VECTOR :: 3*3 numpy array
            unitcell vectors in row wise
      - ifsave          :: bool
            if True, save trajectory to *.xyz and *.traj.

    Returns
    -------
      - wfc_list     :: list of atoms.ase

    Notes
    -----
    格子定数は与えなくても良い．その場合格子定数を保持しないatoms.aseとして出力される．
    '''
    
    # nbands(wfcの数)を取得
    nbands=get_nbands(filename)
    
    # wfcのリスト 
    wfc_list = []

    with open(filename) as f:
        lines = f.read().splitlines()
    
    lines = [l.split() for l in lines]
    for i,l in enumerate(lines) :
        if (i%(nbands+1) == 0) and (i==0) :
            block = []
        elif i%(nbands+1) == 0 :
            block = [] 
            wfc_list.append(block)
        else :
            block.append([float(p) for p in l ]) 
    wfc_list = np.array(wfc_list) * ase.units.Bohr #wfcはbohr. Angへ変換

    
    # Atomsオブジェクトのリストを作成する
    wfc_array=[]

    # He原子を割り当てる
    new_atomic_num=["He" for i in range(nbands)]

    # 座標のリスト
    new_coord=wfc_list

    
    # 格子定数を与えるか与えないか．
    if unitcell_vector.any() == None:
        for i in range(len(wfc_list)):
            mol_with_WC = ase.Atoms(new_atomic_num,
                                    positions=new_coord[i],        
                                    pbc=[1, 1, 1])  
            wfc_array.append(mol_with_WC)
    else:
        for i in range(len(wfc_list)):
            mol_with_WC = ase.Atoms(new_atomic_num,
                                    positions=new_coord[i],        
                                    cell= unitcell_vector,   
                                    pbc=[1, 1, 1])  
            wfc_array.append(mol_with_WC)
            
            
    # traj形式で保存
    #if ifsave == True:
    #    ase.io.write(filename+"_refine.xyz",wfc_array, format="extxyz")
    #    ase.io.write(filename+"_refine.traj",wfc_array, format="traj")
    # ase.atomsのリストを返す
    return wfc_array


def raw_merge_wfc_xyz(wfc_list:list, xyz_list:list):
    '''
    同じtrajectoryから得られたwfcのデータとxyzのデータを合体させて新しいase.atomsリストを作成する．
    入力はどちらもextended xyzであることを想定している．

    input
    ----------------
      - wfc_list :: ase.atoms list
           input wfc list
      - xyz_list :: ase.atoms list
           input xyz list
    '''
    # まず与えられた二つのファイルの長さが同じか判定
    if not len(wfc_list) == len(xyz_list):
        print("ERROR :: steps of 2 files differ")
        print(" steps of wfc_list :: ", len(wfc_list))
        print(" steps of xyz_list :: ", len(xyz_list))
        sys.exit()
    
    # 原子リスト(trajectoryの最初のconfigurationから原子種を取得)
    MERGE_SYMBOL_LIST=wfc_list[0].get_chemical_symbols()+xyz_list[0].get_chemical_symbols()

    # 結晶構造はxyzから取得
    if not wfc_list[0].get_cell().any()==None:
        if not np.allclose(wfc_list[0].get_cell(),xyz_list[0].get_cell()):
            print("WARNING :: unitcell vectors of 2 files differ")
            print(" unitcell of wfc_list :: ",wfc_list[0].get_cell())
            print(" unitcell of xyz_list :: ",xyz_list[0].get_cell())
            sys.exit()
    UNITCELL_VECTOR=xyz_list[0].get_cell()
    
    merged_atoms_list=[]
    
    for i in range(len(wfc_list)):
        # step iでのatomsを作成. MERGE_SYMBOL_LISTと同じくwfc→xyzの順番でappendする．
        tmp_positions=[]
        for j in range(len(wfc_list[i].get_positions())):
            tmp_positions.append(wfc_list[i].get_positions()[j])

        for j in range(len(xyz_list[i].get_positions())):
            tmp_positions.append(xyz_list[i].get_positions()[j])
        # atomsを作成
        tmp_atoms = ase.Atoms(MERGE_SYMBOL_LIST,
                              positions=tmp_positions,        
                              cell= UNITCELL_VECTOR,
                              pbc=[1, 1, 1])  
        merged_atoms_list.append(tmp_atoms)

    # traj形式で保存
    #if ifsave == True:
    #    if prefix == None:
    #        print(" WARNING :: prefix is not given. \n")
    #        print(" prefix is used for output filenames. \n")
    #    ase.io.write(prefix+"_wfc_merged.traj",merged_atoms_list, format="traj")
    #    ase.io.write(prefix+"_wfc_merged.xyz",merged_atoms_list, format="extxyz")
    #    merged_atoms_traj=ase.io.trajectory.Trajectory(prefix+"_wfc_merged.traj")
    #
    return merged_atoms_list
            

class ReadWFC(cpmd.read_core.custom_traj):
    '''
    *.wfcファイルを読み込んで操作するクラス．オプションとしてcppp.xの作るxyzファイルをxyzfilenameを与えればそこから格子定数を読み込んで付加することができる．wfcファイルのみを可視化したい場合にはこれがあると便利．custom_trajクラスを継承して作成しているので細かい関数などについてはそちらも参照すること．
    
    input
    -----------------
      - filename    :: str
            original wfc file from cp.x
    
      - xyzfilename :: str
            xyz files from cppp.x. if given, read unitcellvectors from this file.
    
    Notes
    -----------------
    xyzfilenameにはcppp.xで作成した生のxyzファイルを使うこと！！
    '''
    def __init__(self, filename:str, xyzfilename:str=""):
        #self.filename=filename
        if xyzfilename=="":
            # クラスの継承をする場合
            super().__init__(atoms_list=raw_read_wfc(filename, unitcell_vector=None), unitcell_vector=None, filename=filename)
            #self.UNITCELL_VECTOR=None
            #self.ATOMS_LIST=raw_read_wfc(filename, unitcell_vector=None)
            # self.TRAJ=ase.io.trajectory.Trajectory(filename+"_refine.traj")
        else:
            super().__init__(atoms_list=raw_read_wfc(filename, unitcell_vector=raw_read_unitcell_vector(xyzfilename)), unitcell_vector=raw_read_unitcell_vector(xyzfilename), filename=filename)
            #self.UNITCELL_VECTOR=raw_read_unitcell_vector(xyzfilename)
            #self.ATOMS_LIST=raw_read_wfc(filename, unitcell_vector=raw_read_unitcell_vector(xyzfilename))
            # self.TRAJ=ase.io.trajectory.Trajectory(filename+"_refine.traj")

    #def nglview_traj(self):
    #    return cpmd.read_traj.raw_nglview_traj(self.ATOMS_LIST)
            
    def merge_wfc_xyz(self, xyz_list):
        '''
        xyz_list :: atoms trajectory (list of ase.atoms)
        '''
        merged_atoms=raw_merge_wfc_xyz(self.ATOMS_LIST, xyz_list)
        return cpmd.read_core.custom_traj(atoms_list=merged_atoms)

    # メソッドのoverride
    def save(self):
        cpmd.read_traj.raw_save_aseatoms(self.ATOMS_LIST,  xyz_filename=self.filename+"_refine.xyz")
        return 0
    
