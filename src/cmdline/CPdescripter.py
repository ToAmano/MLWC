#!/usr/bin/env python3
# coding: utf-8

'''
# 2023/04/07
## 2023/4/11 :: 現状記述子のみ作成
## frameに関する並列化を目指す
## 特にここでは1分子の計算を行なっている（とはいえ特にコードに変更は必要ない）
## また，カットオフを6/8 Angstromに変更してある．

### 2023/3/28
#### inputの段階で，格子定数はすでに
####    CPextract.py cpmd addlattice 
#### で追加していることとする．

### 1: IONS+CENTERS.xyz，および格子定数を読み込む． → コードで読み込むためのxyzを作る．
### 2: WCsの割り当てを行う．（itpファイルのbond情報が必須．）→ この段階で一回保存したいんだけどねぇ．
### 3: ML用の記述子を作成する．
'''

import argparse
import sys
import numpy as np
import argparse
# import matplotlib.pyplot as plt
import ase.io
import ase
import numpy as np
# import nglview as nv
from ase.io.trajectory import Trajectory

# import home-made package
# import importlib
# import cpmd

# 物理定数
from include.constants import constant 
# Debye   = 3.33564e-30
# charge  = 1.602176634e-019
# ang      = 1.0e-10 
coef    = constant.Ang*constant.Charge/constant.Debye


def find_input(inputs, str):
    '''
    入力ファイルから特定のキーワードをサーチする．
    
    input
    -------------
      inputs :: [keyword, value]がappendされた2次元配列．keywordを検索し，valueを返す
      
    output
    -------------
      output  :: keywordに対応するvalue.
      
    note 
    -------------
     TODO :: キーワードが複数出てきた時は？
     TODO :: optional keywordがこのままだと扱えない．
    '''
    output = None
    for i in inputs:
        if i[0] == str:
            output=i[1]
            print(" {0} :: {1}".format(str,output))
    if output == None:
        print(" ERROR :: input not found :: {}".format(str))
        return 1
    return output


def main():
    # 
    # * 1-2：入力情報をここにまとめる
    # 

    # コードのバージョンとして，
    #  - CPMDのIONS+CENTERS.xyzを使う場合（ワニエまで読み込む）
    #  - CPMDのIONS_ONLY.xyzを使う場合（ワニエは読まない）
    # があり得る．haswannierフラグで両者を実行可能になっている．

    # コードの構成
    # 1: itpファイルを読み込む（共通）
    # 2: descriptorの場合，xyzの読み込み
    # 3: predictの場合，modeldirの読み込み

    # directory="2022_12_11_5ps_restart_3/"
    ## directory="1ps_12_test/"
    ## stdoutfile="bomd-wan-restart.out"
    ## filename="IONS_ONLY.xyz"
    ## itpfilename="gromacs_input/input1.itp"
    # TODO hard-code :: 二重結合用(実質リング用で，共役でない通常の二重結合は問題ない．)
    double_bonds_pairs = []

    # 
    # * read input file
    from pathlib import Path
    if Path(sys.argv[1]).exists():  # 第一引数がファイルだったら
        inpfilename=sys.argv[1]
        # TODO :: hard code
        fp=open(inpfilename,mode="r")
        inputs = []

        for line in fp.readlines():
            print(line.strip().split('='))
            inputs.append(line.strip().split('=')) # space/改行などを削除
        print("inputs :: {}".format(input))
    else:
        print("ERROR :: inputfile not found")
        return 1

    # read input parameters
    directory=find_input(inputs,"directory")
    # stdoutfile=find_input(inputs,"stdoutfile")
    filename=find_input(inputs,"filename")
    itpfilename=find_input(inputs,"itpfilename")
    savedir=find_input(inputs,"savedir")
    haswannier=int(find_input(inputs,"haswannier")) # int型へ変換


    # * 1-3：トポロジーファイル：itpの読み込み
    # * ボンドの情報を読み込む．
    # *
    import ml.atomtype
    itp_data=ml.atomtype.read_itp(itpfilename)
    bonds_list=itp_data.bonds_list
    NUM_MOL_ATOMS=itp_data.num_atoms_per_mol
    atomic_type=itp_data.atomic_type


    '''
    # * ボンドの情報設定
    # * 基本的にはitpの情報通りにCH，COなどのボンド情報を割り当てる．
    # * ボンドindexの何番がどのボンドになっているかを調べる．
    # * ベンゼン環だけは通常のC-C，C=Cと区別がつかないのでそこは手動にしないとダメかも．

    このボンド情報でボンドセンターの学習を行う．
    '''

    # ring_bonds = double_bonds_pairs
    ring_bonds = []

    ch_bonds = itp_data.ch_bond
    co_bonds = itp_data.co_bond
    oh_bonds = itp_data.oh_bond
    cc_bonds = itp_data.cc_bond

    ring_bond_index = itp_data.ring_bond_index
    ch_bond_index   = itp_data.ch_bond_index
    co_bond_index   = itp_data.co_bond_index
    oh_bond_index   = itp_data.oh_bond_index
    cc_bond_index   = itp_data.cc_bond_index

    o_index = itp_data.o_list
    n_index = itp_data.n_list
    
    print(" ================== ")
    print(" ring_bond_index ", ring_bond_index)
    print(" ch_bond_index   ", ch_bond_index)
    print(" oh_bond_index   ", oh_bond_index)
    print(" co_bond_index   ", co_bond_index)
    print(" cc_bond_index   ", cc_bond_index)
    print(" o_index         ", o_index)
    print(" n_index         ", n_index)
    print(" ================== ")
    
    # *
    # * （escripterを計算する設定の場合）系のパラメータの設定
    # *

    import numpy as np
    import cpmd.read_traj_cpmd

    '''
    誘電関数計算などで利用するパラメータ
    TEMPERATURE: 温度[K]
    VOLUME     : 体積[Ang^3]
    TIMESTEP   : IONS+CENTERS.xyzの時間ステップ[a.u.]
    '''
    
    # TODO :: もしfilemodeがwannieronlyではない場合，wannier部分を除去したい！！
    if haswannier == True:
        print("haswannier=True")
        import cpmd.read_traj_cpmd
        traj, wannier_list=cpmd.read_traj_cpmd.raw_xyz_divide_aseatoms_list(directory+filename)
    else:
        print("haswannier=False")
        traj=ase.io.read(directory+filename,index=":")

    # aseでデータをロード
    # traj=ase.io.read(directory+filename,index=":")

    UNITCELL_VECTORS = traj[0].get_cell() # TODO :: セル情報がない場合にerrorを返す
    # >>> not used for descripter >>>
    # TEMPERATURE      = 300
    # TIMESTEP         = 40*10
    # VOLUME           = np.abs(np.dot(np.cross(UNITCELL_VECTORS[:,0],UNITCELL_VECTORS[:,1]),UNITCELL_VECTORS[:,2]))
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    # 種々のデータをloadする．
    NUM_ATOM:int    = len(traj[0].get_atomic_numbers()) #原子数
    NUM_CONFIG:int  = len(traj) #フレーム数
    # UNITCELL_VECTORS = traj[0].get_cell() #cpmd.read_traj_cpmd.raw_cpmd_read_unitcell_vector("cpmd.read_traj_cpmd/bomd-wan.out.2.0") # tes.get_cell()[:]
    #num_of_bonds = {14:4,6:3,8:2,1:1} #原子の化学結合の手の数

    NUM_MOL = int(NUM_ATOM/NUM_MOL_ATOMS) #UnitCell中の総分子数

    print(" --------  ")
    print(" NUM_ATOM  ::    ", NUM_ATOM )
    print(" NUM_CONFIG ::   ", NUM_CONFIG)
    print(" NUM_MOL    :: ",    NUM_MOL)
    print(" NUM_MOL_ATOMS :: ", NUM_MOL_ATOMS)
    print(" UNITCELL_VECTORS :: ", UNITCELL_VECTORS)
    print(" --------  ")

    elements = {"N":7,"C":6,"O":8,"H":1}
    # atom_id = traj[0].get_chemical_symbols()
    # atom_id = [elements[i] for i in atom_id ]

    #
    # 
    # * 結合リストの作成
    # * 上の分子構造を見てリストを作成する--> 二重結合のリストのみ作る
    # * 二重結合の電子は1つのC=C結合に２つ上下に並ばないケースもある。ベンゼン環上に非局在化しているのが要因か。
    # * 結合１つにワニエ中心１つづつ探し、二重結合は残った電子について探索する


    # TODO :: hard code :: 二重結合だけは，ここでdouble_bondsというのを作成している
    double_bonds = []
    for pair in double_bonds_pairs :
        if pair in bonds_list :
            double_bonds.append(bonds_list.index(pair))
        elif pair[::-1] in bonds_list :
            double_bonds.append(bonds_list.index(pair[::-1]))
        else :
            print("error")

    # * >>>>  1分子の情報をもとに，ボンド情報を系全体に拡張する >>>>>>>>>
    # * 2023/4/16 :: unit_cell_bondsはasign_wcs.pyの中で計算するようにした
    # # TODO :: hard code :: 分子のボンドリスト(bonds)をNUM_MOL回繰り返す
    # # * ボンドをセル内の全ての分子について加える
    # unit_cell_bonds = []
    # for indx in range(NUM_MOL) :
    #     unit_cell_bonds.append([[int(b_pair[0]+NUM_MOL_ATOMS*indx),int(b_pair[1]+NUM_MOL_ATOMS*indx)] for b_pair in bonds_list ]) 


    # ! <<<<<<<<  ここ使ってなくない？
    # # * 分子を構成する原子のインデックスのリストを作成する。（mol_at0をNUM_MOL回繰り返す）
    # mol_at0 = [ i for i in range(NUM_MOL_ATOMS) ]
    # mol_ats = [ [ int(at+NUM_MOL_ATOMS*indx) for at in mol_at0 ] for indx in range(NUM_MOL)]
    # # for indx in range(NUM_MOL) :
    # #    mol_ats.append([ int(at+NUM_MOL_ATOMS*indx) for at in mol_at0 ])
    # ! <<<<<<<<  ここ使ってなくない？


    # * >>>>  1分子の情報をもとに，ボンド情報を系全体に拡張する >>>>>>>>>
    print(" double_bonds :: ", double_bonds)
    # print("unit_cell_bonds::分子ごとの原子の番号のリスト")
    # print("unit_cell_bonds :: ", unit_cell_bonds)
    # print(" -------- ")

    ### 機械学習用のデータ（記述子）を作成する

    # * メソッド化
    import  cpmd.asign_wcs 
    # importlib.reload(cpmd.asign_wcs)
    ASIGN=cpmd.asign_wcs.asign_wcs(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)

    import cpmd.descripter
    # importlib.reload(cpmd.descripter)
    DESC=cpmd.descripter.descripter(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)

    # 全フレームを計算
    frames = len(traj) # フレーム数
    print("frames:: ", frames)

    import joblib

    def calc_descripter_frame(atoms_fr, fr, savedir):
        # * 原子座標とボンドセンターの計算
        # 原子座標,ボンドセンターを分子基準で再計算
        results = ASIGN.aseatom_to_mol_coord_bc(atoms_fr, bonds_list)
        list_mol_coords, list_bond_centers =results
    
        # * ボンドデータをさらにch/coなど種別ごとに分割 & 記述子を計算
        # mu_bondsの中身はchとringで分割する
        #mu_paiは全数をringにアサイン
        #mu_lpOとlpNはゼロ
        # ring
        if len(ring_bond_index) != 0:
            Descs_ring = []
            ring_cent_mol = cpmd.descripter.find_specific_ringcenter(list_bond_centers, ring_bond_index, 8, NUM_MOL)
            i=0 
            for bond_center in ring_cent_mol:
                mol_id = i % NUM_MOL // 1
                Descs_ring.append(DESC.get_desc_bondcent(atoms_fr,bond_center,mol_id))
                i+=1 

        # ch
        Descs_ch=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,ch_bond_index)
        # oh
        Descs_oh=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,oh_bond_index)
        # co
        Descs_co=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,co_bond_index)
        # cc
        Descs_cc=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,cc_bond_index)   
        # oローンペア
        Descs_o = DESC.calc_lonepair_descripter_at_frame(atoms_fr,list_bond_centers, o_index, 8)

        # データが作成できているかの確認（debug）
        # print( " DESCRIPTOR SHAPE ")
        # print(" ring (Descs/data) ::", Descs_ring.shape)
        # print(" ch-bond (Descs/data) ::", Descs_ch.shape)
        # print(" cc-bond (Descs/data) ::", Descs_cc.shape)
        # print(" co-bond (Descs/data) ::", Descs_co.shape)
        # print(" oh-bond (Descs/data) ::", Descs_oh.shape)
        # print(" o-lone (Descs/data) ::", Descs_o.shape)

        # ring
        if len(ring_bond_index) != 0:
            np.savetxt(directory+'Descs_ring_'+str(fr)+'.csv', Descs_ring, delimiter=',')
        # CHボンド
        if len(ch_bond_index) != 0:
            np.savetxt(savedir+'Descs_ch_'+str(fr)+'.csv', Descs_ch, delimiter=',')
        # CCボンド
        if len(cc_bond_index) != 0:
            np.savetxt(savedir+'Descs_cc_'+str(fr)+'.csv', Descs_cc, delimiter=',')
        # # COボンド
        if len(co_bond_index) != 0:
            np.savetxt(savedir+'Descs_co_'+str(fr)+'.csv', Descs_co, delimiter=',')
        # # OHボンド
        if len(oh_bond_index) != 0:
            np.savetxt(savedir+'Descs_oh_'+str(fr)+'.csv', Descs_oh, delimiter=',')
        # Oローンペア
        if len(o_index) != 0:
            np.savetxt(savedir+'Descs_o_'+str(fr)+'.csv', Descs_o, delimiter=',')
            
        
        # >>>> 関数ここまで <<<<<
        
    # * データの保存
    # savedir = directory+"/bulk/0331test/"
    import os
    if not os.path.isdir(savedir):
        os.makedirs(savedir) # mkdir
        
    result = joblib.Parallel(n_jobs=-1, verbose=50)(joblib.delayed(calc_descripter_frame)(atoms_fr,fr,savedir) for fr,atoms_fr in enumerate(traj))
    return 0

if __name__ == '__main__':
    main()
