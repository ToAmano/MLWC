#!/usr/bin/env python3
# coding: utf-8

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
import ml.parse
# import home-made package
# import importlib
# import cpmd

# 物理定数
from include.constants import constant
# Debye   = 3.33564e-30
# charge  = 1.602176634e-019
# ang      = 1.0e-10
coef    = constant.Ang*constant.Charge/constant.Debye


def main():
    import ml.parse
    inputfilename=sys.argv[1]
    inputs_list=ml.parse.read_inputfile(inputfilename)
    input_general, input_descripter, input_predict=ml.parse.locate_tag(inputs_list)
    var_gen=ml.parse.var_general(input_general)
    var_des=ml.parse.var_descripter(input_descripter)
    var_pre=ml.parse.var_predict(input_predict)

    
    #
    # * 1-3：トポロジーファイル：itpの読み込み
    # * ボンドの情報を読み込む．
    # *
    import ml.atomtype
    itp_data=ml.atomtype.read_itp(var_gen.itpfilename)
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

    '''
    # * 計算モードがどうなっているかをチェックする
    パターン1: （単なる予測） 記述子だけ作成
    パターン2: （学習データ作成） ワニエのアサインと双極子の真値計算も実行
    パターン3: (予測&真値との比較) 記述子の作成, ワニエのアサインと双極子モーメント計算
    '''
    import numpy as np
    import cpmd.read_traj_cpmd
    import  cpmd.asign_wcs 

    double_bonds_pairs = []    

    # * descripter計算開始
    if var_des.descmode == "1":
        #
        # * 系のパラメータの設定
        # * 

        # aseでデータをロード
        traj=ase.io.read(var_des.directory+var_des.xyzfilename,index=":")

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

        print(" double_bonds :: ", double_bonds)
        print(" -------- ")
        # * >>>>  double_bondsというか，π電子系のための設定 >>>>>>>>>


        ### 機械学習用のデータ（記述子）を作成する


        # 
        # * メソッド化
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
                np.savetxt(savedir+'Descs_ring_'+str(fr)+'.csv', Descs_ring, delimiter=',')
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
            return 0
            # >>>> 関数ここまで <<<<<
            
        # * データの保存
        # savedir = directory+"/bulk/0331test/"
        import os
        if not os.path.isdir(var_des.savedir):
            os.makedirs(var_des.savedir) # mkdir
            
        result = joblib.Parallel(n_jobs=-1, verbose=50)(joblib.delayed(calc_descripter_frame)(atoms_fr,fr,var_des.savedir) for fr,atoms_fr in enumerate(traj))
        return 0
    
    # * 
    # * パターン2つ目，ワニエのアサインもする場合
    # * descripter計算開始
    if var_des.descmode == "2":
        #
        # * 系のパラメータの設定
        # * 
        
        # desc_mode = 2の場合，trajがwannierを含んでいるので，それを原子とワニエに分割する
        # IONS_only.xyzにwannierを除いたデータを保存（と同時にsupercell情報を載せる．）
        import cpmd.read_traj_cpmd
        traj, wannier_list=cpmd.read_traj_cpmd.raw_xyz_divide_aseatoms_list(var_des.directory+var_des.xyzfilename)

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

        print(" double_bonds :: ", double_bonds)
        print(" -------- ")
        # * >>>>  double_bondsというか，π電子系のための設定 >>>>>>>>>


        ### 機械学習用のデータ（記述子）を作成する


        # 
        # * メソッド化
        # importlib.reload(cpmd.asign_wcs)
        ASIGN=cpmd.asign_wcs.asign_wcs(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)

        import cpmd.descripter
        # importlib.reload(cpmd.descripter)
        DESC=cpmd.descripter.descripter(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)

        # 全フレームを計算
        frames = len(traj) # フレーム数
        print("frames:: ", frames)

        import joblib

        def calc_descripter_frame(atoms_fr, wannier_fr, fr, savedir):
            # * 原子座標とボンドセンターの計算
            # 原子座標,ボンドセンターを分子基準で再計算
            results = ASIGN.aseatom_to_mol_coord_bc(atoms_fr, bonds_list)
            list_mol_coords, list_bond_centers =results
            
            # wcsをbondに割り当て，bondの双極子まで計算
            results_mu = ASIGN.calc_mu_bond_lonepair(wannier_fr,atoms_fr,bonds_list,double_bonds)
            list_mu_bonds,list_mu_pai,list_mu_lpO,list_mu_lpN, list_bond_wfcs,list_pi_wfcs,list_lpO_wfcs,list_lpN_wfcs = results_mu
            # wannnierをアサインしたase.atomsを作成する
            mol_with_WC = cpmd.asign_wcs.make_ase_with_WCs(atoms_fr.get_atomic_numbers(),NUM_MOL, UNITCELL_VECTORS,list_mol_coords,list_bond_centers,list_bond_wfcs,list_pi_wfcs,list_lpO_wfcs,list_lpN_wfcs)
            # 系の全双極子を計算
            # print(" list_mu_bonds {0}, list_mu_pai {1}, list_mu_lpO {2}, list_mu_lpN {3}".format(np.shape(list_mu_bonds),np.shape(list_mu_pai),np.shape(list_mu_lpO),np.shape(list_mu_lpN)))
            ase.io.save(savedir+"molWC_"+str(fr)+".xyz", mol_with_WC)
            Mtot = []
            for i in range(NUM_MOL):
                Mtot.append(np.sum(list_mu_bonds[i],axis=0)+np.sum(list_mu_pai[i],axis=0)+np.sum(list_mu_lpO[i],axis=0)+np.sum(list_mu_lpN[i],axis=0))
            Mtot = np.array(Mtot)
            #unit cellの双極子モーメントの計算
            total_dipole = np.sum(Mtot,axis=0)
            # total_dipole = np.sum(list_mu_bonds,axis=0)+np.sum(list_mu_pai,axis=0)+np.sum(list_mu_lpO,axis=0)+np.sum(list_mu_lpN,axis=0)
            # ワニエセンターのアサイン
            #ワニエ中心を各分子に帰属する
            # results_mu=ASIGN.calc_mu_bond(atoms_fr,results)
            #ワニエ中心の座標を計算する
            # results_wfcs = ASIGN.assign_wfc_to_mol(atoms_fr,results) 
        
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
                np.savetxt(savedir+'Descs_ring_'+str(fr)+'.csv', Descs_ring, delimiter=',')
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
            return total_dipole
            # >>>> 関数ここまで <<<<<
            
        # * データの保存
        # savedir = directory+"/bulk/0331test/"
        import os
        if not os.path.isdir(var_des.savedir):
            os.makedirs(var_des.savedir) # mkdir
            
        result = joblib.Parallel(n_jobs=-1, verbose=50)(joblib.delayed(calc_descripter_frame)(atoms_fr,wannier_fr,fr,var_des.savedir) for fr,(atoms_fr, wannier_fr) in enumerate(zip(traj,wannier_list)))

        # 双極子を保存
        result_dipole = np.array(result)
        np.save(var_des.savedir+"/wannier_dipole.npy", result_dipole)
        
        # atomsを保存
        return 0

if __name__ == '__main__':
    main()

    
    
