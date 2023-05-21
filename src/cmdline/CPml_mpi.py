#!/usr/bin/env python
# coding: utf-8

# * 
# * MPI implementation of mpi4py by mpi4py
# * 

import argparse
import sys
import numpy as np
import argparse
# import matplotlib.pyplot as plt

try:
    import ase.io
except ImportError:
    sys.exit("Error: ase.io not installed")
try:
    import ase
except ImportError:
    sys.exit("Error: ase not installed")
try:
    import mpi4py
except ImportError:
    sys.exit("Error: mpi4py not installed")

# import nglview as nv
from ase.io.trajectory import Trajectory
import ml.parse
# import home-made package
# import importlib
# import cpmd
import cpmd.read_traj_cpmd
import cpmd.asign_wcs 

import torch       # ライブラリ「PyTorch」のtorchパッケージをインポート
import torch.nn as nn  # 「ニューラルネットワーク」モジュールの別名定義


# 物理定数
from include.constants import constant
# Debye   = 3.33564e-30
# charge  = 1.602176634e-019
# ang      = 1.0e-10
coef    = constant.Ang*constant.Charge/constant.Debye

def calc_descripter_frame_descmode1(atoms_fr, fr, savedir, itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS ):
    '''
    記述子の保存：あり
    ワニエの割り当て：なしls
    機会学習:なし
    '''
    if atoms_fr == None:
        return 0
    import cpmd.descripter
    import cpmd.asign_wcs
    # * wannierの割り当て部分のメソッド化
    ASIGN=cpmd.asign_wcs.asign_wcs(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)
    DESC=cpmd.descripter.descripter(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)
    # * 原子座標とボンドセンターの計算
    # 原子座標,ボンドセンターを分子基準で再計算
    results = ASIGN.aseatom_to_mol_coord_bc(atoms_fr, itp_data.bonds_list)
    list_mol_coords, list_bond_centers =results
    # * ボンドデータをさらにch/coなど種別ごとに分割 & 記述子を計算
    # mu_bondsの中身はchとringで分割する
    #mu_paiは全数をringにアサイン
    #mu_lpOとlpNはゼロ
    # ring
    if len(itp_data.ring_bond_index) != 0:
        Descs_ring = []
        ring_cent_mol = cpmd.descripter.find_specific_ringcenter(list_bond_centers, itp_data.ring_bond_index, 8, NUM_MOL)
        i=0 
        for bond_center in ring_cent_mol:
            mol_id = i % NUM_MOL // 1
            Descs_ring.append(DESC.get_desc_bondcent(atoms_fr,bond_center,mol_id))
            i+=1 

    # ch,oh,co,cc,
    Descs_ch=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.ch_bond_index)
    Descs_oh=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.oh_bond_index)
    Descs_co=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.co_bond_index)
    Descs_cc=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.cc_bond_index)   
    # oローンペア
    Descs_o = DESC.calc_lonepair_descripter_at_frame(atoms_fr,list_mol_coords, itp_data.o_list, 8)

    # データが作成できているかの確認（debug）
    # print( " DESCRIPTOR SHAPE ")
    # print(" ring (Descs/data) ::", Descs_ring.shape)
    # print(" ch-bond (Descs/data) ::", Descs_ch.shape)
    # print(" cc-bond (Descs/data) ::", Descs_cc.shape)
    # print(" co-bond (Descs/data) ::", Descs_co.shape)
    # print(" oh-bond (Descs/data) ::", Descs_oh.shape)
    # print(" o-lone (Descs/data) ::", Descs_o.shape)

    # # ring, CHボンド，CCボンド，COボンド，OHボンド，Oローンペアのsave
    # if len(itp_data.ring_bond_index) != 0: np.savetxt(savedir+'Descs_ring_'+str(fr)+'.csv', Descs_ring, delimiter=',')
    # if len(itp_data.ch_bond_index) != 0: np.savetxt(savedir+'Descs_ch_'+str(fr)+'.csv', Descs_ch, delimiter=',')
    # if len(itp_data.cc_bond_index) != 0: np.savetxt(savedir+'Descs_cc_'+str(fr)+'.csv', Descs_cc, delimiter=',')
    # if len(itp_data.co_bond_index) != 0: np.savetxt(savedir+'Descs_co_'+str(fr)+'.csv', Descs_co, delimiter=',')
    # if len(itp_data.oh_bond_index) != 0: np.savetxt(savedir+'Descs_oh_'+str(fr)+'.csv', Descs_oh, delimiter=',')                
    # # Oローンペア
    # if len(itp_data.o_list) != 0: np.savetxt(savedir+'Descs_o_'+str(fr)+'.csv', Descs_o, delimiter=',')
    return 0


def calc_descripter_frame2(atoms_fr, wannier_fr, fr, savedir, itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS, double_bonds):
    '''
    記述子の保存：あり
    ワニエの割り当て：あり
    機会学習:なし
    '''
    import cpmd.descripter
    import cpmd.asign_wcs
    # * wannierの割り当て部分のメソッド化
    ASIGN=cpmd.asign_wcs.asign_wcs(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)
    DESC=cpmd.descripter.descripter(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)

    # * 原子座標とボンドセンターの計算
    # 原子座標,ボンドセンターを分子基準で再計算
    # TODO :: list_mol_coordsを使うのではなく，原子座標からatomsを作り直した方が良い．
    # TODO :: そうしておけば後ろでatomsを使う時にmicのことを気にしなくて良い（？）ので楽かも．
    results = ASIGN.aseatom_to_mol_coord_bc(atoms_fr, itp_data.bonds_list)
    list_mol_coords, list_bond_centers =results
    
    # wcsをbondに割り当て，bondの双極子まで計算
    results_mu = ASIGN.calc_mu_bond_lonepair(wannier_fr,atoms_fr,itp_data.bonds_list,double_bonds)
    list_mu_bonds,list_mu_pai,list_mu_lpO,list_mu_lpN, list_bond_wfcs,list_pi_wfcs,list_lpO_wfcs,list_lpN_wfcs = results_mu
    # wannnierをアサインしたase.atomsを作成する
    mol_with_WC = cpmd.asign_wcs.make_ase_with_WCs(atoms_fr.get_atomic_numbers(),NUM_MOL, UNITCELL_VECTORS,list_mol_coords,list_bond_centers,list_bond_wfcs,list_pi_wfcs,list_lpO_wfcs,list_lpN_wfcs)
    # 系の全双極子を計算
    # print(" list_mu_bonds {0}, list_mu_pai {1}, list_mu_lpO {2}, list_mu_lpN {3}".format(np.shape(list_mu_bonds),np.shape(list_mu_pai),np.shape(list_mu_lpO),np.shape(list_mu_lpN)))
    # ase.io.write(savedir+"molWC_"+str(fr)+".xyz", mol_with_WC)
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
    if len(itp_data.ring_bond_index) != 0:
        Descs_ring = []
        ring_cent_mol = cpmd.descripter.find_specific_ringcenter(list_bond_centers, itp_data.ring_bond_index, 8, NUM_MOL)
        i=0 
        for bond_center in ring_cent_mol:
            mol_id = i % NUM_MOL // 1
            Descs_ring.append(DESC.get_desc_bondcent(atoms_fr,bond_center,mol_id))
            i+=1 

    # ch,oh,co,cc,
    Descs_ch=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.ch_bond_index)
    Descs_oh=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.oh_bond_index)
    Descs_co=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.co_bond_index)
    Descs_cc=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.cc_bond_index)   
    # oローンペア
    Descs_o = DESC.calc_lonepair_descripter_at_frame(atoms_fr,list_mol_coords, itp_data.o_list, 8)

    # データが作成できているかの確認（debug）
    # print( " DESCRIPTOR SHAPE ")
    # print(" ring (Descs/data) ::", Descs_ring.shape)
    # print(" ch-bond (Descs/data) ::", Descs_ch.shape)
    # print(" cc-bond (Descs/data) ::", Descs_cc.shape)
    # print(" co-bond (Descs/data) ::", Descs_co.shape)
    # print(" oh-bond (Descs/data) ::", Descs_oh.shape)
    # print(" o-lone (Descs/data) ::", Descs_o.shape)

    # ring, CHボンド，CCボンド，COボンド，OHボンド，Oローンペアのsave
    if len(itp_data.ring_bond_index) != 0: np.savetxt(savedir+'Descs_ring_'+str(fr)+'.csv', Descs_ring, delimiter=',')
    if len(itp_data.ch_bond_index) != 0: np.savetxt(savedir+'Descs_ch_'+str(fr)+'.csv', Descs_ch, delimiter=',')
    if len(itp_data.cc_bond_index) != 0: np.savetxt(savedir+'Descs_cc_'+str(fr)+'.csv', Descs_cc, delimiter=',')
    if len(itp_data.co_bond_index) != 0: np.savetxt(savedir+'Descs_co_'+str(fr)+'.csv', Descs_co, delimiter=',')
    if len(itp_data.oh_bond_index) != 0: np.savetxt(savedir+'Descs_oh_'+str(fr)+'.csv', Descs_oh, delimiter=',')                
    # Oローンペア
    if len(itp_data.o_list) != 0: np.savetxt(savedir+'Descs_o_'+str(fr)+'.csv', Descs_o, delimiter=',')

    # データが作成できているかの確認（debug）
    # print( " DESCRIPTOR SHAPE ")
    # print(" ring (Descs/data) ::", Descs_ring.shape)
    # print(" ch-bond (Descs/data) ::", Descs_ch.shape)
    # print(" cc-bond (Descs/data) ::", Descs_cc.shape)
    # print(" co-bond (Descs/data) ::", Descs_co.shape)
    # print(" oh-bond (Descs/data) ::", Descs_oh.shape)
    # print(" o-lone (Descs/data) ::", Descs_o.shape)

    # ring, CHボンド, CCボンド, COボンド, OHボンド, Oローンペアの記述子を保存
    if len(itp_data.ring_bond_index) != 0: np.savetxt(savedir+'Descs_ring_'+str(fr)+'.csv', Descs_ring, delimiter=',')
    if len(itp_data.ch_bond_index) != 0: np.savetxt(savedir+'Descs_ch_'+str(fr)+'.csv', Descs_ch, delimiter=',')
    if len(itp_data.cc_bond_index) != 0: np.savetxt(savedir+'Descs_cc_'+str(fr)+'.csv', Descs_cc, delimiter=',')
    if len(itp_data.co_bond_index) != 0: np.savetxt(savedir+'Descs_co_'+str(fr)+'.csv', Descs_co, delimiter=',')
    if len(itp_data.oh_bond_index) != 0: np.savetxt(savedir+'Descs_oh_'+str(fr)+'.csv', Descs_oh, delimiter=',')
    if len(itp_data.o_list) != 0:
        np.savetxt(savedir+'Descs_o_'+str(fr)+'.csv', Descs_o, delimiter=',')
    return mol_with_WC, total_dipole
    # >>>> 関数ここまで <<<<<

# * 記述子をロードして予測させる
def predict_dipole_mode1(fr,desc_dir):
    #
    # * 機械学習用のデータを読み込む
    # *
    #
    global model_ch_2
    global model_co_2
    global model_oh_2
    global model_o_2

    # デバイスの設定    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    nfeatures = 288

    # ring
    # descs_X_ring = np.loadtxt('Descs_ring.csv',delimiter=',')
    # data_y_ring = np.loadtxt('data_y_ring.csv',delimiter=',')

    # CHボンド，COボンド，OHボンド，Oローンペア
    descs_X_ch = np.loadtxt(desc_dir+'Descs_ch_'+str(fr)+'.csv',delimiter=',')
    descs_X_co = np.loadtxt(desc_dir+'Descs_co_'+str(fr)+'.csv',delimiter=',')
    descs_X_oh = np.loadtxt(desc_dir+'Descs_oh_'+str(fr)+'.csv',delimiter=',')
    descs_X_o =  np.loadtxt(desc_dir+'Descs_o_'+str(fr)+'.csv',delimiter=',')

    # オリジナルの記述子を一旦tensorへ
    X_ch = torch.from_numpy(descs_X_ch.astype(np.float32)).clone()
    X_oh = torch.from_numpy(descs_X_oh.astype(np.float32)).clone()
    X_co = torch.from_numpy(descs_X_co.astype(np.float32)).clone()
    X_o  = torch.from_numpy(descs_X_o.astype(np.float32)).clone()

    # 予測
    y_pred_ch  = model_ch_2(X_ch.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
    y_pred_co  = model_co_2(X_co.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
    y_pred_oh  = model_oh_2(X_oh.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
    y_pred_o   = model_o_2(X_o.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()

    # 最後にreshape
    # !! ここは形としては(NUM_MOL*len(bond_index),3)となるが，予測だけする場合NUM_MOLの情報をgetできないので
    # 1! reshape(-1,3)としてしまう．
    
    # TODO : hard code (分子数)
    # NUM_MOL = 64
    y_pred_ch = y_pred_ch.reshape((-1,3))
    y_pred_co = y_pred_co.reshape((-1,3))
    y_pred_oh = y_pred_oh.reshape((-1,3))
    y_pred_o  = y_pred_o.reshape((-1,3))
    # print("DEBUG :: shape ch/co/oh/o :: {0} {1} {2} {3}".format(np.shape(y_pred_ch),np.shape(y_pred_co),np.shape(y_pred_oh),np.shape(y_pred_o)))
    # if fr == 0: # debug
        # print("y_pred_ch ::", y_pred_ch)
        # print("y_pred_co ::", y_pred_co)
        # print("y_pred_oh ::", y_pred_oh)
        # print("y_pred_o  ::", y_pred_o)
        #予測したモデルを使ったUnit Cellの双極子モーメントの計算
    sum_dipole=np.sum(y_pred_ch,axis=0)+np.sum(y_pred_oh,axis=0)+np.sum(y_pred_co,axis=0)+np.sum(y_pred_o,axis=0)
    return sum_dipole


def calc_descripter_frame_and_predict_dipole(atoms_fr, fr, itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS, model_ch_2, model_co_2, model_oh_2, model_o_2):
    
    '''
    機械学習での予測：あり
    ワニエのアサイン：なし
    '''
    if atoms_fr == None:
        return np.array([100,100,100]) # Noneの場合は100,100,100を代入する．もちろんこれはfakeである．
    import cpmd.descripter
    import cpmd.asign_wcs
    # * wannierの割り当て部分のメソッド化
    ASIGN=cpmd.asign_wcs.asign_wcs(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)
    DESC=cpmd.descripter.descripter(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)
    
    # * 原子座標とボンドセンターの計算
    # 原子座標,ボンドセンターを分子基準で再計算
    results = ASIGN.aseatom_to_mol_coord_bc(atoms_fr, itp_data.bonds_list)
    list_mol_coords, list_bond_centers =results

    # * ボンドデータをさらにch/coなど種別ごとに分割 & 記述子を計算
    # mu_bondsの中身はchとringで分割する
    #mu_paiは全数をringにアサイン
    #mu_lpOとlpNはゼロ
    # ring
    if len(itp_data.ring_bond_index) != 0:
        Descs_ring = []
        ring_cent_mol = cpmd.descripter.find_specific_ringcenter(list_bond_centers, itp_data.ring_bond_index, 8, NUM_MOL)
        i=0 
        for bond_center in ring_cent_mol:
            mol_id = i % NUM_MOL // 1
            Descs_ring.append(DESC.get_desc_bondcent(atoms_fr,bond_center,mol_id))
            i+=1 

    # ch,oh,co,cc,
    Descs_ch=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.ch_bond_index)
    Descs_oh=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.oh_bond_index)
    Descs_co=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.co_bond_index)
    Descs_cc=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.cc_bond_index)   
    # oローンペア
    Descs_o = DESC.calc_lonepair_descripter_at_frame(atoms_fr,list_mol_coords, itp_data.o_list, 8)

    # データが作成できているかの確認（debug）
    # print( " DESCRIPTOR SHAPE ")
    # print(" ring (Descs/data) ::", Descs_ring.shape)
    # print(" ch-bond (Descs/data) ::", Descs_ch.shape)
    # print(" cc-bond (Descs/data) ::", Descs_cc.shape)
    # print(" co-bond (Descs/data) ::", Descs_co.shape)
    # print(" oh-bond (Descs/data) ::", Descs_oh.shape)
    # print(" o-lone (Descs/data) ::", Descs_o.shape)

    # オリジナルの記述子を一旦tensorへ
    X_ch = torch.from_numpy(Descs_ch.astype(np.float32)).clone()
    X_oh = torch.from_numpy(Descs_oh.astype(np.float32)).clone()
    X_co = torch.from_numpy(Descs_co.astype(np.float32)).clone()
    X_o  = torch.from_numpy(Descs_o.astype(np.float32)).clone()

    # # 機械学習モデルの変数
    # global model_ch_2
    # global model_co_2
    # global model_oh_2
    # global model_o_2

    # デバイスの設定    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    nfeatures = 288

    # 予測
    y_pred_ch  = model_ch_2(X_ch.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
    y_pred_co  = model_co_2(X_co.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
    y_pred_oh  = model_oh_2(X_oh.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
    y_pred_o   = model_o_2(X_o.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()

    # 最後にreshape
    # !! ここは形としては(NUM_MOL*len(bond_index),3)となるが，予測だけする場合NUM_MOLの情報をgetできないので
    # !! reshape(-1,3)としてしまう．

    # TODO : hard code (分子数)
    # NUM_MOL = 64
    y_pred_ch = y_pred_ch.reshape((-1,3))
    y_pred_co = y_pred_co.reshape((-1,3))
    y_pred_oh = y_pred_oh.reshape((-1,3))
    y_pred_o  = y_pred_o.reshape((-1,3))
    # print("DEBUG :: shape ch/co/oh/o :: {0} {1} {2} {3}".format(np.shape(y_pred_ch),np.shape(y_pred_co),np.shape(y_pred_oh),np.shape(y_pred_o)))
    global rank
    # if fr == 0 : # デバッグ用
    #     print(" DEBUG y_pred shape len(y_pred_*)")
    #     print("y_pred_ch ::", len(y_pred_ch))
    #     print("y_pred_co ::", len(y_pred_co))
    #     print("y_pred_oh ::", len(y_pred_oh))
    #     print("y_pred_o  ::", len(y_pred_o))
    #予測したモデルを使ったUnit Cellの双極子モーメントの計算
    sum_dipole=np.sum(y_pred_ch,axis=0)+np.sum(y_pred_oh,axis=0)+np.sum(y_pred_co,axis=0)+np.sum(y_pred_o,axis=0)

    return sum_dipole



def main():
    import ml.parse
    import include.small
    import os
    import sys
    print(" ==DEBUG== start main() !!")
    # * ここからMPI implementation
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()  
    rank = comm.Get_rank()
    
    # itpファイルの読み込み
    if rank == 0:    
        # * 1-1：コマンドライン引数の読み込み
        inputfilename=sys.argv[1]
        # include.small.if_file_exist(inputfilename) # ファイルの存在確認（どうもmpiだとうまく動かない？）
        is_file = os.path.isfile(inputfilename)
        if not is_file: # itpファイルの存在を確認
            print("ERROR not found the file :: {} !! ".format(inputfilename))    
            sys.exit("1")
        # read itp files
        inputs_list=ml.parse.read_inputfile(inputfilename)
        input_general, input_descripter, input_predict=ml.parse.locate_tag(inputs_list)
        var_gen=ml.parse.var_general(input_general)
        var_des=ml.parse.var_descripter(input_descripter)
        var_pre=ml.parse.var_predict(input_predict)
        print(" finish reading input file {}".format(inputfilename))
        '''
        # * 計算モードがどうなっているかをチェックする
        パターン1: （単なる予測） 記述子だけ作成
        パターン2: （学習データ作成） ワニエのアサインと双極子の真値計算も実行
        パターン3: (予測&真値との比較) 記述子の作成, ワニエのアサインと双極子モーメント計算
        '''
        if_calc_descripter = var_des.calc # 1ならtrue
        if_calc_predict    = var_pre.calc # 1ならtrue

        if not if_calc_descripter and not if_calc_predict:
            print("CALCULATION MODE :: ERROR no calculation !!")
            print(" you have to specify calc flag in your input file")
            return 1

        if if_calc_descripter and not if_calc_predict:
            print("CALCULATION MODE :: descripter only")

        if not if_calc_descripter and if_calc_predict:
            print("CALCULATION MODE :: predict only")
    
        if if_calc_descripter and if_calc_predict:
            print("CALCULATION MODE :: descripter & predict")
    
    
        #
        # * 1-3：トポロジーファイル：itpの読み込み
        # * ボンドの情報を読み込む．
        # *
        include.small.if_file_exist(var_gen.itpfilename) # ファイルの存在確認

        # 実際の読み込み
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
        # 
        # * 結合リストの作成：二重結合だけは現状手で入れないといけない．
        # * 二重結合の電子は1つのC=C結合に２つ上下に並ばないケースもある。ベンゼン環上に非局在化しているのが要因か。
        # * 結合１つにワニエ中心１つづつ探し、二重結合は残った電子について探索する

        # TODO :: hard code :: 二重結合だけは，ここでdouble_bondsというのを作成している
        double_bonds_pairs = []    
        double_bonds = []
        for pair in double_bonds_pairs:
            if pair in bonds_list :
                double_bonds.append(bonds_list.index(pair))
            elif pair[::-1] in bonds_list :
                double_bonds.append(bonds_list.index(pair[::-1]))
            else :
                print("error")
        print(" double_bonds :: ", double_bonds)
        print(" -------- ")
        # * >>>>  double_bondsというか，π電子系のための設定 >>>>>>>>>


    else: #bcastのために必要
        itp_data = None
        if_calc_descripter = None
        if_calc_predict = None
        var_pre = None
        var_gen = None
        var_des = None
        bonds_list = None
        NUM_MOL_ATOMS = None
        double_bonds = None

    # bcast(broadcast variables)
    itp_data = comm.bcast(itp_data, root=0)
    if_calc_descripter = comm.bcast(if_calc_descripter, root=0)
    if_calc_predict = comm.bcast(if_calc_predict, root=0)
    var_des = comm.bcast(var_des, root=0)
    var_gen = comm.bcast(var_gen, root=0)
    var_pre = comm.bcast(var_pre, root=0)
    bonds_list = comm.bcast(bonds_list, root=0)
    NUM_MOL_ATOMS = comm.bcast(NUM_MOL_ATOMS, root=0)
    double_bonds = comm.bcast(double_bonds, root=0) 

    # return 0
    
    if rank == 0:   
        print(" ")
        print(" FINISH reading itp files and setting variables")
        print(" ")
        # ring_bonds = double_bonds_pairs
        ring_bonds = []

        ring_bond_index = itp_data.ring_bond_index

        # o_index = itp_data.o_list
        # n_index = itp_data.n_list


    if if_calc_descripter and not if_calc_predict:
        '''
        descripter計算のみの場合
        '''
        if rank == 0:
            print(" ")
            print(" *****************************************************************")
            print("             calc_descripter:: Reading Trajectory                 ")
            print(" *****************************************************************")
            print(" ")
        # * trajectoryの読み込み
        # aseでデータをロード
        # もしfilemodeがwannieronlyではない場合，wannier部分を除去する．
        if int(var_des.haswannier) == True:
            import cpmd.read_traj_cpmd
            traj, wannier_list=cpmd.read_traj_cpmd.raw_xyz_divide_aseatoms_list(var_des.directory+var_des.xyzfilename)
        else:
            traj=ase.io.read(var_des.directory+var_des.xyzfilename,index=":")

        # *
        # * 系のパラメータの設定
        # * 
        UNITCELL_VECTORS = traj[0].get_cell() # TODO :: セル情報がない場合にerrorを返す

        # 種々のデータをloadする．
        NUM_ATOM:int    = len(traj[0].get_atomic_numbers()) #原子数
        NUM_CONFIG:int  = len(traj) #フレーム数
        # UNITCELL_VECTORS = traj[0].get_cell() #cpmd.read_traj_cpmd.raw_cpmd_read_unitcell_vector("cpmd.read_traj_cpmd/bomd-wan.out.2.0") # tes.get_cell()[:]
        # num_of_bonds = {14:4,6:3,8:2,1:1} #原子の化学結合の手の数

        NUM_MOL = int(NUM_ATOM/NUM_MOL_ATOMS) #UnitCell中の総分子数
        frames = len(traj) # フレーム数

        if rank == 0:
            print(" --------  ")
            print(" NUM_ATOM  ::    ", NUM_ATOM )
            print(" NUM_CONFIG ::   ", NUM_CONFIG)
            print(" NUM_MOL    :: ",    NUM_MOL)
            print(" NUM_MOL_ATOMS :: ", NUM_MOL_ATOMS)
            print(" UNITCELL_VECTORS :: ", UNITCELL_VECTORS)
            print("total frames of trajectory:: ", frames)
            print(" --------  ")

        # elements = {"N":7,"C":6,"O":8,"H":1}


        # * wannierの割り当て部分のメソッド化
        import cpmd.read_traj_cpmd
        import cpmd.asign_wcs 
        ASIGN=cpmd.asign_wcs.asign_wcs(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)
        import cpmd.descripter
        DESC=cpmd.descripter.descripter(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)


    if if_calc_descripter and if_calc_predict: # descripter計算をする場合，trajectoryを読み込む
        if rank == 0:
            print(" ")
            print(" *****************************************************************")
            print("             calc_descripter:: Reading Trajectory                 ")
            print(" *****************************************************************")
            print(" ")
        # * trajectoryの読み込み
        # aseでデータをロード
        # もしfilemodeがwannieronlyではない場合，wannier部分を除去する．
        if int(var_des.haswannier) == True:
            import cpmd.read_traj_cpmd
            traj, wannier_list=cpmd.read_traj_cpmd.raw_xyz_divide_aseatoms_list(var_des.directory+var_des.xyzfilename)
        else:
            # !! mpi実装の場合，最初の構造だけ読み出し．
            # !! ここでまずは系のパラメータを読み込む．
            # !! 真にデータを読み出すのはあと．
            if rank == 0: # !! rank == 0で読み出したいので，ase.io.readが使えない！！
                from cpmd.read_traj_cpmd import raw_cpmd_get_unitcell_xyz,raw_cpmd_get_atomicnum_xyz
                UNITCELL_VECTORS = raw_cpmd_get_unitcell_xyz(var_des.directory+var_des.xyzfilename)
                NUM_ATOM:int     = raw_cpmd_get_atomicnum_xyz(var_des.directory+var_des.xyzfilename)
                # traj=ase.io.read(var_des.directory+var_des.xyzfilename,index=0) 
                # print(traj)
                # print("DEBUG :: size of traj[B] :: ", traj.__sizeof__())
            else:
                UNITCELL_VECTORS = None
                NUM_ATOM = None
            UNITCELL_VECTORS = comm.bcast(UNITCELL_VECTORS, root = 0)
            NUM_ATOM = comm.bcast(NUM_ATOM, root = 0)

                

        # *
        # * 系のパラメータの設定
        # * 
        if  int(var_des.haswannier) == True:
            UNITCELL_VECTORS = traj[0].get_cell() # TODO :: セル情報がない場合にerrorを返す
            # 種々のデータをloadする．
            NUM_ATOM:int    = len(traj[0].get_atomic_numbers()) #原子数
            NUM_CONFIG:int  = len(traj) #フレーム数
            # UNITCELL_VECTORS = traj[0].get_cell() #cpmd.read_traj_cpmd.raw_cpmd_read_unitcell_vector("cpmd.read_traj_cpmd/bomd-wan.out.2.0") # tes.get_cell()[:]
            # num_of_bonds = {14:4,6:3,8:2,1:1} #原子の化学結合の手の数        
        else:
            # UNITCELL_VECTORS = traj.get_cell() # TODO :: セル情報がない場合にerrorを返す            
            # 種々のデータをloadする．
            # NUM_ATOM:int    = len(traj.get_atomic_numbers()) #原子数
            NUM_CONFIG:int  = 1 # !! フレーム数（ここでは使わない）
            # UNITCELL_VECTORS = traj[0].get_cell() #cpmd.read_traj_cpmd.raw_cpmd_read_unitcell_vector("cpmd.read_traj_cpmd/bomd-wan.out.2.0") # tes.get_cell()[:]
            # num_of_bonds = {14:4,6:3,8:2,1:1} #原子の化学結合の手の数


        NUM_MOL = int(NUM_ATOM/NUM_MOL_ATOMS) #UnitCell中の総分子数
        frames = 1 # len(traj) # フレーム数

        if rank == 0:
            print(" --------  ")
            print(" NUM_ATOM  ::    ", NUM_ATOM )
            print(" NUM_CONFIG ::   ", NUM_CONFIG)
            print(" NUM_MOL    :: ",    NUM_MOL)
            print(" NUM_MOL_ATOMS :: ", NUM_MOL_ATOMS)
            print(" UNITCELL_VECTORS :: ", UNITCELL_VECTORS)
            print("total frames of trajectory:: ", frames)
            print(" --------  ")

        # elements = {"N":7,"C":6,"O":8,"H":1}


        # * wannierの割り当て部分のメソッド化
        import cpmd.read_traj_cpmd
        import cpmd.asign_wcs 
        ASIGN=cpmd.asign_wcs.asign_wcs(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)
        import cpmd.descripter
        DESC=cpmd.descripter.descripter(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)


    if if_calc_descripter and not if_calc_predict: # descripter計算のみの場合，記述子を保存して終了
        # * 
        # * パターン1つ目，ワニエのアサインはしないで記述子だけ作成する場合
        # * descripter計算開始
        if var_des.descmode == "1":
            ### 機械学習用のデータ（記述子）を作成する
            # 
            if rank == 0:
                print(" ------ ")
                print(" start making descripters to files ")
                print(" ------ ")
            import os
            if rank == 0:
                if not os.path.isdir(var_des.savedir):
                    os.makedirs(var_des.savedir) # mkdir

            # if not os.path.isdir(var_des.savedir):
            #     os.makedirs(var_des.savedir) # mkdir
            # if var_des.step != None: # stepが決まっている場合はこちらで設定してしまう．
            #     print("STEP is manually set :: {}".format(var_des.step))
            #     traj = traj[:var_des.step]

            import subprocess            
            if rank == 0: # xyzファイルの行数を取得する．
                # !! 注意 :: 実際のline count-1になっている場合があるので，roundで丸める．
                line_count = int(float(subprocess.check_output(['wc', '-l', var_des.directory+var_des.xyzfilename]).decode().split(' ')[0]))
                print("line_count :: {}".format(line_count))
                nsteps = round(float(line_count/(NUM_ATOM+2))) #29 #50001 
                print("nsteps :: {}".format(nsteps))
            else:
                nsteps = None
            nsteps = comm.bcast(nsteps, root=0) # 
            ave, res = divmod(nsteps, size) # averageとresidualを計算
            result_dipole = []
            
            if rank == 0: # filepointerのよみこみ
                filepointer = open(var_des.directory+var_des.xyzfilename)
            
            for i in range(ave): # もしかするとfor文がmpi全てで回っているかも．
                if rank == 0:
                    print("now we are in ave loop ... {}  :: {} {}".format(i,ave,res))
                    read_traj = []
                    for j in range(size):
                        symbols, positions, filepointer = cpmd.read_traj_cpmd.raw_cpmd_read_xyz(filepointer,NUM_ATOM)
                        read_traj.append(positions)
                else:
                    read_traj = None
                    symbols   = None
                    print("now we are in ave loop ... {}  :: {} {} {}/rank".format(i,ave,res,rank))
                # bcast/scatter data
                read_traj = comm.scatter(read_traj,root=0)
                symbols   = comm.bcast(symbols,root=0)
                aseatom   = ase.Atoms( # atomsを作成
                    symbols,
                    positions=read_traj,
                    cell=UNITCELL_VECTORS,
                    pbc=[1, 1, 1]
                )
                fr = size*i+rank
                print(" fr is ... {}  :: {}/loop {}/rank {}/size".format(fr,i,rank,size))
                
                # print(" hello rank {} {}".format(rank, read_traj)) 
                # frに変数が必要
                result_dipole_tmp = calc_descripter_frame_descmode1(aseatom,fr,var_des.savedir,itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)
                result_dipole_tmp = comm.gather(result_dipole_tmp, root=0)  # gatherしても基本全部0のはず
                if rank == 0:
                    print(" finish gather :: {}/ave".format(i))
                    # result_dipole.append(result_dipole_tmp)
            if rank == 0:
                print("")
                print(" Now start final res part ...")
                print("")
            # (ave+1)*size以降のあまりの部分の処理（res != 0の場合にのみ処理する）
            if res != 0:
                if rank == 0:
                    print("now we are in final step... :: {} {}".format(ave,res))
                    read_traj = []
                    for j in range(res):
                        symbols, positions, filepointer = cpmd.read_traj_cpmd.raw_cpmd_read_xyz(filepointer,NUM_ATOM)
                        read_traj.append(positions)
                    for i in range(size - res):
                        read_traj.append(None)
                    if len(read_traj) != size:
                        print("")
                        print("ERROR :: len(read_traj) != size")
                        print("")
                    print("len(read_traj) :: {}".format(len(read_traj)))
                    print("read_traj :: {}".format(read_traj))
                else: # rank != 0
                    read_traj = None
                    symbols   = None
                
                # bcast/scatter data
                read_traj = comm.scatter(read_traj,root=0)
                symbols   = comm.bcast(symbols,root=0)
                if read_traj == None: # sacatterした後にNoneのままだったら，計算しない．
                    aseatom = None
                else:
                    aseatom   = ase.Atoms( # atomsを作成
                        symbols,
                        positions=read_traj,
                        cell=UNITCELL_VECTORS,
                        pbc=[1, 1, 1]
                    )
                fr = ave*size+rank
                print(" fr is ... {}  :: {}/rank {}/size".format(fr,rank,size))
                if rank == 0:
                    print("")
                    print(" finish scattering data ...")
                    print("")
                # print(" hello rank {} {}".format(rank, read_traj))
                # frに変数が必要
                result_dipole_tmp = calc_descripter_frame_descmode1(aseatom,fr,var_des.savedir,itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)
                if rank == 0:
                    print("")
                    print(" finish descripter calculation ...")
                    print("")
                result_dipole_tmp = comm.gather(result_dipole_tmp, root=0) 
                if rank == 0:
                    print("")
                    print(" finish gather data ...")
                    print(" result_dipole_tmp is ... {}".format(result_dipole_tmp))
                    print(" np.shape(result_dipole_tmp) is ... {}".format(np.shape(result_dipole_tmp)))
                if rank == 0:
                    result_dipole.append(result_dipole_tmp)
            if rank == 0: # filepointer
                print("")
                print(" close file pointer ...")
                print("")
                filepointer.close()
                sys.exit(0)
            else:
                print("")
                print(" I am {}/rank and I am going to exit ...".format(rank))
                sys.exit(0)
            if rank == 0:
                print("")
                print(" sys.exit() ...")
                print("")
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
            ### 機械学習用のデータ（記述子）を作成する

            import joblib

            # * データの保存
            # savedir = directory+"/bulk/0331test/"
            import os
            if not os.path.isdir(var_des.savedir):
                os.makedirs(var_des.savedir) # mkdir
            
            if var_des.step != None: # stepが決まっている場合はこちらで設定してしまう．
                print("STEP is manually set :: {}".format(var_des.step))
                traj = traj[:var_des.step]
            # result = joblib.Parallel(n_jobs=-1, verbose=50)(joblib.delayed(calc_descripter_frame)(atoms_fr,wannier_fr,fr,var_des.savedir) for fr,(atoms_fr, wannier_fr) in enumerate(zip(traj,wannier_list)))
            result = joblib.Parallel(n_jobs=-1, verbose=50)(joblib.delayed(calc_descripter_frame2)(atoms_fr,wannier_fr,fr,var_des.savedir,itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS, double_bonds) for fr,(atoms_fr, wannier_fr) in enumerate(zip(traj,wannier_list)))
            
            # xyzデータと双極子データを取得
            result_ase    = [i[0] for i in result]
            result_dipole = [i[1] for i in result]
            
            # aseを保存
            ase.io.write(var_des.savedir+"/mol_WC.xyz", result_ase)

            # 双極子を保存
            result_dipole = np.array(result_dipole)
            np.save(var_des.savedir+"/wannier_dipole.npy", result_dipole)
            
            # atomsを保存
            return 0

    # *
    # * 機械学習をやる場合
    # * 
    if if_calc_predict: 
        if rank == 0:
            print(" ") 
            print(" *****************************************************************")
            print("             calc_predict :: Setting ML model                     ")
            print(" *****************************************************************")
            print(" ")

        import torch       # ライブラリ「PyTorch」のtorchパッケージをインポート
        import torch.nn as nn  # 「ニューラルネットワーク」モジュールの別名定義
        import ml.mlmodel

        # torch.nn.Moduleによるモデルの定義
        if var_pre.modelmode == "normal":
            # # TODO :: hardcode :: nfeatures :: ここはちょっと渡し方が難しいかも．
            # nfeatures = 288
            # print(" nfeatures :: ", nfeatures )
            
            # # 定数（モデル定義時に必要となるもの）
            # INPUT_FEATURES = nfeatures    # 入力（特徴）の数： 記述子の数
            # LAYER1_NEURONS = 100     # ニューロンの数
            # LAYER2_NEURONS = 100     # ニューロンの数
            # #LAYER3_NEURONS = 200     # ニューロンの数
            # #LAYER4_NEURONS = 100     # ニューロンの数
            # OUTPUT_RESULTS = 3      # 出力結果の数： 3

            # class WFC(nn.Module):
            #     def __init__(self):
            #         super().__init__()
                    
            #         # バッチ規格化層
            #         #self.bn1 = nn.BatchNorm1d(INPUT_FEATURES) #バッチ正規化
                    
            #         # 隠れ層：1つ目のレイヤー（layer）
            #         self.layer1 = nn.Linear(
            #             INPUT_FEATURES,                # 入力ユニット数（＝入力層）
            #             LAYER1_NEURONS)                # 次のレイヤーの出力ユニット数
                    
            #         # バッチ規格化層
            #         #self.bn2 = nn.BatchNorm1d(LAYER1_NEURONS) #バッチ正規化   
                    
            #         # 隠れ層：2つ目のレイヤー（layer）
            #         self.layer2 = nn.Linear(
            #             LAYER1_NEURONS,                # 入力ユニット数（＝入力層）
            #             LAYER2_NEURONS)                # 次のレイヤーの出力ユニット数
                    
            #         # バッチ規格化層
            #         #self.bn3 = nn.BatchNorm1d(LAYER2_NEURONS) #バッチ正規化   
                    
            #         # 隠れ層：3つ目のレイヤー（layer）
            #         #self.layer3 = nn.Linear(
            #         #    LAYER2_NEURONS,                # 入力ユニット数（＝入力層）
            #         #    LAYER3_NEURONS)                # 次のレイヤーの出力ユニット数
                    
            #         ## 隠れ層：4つ目のレイヤー（layer）
            #         #self.layer4 = nn.Linear(
            #         #    LAYER3_NEURONS,                # 入力ユニット数（＝入力層）
            #         #    LAYER4_NEURONS)                # 次のレイヤーの出力ユニット数
                    
            #         # 出力層
            #         self.layer_out = nn.Linear(
            #             LAYER2_NEURONS,                # 入力ユニット数
            #             OUTPUT_RESULTS)                # 出力結果への出力ユニット数

            #     def forward(self, x):
                
            #         # フォワードパスを定義
            #         #x = self.bn1(x) #バッチ規格化
            #         x = nn.functional.leaky_relu(self.layer1(x))  
            #         #x = self.bn2(x) #バッチ規格化
            #         x = nn.functional.leaky_relu(self.layer2(x))  
            #         #x = self.bn3(x) #バッチ規格化
            #         #x = nn.functional.leaky_relu(self.layer3(x))  
            #         #x = nn.functional.leaky_relu(self.layer4(x))  
            #         x = self.layer_out(x)  # ※最終層は線形
            #         return x
                
            # モデル（NeuralNetworkクラス）のインスタンス化（これは絶対に必要）
            if rank == 0:
                model_ring = ml.mlmodel.WFC()
                model_ch = ml.mlmodel.WFC()
                model_co = ml.mlmodel.WFC()
                model_oh = ml.mlmodel.WFC()
                model_o = ml.mlmodel.WFC()
            else:
                model_ring = None
                model_ch   = None
                model_co   = None
                model_oh   = None
                model_o    = None
            model_ring = comm.bcast(model_ring,root=0)
            model_ch   = comm.bcast(model_ch,root=0)
            model_co   = comm.bcast(model_co,root=0)
            model_oh   = comm.bcast(model_oh,root=0)
            model_o    = comm.bcast(model_o,root=0)



        if var_pre.modelmode == "rotate":
            print(" ------------------- ")
            print(" modelmode :: rotate ")
            print(" ------------------- ")

            import torch       # ライブラリ「PyTorch」のtorchパッケージをインポート
            import torch.nn as nn  # 「ニューラルネットワーク」モジュールの別名定義

            # nfeatures = 288 # TODO :: hard code 4*12*6=288 # len(train_X_ch[0][0])
            # print(" nfeatures :: ", nfeatures )

            # M = 20 
            # Mb= 6
                    
            # #Embedding Net 
            # nfeatures_enet = int(nfeatures/4) # 72
            # print(nfeatures_enet)
            # # 定数（モデル定義時に必要となるもの）
            # INPUT_FEATURES_enet = nfeatures_enet      # 入力（特徴）の数： 記述子の数
            # LAYER1_NEURONS_enet = 50             # ニューロンの数
            # LAYER2_NEURONS_enet = 50             # ニューロンの数
            # OUTPUT_RESULTS_enet = M*nfeatures_enet    # 出力結果の数： 
            
            # #Fitting Net 
            # nfeatures_fnet = int(M*Mb) 
            # print(nfeatures_fnet)
            # # 定数（モデル定義時に必要となるもの）
            # INPUT_FEATURES_fnet = nfeatures_fnet    # 入力（特徴）の数： 記述子の数
            # LAYER1_NEURONS_fnet = 50     # ニューロンの数
            # LAYER2_NEURONS_fnet = 50     # ニューロンの数
            # OUTPUT_RESULTS_fnet = M      # 出力結果の数：

            
            # # torch.nn.Moduleによるモデルの定義
            # class NET(nn.Module):
            #     def __init__(self):
            #         super().__init__()
            
            #         ##### Embedding Net #####
            #         # 隠れ層：1つ目のレイヤー（layer）
            #         self.Enet_layer1 = nn.Linear(
            #             INPUT_FEATURES_enet,                # 入力ユニット数（＝入力層）
            #             LAYER1_NEURONS_enet)                # 次のレイヤーの出力ユニット数
            
            #         # 隠れ層：2つ目のレイヤー（layer）
            #         self.Enet_layer2 = nn.Linear(
            #             LAYER1_NEURONS_enet,                # 入力ユニット数
            #             LAYER2_NEURONS_enet)                # 次のレイヤーの出力ユニット数
                    
            #         # 出力層
            #         self.Enet_layer_out = nn.Linear(
            #             LAYER2_NEURONS_enet,                # 入力ユニット数
            #             OUTPUT_RESULTS_enet)                # 出力結果への出力ユニット数
                    
            #         ##### Fitting net #####
            #         # 隠れ層：1つ目のレイヤー（layer）
            #         self.Fnet_layer1 = nn.Linear(
            #             INPUT_FEATURES_fnet,                # 入力ユニット数（＝入力層）
            #             LAYER1_NEURONS_fnet)                # 次のレイヤーの出力ユニット数
                    
            #         # 隠れ層：2つ目のレイヤー（layer）
            #         self.Fnet_layer2 = nn.Linear(
            #             LAYER1_NEURONS_fnet,                # 入力ユニット数
            #             LAYER2_NEURONS_fnet)                # 次のレイヤーの出力ユニット数
                    
            #         # 出力層
            #         self.Fnet_layer_out = nn.Linear(
            #         LAYER2_NEURONS_fnet,                # 入力ユニット数
            #             OUTPUT_RESULTS_fnet)                # 出力結果への出力ユニット数
                    
            #     def forward(self, x):
            
            #         #Si(1/Rをカットオフ関数で処理した値）のみを抽出する
            #         Q1 = x[:,::4]
            #         NB = Q1.size()[0]
            #         N  = Q1.size()[1]
            #         # Embedding Netに代入する 
            #         embedded_x = nn.functional.leaky_relu(self.Enet_layer1(Q1))  
            #         embedded_x = nn.functional.leaky_relu(self.Enet_layer2(embedded_x)) 
            #         embedded_x = self.Enet_layer_out(embedded_x)  # ※最終層は線形 
            #         #embedded_xを(ミニバッチデータ数)xMxN (N=MaxAt*原子種数)に変換
            #         embedded_x = torch.reshape(embedded_x,(NB,M,N ))
            #         #入力データをNB x N x 4 の行列に変形  
            #         matQ = torch.reshape(x,(NB,N,4))
            #         #Enetの出力との掛け算
            #         matT = torch.matmul(embedded_x, matQ)
            #         # matTの次元はNB x M x 4 となっている 
            #         #matSを作る(ハイパーパラメータMbで切り詰める)
            #         matS = matT[:,:Mb,:]
            #         #matSの転置行列を作る　→　NB x 4 x Mb となる 
            #         matSt = torch.transpose(matS, 1, 2)
            #         #matDを作る( matTとmatStの掛け算) →　NB x M x Mb となる 
            #         matD = torch.matmul(matT, matSt)
            #         #matDを１次元化する。matD全体をニューラルネットに入力したいので、ベクトル化する。 
            #         matD1 = torch.reshape(matD,(NB,M*Mb))
            #         # fitting Net に代入する 
            #         fitD = nn.functional.leaky_relu(self.Fnet_layer1(matD1))
            #         fitD = nn.functional.leaky_relu(self.Fnet_layer2(fitD)) 
            #         fitD = self.Fnet_layer_out(fitD)  # ※最終層は線形 
            #         # fitDの次元はNB x M となる。これをNB x 1 x Mの行列にする
            #         fitD3 = torch.reshape(fitD,(NB,1,M))
            #         # fttD3とmatTの掛け算 
            #         matW = torch.matmul(fitD3, matT) 
            #         # matWはNb x 1 x  4 になっている。これをNB x 4 の2次元にする
            #         matW2 = torch.reshape(matW,(NB,4))
            #         # はじめの要素はいらないので、切り詰めてx,y,z にする
            #         outW = matW2[:,1:]
                    
            #         return outW
            # # モデル（NeuralNetworkクラス）のインスタンス化
            if rank == 0:
                model_ring = ml.mlmodel.NET()
                model_ch = ml.mlmodel.NET()
                model_co = ml.mlmodel.NET()
                model_oh = ml.mlmodel.NET()
                model_o = ml.mlmodel.NET()
            else:
                model_ring = None
                model_ch   = None
                model_co   = None
                model_oh   = None
                model_o    = None
            model_ring = comm.bcast(model_ring,root=0)
            model_ch   = comm.bcast(model_ch,root=0)
            model_co   = comm.bcast(model_co,root=0)
            model_oh   = comm.bcast(model_oh,root=0)
            model_o    = comm.bcast(model_o,root=0)

            # <<<<<<<  if文ここまで <<<<<<<<
        from torchinfo import summary            
        if rank == 0:
            print(" Finish define ML model")
            summary(model=model_ring)
        
        # 
        # * モデルをロードする場合はこれを利用する
        # model_dir="model_train40percent/"
        # model_ring.load_state_dict(torch.load('model_ring_weight.pth'))
        model_ch.load_state_dict(torch.load(var_pre.model_dir+'model_ch_weight4.pth'))
        model_co.load_state_dict(torch.load(var_pre.model_dir+'model_co_weight4.pth'))
        model_oh.load_state_dict(torch.load(var_pre.model_dir+'model_oh_weight4.pth'))
        model_o.load_state_dict(torch.load(var_pre.model_dir+'model_o_weight4.pth'))
        if rank == 0:
            print(" Finish load ML parameters")


        #
        # * 全データを再予測させる．
        # 

        #GPUが使用可能か確認
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)
        
        if rank == 0:
            # 一旦モデルをcpuへ
            model_ch_2   = model_ch.to(device)
            model_oh_2   = model_oh.to(device)
            model_co_2   = model_co.to(device)
            model_o_2    = model_o.to(device)
        else:
            model_ch_2   = None
            model_oh_2   = None
            model_co_2   = None
            model_o_2    = None

        model_ch_2 = comm.bcast(model_ch_2, root=0)
        model_oh_2 = comm.bcast(model_oh_2, root=0)
        model_co_2 = comm.bcast(model_co_2, root=0)
        model_o_2 = comm.bcast(model_o_2, root=0)

        # ここで定義？
        model_ch_2   = model_ch.to(device)
        model_oh_2   = model_oh.to(device)
        model_co_2   = model_co.to(device)
        model_o_2    = model_o.to(device)

    # *
    # * 予測と機械学習を同時にやる場合
    # * （既に事前準備は完了しているので，最後のcalc_descripter_frameの定義だけ）
    if if_calc_descripter and if_calc_predict: 
        # * 
        # * パターン1つ目，ワニエのアサインはしないで記述子だけ作成する場合
        # * descripter計算開始
        if var_des.descmode == "1":
            import joblib

            # def calc_descripter_frame(atoms_fr, fr):
            #     # * 原子座標とボンドセンターの計算
            #     # 原子座標,ボンドセンターを分子基準で再計算
            #     results = ASIGN.aseatom_to_mol_coord_bc(atoms_fr, bonds_list) # ASIGNがglobal変数になっている
            #     list_mol_coords, list_bond_centers =results
        
            #     # * ボンドデータをさらにch/coなど種別ごとに分割 & 記述子を計算
            #     # mu_bondsの中身はchとringで分割する
            #     #mu_paiは全数をringにアサイン
            #     #mu_lpOとlpNはゼロ
            #     # ring
            #     if len(ring_bond_index) != 0:
            #         Descs_ring = []
            #         ring_cent_mol = cpmd.descripter.find_specific_ringcenter(list_bond_centers, ring_bond_index, 8, NUM_MOL)
            #         i=0 
            #         for bond_center in ring_cent_mol:
            #             mol_id = i % NUM_MOL // 1
            #             Descs_ring.append(DESC.get_desc_bondcent(atoms_fr,bond_center,mol_id)) #DESCがglobal変数になっている
            #             i+=1 

            #     # ch,oh,co,cc,
            #     Descs_ch=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.ch_bond_index)
            #     Descs_oh=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.oh_bond_index)
            #     Descs_co=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.co_bond_index)
            #     Descs_cc=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.cc_bond_index)   
            #     # oローンペア
            #     Descs_o = DESC.calc_lonepair_descripter_at_frame(atoms_fr,list_mol_coords, o_index, 8)

            #     # データが作成できているかの確認（debug）
            #     # print( " DESCRIPTOR SHAPE ")
            #     # print(" ring (Descs/data) ::", Descs_ring.shape)
            #     # print(" ch-bond (Descs/data) ::", Descs_ch.shape)
            #     # print(" cc-bond (Descs/data) ::", Descs_cc.shape)
            #     # print(" co-bond (Descs/data) ::", Descs_co.shape)
            #     # print(" oh-bond (Descs/data) ::", Descs_oh.shape)
            #     # print(" o-lone (Descs/data) ::", Descs_o.shape)

            #     # オリジナルの記述子を一旦tensorへ
            #     X_ch = torch.from_numpy(Descs_ch.astype(np.float32)).clone()
            #     X_oh = torch.from_numpy(Descs_oh.astype(np.float32)).clone()
            #     X_co = torch.from_numpy(Descs_co.astype(np.float32)).clone()
            #     X_o  = torch.from_numpy(Descs_o.astype(np.float32)).clone()
            
            #     # 予測
            #     y_pred_ch  = model_ch_2(X_ch.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
            #     y_pred_co  = model_co_2(X_co.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
            #     y_pred_oh  = model_oh_2(X_oh.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
            #     y_pred_o   = model_o_2(X_o.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
        
            #     # 最後にreshape
            #     # !! ここは形としては(NUM_MOL*len(bond_index),3)となるが，予測だけする場合NUM_MOLの情報をgetできないので
            #     # !! reshape(-1,3)としてしまう．
            
            #     # TODO : hard code (分子数)
            #     # NUM_MOL = 64
            #     y_pred_ch = y_pred_ch.reshape((-1,3))
            #     y_pred_co = y_pred_co.reshape((-1,3))
            #     y_pred_oh = y_pred_oh.reshape((-1,3))
            #     y_pred_o  = y_pred_o.reshape((-1,3))
            #     # print("DEBUG :: shape ch/co/oh/o :: {0} {1} {2} {3}".format(np.shape(y_pred_ch),np.shape(y_pred_co),np.shape(y_pred_oh),np.shape(y_pred_o)))
            #     if fr == 0: # デバッグ用
            #         print("y_pred_ch ::", y_pred_ch)
            #         print("y_pred_co ::", y_pred_co)
            #         print("y_pred_oh ::", y_pred_oh)
            #         print("y_pred_o  ::", y_pred_o)
            #     #予測したモデルを使ったUnit Cellの双極子モーメントの計算
            #     sum_dipole=np.sum(y_pred_ch,axis=0)+np.sum(y_pred_oh,axis=0)+np.sum(y_pred_co,axis=0)+np.sum(y_pred_o,axis=0)

            #     return sum_dipole
            #     # >>>> 関数ここまで <<<<<

            # # * 計算及びデータの保存
            # # savedir = directory+"/bulk/0331test/"
            # import os
            # if not os.path.isdir(var_des.savedir):
            #     os.makedirs(var_des.savedir) # mkdir
            # if var_des.step != None: # stepが決まっている場合はこちらで設定してしまう．
            #     print("STEP is manually set :: {}".format(var_des.step))
            #     traj = traj[:var_des.step]
                
            # # * ここからMPI implementation
            # from mpi4py import MPI
            # comm = MPI.COMM_WORLD
            # size = comm.Get_size()  
            # rank = comm.Get_rank()
            
            # # !! >>> 古い実装 >>>
            # # !! この実装だと，最初にtrajとして全trajectoryを読み出すのでかなり時間がかかってしまう．
            # # !! 新しい実装で，都度データを読み出す形式に変更．
            # # trajデータをnprocs個に分割
            # if rank == 0:
            #     nsteps = len(traj)  # 50001
            #     # 各サブタスクのサイズを決定
            #     # 基本的に各processにave個割り当てるが，resだけ余っている分を最初のres個のprocessにひとつづつ割り当てる．
            #     ave, res = divmod(nsteps, size)
            #     print("preparing data to scatter...")
            #     print(" ave = {0} and res = {1} ".format(ave,res))
            #     counts = [ave + 1 if p < res else ave for p in range(size)]
            #     print(counts)
            #     # 各サブタスクの開始インデックスと終了インデックスを決定
            #     starts = [sum(counts[:p]) for p in range(size)]
            #     ends = [sum(counts[:p+1]) for p in range(size)]

            #     # 開始インデックスと終了インデックスをデータに保存
            #     data = [(starts[p], ends[p]) for p in range(size)]
            #     print("data {}".format(data))
            #     print("len(data) = {}".format(len(data)))

            #     # traj を分割して，各プロセッサーに送るようにする．
            #     # traj = [[] for i in range(size)]
            #     new_traj = [[ traj[i] for i in range(starts[p],ends[p])] for p in range(size) ]
            # else:
            #     data = None
            #     traj = None
            #     new_traj = None

            # data = comm.scatter(data, root=0)
            # # traj = comm.scatter(traj, root=0)
            # new_traj = comm.scatter(new_traj,root=0)
            # print("hello !! data is {} ~ {}".format(data[0],data[1]))

            # # traj = ase.io.read("gromacs_trajectory_cell.xyz", index="{0}:{1}".format(data[0],data[1]))
            # print(" hello rank {},finish reading traj :: {} {}".format(rank,len(new_traj),new_traj[0].get_positions()[0]))
            # # print("rank {} :: traj is ... {}".format(rank, traj))

            # result_dipole = np.array([ calc_descripter_frame_and_predict_dipole(atoms_fr,fr,itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS, model_ch_2, model_co_2, model_oh_2, model_o_2) for fr,atoms_fr in enumerate(new_traj) ])
            # print("hello rank {}, len(result_dipole) is {}, ".format(rank, np.shape(result_dipole)))
            # # !! ここは注意が必要で，result_dipoleの形は[ [processor1], [processor2], ... ]となっている．
            # # !! 従って，単にnp.reshapeするだけだけではダメ．
            # result_dipole = comm.gather(result_dipole, root=0) 
            # # !! <<< ここまで古い実装 <<<
            
            # !! <<< ここから新しい実装 <<<
            import os
            if rank == 0:
                if not os.path.isdir(var_des.savedir):
                    os.makedirs(var_des.savedir) # mkdir

            # if not os.path.isdir(var_des.savedir):
            #     os.makedirs(var_des.savedir) # mkdir
            # if var_des.step != None: # stepが決まっている場合はこちらで設定してしまう．
            #     print("STEP is manually set :: {}".format(var_des.step))
            #     traj = traj[:var_des.step]

            import subprocess            
            if rank == 0: # xyzファイルの行数を取得する．
                # !! 注意 :: 実際のline count-1になっている場合があるので，roundで丸める．
                line_count = int(float(subprocess.check_output(['wc', '-l', var_des.directory+var_des.xyzfilename]).decode().split(' ')[0]))
                print("line_count :: {}".format(line_count))
                nsteps = round(float(line_count/(NUM_ATOM+2))) #29 #50001 
                print("nsteps :: {}".format(nsteps))
            else:
                nsteps = None
            nsteps = comm.bcast(nsteps, root=0)
            ave, res = divmod(nsteps, size) # averageとresidualを計算
            result_dipole = []
            
            if rank == 0: # filepointer
                filepointer = open(var_des.directory+var_des.xyzfilename)
            
            for i in range(ave):
                if rank == 0:
                    print("now we are in ... {}  :: {} {}".format(i,ave,res))
                    read_traj = []
                    for j in range(size):
                        symbols, positions, filepointer = cpmd.read_traj_cpmd.raw_cpmd_read_xyz(filepointer,NUM_ATOM)
                        read_traj.append(positions)
                else:
                    read_traj = None
                    symbols   = None

                # bcast/scatter data
                read_traj = comm.scatter(read_traj,root=0)
                symbols   = comm.bcast(symbols,root=0)
                aseatom   = ase.Atoms( # atomsを作成
                    symbols,
                    positions=read_traj,
                    cell=UNITCELL_VECTORS,
                    pbc=[1, 1, 1]
                )
                
                # print(" hello rank {} {}".format(rank, read_traj))
                result_dipole_tmp = calc_descripter_frame_and_predict_dipole(aseatom,0,itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS, model_ch_2, model_co_2, model_oh_2, model_o_2) 
                result_dipole_tmp = comm.gather(result_dipole_tmp, root=0) 
                if rank == 0:
                    result_dipole.append(result_dipole_tmp)
            
            # (ave+1)*size以降のあまりの部分の処理（res != 0の場合にのみ処理する）
            if res != 0:
                if rank == 0:
                    print("now we are in final step... :: {} {}".format(ave,res))
                    read_traj = []
                    for j in range(res):
                        symbols, positions, filepointer = cpmd.read_traj_cpmd.raw_cpmd_read_xyz(filepointer,NUM_ATOM)
                        read_traj.append(positions)
                    for i in range(size - res):
                        read_traj.append(None)
                    print("len(read_traj) :: {}".format(len(read_traj)))
                else: # rank != 0
                    read_traj = None
                    symbols   = None
                
                # bcast/scatter data
                read_traj = comm.scatter(read_traj,root=0)
                symbols   = comm.bcast(symbols,root=0)
                if read_traj == None:
                    aseatom = None
                else:
                    aseatom   = ase.Atoms( # atomsを作成
                        symbols,
                        positions=read_traj,
                        cell=UNITCELL_VECTORS,
                        pbc=[1, 1, 1]
                    )

                # print(" hello rank {} {}".format(rank, read_traj))
                result_dipole_tmp = calc_descripter_frame_and_predict_dipole(aseatom,0,itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS, model_ch_2, model_co_2, model_oh_2, model_o_2) 
                result_dipole_tmp = comm.gather(result_dipole_tmp, root=0) 
                if rank == 0:
                    result_dipole.append(result_dipole_tmp)

            if rank == 0:
                print("result_dipole ...")
                print(result_dipole)
                # answer_result_dipole = [i for j in result_dipole for i in j] # こういう書き方もある．https://qiita.com/propella/items/fa64b40b6f45d4f32cbc
                answer_result_dipole = []
                count = 0 # nstepsになったら終了
                for i in result_dipole: # i = [processor]
                    for j in i: # j = [fr0,fr1,...]
                        answer_result_dipole.append(j)
                        count += 1
                        if count == nsteps:
                            break

                # 双極子を保存
                answer_result_dipole = np.array(answer_result_dipole)
                print("np.shape(answer_result_dipole)", np.shape(answer_result_dipole))
                # np.save(var_des.savedir+"/wannier_dipole.npy", result_dipole)
                np.save(var_des.savedir+"/result_dipole.npy",answer_result_dipole)
                print(" finish saving data")
                print("answer_result_dipole is ... ",answer_result_dipole)

            # result_dipole = joblib.Parallel(n_jobs=-1, verbose=50)(joblib.delayed(calc_descripter_frame)(atoms_fr,fr) for fr,atoms_fr in enumerate(traj))
            # result_dipole = joblib.Parallel(n_jobs=-1, verbose=50)(joblib.delayed(calc_descripter_frame_and_predict_dipole)(atoms_fr,fr,itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS) for fr,atoms_fr in enumerate(traj))
            print("finish all procedure !!")
            return 0 # これが動いていない？
    return 0 

if __name__ == '__main__':
    main()

