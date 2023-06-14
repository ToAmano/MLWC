#!/usr/bin/env python
# coding: utf-8

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


import torch       # ライブラリ「PyTorch」のtorchパッケージをインポート
import torch.nn as nn  # 「ニューラルネットワーク」モジュールの別名定義

from ase.io.trajectory import Trajectory
import ml.parse # my package
# import home-made package
# import importlib
# import cpmd

# 物理定数
from include.constants import constant
# Debye   = 3.33564e-30
# charge  = 1.602176634e-019
# ang      = 1.0e-10
coef    = constant.Ang*constant.Charge/constant.Debye


def calc_descripter_frame_descmode1(atoms_fr, fr, savedir, itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS ):
    '''
    計算を独立させた場合にどうなるかの実験
    '''
    import cpmd.descripter
    import cpmd.asign_wcs
    # * wannierの割り当て部分のメソッド化
    ASIGN=cpmd.asign_wcs.asign_wcs(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)
    DESC=cpmd.descripter.descripter(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)
    # * 原子座標とボンドセンターの計算
    # 原子座標,ボンドセンターを分子基準で再計算
    results = ASIGN.aseatom_to_mol_coord_bc(atoms_fr, itp_data, itp_data.bonds_list)
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
        np.savetxt(savedir+'Descs_ring_'+str(fr)+'.csv', Descs_ring, delimiter=',')
        del Descs_ring
    
    # ボンドが存在すれば記述子計算&保存&変数削除
    if len(itp_data.ch_bond_index) != 0:
        Descs_ch=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.ch_bond_index)
        np.savetxt(savedir+'Descs_ch_'+str(fr)+'.csv', Descs_ch, delimiter=',')
        del Descs_ch
    if len(itp_data.cc_bond_index) != 0:
        Descs_cc=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.cc_bond_index)
        np.savetxt(savedir+'Descs_cc_'+str(fr)+'.csv', Descs_cc, delimiter=',')
        del Descs_cc
    if len(itp_data.co_bond_index) != 0:
        Descs_co=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.co_bond_index)
        np.savetxt(savedir+'Descs_co_'+str(fr)+'.csv', Descs_co, delimiter=',')
        del Descs_co
    if len(itp_data.oh_bond_index) != 0:
        Descs_oh=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.oh_bond_index)
        np.savetxt(savedir+'Descs_oh_'+str(fr)+'.csv', Descs_oh, delimiter=',')
        del Descs_oh
    if len(itp_data.o_list) != 0:
        Descs_o = DESC.calc_lonepair_descripter_at_frame(atoms_fr,list_mol_coords, itp_data.o_list, 8)
        np.savetxt(savedir+'Descs_o_'+str(fr)+'.csv', Descs_o, delimiter=',')
        del Descs_o

    
    # # ch,oh,co,cc,
    # Descs_oh=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.oh_bond_index)
    # Descs_co=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.co_bond_index)
    # Descs_cc=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.cc_bond_index)   
    # # oローンペア
    # Descs_o = DESC.calc_lonepair_descripter_at_frame(atoms_fr,list_mol_coords, itp_data.o_list, 8)
    # # データが作成できているかの確認（debug）
    # # print( " DESCRIPTOR SHAPE ")
    # # print(" ring (Descs/data) ::", Descs_ring.shape)
    # # print(" ch-bond (Descs/data) ::", Descs_ch.shape)
    # # print(" cc-bond (Descs/data) ::", Descs_cc.shape)
    # # print(" co-bond (Descs/data) ::", Descs_co.shape)
    # # print(" oh-bond (Descs/data) ::", Descs_oh.shape)
    # # print(" o-lone (Descs/data) ::", Descs_o.shape)
    
    # # ring, CHボンド，CCボンド，COボンド，OHボンド，Oローンペアのsave
    # if len(itp_data.ring_bond_index) != 0: np.savetxt(savedir+'Descs_ring_'+str(fr)+'.csv', Descs_ring, delimiter=',')
    # if len(itp_data.ch_bond_index) != 0: np.savetxt(savedir+'Descs_ch_'+str(fr)+'.csv', Descs_ch, delimiter=',')
    # if len(itp_data.cc_bond_index) != 0: np.savetxt(savedir+'Descs_cc_'+str(fr)+'.csv', Descs_cc, delimiter=',')
    # if len(itp_data.co_bond_index) != 0: np.savetxt(savedir+'Descs_co_'+str(fr)+'.csv', Descs_co, delimiter=',')
    # if len(itp_data.oh_bond_index) != 0: np.savetxt(savedir+'Descs_oh_'+str(fr)+'.csv', Descs_oh, delimiter=',')                
    # # Oローンペア
    # if len(itp_data.o_list) != 0: np.savetxt(savedir+'Descs_o_'+str(fr)+'.csv', Descs_o, delimiter=',')
    return 0

from scipy.spatial import distance
def calc_descripter_frame2(atoms_fr, wannier_fr, fr, savedir, itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS, double_bonds):
    '''
    ワニエのアサインもやるタイプ
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
    results = ASIGN.aseatom_to_mol_coord_bc(atoms_fr, itp_data, itp_data.bonds_list)
    list_mol_coords, list_bond_centers =results
    
    # そもそものwcsがちゃんとしているかの確認
    # 全てのワニエ間の距離を確認し，あまりに小さい場合に警告を出す（CPMDのワニエ計算が失敗している可能性あり）
    test_wan = np.array(wannier_fr)
    test_wan_distances = distance.cdist(test_wan,test_wan,metric='euclidean')
    # print(test_wan_distances) 
    if test_wan_distances[test_wan_distances>0].any() < 0.2:
        print("ERROR :: wcs are too small !! :: check CPMD calculation")
    
    # wcsをbondに割り当て，bondの双極子まで計算
    # !! 注意 :: calc_mu_bond_lonepairの中で，再度raw_aseatom_to_mol_coord_bcを呼び出して原子/BCのMIC座標を計算している．
    results_mu = ASIGN.calc_mu_bond_lonepair(wannier_fr,atoms_fr,itp_data.bonds_list,itp_data,double_bonds)
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
    #unit cellの全双極子モーメントの計算
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

    # ch,oh,co,ccの記述子&真値を計算 
    if len(itp_data.ch_bond_index) != 0:
        Descs_ch=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.ch_bond_index)
        np.savetxt(savedir+'Descs_ch_'+str(fr)+'.csv', Descs_ch, delimiter=',')
        if np.max(Descs_ch) > 5.0: # 記述子が大きすぎる場合に警告
            print(" WARNING :: Descs_ch is too large !! :: {}".format(np.max(Descs_ch)))
        del Descs_ch
        True_y_ch=DESC.calc_bondmu_descripter_at_frame(list_mu_bonds, itp_data.ch_bond_index)
        np.savetxt(savedir+'True_y_ch_'+str(fr)+'.csv', True_y_ch, delimiter=',')
        del True_y_ch
    if len(itp_data.oh_bond_index) != 0:
        Descs_oh=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.oh_bond_index)
        np.savetxt(savedir+'Descs_oh_'+str(fr)+'.csv', Descs_oh, delimiter=',')
        if np.max(Descs_oh) > 5.0: # 記述子が大きすぎる場合に警告
            print(" WARNING :: Descs_oh is too large !! :: {}".format(np.max(Descs_oh)))
        del Descs_oh
        True_y_oh=DESC.calc_bondmu_descripter_at_frame(list_mu_bonds, itp_data.oh_bond_index)
        np.savetxt(savedir+'True_y_oh_'+str(fr)+'.csv', True_y_oh, delimiter=',')
        del True_y_oh
    if len(itp_data.co_bond_index) != 0:
        Descs_co=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.co_bond_index)
        np.savetxt(savedir+'Descs_co_'+str(fr)+'.csv', Descs_co, delimiter=',')
        if np.max(Descs_co) > 5.0:
            print(" WARNING :: Descs_co is too large !! :: {}".format(np.max(Descs_co)))
        del Descs_co
        True_y_co=DESC.calc_bondmu_descripter_at_frame(list_mu_bonds, itp_data.co_bond_index)
        np.savetxt(savedir+'True_y_co_'+str(fr)+'.csv', True_y_co, delimiter=',')
        del True_y_co
    if len(itp_data.cc_bond_index) != 0:
        Descs_cc=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.cc_bond_index)
        np.savetxt(savedir+'Descs_cc_'+str(fr)+'.csv', Descs_cc, delimiter=',')
        if np.max(Descs_cc) > 5.0:
            print(" WARNING :: Descs_cc is too large !! :: {}".format(np.max(Descs_cc)))
        del Descs_cc
        True_y_cc=DESC.calc_bondmu_descripter_at_frame(list_mu_bonds, itp_data.cc_bond_index)
        np.savetxt(savedir+'True_y_cc_'+str(fr)+'.csv', True_y_cc, delimiter=',')
        del True_y_cc
    if len(itp_data.o_list) != 0:
        Descs_o = DESC.calc_lonepair_descripter_at_frame(atoms_fr,list_mol_coords, itp_data.o_list, 8)
        np.savetxt(savedir+'Descs_o_'+str(fr)+'.csv', Descs_o, delimiter=',')
        if np.max(Descs_o) > 5.0:
            print(" WARNING :: Descs_o is too large !! :: {}".format(np.max(Descs_o)))
        del Descs_o
        True_y_o = np.array(list_mu_lpO).reshape(-1,3) # !! 形に注意
        np.savetxt(savedir+'True_y_o_'+str(fr)+'.csv', True_y_o, delimiter=',')
        del True_y_o        

    # Descs_ch=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.ch_bond_index)
    # Descs_oh=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.oh_bond_index)
    # Descs_co=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.co_bond_index)
    # Descs_cc=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.cc_bond_index)   
    # # oローンペア
    # Descs_o = DESC.calc_lonepair_descripter_at_frame(atoms_fr,list_mol_coords, itp_data.o_list, 8)

    # # データが作成できているかの確認（debug）
    # print( " DESCRIPTOR SHAPE :: should be 2D array, (NUM_MOL*NUM_bond, features)")
    # # print(" ring (Descs/data) ::", Descs_ring.shape)
    # print(" ch-bond (Descs/data) ::", Descs_ch.shape)
    # print(" cc-bond (Descs/data) ::", Descs_cc.shape)
    # print(" co-bond (Descs/data) ::", Descs_co.shape)
    # print(" oh-bond (Descs/data) ::", Descs_oh.shape)
    # print(" o-lone (Descs/data) ::", Descs_o.shape)

    # # ch,oh,co,ccのdipoleの真値
    # True_y_ch=DESC.calc_bondmu_descripter_at_frame(list_mu_bonds, itp_data.ch_bond_index)
    # True_y_oh=DESC.calc_bondmu_descripter_at_frame(list_mu_bonds, itp_data.oh_bond_index)
    # True_y_co=DESC.calc_bondmu_descripter_at_frame(list_mu_bonds, itp_data.co_bond_index)
    # True_y_cc=DESC.calc_bondmu_descripter_at_frame(list_mu_bonds, itp_data.cc_bond_index)

    # # oローンペア
    # True_y_o = np.array(list_mu_lpO).reshape(-1,3) # !! 形に注意

    # print( " TRUE DATA SHAPE :: should be 2D array, (NUM_MOL*NUM_bond, 3)")
    # print(" ch-bond (Descs/data) ::", True_y_ch.shape)
    # print(" cc-bond (Descs/data) ::", True_y_cc.shape)
    # print(" co-bond (Descs/data) ::", True_y_co.shape)
    # print(" oh-bond (Descs/data) ::", True_y_oh.shape)
    # print(" o-lone (Descs/data) ::",  True_y_o.shape)


    # # ring, CHボンド，CCボンド，COボンド，OHボンド，Oローンペアのsave
    # if len(itp_data.ring_bond_index) != 0: np.savetxt(savedir+'Descs_ring_'+str(fr)+'.csv', Descs_ring, delimiter=',')
    # if len(itp_data.ch_bond_index) != 0: np.savetxt(savedir+'Descs_ch_'+str(fr)+'.csv', Descs_ch, delimiter=',')
    # if len(itp_data.cc_bond_index) != 0: np.savetxt(savedir+'Descs_cc_'+str(fr)+'.csv', Descs_cc, delimiter=',')
    # if len(itp_data.co_bond_index) != 0: np.savetxt(savedir+'Descs_co_'+str(fr)+'.csv', Descs_co, delimiter=',')
    # if len(itp_data.oh_bond_index) != 0: np.savetxt(savedir+'Descs_oh_'+str(fr)+'.csv', Descs_oh, delimiter=',')                
    # # Oローンペア
    # if len(itp_data.o_list) != 0: np.savetxt(savedir+'Descs_o_'+str(fr)+'.csv', Descs_o, delimiter=',')

    # # ring, CHボンド，CCボンド，COボンド，OHボンド，Oローンペアの真値のsave
    # # if len(itp_data.ring_bond_index) != 0: np.savetxt(savedir+'Descs_ring_'+str(fr)+'.csv', True_y_ring, delimiter=',')
    # if len(itp_data.ch_bond_index) != 0: np.savetxt(savedir+'True_y_ch_'+str(fr)+'.csv', True_y_ch, delimiter=',')
    # if len(itp_data.cc_bond_index) != 0: np.savetxt(savedir+'True_y_cc_'+str(fr)+'.csv', True_y_cc, delimiter=',')
    # if len(itp_data.co_bond_index) != 0: np.savetxt(savedir+'True_y_co_'+str(fr)+'.csv', True_y_co, delimiter=',')
    # if len(itp_data.oh_bond_index) != 0: np.savetxt(savedir+'True_y_oh_'+str(fr)+'.csv', True_y_oh, delimiter=',')                
    # # Oローンペア
    # if len(itp_data.o_list) != 0: np.savetxt(savedir+'True_y_o_'+str(fr)+'.csv', True_y_o, delimiter=',')

    return mol_with_WC, total_dipole
    # >>>> 関数ここまで <<<<<

class WFC(nn.Module):
    # TODO :: hardcode :: nfeatures :: ここはちょっと渡し方が難しいかも．
    nfeatures = 288
    print(" nfeatures :: ", nfeatures )
    
    # 定数（モデル定義時に必要となるもの）
    INPUT_FEATURES = nfeatures    # 入力（特徴）の数： 記述子の数
    LAYER1_NEURONS = 100     # ニューロンの数
    LAYER2_NEURONS = 100     # ニューロンの数
    #LAYER3_NEURONS = 200     # ニューロンの数
    #LAYER4_NEURONS = 100     # ニューロンの数
    OUTPUT_RESULTS = 3      # 出力結果の数： 3
    def __init__(self):
        super().__init__()
        
        # バッチ規格化層
        #self.bn1 = nn.BatchNorm1d(INPUT_FEATURES) #バッチ正規化
        
        # 隠れ層：1つ目のレイヤー（layer）
        self.layer1 = nn.Linear(
            self.INPUT_FEATURES,                # 入力ユニット数（＝入力層）
            self.LAYER1_NEURONS)                # 次のレイヤーの出力ユニット数
        
        # バッチ規格化層
        #self.bn2 = nn.BatchNorm1d(LAYER1_NEURONS) #バッチ正規化   
        
        # 隠れ層：2つ目のレイヤー（layer）
        self.layer2 = nn.Linear(
            self.LAYER1_NEURONS,                # 入力ユニット数（＝入力層）
            self.LAYER2_NEURONS)                # 次のレイヤーの出力ユニット数
        
        # バッチ規格化層
        #self.bn3 = nn.BatchNorm1d(LAYER2_NEURONS) #バッチ正規化   
        
        # 隠れ層：3つ目のレイヤー（layer）
        #self.layer3 = nn.Linear(
        #    LAYER2_NEURONS,                # 入力ユニット数（＝入力層）
        #    LAYER3_NEURONS)                # 次のレイヤーの出力ユニット数
        
        ## 隠れ層：4つ目のレイヤー（layer）
        #self.layer4 = nn.Linear(
        #    LAYER3_NEURONS,                # 入力ユニット数（＝入力層）
        #    LAYER4_NEURONS)                # 次のレイヤーの出力ユニット数
        
        # 出力層
        self.layer_out = nn.Linear(
            self.LAYER2_NEURONS,                # 入力ユニット数
            self.OUTPUT_RESULTS)                # 出力結果への出力ユニット数

    def forward(self, x):
    
        # フォワードパスを定義
        #x = self.bn1(x) #バッチ規格化
        x = nn.functional.leaky_relu(self.layer1(x))  
        #x = self.bn2(x) #バッチ規格化
        x = nn.functional.leaky_relu(self.layer2(x))  
        #x = self.bn3(x) #バッチ規格化
        #x = nn.functional.leaky_relu(self.layer3(x))  
        #x = nn.functional.leaky_relu(self.layer4(x))  
        x = self.layer_out(x)  # ※最終層は線形
        return x


# torch.nn.Moduleによるモデルの定義
class NET(nn.Module):
    nfeatures = 288 # TODO :: hard code 4*12*6=288 # len(train_X_ch[0][0])
    print(" nfeatures :: ", nfeatures )

    M = 20 
    Mb= 6
            
    #Embedding Net 
    nfeatures_enet = int(nfeatures/4) # 72
    print(nfeatures_enet)
    # 定数（モデル定義時に必要となるもの）
    INPUT_FEATURES_enet = nfeatures_enet      # 入力（特徴）の数： 記述子の数
    LAYER1_NEURONS_enet = 50             # ニューロンの数
    LAYER2_NEURONS_enet = 50             # ニューロンの数
    OUTPUT_RESULTS_enet = M*nfeatures_enet    # 出力結果の数： 

    #Fitting Net 
    nfeatures_fnet = int(M*Mb) 
    print(nfeatures_fnet)
    # 定数（モデル定義時に必要となるもの）
    INPUT_FEATURES_fnet = nfeatures_fnet    # 入力（特徴）の数： 記述子の数
    LAYER1_NEURONS_fnet = 50     # ニューロンの数
    LAYER2_NEURONS_fnet = 50     # ニューロンの数
    OUTPUT_RESULTS_fnet = M      # 出力結果の数：

    def __init__(self):
        super().__init__()

        ##### Embedding Net #####
        # 隠れ層：1つ目のレイヤー（layer）
        self.Enet_layer1 = nn.Linear(
            self.INPUT_FEATURES_enet,                # 入力ユニット数（＝入力層）
            self.LAYER1_NEURONS_enet)                # 次のレイヤーの出力ユニット数

        # 隠れ層：2つ目のレイヤー（layer）
        self.Enet_layer2 = nn.Linear(
            self.LAYER1_NEURONS_enet,                # 入力ユニット数
            self.LAYER2_NEURONS_enet)                # 次のレイヤーの出力ユニット数
        
        # 出力層
        self.Enet_layer_out = nn.Linear(
            self.LAYER2_NEURONS_enet,                # 入力ユニット数
            self.OUTPUT_RESULTS_enet)                # 出力結果への出力ユニット数
        
        ##### Fitting net #####
        # 隠れ層：1つ目のレイヤー（layer）
        self.Fnet_layer1 = nn.Linear(
            self.INPUT_FEATURES_fnet,                # 入力ユニット数（＝入力層）
            self.LAYER1_NEURONS_fnet)                # 次のレイヤーの出力ユニット数
        
        # 隠れ層：2つ目のレイヤー（layer）
        self.Fnet_layer2 = nn.Linear(
            self.LAYER1_NEURONS_fnet,                # 入力ユニット数
            self.LAYER2_NEURONS_fnet)                # 次のレイヤーの出力ユニット数
        
        # 出力層
        self.Fnet_layer_out = nn.Linear(
            self.LAYER2_NEURONS_fnet,                # 入力ユニット数
            self.OUTPUT_RESULTS_fnet)                # 出力結果への出力ユニット数
        
    def forward(self, x):

        #Si(1/Rをカットオフ関数で処理した値）のみを抽出する
        Q1 = x[:,::4]
        NB = Q1.size()[0]
        N  = Q1.size()[1]
        # Embedding Netに代入する 
        embedded_x = nn.functional.leaky_relu(self.Enet_layer1(Q1))  
        embedded_x = nn.functional.leaky_relu(self.Enet_layer2(embedded_x)) 
        embedded_x = self.Enet_layer_out(embedded_x)  # ※最終層は線形 
        #embedded_xを(ミニバッチデータ数)xMxN (N=MaxAt*原子種数)に変換
        embedded_x = torch.reshape(embedded_x,(NB,self.M,N ))
        #入力データをNB x N x 4 の行列に変形  
        matQ = torch.reshape(x,(NB,N,4))
        #Enetの出力との掛け算
        matT = torch.matmul(embedded_x, matQ)
        # matTの次元はNB x M x 4 となっている 
        #matSを作る(ハイパーパラメータMbで切り詰める)
        matS = matT[:,:self.Mb,:]
        #matSの転置行列を作る　→　NB x 4 x Mb となる 
        matSt = torch.transpose(matS, 1, 2)
        #matDを作る( matTとmatStの掛け算) →　NB x M x Mb となる 
        matD = torch.matmul(matT, matSt)
        #matDを１次元化する。matD全体をニューラルネットに入力したいので、ベクトル化する。 
        matD1 = torch.reshape(matD,(NB,self.M*self.Mb))
        # fitting Net に代入する 
        fitD = nn.functional.leaky_relu(self.Fnet_layer1(matD1))
        fitD = nn.functional.leaky_relu(self.Fnet_layer2(fitD)) 
        fitD = self.Fnet_layer_out(fitD)  # ※最終層は線形 
        # fitDの次元はNB x M となる。これをNB x 1 x Mの行列にする
        fitD3 = torch.reshape(fitD,(NB,1,self.M))
        # fttD3とmatTの掛け算 
        matW = torch.matmul(fitD3, matT) 
        # matWはNb x 1 x  4 になっている。これをNB x 4 の2次元にする
        matW2 = torch.reshape(matW,(NB,4))
        # はじめの要素はいらないので、切り詰めてx,y,z にする
        outW = matW2[:,1:]
        
        return outW


# * ここから予測させる，すなわちここからデータをロードして並列化
# def predict_dipole_mode1(fr,desc_dir):
#     #
#     # * 機械学習用のデータを読み込む
#     # *
#     #
#     global model_ch_2
#     global model_co_2
#     global model_oh_2
#     global model_o_2

#     # デバイスの設定    
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     nfeatures = 288

#     # ring
#     # descs_X_ring = np.loadtxt('Descs_ring.csv',delimiter=',')
#     # data_y_ring = np.loadtxt('data_y_ring.csv',delimiter=',')

#     # CHボンド，COボンド，OHボンド，Oローンペア
#     descs_X_ch = np.loadtxt(desc_dir+'Descs_ch_'+str(fr)+'.csv',delimiter=',')
#     descs_X_co = np.loadtxt(desc_dir+'Descs_co_'+str(fr)+'.csv',delimiter=',')
#     descs_X_oh = np.loadtxt(desc_dir+'Descs_oh_'+str(fr)+'.csv',delimiter=',')
#     descs_X_o =  np.loadtxt(desc_dir+'Descs_o_'+str(fr)+'.csv',delimiter=',')

#     # オリジナルの記述子を一旦tensorへ
#     X_ch = torch.from_numpy(descs_X_ch.astype(np.float32)).clone()
#     X_oh = torch.from_numpy(descs_X_oh.astype(np.float32)).clone()
#     X_co = torch.from_numpy(descs_X_co.astype(np.float32)).clone()
#     X_o  = torch.from_numpy(descs_X_o.astype(np.float32)).clone()

#     # 予測
#     y_pred_ch  = model_ch_2(X_ch.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
#     y_pred_co  = model_co_2(X_co.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
#     y_pred_oh  = model_oh_2(X_oh.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
#     y_pred_o   = model_o_2(X_o.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()

#     # 最後にreshape
#     # !! ここは形としては(NUM_MOL*len(bond_index),3)となるが，予測だけする場合NUM_MOLの情報をgetできないので
#     # 1! reshape(-1,3)としてしまう．
    
#     # TODO : hard code (分子数)
#     # NUM_MOL = 64
#     y_pred_ch = y_pred_ch.reshape((-1,3))
#     y_pred_co = y_pred_co.reshape((-1,3))
#     y_pred_oh = y_pred_oh.reshape((-1,3))
#     y_pred_o  = y_pred_o.reshape((-1,3))
#     print("DEBUG :: shape ch/co/oh/o :: {0} {1} {2} {3}".format(np.shape(y_pred_ch),np.shape(y_pred_co),np.shape(y_pred_oh),np.shape(y_pred_o)))
#     if fr == 0:
#         print("y_pred_ch ::", y_pred_ch)
#         print("y_pred_co ::", y_pred_co)
#         print("y_pred_oh ::", y_pred_oh)
#         print("y_pred_o  ::", y_pred_o)
#         #予測したモデルを使ったUnit Cellの双極子モーメントの計算
#     sum_dipole=np.sum(y_pred_ch,axis=0)+np.sum(y_pred_oh,axis=0)+np.sum(y_pred_co,axis=0)+np.sum(y_pred_o,axis=0)
#     return sum_dipole


def calc_descripter_frame_and_predict_dipole(atoms_fr, fr, itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS,model_ch_2,model_oh_2,model_cc_2,model_co_2,model_o_2):
    
    '''
    機械学習での予測：あり
    ワニエのアサイン：なし
    '''
    import cpmd.descripter
    import cpmd.asign_wcs
    # * wannierの割り当て部分のメソッド化
    ASIGN=cpmd.asign_wcs.asign_wcs(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)
    DESC=cpmd.descripter.descripter(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)
    
    # * 原子座標とボンドセンターの計算
    # 原子座標,ボンドセンターを分子基準で再計算
    results = ASIGN.aseatom_to_mol_coord_bc(atoms_fr, itp_data, itp_data.bonds_list)
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
    X_cc = torch.from_numpy(Descs_cc.astype(np.float32)).clone()
    X_o  = torch.from_numpy(Descs_o.astype(np.float32)).clone()

    #
    # global model_ch_2
    # global model_co_2
    # global model_oh_2
    # global model_o_2
    # global model_cc_2

    print(" == DEBUG in a function == ")
    print("model_ch_2 :: {}".format(model_ch_2))

    # デバイスの設定    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    nfeatures = 288

    # 予測
    if model_ch_2 != None: y_pred_ch  = model_ch_2(X_ch.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
    if model_co_2 != None: y_pred_co  = model_co_2(X_co.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
    if model_oh_2 != None: y_pred_oh  = model_oh_2(X_oh.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
    if model_cc_2 != None: y_pred_cc  = model_cc_2(X_cc.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
    if model_o_2  != None: y_pred_o   = model_o_2(X_o.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()

    # 最後にreshape
    # !! ここは形としては(NUM_MOL*len(bond_index),3)となるが，予測だけする場合NUM_MOLの情報をgetできないので
    # !! reshape(-1,3)としてしまう．

    # TODO : hard code (分子数)
    # NUM_MOL = 64
    if model_ch_2 != None: y_pred_ch = y_pred_ch.reshape((-1,3))
    if model_co_2 != None: y_pred_co = y_pred_co.reshape((-1,3))
    if model_oh_2 != None: y_pred_oh = y_pred_oh.reshape((-1,3))
    if model_cc_2 != None: y_pred_cc = y_pred_cc.reshape((-1,3))    
    if model_o_2  != None: y_pred_o  = y_pred_o.reshape((-1,3))
    
    # print("DEBUG :: shape ch/co/oh/o :: {0} {1} {2} {3}".format(np.shape(y_pred_ch),np.shape(y_pred_co),np.shape(y_pred_oh),np.shape(y_pred_o)))
    if fr == 0: # デバッグ用
        print("y_pred_ch ::", y_pred_ch)
        print("y_pred_co ::", y_pred_co)
        print("y_pred_oh ::", y_pred_oh)
        print("y_pred_o  ::", y_pred_o)
    #予測したモデルを使ったUnit Cellの双極子モーメントの計算
    sum_dipole=np.sum(y_pred_ch,axis=0)+np.sum(y_pred_oh,axis=0)+np.sum(y_pred_co,axis=0)+np.sum(y_pred_o,axis=0) #+np.sum(y_pred_cc,axis=0)
    
    return sum_dipole


def main():
    import ml.parse
    import include.small
    import os
    import time
    
    # * 1-1：コマンドライン引数の読み込み
    inputfilename=sys.argv[1]
    include.small.if_file_exist(inputfilename) # ファイルの存在確認

    inputs_list=ml.parse.read_inputfile(inputfilename)
    input_general, input_descripter, input_predict=ml.parse.locate_tag(inputs_list)
    var_gen=ml.parse.var_general(input_general)
    var_des=ml.parse.var_descripter(input_descripter)
    var_pre=ml.parse.var_predict(input_predict)

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
    # 拡張子（molかgro）に応じてread_itpかread_molかを切り替える
    if var_gen.itpfilename.endswith(".itp"):
        itp_data=ml.atomtype.read_itp(var_gen.itpfilename)
    elif var_gen.itpfilename.endswith(".mol"):
        itp_data=ml.atomtype.read_mol(var_gen.itpfilename)
    else:
        print("ERROR :: itpfilename does not end with itp or mol")
        sys.exit(1)
    bonds_list=itp_data.bonds_list
    NUM_MOL_ATOMS=itp_data.num_atoms_per_mol
    # atomic_type=itp_data.atomic_type

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
    
    import numpy as np
    import cpmd.read_traj_cpmd
    import  cpmd.asign_wcs 
    
    double_bonds_pairs = []    
    
    print(" ")
    print("             finish reading itp file                              ")
    print(" *****************************************************************")
    print(" ")
    
    
    if if_calc_descripter: # descripter計算をする場合，trajectoryを読み込む
        print(" ")
        print(" *****************************************************************")
        print("             calc_descripter:: Reading Trajectory                 ")
        print(" *****************************************************************")
        print(" ")
        # * cpu数（スレッド数）の確認 https://hawk-tech-blog.com/python-learn-count-cpu/
        print(" maximum concurrent workers :: {}".format(os.cpu_count()))
        # * OMP_NUM_THREADSを取得
        OMP_NUM_THREADS=int(os.environ['OMP_NUM_THREADS'])
        print(" OMP_NUM_THREADS :: {}".format(OMP_NUM_THREADS))
        # * trajectoryの読み込み
        # aseでデータをロードする前に，ファイルの大きさを確認して，大きすぎる場合には警告を出す
        # ファイルサイズを取得
        file_size = os.path.getsize(var_des.directory+var_des.xyzfilename)
        # byteをKB→MBに変換して小数点以下2位に四捨五入
        file_size = file_size / 1024 / 1024
        print(" input xyz file size is ... {} MB".format(file_size))
        print(" We recommend to use less than 5GB. because reading too large file consumes too much memory.")
        
        # aseでデータをロード
        # もしfilemodeがwannieronlyではない場合，wannier部分を除去する．
        if int(var_des.haswannier) == True:
            import cpmd.read_traj_cpmd
            time_start = time.time()
            traj, wannier_list=cpmd.read_traj_cpmd.raw_xyz_divide_aseatoms_list(var_des.directory+var_des.xyzfilename)
            time_end = time.time()
            print(" Finish reading trajectory via cpmd.read_traj_cpmd.raw_xyz_divide_aseatoms_list. Time is {} sec.".format(time_end-time_start))
        else:
            time_start = time.time()
            traj=ase.io.read(var_des.directory+var_des.xyzfilename,index=slice(0,None,var_des.interval))
            time_end = time.time()
            print(" Finish reading trajectory via ase.io.read. Time is {} sec.".format(time_end-time_start))
        # * traj変数の大きさを出力
        print(" Size of variable traj is ... {} KB. ".format(sys.getsizeof(traj)/1000))
            
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
        
        print(" --------  ")
        print(" NUM_ATOM  ::    ", NUM_ATOM )
        print(" NUM_CONFIG ::   ", NUM_CONFIG)
        print(" NUM_MOL    :: ",    NUM_MOL)
        print(" NUM_MOL_ATOMS :: ", NUM_MOL_ATOMS)
        print(" UNITCELL_VECTORS :: ", UNITCELL_VECTORS)
        print("total frames of trajectory:: ", frames)
        print(" --------  ")
        
        elements = {"N":7,"C":6,"O":8,"H":1}
        
        # 
        # * 結合リストの作成：二重結合だけは現状手で入れないといけない．
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
        
        # * wannierの割り当て部分のメソッド化
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
            import joblib
            # 
            # * データの保存
            # savedir = directory+"/bulk/0331test/"
            import os
            if not os.path.isdir(var_des.savedir):
                os.makedirs(var_des.savedir) # mkdir
            if var_des.step != None: # stepが決まっている場合はこちらで設定してしまう．
                print("STEP is manually set :: {}".format(var_des.step))
                traj = traj[:var_des.step]
            result = joblib.Parallel(n_jobs=-1, verbose=50)(joblib.delayed(calc_descripter_frame_descmode1)(atoms_fr,fr,var_des.savedir,itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS) for fr,atoms_fr in enumerate(traj))
            # result = joblib.Parallel(n_jobs=-1, verbose=50)(joblib.delayed(calc_descripter_frame)(atoms_fr,fr,var_des.savedir) for fr,atoms_fr in enumerate(traj))
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

            # def calc_descripter_frame(atoms_fr, wannier_fr, fr, savedir):
            #     # * 原子座標とボンドセンターの計算
            #     # 原子座標,ボンドセンターを分子基準で再計算
            #     # TODO :: ここで作った原子座標から，atomsを作り直した方が良い．
            #     # TODO :: そうしておけば後ろでatomsを使う時にmicのことを気にしなくて良い（？）ので楽かも．
            #     results = ASIGN.aseatom_to_mol_coord_bc(atoms_fr, itp_data, bonds_list)
            #     list_mol_coords, list_bond_centers =results
                
            #     # wcsをbondに割り当て，bondの双極子まで計算
            #     results_mu = ASIGN.calc_mu_bond_lonepair(wannier_fr,atoms_fr,bonds_list,itp_data,double_bonds)
            #     list_mu_bonds,list_mu_pai,list_mu_lpO,list_mu_lpN, list_bond_wfcs,list_pi_wfcs,list_lpO_wfcs,list_lpN_wfcs = results_mu
            #     # wannnierをアサインしたase.atomsを作成する
            #     mol_with_WC = cpmd.asign_wcs.make_ase_with_WCs(atoms_fr.get_atomic_numbers(),NUM_MOL, UNITCELL_VECTORS,list_mol_coords,list_bond_centers,list_bond_wfcs,list_pi_wfcs,list_lpO_wfcs,list_lpN_wfcs)
            #     # 系の全双極子を計算
            #     # print(" list_mu_bonds {0}, list_mu_pai {1}, list_mu_lpO {2}, list_mu_lpN {3}".format(np.shape(list_mu_bonds),np.shape(list_mu_pai),np.shape(list_mu_lpO),np.shape(list_mu_lpN)))
            #     # ase.io.write(savedir+"molWC_"+str(fr)+".xyz", mol_with_WC)
            #     Mtot = []
            #     for i in range(NUM_MOL):
            #         Mtot.append(np.sum(list_mu_bonds[i],axis=0)+np.sum(list_mu_pai[i],axis=0)+np.sum(list_mu_lpO[i],axis=0)+np.sum(list_mu_lpN[i],axis=0))
            #     Mtot = np.array(Mtot)
            #     #unit cellの双極子モーメントの計算
            #     total_dipole = np.sum(Mtot,axis=0)
            #     # total_dipole = np.sum(list_mu_bonds,axis=0)+np.sum(list_mu_pai,axis=0)+np.sum(list_mu_lpO,axis=0)+np.sum(list_mu_lpN,axis=0)
            #     # ワニエセンターのアサイン
            #     #ワニエ中心を各分子に帰属する
            #     # results_mu=ASIGN.calc_mu_bond(atoms_fr,results)
            #     #ワニエ中心の座標を計算する
            #     # results_wfcs = ASIGN.assign_wfc_to_mol(atoms_fr,results) 
            
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
            #             Descs_ring.append(DESC.get_desc_bondcent(atoms_fr,bond_center,mol_id))
            #             i+=1 

            #     # ch, oh, co, cc
            #     Descs_ch=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,ch_bond_index)
            #     Descs_oh=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,oh_bond_index)
            #     Descs_co=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,co_bond_index)
            #     Descs_cc=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,cc_bond_index)   
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

            #     # ring, CHボンド, CCボンド, COボンド, OHボンド, Oローンペアの記述子を保存
            #     if len(ring_bond_index) != 0: np.savetxt(savedir+'Descs_ring_'+str(fr)+'.csv', Descs_ring, delimiter=',')
            #     if len(ch_bond_index) != 0: np.savetxt(savedir+'Descs_ch_'+str(fr)+'.csv', Descs_ch, delimiter=',')
            #     if len(cc_bond_index) != 0: np.savetxt(savedir+'Descs_cc_'+str(fr)+'.csv', Descs_cc, delimiter=',')
            #     if len(co_bond_index) != 0: np.savetxt(savedir+'Descs_co_'+str(fr)+'.csv', Descs_co, delimiter=',')
            #     if len(oh_bond_index) != 0: np.savetxt(savedir+'Descs_oh_'+str(fr)+'.csv', Descs_oh, delimiter=',')
            #     if len(o_index) != 0:
            #         np.savetxt(savedir+'Descs_o_'+str(fr)+'.csv', Descs_o, delimiter=',')
            #     return mol_with_WC, total_dipole
            #     # >>>> 関数ここまで <<<<<
                
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
        print(" ")
        print(" *****************************************************************")
        print("             calc_predict :: Setting ML model                     ")
        print(" *****************************************************************")
        print(" ")

        # torch.nn.Moduleによるモデルの定義
        if var_pre.modelmode == "normal":
            print(" ------------------- ")
            print(" modelmode :: normal ")
            print(" ------------------- ")
            # TODO :: hardcode :: nfeatures :: ここはちょっと渡し方が難しいかも．
            nfeatures = 288
            print(" nfeatures :: ", nfeatures )
            
            # モデル（NeuralNetworkクラス）のインスタンス化（これは絶対に必要）
            model_ring = WFC()
            model_ch = WFC()
            model_co = WFC()
            model_cc = WFC()
            model_oh = WFC()
            model_o = WFC()

        if var_pre.modelmode == "rotate":
            print(" ------------------- ")
            print(" modelmode :: rotate ")
            print(" ------------------- ")
        
            # # モデル（NeuralNetworkクラス）のインスタンス化
            model_ring = NET()
            model_ch = NET()
            model_co = NET()
            model_cc = NET()
            model_oh = NET()
            model_o = NET()
            # <<<<<<<  if文ここまで <<<<<<<<
        try:
            from torchinfo import summary
            summary(model=model_ring)
        except ImportError:
            print("WARNING :: torchinfo is not installed. skip printing model summary.")
            print("WARNING :: This has no effect on the calculation.")
        
        # 
        # * モデルをロードする場合はこれを利用する
        
        print(" ------------------- ")
        print(" loading ML variables from modeldir ... ")
        print("")
        # model_dir="model_train40percent/"
        # model_ring.load_state_dict(torch.load('model_ring_weight.pth'))

        #GPUが使用可能か確認
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("device :: check if use GPU :: {}".format(device))
        import os
        import torch.multiprocessing as mp
        if os.path.isfile(var_pre.model_dir+'model_ch_weight4.pth'):
            model_ch.load_state_dict(torch.load(var_pre.model_dir+'model_ch_weight4.pth'))
            model_ch_2 = model_ch.to(device) # 一旦モデルをcpuへ
            print("model_ch_2 :: {}".format(model_ch_2))
            model_ch_2.share_memory() #https://knto-h.hatenablog.com/entry/2018/05/22/130745
        else:
            print("model_ch_2 is not loaded")
        if os.path.isfile(var_pre.model_dir+'model_co_weight4.pth'):
            model_co.load_state_dict(torch.load(var_pre.model_dir+'model_co_weight4.pth'))
            model_co_2 = model_co.to(device)
            print("model_co_2 :: {}".format(model_co_2))
        else:
            print("model_co_2 is not loaded")
        if os.path.isfile(var_pre.model_dir+'model_oh_weight4.pth'):
            model_oh.load_state_dict(torch.load(var_pre.model_dir+'model_oh_weight4.pth'))
            model_oh_2 = model_oh.to(device)
            print("model_oh_2 :: {}".format(model_oh_2))
        else:
            print("model_oh_2 is not loaded")
        if os.path.isfile(var_pre.model_dir+'model_cc_weight4.pth'):
            model_cc.load_state_dict(torch.load(var_pre.model_dir+'model_cc_weight4.pth'))
            model_cc_2 = model_cc.to(device)
            print("model_cc_2 :: {}".format(model_cc_2))
        else:
            model_cc_2 = None
            print("model_cc_2 is not loaded")
        if os.path.isfile(var_pre.model_dir+'model_o_weight4.pth'):
            model_o.load_state_dict(torch.load(var_pre.model_dir+'model_o_weight4.pth'))
            model_o_2  = model_o.to(device)
            print("model_o_2 :: {}".format(model_o_2))
        else:
            print("model_o_2 is not loaded")

        #
        # * 全データを再予測させる．
        # 

    if not if_calc_descripter and if_calc_predict: 
        # * ここから予測させる，すなわちここからデータをロードして並列化
        def predict_dipole_mode1(fr,desc_dir):
            #
            # * 機械学習用のデータを読み込む
            # *
            #
            # global model_ch_2
            # global model_co_2
            # global model_oh_2
            # global model_o_2
            
            # デバイスの設定    
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            nfeatures = 288 # TODO :: hard code :: ここはちょっと渡し方が難しいかも．
            
            # フラグの設定
            flag_ch = False
            flag_co = False
            flag_oh = False
            flag_o  = False
            flag_cc = False
            
            # ring
            # descs_X_ring = np.loadtxt('Descs_ring.csv',delimiter=',')
            # data_y_ring = np.loadtxt('data_y_ring.csv',delimiter=',')
            
            # CHボンド，COボンド，OHボンド，CCボンド，Oローンペア
            # !! ファイルが存在すれば読み込む&予測
            if os.path.isfile(desc_dir+'Descs_ch_'+str(fr)+'.csv'):
                flag_ch = True
                descs_X_ch = np.loadtxt(desc_dir+'Descs_ch_'+str(fr)+'.csv',delimiter=',')
                X_ch = torch.from_numpy(descs_X_ch.astype(np.float32)).clone() # オリジナルの記述子を一旦tensorへ
                y_pred_ch  = model_ch_2(X_ch.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()   # 予測
                y_pred_ch = y_pred_ch.reshape((-1,3)) # # !! ここは形としては(NUM_MOL*len(bond_index),3)となるが，予測だけする場合NUM_MOLの情報をgetできないのでreshape(-1,3)としてしまう．
                del descs_X_ch
            if os.path.isfile(desc_dir+'Descs_co_'+str(fr)+'.csv'):
                flag_co = True
                descs_X_co = np.loadtxt(desc_dir+'Descs_co_'+str(fr)+'.csv',delimiter=',')
                X_co = torch.from_numpy(descs_X_co.astype(np.float32)).clone()
                y_pred_co  = model_co_2(X_co.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
                y_pred_co = y_pred_co.reshape((-1,3))
                del descs_X_co
            if os.path.isfile(desc_dir+'Descs_oh_'+str(fr)+'.csv'):
                flag_oh = True
                descs_X_oh = np.loadtxt(desc_dir+'Descs_oh_'+str(fr)+'.csv',delimiter=',')
                X_oh = torch.from_numpy(descs_X_oh.astype(np.float32)).clone()
                y_pred_oh  = model_oh_2(X_oh.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
                y_pred_oh = y_pred_oh.reshape((-1,3))
                del descs_X_oh
            if os.path.isfile(desc_dir+'Descs_o_'+str(fr)+'.csv'):  
                flag_o = True          
                descs_X_o =  np.loadtxt(desc_dir+'Descs_o_'+str(fr)+'.csv',delimiter=',')
                X_o  = torch.from_numpy(descs_X_o.astype(np.float32)).clone()
                y_pred_o   = model_o_2(X_o.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
                y_pred_o = y_pred_o.reshape((-1,3))
                del descs_X_o
            if os.path.isfile(desc_dir+'Descs_cc_'+str(fr)+'.csv'):
                flag_cc = True
                descs_X_cc = np.loadtxt(desc_dir+'Descs_cc_'+str(fr)+'.csv',delimiter=',')
                X_cc  = torch.from_numpy(descs_X_cc.astype(np.float32)).clone()
                y_pred_cc   = model_cc_2(X_cc.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
                y_pred_cc = y_pred_cc.reshape((-1,3))
                del descs_X_cc
            
            # print("DEBUG :: shape ch/co/oh/o :: {0} {1} {2} {3}".format(np.shape(y_pred_ch),np.shape(y_pred_co),np.shape(y_pred_oh),np.shape(y_pred_o)))
            # if fr == 0:
            #     print("y_pred_ch ::", y_pred_ch)
            #     print("y_pred_co ::", y_pred_co)
            #     print("y_pred_oh ::", y_pred_oh)
            #     print("y_pred_o  ::", y_pred_o)
            #予測したモデルを使ったUnit Cellの双極子モーメントの計算（sum_dipole:np.ndarray(3)）
            # sum_dipole=np.sum(y_pred_ch,axis=0)+np.sum(y_pred_oh,axis=0)+np.sum(y_pred_co,axis=0)+np.sum(y_pred_o,axis=0)
            sum_dipole = np.zeros(3)
            if flag_ch: sum_dipole += np.sum(y_pred_ch,axis=0)
            if flag_cc: sum_dipole += np.sum(y_pred_cc,axis=0)
            if flag_oh: sum_dipole += np.sum(y_pred_oh,axis=0)
            if flag_co: sum_dipole += np.sum(y_pred_co,axis=0)
            if flag_o: sum_dipole += np.sum(y_pred_o,axis=0)
            return sum_dipole
        #     # >>>> 関数ここまで <<<<<
        #
        # * ここから予測させる，すなわちここからデータをロードして並列化
        def predict_dipole(fr,desc_dir):
            #
            # * 機械学習用のデータを読み込む
            # *
            #
            import numpy as np
            # ring
            # descs_X_ring = np.loadtxt('Descs_ring.csv',delimiter=',')
            # data_y_ring = np.loadtxt('data_y_ring.csv',delimiter=',')

            # CHボンド，COボンド，OHボンド，Oローンペア
            descs_X_ch = np.loadtxt(desc_dir+'Descs_ch_'+str(fr)+'.csv',delimiter=',')
            descs_X_co = np.loadtxt(desc_dir+'Descs_co_'+str(fr)+'.csv',delimiter=',')
            descs_X_cc = np.loadtxt(desc_dir+'Descs_cc_'+str(fr)+'.csv',delimiter=',')
            descs_X_oh = np.loadtxt(desc_dir+'Descs_oh_'+str(fr)+'.csv',delimiter=',')
            descs_X_o =  np.loadtxt(desc_dir+'Descs_o_'+str(fr)+'.csv',delimiter=',')

            # オリジナルの記述子を一旦tensorへ
            X_ch = torch.from_numpy(descs_X_ch.astype(np.float32)).clone()
            X_oh = torch.from_numpy(descs_X_oh.astype(np.float32)).clone()
            X_co = torch.from_numpy(descs_X_co.astype(np.float32)).clone()
            X_cc = torch.from_numpy(descs_X_cc.astype(np.float32)).clone()
            X_o  = torch.from_numpy(descs_X_o.astype(np.float32)).clone()

            # 予測
            y_pred_ch  = model_ch_2(X_ch.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
            y_pred_co  = model_co_2(X_co.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
            y_pred_oh  = model_oh_2(X_oh.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
            if model_cc_2 != None: y_pred_cc  = model_cc_2(X_cc.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
            y_pred_o   = model_o_2(X_o.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
        
            # 最後にreshape
            # !! ここは形としては(NUM_MOL*len(bond_index),3)となるが，予測だけする場合NUM_MOLの情報をgetできないので
            # 1! reshape(-1,3)としてしまう．
            
            # TODO : hard code (分子数)
            # NUM_MOL = 64
            y_pred_ch = y_pred_ch.reshape((-1,3))
            y_pred_co = y_pred_co.reshape((-1,3))
            y_pred_oh = y_pred_oh.reshape((-1,3))
            if model_cc_2 != None: y_pred_cc = y_pred_cc.reshape((-1,3))
            y_pred_o  = y_pred_o.reshape((-1,3))
            print("DEBUG :: shape ch/co/oh/o :: {0} {1} {2} {3}".format(np.shape(y_pred_ch),np.shape(y_pred_co),np.shape(y_pred_oh),np.shape(y_pred_o)))
            if fr == 0:
                print("y_pred_ch ::", y_pred_ch)
                print("y_pred_co ::", y_pred_co)
                print("y_pred_oh ::", y_pred_oh)
                # print("y_pred_cc ::", y_pred_cc)
                print("y_pred_o  ::", y_pred_o)
            #予測したモデルを使ったUnit Cellの双極子モーメントの計算
            sum_dipole=np.sum(y_pred_ch,axis=0)+np.sum(y_pred_oh,axis=0)+np.sum(y_pred_co,axis=0)+np.sum(y_pred_o,axis=0)+np.sum(y_pred_cc,axis=0)
            return sum_dipole
            
        import joblib
        
        # 構造の数をcsvファイルから計算する
        import os
        count_csv = 0
        for file in os.listdir(var_pre.desc_dir):
            base, ext = os.path.splitext(file)
            if ext == ".csv":
                count_csv = count_csv+1
        num_structure=int(count_csv/var_pre.bondspecies) # TODO :: hard code :: 今は4つの結合種があるのでこうしているが，本来はこれではダメ
        print(" ------------------ ")
        print("!! caution :: bondspecies :: {}".format(var_pre.bondspecies))
        print("!! caution :: num_structure :: {}".format(num_structure))
        
        # hard code :: 計算した構造の数 50001
        # result_dipole = joblib.Parallel(n_jobs=-1, verbose=50)(joblib.delayed(predict_dipole)(fr,var_pre.desc_dir) for fr in range(num_structure)) #
        result_dipole = joblib.Parallel(n_jobs=-1, verbose=50)(joblib.delayed(predict_dipole_mode1)(fr,var_pre.desc_dir) for fr in range(num_structure)) #
        import numpy as np
        result_dipole = np.array(result_dipole)
        np.save(var_pre.desc_dir+"/result_dipole.npy",result_dipole)
        return result_dipole

    # *
    # * 予測と機械学習を同時にやる場合
    # * （既に事前準備は完了しているので，最後のcalc_descripter_frameの定義だけ）
    if if_calc_descripter and if_calc_predict: 
        # * 
        # * パターン1つ目，ワニエのアサインはしないで記述子だけ作成する場合
        if var_des.descmode == "1":
            import joblib
            print(" ------------------- ")
            print(" descripter/predict/non-wannier ")
            print(" ------------------- ")
            def calc_descripter_frame_and_predict_dipole(atoms_fr, fr, itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS):
                '''
                機械学習での予測：あり
                ワニエのアサイン：なし
                '''
                import cpmd.descripter
                import cpmd.asign_wcs
                # * wannierの割り当て部分のメソッド化
                ASIGN=cpmd.asign_wcs.asign_wcs(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)
                DESC=cpmd.descripter.descripter(NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS)
                
                # * 原子座標とボンドセンターの計算
                # 原子座標,ボンドセンターを分子基準で再計算
                results = ASIGN.aseatom_to_mol_coord_bc(atoms_fr, itp_data, itp_data.bonds_list)
                list_mol_coords, list_bond_centers =results
                # BCをアサインしたase.atomsを作成する
                mol_with_BC = cpmd.asign_wcs.make_ase_with_BCs(atoms_fr.get_atomic_numbers(),NUM_MOL, UNITCELL_VECTORS,list_mol_coords,list_bond_centers)


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
                
                # print(" == DEBUG in a function == ")
                # print("model_ch_2 :: {}".format(model_ch_2))
                
                # デバイスの設定    
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                nfeatures = 288 # TODO :: hard code
                sum_dipole=np.zeros(3)
                # 予測& reshape 
                # !! ここは形としては(NUM_MOL*len(bond_index),3)となるが，予測だけする場合NUM_MOLの情報をgetできないので
                # !! reshape(-1,3)としてしまう．
                if model_ch_2 != None: 
                    Descs_ch=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.ch_bond_index)
                    X_ch = torch.from_numpy(Descs_ch.astype(np.float32)).clone() # オリジナルの記述子を一旦tensorへ
                    y_pred_ch  = model_ch_2(X_ch.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
                    y_pred_ch = y_pred_ch.reshape((-1,3))
                    del Descs_ch
                    sum_dipole += np.sum(y_pred_ch,axis=0) #双極子に加算
                if model_co_2 != None:
                    Descs_co=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.co_bond_index)
                    X_co = torch.from_numpy(Descs_co.astype(np.float32)).clone() # オリジナルの記述子を一旦tensorへ
                    y_pred_co  = model_co_2(X_co.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
                    y_pred_co = y_pred_co.reshape((-1,3))
                    del Descs_co
                    sum_dipole += np.sum(y_pred_co,axis=0)
                if model_oh_2 != None:
                    Descs_oh=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.oh_bond_index)
                    X_oh = torch.from_numpy(Descs_oh.astype(np.float32)).clone() # オリジナルの記述子を一旦tensorへ
                    y_pred_oh  = model_oh_2(X_oh.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
                    y_pred_oh = y_pred_oh.reshape((-1,3))
                    del Descs_oh
                    sum_dipole += np.sum(y_pred_oh,axis=0)
                if model_cc_2 != None:
                    Descs_cc=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.cc_bond_index)
                    X_cc = torch.from_numpy(Descs_cc.astype(np.float32)).clone() # オリジナルの記述子を一旦tensorへ
                    y_pred_cc  = model_cc_2(X_cc.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
                    y_pred_cc = y_pred_cc.reshape((-1,3))  
                    del Descs_cc
                    sum_dipole += np.sum(y_pred_cc,axis=0)
                if model_o_2  != None:
                    Descs_o = DESC.calc_lonepair_descripter_at_frame(atoms_fr,list_mol_coords, itp_data.o_list, 8)
                    X_o  = torch.from_numpy(Descs_o.astype(np.float32)).clone() # オリジナルの記述子を一旦tensorへ
                    y_pred_o   = model_o_2(X_o.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
                    y_pred_o  = y_pred_o.reshape((-1,3))
                    del Descs_o
                    sum_dipole += np.sum(y_pred_o,axis=0)
                
                # print("DEBUG :: shape ch/co/oh/o :: {0} {1} {2} {3}".format(np.shape(y_pred_ch),np.shape(y_pred_co),np.shape(y_pred_oh),np.shape(y_pred_o)))
                if fr == 0: # デバッグ用
                    print("y_pred_ch ::", y_pred_ch)
                    print("y_pred_co ::", y_pred_co)
                    print("y_pred_oh ::", y_pred_oh)
                    print("y_pred_o  ::", y_pred_o)
                #予測したモデルを使ったUnit Cellの双極子モーメントの計算
                # sum_dipole=np.sum(y_pred_ch,axis=0)+np.sum(y_pred_oh,axis=0)+np.sum(y_pred_co,axis=0)+np.sum(y_pred_o,axis=0) #+np.sum(y_pred_cc,axis=0)
                # print("sum_dipole ::", sum_dipole) # !! debug
                return mol_with_BC, sum_dipole
            #     # >>>> 関数ここまで <<<<<

            # * データの保存
            # savedir = directory+"/bulk/0331test/"
            print(" == DEBUG == ")
            print(model_ch_2)
            import os
            if not os.path.isdir(var_des.savedir):
                os.makedirs(var_des.savedir) # mkdir
            if var_des.step != None: # stepが決まっている場合はこちらで設定してしまう．
                print("STEP is manually set :: {}".format(var_des.step))
                traj = traj[:var_des.step]
            # result_dipole = joblib.Parallel(n_jobs=-1, verbose=50)(joblib.delayed(calc_descripter_frame)(atoms_fr,fr) for fr,atoms_fr in enumerate(traj))
            print(" == DEBUG before parallel ==")
            print("model_ch_2 :: {}".format(model_ch_2))
            
            # * debug versionとして，joblibを使わないパターンを作っておこう
            if not __debug__ :
                print(" debug mode for large file !! ")
                # num_trajをco_workersで分割して処理するので，繰り返し回数とあまりを計算する（mpiと同じ処理）
                cpu_size=os.cpu_count()
                ave, res = divmod(len(traj), cpu_size)
                print("ave :: {}, res :: {}".format(ave,res))
                for i in range(ave):
                    print("now we are in loop {}/i  :: {}/ave {}/res".format(i,ave,res))
                    print("len(traj[i*cpu_size:(i+1)*cpu_size]) :: {}".format(len(traj[i*cpu_size:(i+1)*cpu_size])))
                    # trajをcpu_sizeだけ読んでjoblibに渡す
                    for fr,atoms_fr in enumerate(traj[i*cpu_size:(i+1)*cpu_size]):
                        result_dipole = calc_descripter_frame_and_predict_dipole(atoms_fr,fr,itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS) 
                        print(result_dipole)
                    print(" finish save step :: {}".format(i))
                # 最後のあまりの部分
                # print(" Now starting final res part !!")
                # result_dipole = joblib.Parallel(n_jobs=-1, verbose=50)(joblib.delayed(calc_descripter_frame_and_predict_dipole)(atoms_fr,fr,itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS) for fr,atoms_fr in enumerate(traj[ave*cpu_size:]))
                # print(" np.shape(result_dipole) == ave ?? :: {}".format(np.shape(result_dipole)))
                # result_dipole = np.array(result_dipole)
                # # np.save(var_des.savedir+"/wannier_dipole.npy", result_dipole)
                # np.save(var_des.savedir+"/result_dipole_res.npy",result_dipole)
                return 0
            
            # trajの大きさによって，Parallelの挙動を変える．
            if sys.getsizeof(traj)/1000 > 0: # < 100: # 100KB以下の場合は通常のjoblibを使う．
                print(" xyz file is not very large, and we use normal joblib calculation !! ")
                # result_dipole = joblib.Parallel(n_jobs=-1, verbose=50,require='sharedmem')(joblib.delayed(calc_descripter_frame_and_predict_dipole)(atoms_fr,fr,itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS) for fr,atoms_fr in enumerate(traj))
                result = joblib.Parallel(n_jobs=OMP_NUM_THREADS, verbose=50)(joblib.delayed(calc_descripter_frame_and_predict_dipole)(atoms_fr,fr,itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS) for fr,atoms_fr in enumerate(traj))
                
                # xyzデータと双極子データを取得
                result_ase    = [i[0] for i in result]
                result_dipole = np.array([i[1] for i in result])
                # aseを保存
                ase.io.write(var_des.savedir+"/mol_BC.xyz", result_ase)
                # 双極子を保存
                np.save(var_des.savedir+"/result_dipole.npy", result_dipole)
                return 0
            else: # その他の場合，trajを分割して処理する．
                print(" xyz file is very large, and we induce different calculation type !! ")
                # num_trajをco_workersで分割して処理するので，繰り返し回数とあまりを計算する（mpiと同じ処理）
                cpu_size=os.cpu_count()*100 # 試しに100倍くらいで試してみると？
                cpu_size=OMP_NUM_THREADS*100 # cpu_countではなく，OMP_NUM_THREADSを使う．
                ave, res = divmod(len(traj), cpu_size)
                print("ave :: {}, res :: {}".format(ave,res))
                for i in range(ave):
                    print("now we are in loop {}/i  :: {}/ave {}/res".format(i,ave,res))
                    print("len(traj[i*cpu_size:(i+1)*cpu_size]) :: {}".format(len(traj[i*cpu_size:(i+1)*cpu_size])))
                    # trajをcpu_sizeだけ読んでjoblibに渡す
                    tmp_traj = traj[i*cpu_size:(i+1)*cpu_size]
                    result_dipole = joblib.Parallel(n_jobs=-1, verbose=50)(joblib.delayed(calc_descripter_frame_and_predict_dipole)(atoms_fr,fr,itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS) for fr,atoms_fr in enumerate(tmp_traj))
                    print(" result_dipole is ... {}".format(result_dipole))
                    result_dipole = np.array(result_dipole)
                    print(" result_dipole is ... {}".format(result_dipole))
                    # np.save(var_des.savedir+"/wannier_dipole.npy", result_dipole)
                    np.save(var_des.savedir+"/result_dipole_"+str(i)+".npy",result_dipole)
                    print(" finish save step :: {}".format(i))
                # 最後のあまりの部分
                print(" Now starting final res part !!")
                result_dipole = joblib.Parallel(n_jobs=-1, verbose=50)(joblib.delayed(calc_descripter_frame_and_predict_dipole)(atoms_fr,fr,itp_data, NUM_MOL,NUM_MOL_ATOMS,UNITCELL_VECTORS) for fr,atoms_fr in enumerate(traj[ave*cpu_size:]))
                print(" np.shape(result_dipole) == ave ?? :: {}".format(np.shape(result_dipole)))
                result_dipole = np.array(result_dipole)
                # np.save(var_des.savedir+"/wannier_dipole.npy", result_dipole)
                np.save(var_des.savedir+"/result_dipole_res.npy",result_dipole)
                
                return 0

        # * 
        # * パターン2つ目，ワニエのアサインもする場合
        # * descripter計算開始
        if var_des.descmode == "2":
            import joblib
            print(" ------------------- ")
            print(" descripter/predict/wannier ")
            print(" ------------------- ")
            

                #             # !! 注意 :: 実際のline count-1になっている場合があるので，roundで丸める．
                # line_count = int(float(subprocess.check_output(['wc', '-l', var_des.directory+var_des.xyzfilename]).decode().split(' ')[0]))
                # print("line_count :: {}".format(line_count))
                # nsteps = round(float(line_count/(NUM_ATOM+2))) #29 #50001 
                # print("nsteps :: {}".format(nsteps))

            
            def calc_descripter_frame(atoms_fr, wannier_fr, itp_data, NUM_MOL, NUM_MOL_ATOMS, UNITCELL_VECTORS):
                # * 原子座標とボンドセンターの計算
                # 原子座標,ボンドセンターを分子基準で再計算
                # TODO :: ここで作った原子座標から，atomsを作り直した方が良い．
                # TODO :: そうしておけば後ろでatomsを使う時にmicのことを気にしなくて良い（？）ので楽かも．
                results = ASIGN.aseatom_to_mol_coord_bc(atoms_fr, itp_data, bonds_list)
                list_mol_coords, list_bond_centers =results
                
                # wcsをbondに割り当て，bondの双極子まで計算
                results_mu = ASIGN.calc_mu_bond_lonepair(wannier_fr,atoms_fr,bonds_list,itp_data,double_bonds)
                list_mu_bonds,list_mu_pai,list_mu_lpO,list_mu_lpN, list_bond_wfcs,list_pi_wfcs,list_lpO_wfcs,list_lpN_wfcs = results_mu
                # wannnierをアサインしたase.atomsを作成する
                mol_with_WC = cpmd.asign_wcs.make_ase_with_WCs(atoms_fr.get_atomic_numbers(),NUM_MOL, UNITCELL_VECTORS,list_mol_coords,list_bond_centers,list_bond_wfcs,list_pi_wfcs,list_lpO_wfcs,list_lpN_wfcs)
                # * 系の全双極子を計算
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
            
                # デバイスの設定    
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                nfeatures = 288
                sum_dipole=np.zeros(3)


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
                # 
                # ch, oh, co, ccの計算
                if len(itp_data.ch_bond_index) != 0:
                    Descs_ch=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.ch_bond_index)
                    X_ch = torch.from_numpy(Descs_ch.astype(np.float32)).clone()
                    y_pred_ch  = model_ch_2(X_ch.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()   # 予測
                    y_pred_ch = y_pred_ch.reshape((-1,3)) # # !! ここは形としては(NUM_MOL*len(bond_index),3)となるが，予測だけする場合NUM_MOLの情報をgetできないのでreshape(-1,3)としてしまう．
                    del Descs_ch                
                    sum_dipole += np.sum(y_pred_ch,axis=0) #双極子に加算
                if len(itp_data.co_bond_index) != 0:
                    Descs_co=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.co_bond_index)
                    X_co = torch.from_numpy(Descs_co.astype(np.float32)).clone() # オリジナルの記述子を一旦tensorへ
                    y_pred_co  = model_co_2(X_co.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
                    y_pred_co = y_pred_co.reshape((-1,3))
                    del Descs_co
                    sum_dipole += np.sum(y_pred_co,axis=0)
                if len(itp_data.oh_bond_index) != 0:
                    Descs_oh=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.oh_bond_index)
                    X_oh = torch.from_numpy(Descs_oh.astype(np.float32)).clone() # オリジナルの記述子を一旦tensorへ
                    y_pred_oh  = model_oh_2(X_oh.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
                    y_pred_oh = y_pred_oh.reshape((-1,3))
                    del Descs_oh
                    sum_dipole += np.sum(y_pred_oh,axis=0)
                if len(itp_data.cc_bond_index) != 0:
                    Descs_cc=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,itp_data.cc_bond_index)
                    X_cc = torch.from_numpy(Descs_cc.astype(np.float32)).clone() # オリジナルの記述子を一旦tensorへ
                    y_pred_cc  = model_cc_2(X_cc.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
                    y_pred_cc = y_pred_cc.reshape((-1,3))  
                    del Descs_cc
                    sum_dipole += np.sum(y_pred_cc,axis=0)
                if len(itp_data.o_list) != 0:
                    Descs_o = DESC.calc_lonepair_descripter_at_frame(atoms_fr,list_mol_coords, itp_data.o_list, 8)
                    X_o  = torch.from_numpy(Descs_o.astype(np.float32)).clone() # オリジナルの記述子を一旦tensorへ
                    y_pred_o   = model_o_2(X_o.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
                    y_pred_o  = y_pred_o.reshape((-1,3))
                    del Descs_o
                    sum_dipole += np.sum(y_pred_o,axis=0)

                # Descs_ch=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,ch_bond_index)
                # Descs_oh=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,oh_bond_index)
                # Descs_co=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,co_bond_index)
                # Descs_cc=DESC.calc_bond_descripter_at_frame(atoms_fr,list_bond_centers,cc_bond_index)   
                # # oローンペア
                # Descs_o = DESC.calc_lonepair_descripter_at_frame(atoms_fr,list_mol_coords, o_index, 8)

                # # データが作成できているかの確認（debug）
                # # print( " DESCRIPTOR SHAPE ")
                # # print(" ring (Descs/data) ::", Descs_ring.shape)
                # # print(" ch-bond (Descs/data) ::", Descs_ch.shape)
                # # print(" cc-bond (Descs/data) ::", Descs_cc.shape)
                # # print(" co-bond (Descs/data) ::", Descs_co.shape)
                # # print(" oh-bond (Descs/data) ::", Descs_oh.shape)
                # # print(" o-lone (Descs/data) ::", Descs_o.shape)

                # # オリジナルの記述子を一旦tensorへ
                # X_ch = torch.from_numpy(Descs_ch.astype(np.float32)).clone()
                # X_oh = torch.from_numpy(Descs_oh.astype(np.float32)).clone()
                # X_co = torch.from_numpy(Descs_co.astype(np.float32)).clone()
                # X_cc = torch.from_numpy(Descs_cc.astype(np.float32)).clone()
                # X_o  = torch.from_numpy(Descs_o.astype(np.float32)).clone()

                # # 予測
                # y_pred_ch  = model_ch_2(X_ch.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
                # y_pred_co  = model_co_2(X_co.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
                # y_pred_cc  = model_cc_2(X_cc.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
                # y_pred_oh  = model_oh_2(X_oh.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
                # y_pred_o   = model_o_2(X_o.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
        
                # # 最後にreshape
                # # !! ここは形としては(NUM_MOL*len(bond_index),3)となるが，予測だけする場合NUM_MOLの情報をgetできないので
                # # !! reshape(-1,3)としてしまう．
            
                # # TODO : hard code (分子数)
                # # NUM_MOL = 64
                # y_pred_ch = y_pred_ch.reshape((-1,3))
                # y_pred_co = y_pred_co.reshape((-1,3))
                # y_pred_cc = y_pred_cc.reshape((-1,3))
                # y_pred_oh = y_pred_oh.reshape((-1,3))
                # y_pred_o  = y_pred_o.reshape((-1,3))
                # # print("DEBUG :: shape ch/co/oh/o :: {0} {1} {2} {3}".format(np.shape(y_pred_ch),np.shape(y_pred_co),np.shape(y_pred_oh),np.shape(y_pred_o)))
                # if fr == 0: # デバッグ用
                #     print("y_pred_ch ::", y_pred_ch)
                #     print("y_pred_co ::", y_pred_co)
                #     print("y_pred_oh ::", y_pred_oh)
                #     print("y_pred_o  ::", y_pred_o)
                # #予測したモデルを使ったUnit Cellの双極子モーメントの計算
                # sum_dipole=np.sum(y_pred_ch,axis=0)+np.sum(y_pred_oh,axis=0)+np.sum(y_pred_co,axis=0)+np.sum(y_pred_cc,axis=0)+np.sum(y_pred_o,axis=0)

                return total_dipole, sum_dipole  #               return mol_with_WC, total_dipole
                # >>>> 関数ここまで <<<<<

            # * データの保存
            # savedir = directory+"/bulk/0331test/"
            import os
            if not os.path.isdir(var_des.savedir):
                os.makedirs(var_des.savedir) # mkdir
            
            if var_des.step != None: # stepが決まっている場合はこちらで設定してしまう．
                print("STEP is manually set :: {}".format(var_des.step))
                traj = traj[:var_des.step]
            result_dipoles = joblib.Parallel(n_jobs=-1, verbose=50)(joblib.delayed(calc_descripter_frame)(atoms_fr,wannier_fr,itp_data, NUM_MOL, NUM_MOL_ATOMS, UNITCELL_VECTORS) for atoms_fr, wannier_fr in zip(traj,wannier_list))
            print("len(result_dipoles) :: {}".format(len(result_dipoles)))
            print("len(result_dipoles[0]) :: {}".format(len(result_dipoles[0])))
            print("len(result_dipoles[1]) :: {}".format(len(result_dipoles[1])))
            
            # 双極子を保存
            wannier_dipole = [] # 真値
            result_dipole = [] # 予測値
            for i in result_dipoles:
                wannier_dipole.append(i[0])
                result_dipole.append(i[1])
            print("len(wannier_dipole) :: {}".format(len(wannier_dipole)))
            print("len(result_dipole) :: {}".format(len(result_dipole)))
            
            np.save(var_des.savedir+"/wannier_dipole.npy", wannier_dipole)
            np.save(var_des.savedir+"/result_dipole.npy",result_dipole)
            
            # atomsを保存
            return 0

if __name__ == '__main__':
    main()

    
    
