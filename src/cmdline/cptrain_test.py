#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations # fugaku上のpython3.8で型指定をする方法（https://future-architect.github.io/articles/20201223/）

import argparse
import sys
import numpy as np
import argparse
import sys
import os
from typing import Tuple, Set
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

import argparse
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



def command_mltrain_test(args)-> int:
    """mltrain train 
        wrapper for mltrain
    Args:
        args (_type_): _description_
    """
    mltest(args.input)
    return 0


def mltest(model_filename:str, xyz_filename:str, itp_filename:str, bond_name:str)->None:
    """_summary_
    
    Args:
        yaml_filename (str): _description_

    Returns:
        _type_: _description_
    """
    print(" ")
    print(" --------- ")
    print(" subcommand test :: validation for ML models")
    print(" ") 
    
    # * モデルのロード ( torch scriptで読み込み)
    # https://take-tech-engineer.com/pytorch-model-save-load/
    import torch       # ライブラリ「PyTorch」のtorchパッケージをインポート
    import torch.nn as nn  # 「ニューラルネットワーク」モジュールの別名定義
    model = torch.jit.load(model_filename)
    
    #
    print(" ==========  Model Parameter informations  ============ ")
    print(f" M         = {model.M}")
    print(f" Mb        = {model.Mb}")
    print(f" nfeatures = {model.nfeatures}")
    MaxAt:int = int(model.nfeatures/4/3)
    print(f" MaxAt     = {MaxAt}")
    try:
        print(f" Rcs = {model.Rcs}")
        print(f" Rc = {model.Rc}")
        print(f" type = {model.bondtype}")
        bond_name:str = model.bondtype # 上書き
        Rcs:float = model.Rcs
        Rc:float  = model.Rc
    except:
        print(" WARNING :: model is old (not include Rc, Rcs, type)")
        Rcs:float = 4.0 # default value
        Rc:float  = 6.0 # default value
    print(" ====================== ")
    
    
    # * itpデータの読み込み
    # note :: itpファイルは記述子からデータを読み込む場合は不要なのでコメントアウトしておく
    import ml.atomtype
    # 実際の読み込み
    if itp_filename.endswith(".itp"):
        itp_data=ml.atomtype.read_itp(itp_filename)
    elif itp_filename.endswith(".mol"):
        itp_data=ml.atomtype.read_mol(itp_filename)
    else:
        print("ERROR :: itp_filename should end with .itp or .mol")
    # bonds_list=itp_data.bonds_list
    NUM_MOL_ATOMS=itp_data.num_atoms_per_mol
    # atomic_type=itp_data.atomic_type
    
    
    # * 検証用トラジェクトリファイルのロード
    import ase
    import ase.io
    print(" Loading xyz file :: ",xyz_filename)
    atoms_list = ase.io.read(xyz_filename,index=":")
    
    # * xyzからatoms_wanクラスを作成する．
    # note :: datasetから分離している理由は，wannierの割り当てを並列計算でやりたいため．
    import importlib
    import cpmd.class_atoms_wan 
    importlib.reload(cpmd.class_atoms_wan)

    print(" splitting atoms into atoms and WCs")
    atoms_wan_list = []
    for atoms in atoms_list:
        atoms_wan_list.append(cpmd.class_atoms_wan.atoms_wan(atoms,NUM_MOL_ATOMS,itp_data))
        
    # 
    # 
    # * まずwannierの割り当てを行う．
    # TODO :: joblibでの並列化を試したが失敗した．
    # TODO :: どうもjoblibだとインスタンス変数への代入はうまくいかないっぽい．
    print(" Assigning Wannier Centers")
    for atoms_wan_fr in atoms_wan_list:
        y = lambda x:x._calc_wcs()
        y(atoms_wan_fr)
    print(" Finish Assigning Wannier Centers")
    
    # atoms_wan_fr._calc_wcs() for atoms_wan_fr in atoms_wan_list
    
    
    # * データセットの作成およびデータローダの設定
    import importlib
    import ml.ml_dataset 
    importlib.reload(ml.ml_dataset)
    # make dataset
    # 第二変数で訓練したいボンドのインデックスを指定する．
    # 第三変数は記述子のタイプを表す
    if bond_name == "CH":
            calculate_bond = itp_data.ch_bond_index
    elif bond_name == "OH":
            calculate_bond = itp_data.oh_bond_index
    elif bond_name == "CO":
            calculate_bond = itp_data.co_bond_index
    elif bond_name == "CC":
        calculate_bond = itp_data.cc_bond_index
    elif bond_name == "O":
        calculate_bond = itp_data.o_bond_index 
    elif bond_name == "COC":
        print("INVOKE COC")
    elif bond_name == "COH":
        print("INVOKE COH")
    else:
        raise ValueError(f"ERROR :: bond_name should be CH,OH,CO,CC or O {bond_name}")
        
    # set dataset
    if bond_name in ["CH", "OH", "CO", "CC"]:
        dataset = ml.ml_dataset.DataSet_xyz(atoms_wan_list, calculate_bond,"allinone",Rcs=4, Rc=6, MaxAt=24,bondtype="bond")
    elif bond_name == "O":
        dataset = ml.ml_dataset.DataSet_xyz(atoms_wan_list, calculate_bond,"allinone",Rcs=4, Rc=6, MaxAt=24,bondtype="lonepair")
    elif bond_name == "COC":        
        dataset = ml.ml_dataset.DataSet_xyz_coc(atoms_wan_list, itp_data,"allinone",Rcs=4, Rc=6, MaxAt=24, bondtype="coc")
    elif bond_name == "COH": 
        dataset = ml.ml_dataset.DataSet_xyz_coc(atoms_wan_list, itp_data,"allinone",Rcs=4, Rc=6, MaxAt=24, bondtype="coh")
    else:
        raise ValueError("ERROR :: bond_name should be CH,OH,CO,CC or O")
    
    # dataset = ml.ml_dataset.DataSet_xyz(atoms_wan_list, calculate_bond,"allinone",Rcs=Rcs, Rc=Rc, MaxAt=MaxAt)

    # データローダーの定義
    # !! TODO :: hard code :: batch_size=32
    dataloader_valid = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False,drop_last=False, pin_memory=True, num_workers=0)
    
    # pred, trueのリストを作成
    pred_list = []
    true_list = []
    
    # * モデルによるテスト
    model.eval() # モデルを推論モードに変更 (BN)
    with torch.no_grad(): # https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
        for data in dataloader_valid:
            # self.logger.debug("start batch valid")
            if data[0].dim() == 3: # 3次元の場合[NUM_BATCH,NUM_BOND,288]はデータを整形する
                # TODO :: torch.reshape(data[0], (-1, 288)) does not work !!
                for data_1 in zip(data[0],data[1]):
                    # self.logger.debug(f" DEBUG :: data_1[0].shape = {data_1[0].shape} : data_1[1].shape = {data_1[1].shape}")
                    # self.batch_step(data_1,validation=True)
                    x = data_1[0]
                    y = data_1[1]
                    y_pred = model(x)
                    pred_list.append(y_pred.to("cpu").detach().numpy())
                    true_list.append(y.detach().numpy())
            if data[0].dim() == 2: # 2次元の場合はそのまま
                # self.batch_step(data,validation=True)
                x = data_1[0]
                y = data_1[1]
                y_pred = model(x)
                pred_list.append(y_pred.to("cpu").detach().numpy())
                true_list.append(y.detach().numpy())
            # lossを計算?
            np_loss = np.sqrt(np.mean((y_pred.to("cpu").detach().numpy()-y.detach().numpy())**2))  #損失のroot，RSMEと同じ
    #
    pred_list = np.array(pred_list).reshape(-1,3)
    true_list = np.array(true_list).reshape(-1,3)

    # save results
    print(" ======")
    print("  Finish testing.")
    print("  Save results as pred_true_list.txt")
    print(" ")
    print(np.shape(pred_list))
    print(np.shape(true_list))
    np.savetxt("pred_list.txt",pred_list)
    np.savetxt("true_list.txt",true_list)
    # make figures
    make_figure(pred_list,true_list)
    plot_residure_density(pred_list, true_list)
    return 0

def make_figure(pred_list:np.array,true_list:np.array)->None:
    import matplotlib.pyplot as plt
    import numpy as np
    # calculate RSME
    rmse = np.sqrt(np.mean((true_list-pred_list)**2))
    print(" RSME = {0}".format(rmse))
    # plot figure
    fig, ax = plt.subplots(figsize=(8,5),tight_layout=True) # figure, axesオブジェクトを作成
    scatter1=ax.scatter(np.linalg.norm(pred_list,axis=1),np.linalg.norm(true_list,axis=1),alpha=0.2,s=5,label="RMSE={}".format(rmse))
    # 各要素で設定したい文字列の取得
    xticklabels = ax.get_xticklabels()
    yticklabels = ax.get_yticklabels()
    xlabel="ML predicted dipole [D]"
    ylabel="DFT simulated dipole [D]"
    # 各要素の設定を行うsetコマンド
    ax.set_xlabel(xlabel,fontsize=22)
    ax.set_ylabel(ylabel,fontsize=22)
    # ax.set_xlim(0,3)
    # ax.set_ylim(0,3)
    ax.grid()
    ax.tick_params(axis='x', labelsize=20 )
    ax.tick_params(axis='y', labelsize=20 )
    # ax.legend = ax.legend(*scatter.legend_elements(prop="colors"),loc="upper left", title="Ranking")
    lgnd=ax.legend(loc="upper left",fontsize=20)
    for handle in lgnd.legendHandles:
        handle.set_sizes([30])
        handle.set_alpha([1.0])
    fig.savefig("pred_true_norm.png")
    # FINISH FUNCTION


def calculate_gaussian_kde(data_x:np.array,data_y:np.array)-> Tuple[np.array, np.array, np.array]:
    """calculate gaussian kde using scipy.stats.gaussian_kde

    Args:
        data_x (np.array): _description_
        data_y (np.array): _description_

    Returns:
        np.array, np.array, np.array: _description_
    """

    # https://runtascience.hatenablog.com/entry/2021/05/06/%E3%80%90Matplotlib%E3%80%91python%E3%81%A7%E5%AF%86%E5%BA%A6%E3%83%97%E3%83%AD%E3%83%83%E3%83%88%28Density_plot%29
    from scipy.stats import gaussian_kde
    # KDE probability
    x = data_x
    y = data_y
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    # zの値で並び替え→x,yも並び替える
    idx = z.argsort() 
    x, y, z = x[idx], y[idx], z[idx]
    return x,y,z

def plot_residure_density(pred_list:np.array, true_list:np.array, limit:bool=True):
    '''
    学習結果をplotする関数．
    こちらではtrain/validの区別なく，全てのデータをまとめて，代わりにdensity mapで表示する
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    print(" ========= ")
    print(" calculate density map (takes a few minutes)")
    print(" ")
    print(" ")
    
    # RSMEを計算する
    rmse = np.sqrt(np.mean((true_list-pred_list)**2))
    print(" RSME_train = {0}".format(rmse))
    
    # matplotlibで複数のプロットをまとめる．
    # https://python-academia.com/matplotlib-multiplegraphs/
    # グラフを表示する領域を，figオブジェクトとして作成。
    fig = plt.figure(figsize = (15,5), facecolor='lightblue')
    
    #グラフを描画するsubplot領域を作成。
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    
    #各subplot領域にデータを渡す
    # TODO :: RSME，決定係数Rを同時に表示するようにする

    # KDE probability
    x,y,z = calculate_gaussian_kde(pred_list[:,0], true_list[:,0])
    im = ax1.scatter(x, y, c=z, s=50, cmap="jet")
    fig.colorbar(im)

    x,y,z = calculate_gaussian_kde(pred_list[:,1], true_list[:,1])
    im = ax2.scatter(x, y, c=z, s=50, cmap="jet")
    fig.colorbar(im)

    x,y,z = calculate_gaussian_kde(pred_list[:,2], true_list[:,2])
    im = ax3.scatter(x, y, c=z, s=50, cmap="jet")
    fig.colorbar(im)

    #タイトル
    ax1.set_title("Dipole_x")
    ax2.set_title("Dipole_y")
    ax3.set_title("Dipole_z")

    #各subplotにxラベルを追加
    ax1.set_xlabel("ML dipole [D]")
    ax2.set_xlabel("ML dipole [D]")
    ax3.set_xlabel("ML dipole [D]")

    ax1.set_ylabel("DFT dipole [D]")
    ax2.set_ylabel("DFT dipole [D]")
    ax3.set_ylabel("DFT dipole [D]")

    # 凡例表示
    ax1.legend(loc = 'upper left') 
    ax2.legend(loc = 'upper left') 
    ax3.legend(loc = 'upper left') 

    # grid表示
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    fig.savefig("pred_true_density.png")
    return 0

def command_cptrain_test(args)-> int:
    """mltrain train 
        wrapper for mltrain
    Args:
        args (_type_): _description_
    """
    mltest(args.model,args.xyz,args.mol, args.bond)
    return 0
