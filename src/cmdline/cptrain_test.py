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
import time
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
    import time
    print(" ")
    print(" --------- ")
    print(" subcommand test :: validation for ML models")
    print(" ") 
    
    # * モデルのロード ( torch scriptで読み込み)
    # https://take-tech-engineer.com/pytorch-model-save-load/
    import torch 
    # check cpu/gpu/mps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_filename).to(device)
    
    #
    print(" ==========  Model Parameter informations  ============ ")
    try:
        print(f" M         = {model.M}")
    except:
        print("The model do not contain M")
    try:           
        print(f" Mb        = {model.Mb}")
    except:
        print("The model do not contain Mb")
    try:
        print(f" nfeatures = {model.nfeatures}")
        MaxAt:int = int(model.nfeatures/4/3)
        print(f" MaxAt     = {MaxAt}")
    except:
        print("The model do not contain nfeatures")
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
    print(f" Finish loading xyz file. len(traj) = {len(atoms_list)}")
    
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
    import ml.dataset.mldataset_xyz
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
        calculate_bond = itp_data.o_list 
    elif bond_name == "COC":
        print("INVOKE COC")
    elif bond_name == "COH":
        print("INVOKE COH")
    else:
        raise ValueError(f"ERROR :: bond_name should be CH,OH,CO,CC or O {bond_name}")
        
    # set dataset
    if bond_name in ["CH", "OH", "CO", "CC"]:
        dataset = ml.dataset.mldataset_xyz.DataSet_xyz(atoms_wan_list, calculate_bond,"allinone",Rcs=4, Rc=6, MaxAt=24,bondtype="bond")
    elif bond_name == "O":
        dataset = ml.dataset.mldataset_xyz.DataSet_xyz(atoms_wan_list, calculate_bond,"allinone",Rcs=4, Rc=6, MaxAt=24,bondtype="lonepair")
    elif bond_name == "COC":        
        dataset = ml.dataset.mldataset_xyz.DataSet_xyz_coc(atoms_wan_list, itp_data,"allinone",Rcs=4, Rc=6, MaxAt=24, bondtype="coc")
    elif bond_name == "COH": 
        dataset = ml.dataset.mldataset_xyz.DataSet_xyz_coc(atoms_wan_list, itp_data,"allinone",Rcs=4, Rc=6, MaxAt=24, bondtype="coh")
    else:
        raise ValueError("ERROR :: bond_name should be CH,OH,CO,CC or O")
    
    # データローダーの定義
    # !! TODO :: hard code :: batch_size=32
    dataloader_valid = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False,drop_last=False, pin_memory=True, num_workers=0)
    
    # pred, trueのリストを作成
    pred_list = []
    true_list = []
    
    
    # * Test by models
    start_time = time.perf_counter() # start time check
    model.eval() # model to evaluation mode
    with torch.no_grad(): # https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
        for data in dataloader_valid:
            # self.logger.debug("start batch valid")
            if data[0].dim() == 3: # 3次元の場合[NUM_BATCH,NUM_BOND,288]はデータを整形する
                # TODO :: torch.reshape(data[0], (-1, 288)) does not work !!
                for data_1 in zip(data[0],data[1]):
                    # self.logger.debug(f" DEBUG :: data_1[0].shape = {data_1[0].shape} : data_1[1].shape = {data_1[1].shape}")
                    # self.batch_step(data_1,validation=True)
                    x = data_1[0].to(device) # modve descriptor to device
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
    end_time = time.perf_counter() #計測終了
    # RSMEを計算する
    rmse = np.sqrt(np.mean((true_list-pred_list)**2))
    from sklearn.metrics import r2_score
    # save results
    print(" ======")
    print("  Finish testing.")
    print("  Save results as pred_true_list.txt")
    print(f" RSME_train = {rmse}")
    print(f' r^2        = {r2_score(true_list,pred_list)}')
    print(" ")
    print(' ELAPSED TIME  {:.2f}'.format((end_time-start_time))) 
    print(np.shape(pred_list))
    print(np.shape(true_list))
    np.savetxt("pred_list.txt",pred_list)
    np.savetxt("true_list.txt",true_list)
    # make figures
    import ml.ml_train
    ml.ml_train.make_figure(pred_list,true_list)
    ml.ml_train.plot_residure_density(pred_list, true_list)
    return 0

def command_cptrain_test(args)-> int:
    """mltrain train 
        wrapper for mltrain
    Args:
        args (_type_): _description_
    """
    mltest(args.model,args.xyz,args.mol, args.bond)
    return 0
