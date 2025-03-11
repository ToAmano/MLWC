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



def command_mltrain_pca(args)-> int:
    """mltrain train 
        wrapper for mltrain
    Args:
        args (_type_): _description_
    """
    mlpca(args.input)
    return 0


def mlpca(model_filename:str, data:list, bond_name:str)->None:
    """_summary_
    
    Args:
        yaml_filename (str): _description_

    Returns:
        _type_: _description_
    """
    import time
    print(" ")
    print(" --------- ")
    print(" subcommand pca :: Visualize the PCA of the descriptors")
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

    # Initialize an empty list to hold all descriptors
    all_descriptors = []
    all_labels = []
    

    # * データの読み込み
    for counter,struc in enumerate(data):
        itp_filename:str = struc["mol"]
        xyz_filename:str = struc["xyz"] 
        
    
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
        import cpmd.class_atoms_wan 
        print(" splitting atoms into atoms and WCs")
        atoms_wan_list:list = []
        for atoms in atoms_list: # loop over atoms
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
                calculate_bond = itp_data.bond_index['CH_1_bond']
        elif bond_name == "OH":
                calculate_bond = itp_data.bond_index['OH_1_bond']
        elif bond_name == "CO":
                calculate_bond = itp_data.bond_index['CO_1_bond']
        elif bond_name == "CC":
            calculate_bond = itp_data.bond_index['CC_1_bond']
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
                        x = data_1[0].numpy() # modve descriptor to device
                        print(np.shape(x))
                        all_descriptors.append(x) # modve descriptor to device
                        all_labels.append(np.shape(x)[0]*[counter])
                        # y = data_1[1]
                        # y_pred = model(x)
                        # pred_list.append(y_pred.to("cpu").detach().numpy())
                        # true_list.append(y.detach().numpy())
                if data[0].dim() == 2: # 2次元の場合はそのまま [NUM_BATCH,288]
                    # self.batch_step(data,validation=True)
                    x = data_1[0].numpy()
                    print(np.shape(x))
                    all_descriptors.append(x) 
                    all_labels.append(np.shape(x)[0]*[counter])
                    # x = data_1[0]
                    # y = data_1[1]
                    # y_pred = model(x)
                    # pred_list.append(y_pred.to("cpu").detach().numpy())
                    # true_list.append(y.detach().numpy())
                # lossを計算?
                # np_loss = np.sqrt(np.mean((y_pred.to("cpu").detach().numpy()-y.detach().numpy())**2))  #損失のroot，RSMEと同じ
        #
    # リスト内のすべてのデータを結合してnumpy arrayに変換
    all_descriptors = np.concatenate(all_descriptors, axis=0)
    all_descriptors = np.array(all_descriptors).reshape(-1,model.nfeatures)
    all_labels = np.concatenate(all_labels, axis=0)
    print(all_descriptors.shape)
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler


    # https://zero2one.jp/learningblog/dimension-reduction-with-python/
    # Initialize PCA with 2 components for 2D visualization
    pca = PCA(n_components=10)
    # データの標準化
    # scaler = StandardScaler()
    
    # Fit and transform the descriptors with PCA
    pca_result = pca.fit_transform(all_descriptors)

    # 主成分軸のベクトル（重み）
    principal_components = pca.components_

    # 主成分ベクトルの形を確認
    print("Principal Components Shape:", principal_components.shape)

    # 最初の主成分ベクトルを表示
    print("Principal Component 1 (first axis):", principal_components[0])

    
    import matplotlib.pyplot as plt
    # Scatter plot of the PCA results
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=all_labels,cmap='viridis', alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Descriptors')
    plt.savefig("test_pca_12.png")
    plt.cla()

    
    import matplotlib.pyplot as plt
    # Scatter plot of the PCA results
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 2], c=all_labels,cmap='viridis', alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Descriptors')
    plt.savefig("test_pca_13.png")
    plt.cla()

    # 元の特徴量と同じ数で主成分分析
    plt.figure(figsize=(8, 6))
    plt.bar([n for n in range(1, len(pca.explained_variance_ratio_)+1)], pca.explained_variance_ratio_)
    plt.savefig("test_pca_ratio.png")
    return 0

def command_cptrain_pca(args)-> int:
    """mltrain train 
        wrapper for mltrain
    Args:
        args (_type_): _description_
    """
    # read input yaml file
    import yaml
    with open(args.input) as file:
        yml = yaml.safe_load(file)
        print(yml)
    model = yml["model"]
    bond = yml["bond"]
    data = yml["data"]
    mlpca(model,data, bond)
    return 0
