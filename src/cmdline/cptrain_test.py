#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations # fugaku上のpython3.8で型指定をする方法（https://future-architect.github.io/articles/20201223/）

import argparse
import sys
import numpy as np
import argparse
import sys
import os
# import matplotlib.pyplot as plt


# python version check
from include.small import python_version_check
python_version_check()


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


# def mltrain(yaml_filename:str)->None:

#     # parser, args = parse_cml_args(sys.argv[1:])

#     # if hasattr(args, "handler"):
#     #    args.handler(args)
#     #else:
#     #    parser.print_help()
    
#     #
#     #* logging levelの設定
#     #* Trainerクラス内ではloggingを使って出力しているので必須

#     import sys

#     # INFO以上のlogを出力
#     # set_up_script_logger(None, verbose="INFO")

#     #
#     # * モデルのロード（NET_withoutBNは従来通りのモデル）
#     # !! モデルは何を使っても良いが，インスタンス変数として
#     # !! self.modelname
#     # !! だけは絶対に指定しないといけない．chやohなどを区別するためにTrainerクラスでこの変数を利用している
#     import ml.mlmodel
#     import importlib
#     importlib.reload(ml.mlmodel)

#     # *  モデル（NeuralNetworkクラス）のインスタンス化
#     model = ml.mlmodel.NET_withoutBN(input_model.modelname, input_model.nfeature, input_model.M, input_model.Mb)


#     #from torchinfo import summary
#     #summary(model=model_ring)

#     #
#     # * データ（記述子と真の双極子）をload
#     import numpy as np
#     for filename in input_data.file_list:
#         print(f"Reading input descriptor :: {filename}_descs.npy")
#         print(f"Reading input truevalues :: {filename}_true.npy")
#         descs_x = np.load(filename+"_descs.npy")
#         descs_y = np.load(filename+"_true.npy")
    
#     # 記述子の形は，(フレーム数*ボンド数，記述子の次元数)となっている．これが前提なので注意
#     print(f"shape descs_x :: {np.shape(descs_x)}")
#     print(f"shape descs_y :: {np.shape(descs_y)}")
#     print("Finish reading desc and true_y")
#     print(f"max descs_x   :: {np.max(descs_x)}")


#     #
#     # * データセットの作成およびデータローダの設定

#     import importlib
#     import ml.ml_dataset
#     importlib.reload(ml.ml_dataset)

#     # make dataset
#     dataset = ml.ml_dataset.DataSet_custom(descs_x,descs_y)


#     #
#     # * 訓練用クラスのimport
#     import ml.ml_train
#     import importlib
#     importlib.reload(ml.ml_train)

#     #
#     # TODO :: schedulerの実装がまだできておらず，learning rateは固定値しか受け付けない．
#     Train = ml.ml_train.Trainer(
#         model,  # モデルの指定
#         device     = input_train.device,   # Torchのdevice
#         batch_size = input_train.batch_size,  # 訓練のバッチサイズ
#         validation_batch_size = input_train.validation_batch_size, # validationのバッチサイズ
#         max_epochs    = input_train.max_epochs,
#         learning_rate = input_train.learning_rate, # starting learning rate
#         n_train       = input_train.n_train, # データ数（xyzのフレーム数ではないので注意．純粋なデータ数）
#         n_val         = input_train.n_val,
#         modeldir      = input_train.modeldir,
#         restart       = input_train.restart)

#     #
#     # * データをtrain/validで分割
#     # note :: 分割数はn_trainとn_valでTrainer引数として指定
#     Train.set_dataset(dataset)
#     # DEEPMD INFO    -----------------------------------------------------------------
#     # DEEPMD INFO    ---Summary of DataSystem: training     ----------------------------------
#     # DEEPMD INFO    found 1 system(s):
#     # DEEPMD INFO                                 system  natoms  bch_sz   n_bch   prob  pbc
#     # DEEPMD INFO               ../00.data/training_data       5       7      23  1.000    T
#     # DEEPMD INFO    -------------------------------------------------------------------------
#     # DEEPMD INFO    ---Summary of DataSystem: validation   ----------------------------------
#     # DEEPMD INFO    found 1 system(s):
#     # DEEPMD INFO                                 system  natoms  bch_sz   n_bch   prob  pbc
#     # DEEPMD INFO             ../00.data/validation_data       5       7       5  1.000    T
#     # DEEPMD INFO    -------------------------------------------------------------------------
#     # training
#     Train.train()


def mltest(model_filename:str, xyz_filename:str, itp_filename:str)->None:
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
    
    # * itpデータの読み込み
    # note :: itpファイルは記述子からデータを読み込む場合は不要なのでコメントアウトしておく
    import ml.atomtype
    # 実際の読み込み
    itp_data=ml.atomtype.read_itp(itp_filename)
    bonds_list=itp_data.bonds_list
    NUM_MOL_ATOMS=itp_data.num_atoms_per_mol
    atomic_type=itp_data.atomic_type
    
    
    # * 検証用トラジェクトリファイルのロード
    import ase
    import ase.io
    atoms_list = ase.io.read(xyz_filename,index=":")
    
    # * xyzからatoms_wanクラスを作成する．
    # note :: datasetから分離している理由は，wannierの割り当てを並列計算でやりたいため．
    import importlib
    import cpmd.class_atoms_wan 
    importlib.reload(cpmd.class_atoms_wan)

    atoms_wan_list = []
    for atoms in atoms_list:
        atoms_wan_list.append(cpmd.class_atoms_wan.atoms_wan(atoms,NUM_MOL_ATOMS,itp_data))
        
    # 
    # 
    # * まずwannierの割り当てを行う．
    # TODO :: joblibでの並列化を試したが失敗した．
    # TODO :: どうもjoblibだとインスタンス変数への代入はうまくいかないっぽい．
    for atoms_wan_fr in atoms_wan_list:
        y = lambda x:x._calc_wcs()
        y(atoms_wan_fr)
        
    # atoms_wan_fr._calc_wcs() for atoms_wan_fr in atoms_wan_list
    
    
    # * データセットの作成およびデータローダの設定
    import importlib
    import ml.ml_dataset 
    importlib.reload(ml.ml_dataset)
    # make dataset
    # 第二変数で訓練したいボンドのインデックスを指定する．
    # 第三変数は記述子のタイプを表す
    # !! TODO :: hard code :: itp_data.ch_bond_index
    dataset_ch = ml.ml_dataset.DataSet_xyz(atoms_wan_list, itp_data.ch_bond_index,"allinone")

    # データローダーの定義
    dataloader_valid = torch.utils.data.DataLoader(dataset_ch, batch_size=32, shuffle=False,drop_last=False, pin_memory=True, num_workers=0)
    
    # pred, trueのリストを作成
    pred_true_list = []
    
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
            if data[0].dim() == 2: # 2次元の場合はそのまま
                # self.batch_step(data,validation=True)
                x = data_1[0]
                y = data_1[1]
                y_pred = model(x)
            # lossを計算?
            np_loss = np.sqrt(np.mean((y_pred.to("cpu").detach().numpy()-y.detach().numpy())**2))  #損失のroot，RSMEと同じ
            # append results
            pred_true_list.append([y_pred.to("cpu").detach().numpy(),y.detach().numpy()])
    # save results
    print(" ======")
    print("  Finish testing.")
    print("  Save results as pred_true_list.txt")
    print(" ")
    np.savetxt("pred_true_list.txt",np.array(pred_true_list))    
    return 0


def command_mltrain_test(args)-> int:
    """mltrain train 
        wrapper for mltrain
    Args:
        args (_type_): _description_
    """
    mltest(args.input)
    return 0
