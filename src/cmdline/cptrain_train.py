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



def set_up_script_logger(logfile: str, verbose: str = "INFO"):
    import logging
    # Configure the root logger so stuff gets printed
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [
        logging.StreamHandler(sys.stderr),
        logging.StreamHandler(sys.stdout),
    ]
    level = getattr(logging, verbose.upper())
    root_logger.handlers[0].setLevel(level)
    root_logger.handlers[1].setLevel(logging.INFO)
    if logfile is not None:
        root_logger.addHandler(logging.FileHandler(logfile, mode="w"))
        root_logger.handlers[-1].setLevel(level)
    return root_logger



class variables_model:
    def __init__(self,yml:dict) -> None:
        # parse yaml files1: model
        self.modelname:str = yml["model"]["modelname"]
        self.nfeature:int  = int(yml["model"]["nfeature"])
        self.M:int         = int(yml["model"]["M"])
        self.Mb:int        = int(yml["model"]["Mb"])

class variables_data:
    def __init__(self,yml:dict) -> None:
        # parse yaml files1: model
        self.type      = yml["data"]["type"]
        self.file_list = yml["data"]["file"]

class variables_training:
    def __init__(self,yml:dict) -> None:        
        # parse yaml 2: training
        self.device     = yml["training"]["device"]   # Torchのdevice
        self.batch_size:int             = int(yml["training"]["batch_size"])  # 訓練のバッチサイズ
        self.validation_batch_size:int  = int(yml["training"]["validation_batch_size"]) # validationのバッチサイズ
        self.max_epochs:int             = int(yml["training"]["max_epochs"])
        self.learning_rate:float        = float(yml["training"]["learning_rate"]) # starting learning rate
        self.n_train:int                = int(yml["training"]["n_train"]) # データ数（xyzのフレーム数ではないので注意．純粋なデータ数）
        self.n_val:int                  = int(yml["training"]["n_val"])
        self.modeldir              = yml["training"]["modeldir"]
        self.restart               = yml["training"]["restart"]


def mltrain(yaml_filename:str)->None:

    # parser, args = parse_cml_args(sys.argv[1:])

    # if hasattr(args, "handler"):
    #    args.handler(args)
    #else:
    #    parser.print_help()
    
    #
    #* logging levelの設定
    #* Trainerクラス内ではloggingを使って出力しているので必須

    import sys

    # INFO以上のlogを出力
    set_up_script_logger(None, verbose="INFO")

    # read input yaml file
    import yaml
    with open(yaml_filename) as file:
        yml = yaml.safe_load(file)
        print(yml)
    input_model = variables_model(yml)
    input_train = variables_training(yml)
    input_data  = variables_data(yml)
    

    #
    # * モデルのロード（NET_withoutBNは従来通りのモデル）
    # !! モデルは何を使っても良いが，インスタンス変数として
    # !! self.modelname
    # !! だけは絶対に指定しないといけない．chやohなどを区別するためにTrainerクラスでこの変数を利用している
    import ml.mlmodel
    import importlib
    importlib.reload(ml.mlmodel)

    # *  モデル（NeuralNetworkクラス）のインスタンス化
    model = ml.mlmodel.NET_withoutBN(input_model.modelname, input_model.nfeature, input_model.M, input_model.Mb)

    from torchinfo import summary
    summary(model=model)

    #from torchinfo import summary
    #summary(model=model_ring)

    #
    # * データ（記述子と真の双極子）をload
    import numpy as np
    for filename in input_data.file_list:
        print(f"Reading input descriptor :: {filename}_descs.npy")
        print(f"Reading input truevalues :: {filename}_true.npy")
        descs_x = np.load(filename+"_descs.npy")
        descs_y = np.load(filename+"_true.npy")
    
    # 記述子の形は，(フレーム数*ボンド数，記述子の次元数)となっている．これが前提なので注意
    print(f"shape descs_x :: {np.shape(descs_x)}")
    print(f"shape descs_y :: {np.shape(descs_y)}")
    print("Finish reading desc and true_y")
    print(f"max descs_x   :: {np.max(descs_x)}")


    #
    # * データセットの作成およびデータローダの設定

    import importlib
    import ml.ml_dataset
    importlib.reload(ml.ml_dataset)

    # make dataset
    dataset = ml.ml_dataset.DataSet_custom(descs_x,descs_y)


    #
    # * 訓練用クラスのimport
    import ml.ml_train
    import importlib
    importlib.reload(ml.ml_train)

    #
    # TODO :: schedulerの実装がまだできておらず，learning rateは固定値しか受け付けない．
    Train = ml.ml_train.Trainer(
        model,  # モデルの指定
        device     = input_train.device,   # Torchのdevice
        batch_size = input_train.batch_size,  # 訓練のバッチサイズ
        validation_batch_size = input_train.validation_batch_size, # validationのバッチサイズ
        max_epochs    = input_train.max_epochs,
        learning_rate = input_train.learning_rate, # starting learning rate
        n_train       = input_train.n_train, # データ数（xyzのフレーム数ではないので注意．純粋なデータ数）
        n_val         = input_train.n_val,
        modeldir      = input_train.modeldir,
        restart       = input_train.restart)

    #
    # * データをtrain/validで分割
    # note :: 分割数はn_trainとn_valでTrainer引数として指定
    Train.set_dataset(dataset)
    # DEEPMD INFO    -----------------------------------------------------------------
    # DEEPMD INFO    ---Summary of DataSystem: training     ----------------------------------
    # DEEPMD INFO    found 1 system(s):
    # DEEPMD INFO                                 system  natoms  bch_sz   n_bch   prob  pbc
    # DEEPMD INFO               ../00.data/training_data       5       7      23  1.000    T
    # DEEPMD INFO    -------------------------------------------------------------------------
    # DEEPMD INFO    ---Summary of DataSystem: validation   ----------------------------------
    # DEEPMD INFO    found 1 system(s):
    # DEEPMD INFO                                 system  natoms  bch_sz   n_bch   prob  pbc
    # DEEPMD INFO             ../00.data/validation_data       5       7       5  1.000    T
    # DEEPMD INFO    -------------------------------------------------------------------------
    # training
    Train.train()
    # FINISH FUNCTION


def command_cptrain_train(args)-> int:
    """mltrain train 
        wrapper for mltrain
    Args:
        args (_type_): _description_
    """
    mltrain(args.input)
    return 0