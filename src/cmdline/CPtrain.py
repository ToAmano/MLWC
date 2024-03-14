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




def command_help(args):
    print(parser.parse_args([args.command, "--help"]))


def parse_cml_args(cml):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    
    # * ------------
    # cptrain train
    parser_train = subparsers.add_parser("train", help="train models")
    # parser_cpmd.set_defaults(handler=command_cpmd)

    # create sub-parser for sub-command cool
    # cpmd_sub_parsers = parser_train.add_subparsers(help='sub-command help')

    # 
    parser_train.add_argument("-i", "--input", \
                        help='input file name. .\n', \
                        default="train.yaml"
                        )

    parser_train.set_defaults(handler=command_mltrain_train)

    
    return parser, parser.parse_args(cml)   


def command_mltrain_train(args)-> int:
    """mltrain train 
        wrapper for mltrain
    Args:
        args (_type_): _description_
    """
    mltrain(args.input)
    return 0


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
        self.nfeature:int  = yml["model"]["nfeature"]
        self.M:int         = yml["model"]["M"]
        self.Mb:int        = yml["model"]["Mb"]

class variables_data:
    def __init__(self,yml:dict) -> None:
        # parse yaml files1: model
        self.type      = yml["data"]["descriptor"]
        self.file_list = yml["data"]["file"]

class variables_training:
    def __init__(self,yml:dict) -> None:        
        # parse yaml 2: training
        self.device     = yml["training"]["device"]   # Torchのdevice
        self.batch_size = yml["training"]["batch_size"]  # 訓練のバッチサイズ
        self.validation_batch_size = yml["training"]["validation_batch_size"] # validationのバッチサイズ
        self.max_epochs            = yml["training"]["max_epochs"]
        self.learning_rate         = yml["training"]["learning_rate"] # starting learning rate
        self.n_train               = yml["training"]["n_train"] # データ数（xyzのフレーム数ではないので注意．純粋なデータ数）
        self.n_val                 = yml["training"]["n_val"]
        self.modeldir              = yml["training"]["modeldir"]
        self.restart               = yml["training"]["restart"]


def mltrain(yaml_filename:str):

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
    # training
    Train.train()


def main():
    '''
         Simple script for plotting CP.x output
        Usage:
        $ python CPextract.py file

        For details of available options, please type
        $ python CPextract.py -h
    '''
    print(" ")
    print(" *****************************************************************")
    print("                       CPtrain.py                                 ")
    print("                       Version. 0.0.1                             ")
    print(" *****************************************************************")
    print(" ")

    parser, args = parse_cml_args(sys.argv[1:])

    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()

# 
if __name__ == '__main__':
    main()
