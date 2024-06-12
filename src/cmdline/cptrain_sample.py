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


def output_yaml()->int:
    # yamlを出力する．
    # import yaml
    
    # yml = {'member': ['name:Taro Yamada address:Hokkaido', 
    #                   'name:Ichiro Tanaka address:Tokyo', 
    #                   'name:Jiro Sato address:Okinawa']}
    # with open('test_out.yaml', 'w') as file:
    #     yaml.dump(yml, file, default_flow_style=False)
    config_yaml="""\
    model:
        modelname: test  # specify name
        nfeature:  288   # length of descriptor
        M:         20    # M  (embedding matrix size)
        Mb:        6     # Mb (embedding matrix size, smaller than M)

    learning_rate:
        type: fix

    loss:
        type: mse        # mean square error

    data:
        type: descriptor # or xyz
        file:
        - "descs_bulk/cc"
        bond_name: "CH"  # bond name (CH,CC,OH,CO)

    traininig:
        device:     cpu # Torchのdevice
        batch_size: 32  # batch size for training 
        validation_vatch_size: 32 # batch size for validation
        max_epochs: 40
        learnint_rate: 1e-2 # starting learning rate
        n_train: 2100000    # the number of training data
        n_val:     10000    # the number of validation data
        modeldir:  model_test # directory to save models
        restart:   False    # If restart training 
"""
    print(config_yaml)
    return 0

# --------------------------------
# 以下CPtrain.pyからロードする関数たち
# --------------------------------

def command_sample(args):
    # if args.type == "yaml":
    output_yaml()
    return 0

