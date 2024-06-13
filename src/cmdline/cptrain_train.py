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



def set_up_script_logger(logfile: str, verbose: str = "CRITICAL"):
    """_summary_
    No 
    -----
    Logging levels:

    +---------+--------------+----------------+----------------+----------------+
    |         | our notation | python logging | tensorflow cpp | OpenMP         |
    +=========+==============+================+================+================+
    | debug   | 10           | 10             | 0              | 1/on/true/yes  |
    +---------+--------------+----------------+----------------+----------------+
    | info    | 20           | 20             | 1              | 0/off/false/no |
    +---------+--------------+----------------+----------------+----------------+
    | warning | 30           | 30             | 2              | 0/off/false/no |
    +---------+--------------+----------------+----------------+----------------+
    | error   | 40           | 40             | 3              | 0/off/false/no |
    +---------+--------------+----------------+----------------+----------------+
    Args:
        logfile (str): _description_
        verbose (str, optional): _description_. Defaults to "CRITICAL".

    Returns:
        _type_: _description_
    """
    import logging
    formatter = logging.Formatter('%(asctime)s %(name)s %(funcName)s [%(levelname)s]: %(message)s')
    # Configure the root logger so stuff gets printed
    root_logger = logging.getLogger() # root logger
    root_logger.setLevel(logging.DEBUG) # default level is INFO
    level = getattr(logging, verbose.upper())  # convert string to log level (default INFO)
    
    # setup stdout logger
    # INFO以下のログを標準出力する
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)
    
        
    # root_logger.handlers = [
    #     logging.StreamHandler(sys.stderr),
    #     logging.StreamHandler(sys.stdout),
    # ]
    # root_logger.handlers[0].setLevel(level)        # stderr
    # root_logger.handlers[1].setLevel(logging.INFO) # stdout
    if logfile is not None: # add log file
        root_logger.addHandler(logging.FileHandler(logfile, mode="w"))
        root_logger.handlers[-1].setLevel(level)
    return root_logger


def _format_name_length(name, width):
    if len(name) <= width:
        return "{: >{}}".format(name, width)
    else:
        name = name[-(width - 3) :]
        name = "-- " + name
        return name

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
        self.itp_file  = yml["data"]["itp_file"]
        self.bond_name = yml["data"]["bond_name"]
        # Validate the values
        self._validate_values()
    
    def _validate_values(self):
        if self.bond_name not in ["CH", "OH","CO","CC","O"]:
            raise ValueError("ERROR :: bond_name should be CH,OH,CO,CC or O")
    

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
    root_logger = set_up_script_logger(None, verbose="INFO")
    root_logger.info("Start logging")

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

    # * データのロード
    root_logger.info(" -------------------------------------- ")
    if input_data.type == "xyz":
        print("data type :: xyz")
        # * itpデータの読み込み
        # note :: itpファイルは記述子からデータを読み込む場合は不要なのでコメントアウトしておく
        import ml.atomtype
        # 実際の読み込み
        import os
        if not os.path.isfile(input_data.itp_file):
            root_logger.error(f"ERROR :: itp file {input_data.itp_file} does not exist")
        if input_data.itp_file.endswith(".itp"):
            itp_data=ml.atomtype.read_itp(input_data.itp_file)
        elif input_data.itp_file.endswith(".mol"):
            itp_data=ml.atomtype.read_mol(input_data.itp_file)
        else:
            print("ERROR :: itp_filename should end with .itp or .mol")
        # bonds_list=itp_data.bonds_list
        NUM_MOL_ATOMS=itp_data.num_atoms_per_mol
        # atomic_type=itp_data.atomic_type
        
        # * load trajectories
        import ase
        import ase.io
        root_logger.info(f" Loading xyz file :: {input_data.file_list}")
        atoms_list = []
        for xyz_filename in input_data.file_list:
            tmp_atoms = ase.io.read(xyz_filename,index=":")
            atoms_list.append(tmp_atoms)
            print(f" len xyz == {len(tmp_atoms)}")
        root_logger.info(" Finish loading xyz file...")
        root_logger.info(f" The number of trajectories are {len(atoms_list)}")
        root_logger.info("")        
        root_logger.info(" ----------------------------------------------------------------- ")
        root_logger.info(" ---Summary of DataSystem: training     ---------------------------------- ")
        root_logger.info("found %d system(s):" % len(input_data.file_list))
        root_logger.info(
            ("%s  " % _format_name_length("system", 42))
            + ("%6s  %6s  %6s" % ("natoms", "bch_sz", "n_bch"))
        )
        for xyz_filename,atoms in zip(input_data.file_list,atoms_list):
            root_logger.info(
                "%s  %6d  %6d  %6d"
                % (
                    xyz_filename,
                    len(atoms), # num of atoms
                    input_train.batch_size,
                    int(len(atoms)/input_train.batch_size),
                )
            )
        root_logger.info(
            "--------------------------------------------------------------------------------------"
        )
        
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
        
        
        # * xyzからatoms_wanクラスを作成する．
        # note :: datasetから分離している理由は，wannierの割り当てを並列計算でやりたいため．
        import importlib
        import cpmd.class_atoms_wan 
        importlib.reload(cpmd.class_atoms_wan)

        root_logger.info(" splitting atoms into atoms and WCs")
        atoms_wan_list = []
        for atoms in atoms_list[0]: # TODO 最初のatomsのみ利用
            atoms_wan_list.append(cpmd.class_atoms_wan.atoms_wan(atoms,NUM_MOL_ATOMS,itp_data))
            
        # 
        # 
        # * まずwannierの割り当てを行う．
        # TODO :: joblibでの並列化を試したが失敗した．
        # TODO :: どうもjoblibだとインスタンス変数への代入はうまくいかないっぽい．
        root_logger.info(" Assigning Wannier Centers")
        for atoms_wan_fr in atoms_wan_list:
            y = lambda x:x._calc_wcs()
            y(atoms_wan_fr)
        root_logger.info(" Finish Assigning Wannier Centers")
        
        # TODO :: 割当後のデータを保存する．
        # atoms_wan_fr._calc_wcs() for atoms_wan_fr in atoms_wan_list
        
        
        # * データセットの作成およびデータローダの設定
        import importlib
        import ml.ml_dataset 
        importlib.reload(ml.ml_dataset)
        # make dataset
        # 第二変数で訓練したいボンドのインデックスを指定する．
        # 第三変数は記述子のタイプを表す
        if input_data.bond_name == "CH":
            calculate_bond = itp_data.ch_bond_index
        elif input_data.bond_name == "OH":
            calculate_bond = itp_data.oh_bond_index
        elif input_data.bond_name == "CO":
            calculate_bond = itp_data.co_bond_index
        elif input_data.bond_name == "CC":
            calculate_bond = itp_data.cc_bond_index
        elif input_data.bond_name == "O":
            calculate_bond = itp_data.o_bond_index 
        else:
            print("ERROR :: bond_name should be CH,OH,CO,CC or O")
            sys.exit(1) 
        
        dataset = ml.ml_dataset.DataSet_xyz(atoms_wan_list, calculate_bond,"allinone",Rcs=4, Rc=6, MaxAt=24)

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

    elif input_data.type == "descriptor":    
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
        device     = torch.device(input_train.device),   # Torchのdevice
        batch_size = input_train.batch_size,  # batch size for training (recommend: 32)
        validation_batch_size = input_train.validation_batch_size, # batch size for validation (recommend: 32)
        max_epochs    = input_train.max_epochs,
        learning_rate = input_train.learning_rate, # starting learning rate
        n_train       = input_train.n_train, # num of data （xyz frame for xyz data type/ data number for descriptor data type)
        n_val         = input_train.n_val,
        modeldir      = input_train.modeldir,
        restart       = input_train.restart)

    #
    # * データをtrain/validで分割
    # note :: 分割数はn_trainとn_valでTrainer引数として指定
    Train.set_dataset(dataset)
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