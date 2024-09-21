# -*- coding: utf-8 -*-
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
import ml.dataset.mldataset_xyz
import ml.model.mlmodel_basic

# 物理定数
from include.constants import constant
# Debye   = 3.33564e-30
# charge  = 1.602176634e-019
# ang      = 1.0e-10
coef    = constant.Ang*constant.Charge/constant.Debye

import cmdline.cptrain_train_io


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
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    if len(name) <= width:
        return "{: >{}}".format(name, width)
    else:
        name = name[-(width - 3) :]
        name = "-- " + name
        return name

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
    import numpy as np
    import ml.model.mlmodel_basic

    # INFO以上のlogを出力
    root_logger = set_up_script_logger(None, verbose="INFO")
    root_logger.info("Start logging")

    # read input yaml file
    import yaml
    with open(yaml_filename) as file:
        yml = yaml.safe_load(file)
        print(yml)
    input_model = cmdline.cptrain_train_io.variables_model(yml)
    input_train = cmdline.cptrain_train_io.variables_training(yml)
    input_data  = cmdline.cptrain_train_io.variables_data(yml)
    
    
    #
    # * load models
    # TODO :: utilize other models than NET_withoutBN
    # !! モデルは何を使っても良いが，インスタンス変数として
    # !! self.modelname
    # !! だけは絶対に指定しないといけない．chやohなどを区別するためにTrainerクラスでこの変数を利用している
    # * Construct instance of NN model (NeuralNetwork class) 
    torch.manual_seed(input_model.seed)
    np.random.seed(input_model.seed)
    model = ml.model.mlmodel_basic.NET_withoutBN(
        modelname=input_model.modelname,
        nfeatures=input_model.nfeature,
        M=input_model.M,
        Mb=input_model.Mb,
        bondtype=input_data.bond_name,
        hidden_layers_enet=input_model.hidden_layers_enet,
        hidden_layers_fnet=input_model.hidden_layers_fnet)

    #from torchinfo import summary
    #summary(model=model_ring)

    # * load data (xyz or descriptor)
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
        # TODO :: ここで変数を定義してるのはあまりよろしくない．
        NUM_MOL_ATOMS:int=itp_data.num_atoms_per_mol
        root_logger.info(f" The number of atoms in a single molecule :: {NUM_MOL_ATOMS}")
        # atomic_type=itp_data.atomic_type
        
        # * load trajectories
        import ase
        import ase.io
        root_logger.info(f" Loading xyz file :: {input_data.file_list}")
        # check atomic arrangement is consistent with itp/mol files
        for xyz_filename in input_data.file_list:
            tmp_atoms = ase.io.read(xyz_filename,index="1")
            print(tmp_atoms.get_chemical_symbols()[:NUM_MOL_ATOMS])
            if tmp_atoms.get_chemical_symbols()[:NUM_MOL_ATOMS] != itp_data.atom_list:
                raise ValueError("configuration different for xyz and itp !!")
        
        atoms_list:list = []
        for xyz_filename in input_data.file_list:
            tmp_atoms = ase.io.read(xyz_filename,index=":")
            atoms_list.append(tmp_atoms)
            print(f" len xyz == {len(tmp_atoms)}")
        root_logger.info(" Finish loading xyz file...")
        root_logger.info(f" The number of trajectories are {len(atoms_list)}")
        root_logger.info("")        
        root_logger.info(" ----------------------------------------------------------------------- ")
        root_logger.info(" -----------  Summary of training Data --------------------------------- ")
        root_logger.info("found %d system(s):" % len(input_data.file_list))
        root_logger.info(
            ("%s  " % _format_name_length("system", 42))
            + ("%6s  %6s  %6s %6s" % ("nun_frames", "batch_size", "num_batch", "natoms(include WC)"))
        )
        for xyz_filename,atoms in zip(input_data.file_list,atoms_list):
            root_logger.info(
                "%s  %6d  %6d  %6d %6d"
                % (
                    xyz_filename,
                    len(atoms), # num of frames
                    input_train.batch_size,
                    int(len(atoms)/input_train.batch_size),
                    len(atoms[0].get_atomic_numbers()),
                )
            )
        root_logger.info(
            "--------------------------------------------------------------------------------------"
        )
        
        # * convert xyz to atoms_wan 
        import cpmd.class_atoms_wan 

        root_logger.info(" splitting atoms into atoms and WCs")
        atoms_wan_list:list = []
        # for atoms in atoms_list[0]: 
        for traj in atoms_list: # loop over trajectories
            print(f" NEW TRAJ :: {len(traj)}")
            for atoms in traj: # loop over atoms
                atoms_wan_list.append(cpmd.class_atoms_wan.atoms_wan(atoms,NUM_MOL_ATOMS,itp_data))
            
        # 
        # 
        # * Assign WCs
        # TODO :: joblibでの並列化を試したが失敗した．
        # TODO :: どうもjoblibだとインスタンス変数への代入はうまくいかないっぽい．
        # TODO :: 代替案としてpytorchによる高速割り当てアルゴリズムを実装中．
        root_logger.info(" Assigning Wannier Centers")
        for atoms_wan_fr in atoms_wan_list:
            y = lambda x:x._calc_wcs()
            y(atoms_wan_fr)
        root_logger.info(" Finish Assigning Wannier Centers")
        
        # save assined results
        # TODO :: 割当後のデータをより洗練されたフォーマットで保存する．
        result_atoms = []
        for atoms_wan_fr in atoms_wan_list:
            result_atoms.append(atoms_wan_fr.make_atoms_with_wc())
        ase.io.write("mol_with_WC.xyz",result_atoms)
        # save total dipole moment
        
        
        # * dataset/dataloader 
        import ml.dataset.mldataset_xyz
        # make dataset
        if input_data.bond_name == "CH":
            calculate_bond = itp_data.ch_bond_index
        elif input_data.bond_name == "OH":
            calculate_bond = itp_data.oh_bond_index
        elif input_data.bond_name == "CO":
            calculate_bond = itp_data.co_bond_index
        elif input_data.bond_name == "CC":
            calculate_bond = itp_data.cc_bond_index
        elif input_data.bond_name == "O":
            calculate_bond = itp_data.o_list
        elif input_data.bond_name == "COC":
            print("INVOKE COC")
        elif input_data.bond_name == "COH":
            print("INVOKE COH")
        else:
            raise ValueError("ERROR :: bond_name should be CH,OH,CO,CC or O")
        
        # set dataset
        if input_data.bond_name in ["CH", "OH", "CO", "CC"]:
            dataset = ml.dataset.mldataset_xyz.DataSet_xyz(atoms_wan_list, calculate_bond,"allinone",Rcs=4, Rc=6, MaxAt=24,bondtype="bond")
        elif input_data.bond_name == "O":
            dataset = ml.dataset.mldataset_xyz.DataSet_xyz(atoms_wan_list, calculate_bond,"allinone",Rcs=4, Rc=6, MaxAt=24,bondtype="lonepair")
        elif input_data.bond_name == "COC":   
            print("INVOKE COC")     
            dataset = ml.dataset.mldataset_xyz.DataSet_xyz_coc(atoms_wan_list, itp_data,"allinone",Rcs=4, Rc=6, MaxAt=24, bondtype="coc")
        elif input_data.bond_name == "COH": 
            print("INVOKE COH")
            dataset = ml.dataset.mldataset_xyz.DataSet_xyz_coc(atoms_wan_list, itp_data,"allinone",Rcs=4, Rc=6, MaxAt=24, bondtype="coh")
        else:
            raise ValueError("ERROR :: bond_name should be CH,OH,CO,CC or O")

    elif input_data.type == "descriptor":  # calculation from descriptor 
        import numpy as np
        for filename in input_data.file_list:
            print(f"Reading input descriptor :: {filename}_descs.npy")
            print(f"Reading input truevalues :: {filename}_true.npy")
            descs_x = np.load(filename+"_descs.npy")
            descs_y = np.load(filename+"_true.npy")

            # !! 記述子の形は，(フレーム数*ボンド数，記述子の次元数)となっている．これが前提なので注意
            print(f"shape descs_x :: {np.shape(descs_x)}")
            print(f"shape descs_y :: {np.shape(descs_y)}")
            print("Finish reading desc and true_y")
            print(f"max descs_x   :: {np.max(descs_x)}")
            #
            # * dataset/dataloader
            import ml.dataset.mldataset_descs
            # make dataset
            dataset = ml.dataset.mldataset_descs.DataSet_descs(descs_x,descs_y)


    #
    import ml.ml_train
    #
    Train = ml.ml_train.Trainer(
        model,  # model 
        device     = torch.device(input_train.device),   # Torch device(cpu/cuda/mps)
        batch_size = input_train.batch_size,  # batch size for training (recommend: 32)
        validation_batch_size = input_train.validation_batch_size, # batch size for validation (recommend: 32)
        max_epochs    = input_train.max_epochs,
        learning_rate = input_train.learning_rate, # dict of scheduler
        n_train       = input_train.n_train, # num of data （xyz frame for xyz data type/ data number for descriptor data type)
        n_val         = input_train.n_val,
        modeldir      = input_train.modeldir,
        restart       = input_train.restart)

    #
    # * decompose dateset into train/valid
    # note :: the numbr of train/valid data is set by n_train/n_val
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
