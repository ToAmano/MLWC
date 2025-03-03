# -*- coding: utf-8 -*-
from __future__ import annotations # fugaku上のpython3.8で型指定をする方法（https://future-architect.github.io/articles/20201223/）

import numpy as np
import ase
import ase.io
import yaml
import os
import sys
import torch       

import argparse
from ase.io.trajectory import Trajectory
import ml.parse # my package
import ml.dataset.mldataset_xyz
import ml.ml_train

import cmdline.cptrain_train.cptrain_train_io as cptrain_train_io

from ml.dataset.mldataset_xyz import ConcreteFactory_xyz, ConcreteFactory_xyz_coc
from ml.dataset.mldataset_abstract import DataSetContext
from ml.model.mlmodel_basic import NET_withoutBN

from include.mlwc_logger import setup_library_logger
logger = setup_library_logger("MLWC."+__name__)


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

    # * 1 :: read input yaml file
    with open(yaml_filename) as file:
        yml = yaml.safe_load(file)
        print(yml)
    input_model = cptrain_train_io.variables_model(yml)
    input_train = cptrain_train_io.variables_training(yml)
    input_data  = cptrain_train_io.variables_data(yml)
    
    #
    # * 2 :: load models
    # TODO :: utilize other models than NET_withoutBN
    # !! モデルは何を使っても良いが，インスタンス変数として
    # !! self.modelname
    # !! だけは絶対に指定しないといけない．chやohなどを区別するためにTrainerクラスでこの変数を利用している
    # * Construct instance of NN model (NeuralNetwork class) 
    torch.manual_seed(input_model.seed)
    np.random.seed(input_model.seed)
    # TODO モデルの生成をfactoryパターンで行うように変更する
    # https://www.smartbowwow.com/2019/04/pythonsignaturefactorytips.html
    model = NET_withoutBN(
        modelname=input_model.modelname,
        nfeatures=input_model.nfeature,
        M=input_model.M,
        Mb=input_model.Mb,
        bondtype=input_data.bond_name,
        hidden_layers_enet=input_model.hidden_layers_enet,
        hidden_layers_fnet=input_model.hidden_layers_fnet,
        list_atomim_number=input_model.list_atomim_number,
        list_descriptor_length=input_model.list_descriptor_length)

    #from torchinfo import summary
    #summary(model=model_ring)

    # * 3:: load data (xyz or descriptor)
    logger.info(" -------------------------------------- ")
    if input_data.type == "xyz":
        print("data type :: xyz")
        # * itpデータの読み込み
        # note :: itpファイルは記述子からデータを読み込む場合は不要なのでコメントアウトしておく
        import ml.atomtype
        # 実際の読み込み
        if not os.path.isfile(input_data.itp_file):
            logger.error(f"ERROR :: itp file {input_data.itp_file} does not exist")
        if input_data.itp_file.endswith(".itp"):
            itp_data=ml.atomtype.read_itp(input_data.itp_file)
        elif input_data.itp_file.endswith(".mol"):
            itp_data=ml.atomtype.read_mol(input_data.itp_file)
        else:
            raise ValueError("ERROR :: itp_filename should end with .itp or .mol :: {input_data.itp_file}")
        # bonds_list=itp_data.bonds_list
        # TODO :: ここで変数を定義してるのはあまりよろしくない．
        NUM_MOL_ATOMS:int=itp_data.num_atoms_per_mol
        logger.info(f" The number of atoms in a single molecule :: {NUM_MOL_ATOMS}")
        # atomic_type=itp_data.atomic_type
        
        # * load trajectories
        logger.info(f" Loading xyz file :: {input_data.file_list}")
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
        logger.info(" Finish loading xyz file...")
        logger.info(f" The number of trajectories are {len(atoms_list)}")
        logger.info("")        
        logger.info(" ----------------------------------------------------------------------- ")
        logger.info(" -----------  Summary of training Data --------------------------------- ")
        logger.info("found %d system(s):" % len(input_data.file_list))
        logger.info(
            ("%s  " % _format_name_length("system", 42))
            + ("%6s  %6s  %6s %6s" % ("nun_frames", "batch_size", "num_batch", "natoms(include WC)"))
        )
        for xyz_filename,atoms in zip(input_data.file_list,atoms_list):
            logger.info(
                "%s  %6d  %6d  %6d %6d"
                % (
                    xyz_filename,
                    len(atoms), # num of frames
                    input_train.batch_size,
                    int(len(atoms)/input_train.batch_size),
                    len(atoms[0].get_atomic_numbers()),
                )
            )
        logger.info(
            "--------------------------------------------------------------------------------------"
        )
        
        # * convert xyz to atoms_wan 
        import cpmd.class_atoms_wan 

        logger.info(" splitting atoms into atoms and WCs")
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
        logger.info(" Assigning Wannier Centers")
        for atoms_wan_fr in atoms_wan_list:
            y = lambda x:x._calc_wcs()
            y(atoms_wan_fr)
        logger.info(" Finish Assigning Wannier Centers")
        
        # save assined results
        # TODO :: 割当後のデータをより洗練されたフォーマットで保存する．
        result_atoms = []
        for atoms_wan_fr in atoms_wan_list:
            result_atoms.append(atoms_wan_fr.make_atoms_with_wc())
        ase.io.write("mol_with_WC.xyz",result_atoms)
        # save total dipole moment
        
        # * dataset/dataloader         
        # set dataset
        # https://yiskw713.hatenablog.com/entry/2023/01/22/151940
        strategy_map = {
        "CH": [ConcreteFactory_xyz(), itp_data.bond_index['CH_1_bond'], "bond"], 
        "OH": [ConcreteFactory_xyz(), itp_data.bond_index['OH_1_bond'], "bond"],
        "CO": [ConcreteFactory_xyz(), itp_data.bond_index['CO_1_bond'], "bond"],
        "CC": [ConcreteFactory_xyz(), itp_data.bond_index['CC_1_bond'], "bond"],
        "O":  [ConcreteFactory_xyz(), itp_data.o_list, "lonepair"],
        "COC": [ConcreteFactory_xyz_coc(), itp_data, "coc"],
        "COH": [ConcreteFactory_xyz_coc(), itp_data, "coh"]
        }
        
        # loop over bondtype
        for bond_name,modeldir,modelname in zip(input_data.bond_name,input_train.modeldir,input_model.modelname):
            # save input file to output directory
            with open(modeldir+'/input.yaml','w')as f:
                yaml.dump(yml, f, default_flow_style=False, allow_unicode=True)

            # extract dataset parameters ftom input_data.bond_name(CH,OH,CO,CC,O,COC,COH)
            strategy = strategy_map.get(bond_name)[0]
            calculate_bond = strategy_map.get(bond_name)[1]
            bondtype = strategy_map.get(bond_name)[2]
            logger.info(" -----------  Summary of dataset ------------ ")
            logger.info(f"  bond_name         :: {bond_name}")
            logger.info(f"  calculate_bond    :: {calculate_bond}")
            logger.info(f"  bondtype          :: {bondtype}")
            logger.info(f"  dataset_function  :: {strategy.__class__.__name__}")
            logger.info(" -------------------------------------------- ")
            if strategy is None:
                raise ValueError(f"Unsupported bond_name: {bond_name}")

            # TODO モデルの生成をfactoryパターンで行うように変更する
            # https://www.smartbowwow.com/2019/04/pythonsignaturefactorytips.html
            model = NET_withoutBN(
                modelname=modelname, # loop variable
                nfeatures=input_model.nfeature,
                M=input_model.M,
                Mb=input_model.Mb,
                bondtype=bond_name, # loop variable
                hidden_layers_enet=input_model.hidden_layers_enet,
                hidden_layers_fnet=input_model.hidden_layers_fnet,
                list_atomim_number=input_model.list_atomim_number,
                list_descriptor_length=input_model.list_descriptor_length)

            # make dataset
            dataset = DataSetContext(strategy).create_dataset(atoms_wan_list, 
                                            calculate_bond, 
                                            "allinone",
                                            Rcs=input_model.Rcs, Rc=input_model.Rc, 
                                            MaxAt=24,bondtype=bondtype
                                            )    
            #
            # ここからtraining
            import ml.ml_train
            Train = ml.ml_train.Trainer(
                model,  # model 
                device     = torch.device(input_train.device),   # Torch device(cpu/cuda/mps)
                batch_size = input_train.batch_size,  # batch size for training (recommend: 32)
                validation_batch_size = input_train.validation_batch_size, # batch size for validation (recommend: 32)
                max_epochs    = input_train.max_epochs,
                learning_rate = input_train.learning_rate, # dict of scheduler
                n_train       = input_train.n_train, # num of data （xyz frame for xyz data type/ data number for descriptor data type)
                n_val         = input_train.n_val,
                modeldir      = modeldir, # loop variable
                restart       = input_train.restart)
            #
            # * decompose dateset into train/valid
            # note :: the numbr of train/valid data is set by n_train/n_val
            Train.set_dataset(dataset)
            # training
            Train.train()
            # FINISH FUNCTION



    elif input_data.type == "descriptor":  # calculation from descriptor 
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

        # ここからtraining
        #
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
