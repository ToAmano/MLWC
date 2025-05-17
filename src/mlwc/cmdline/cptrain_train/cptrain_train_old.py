# -*- coding: utf-8 -*-
"""
This module implements the training process for the ML model.
It reads the input parameters from a YAML file, loads the data,
constructs the neural network model, and trains the model.

"""

# fugaku上のpython3.8で型指定をする方法（https://future-architect.github.io/articles/20201223/）
from __future__ import annotations

import os

import ase
import ase.io
import numpy as np
import torch
import yaml

import mlwc.cmdline.cptrain_train.cptrain_train_io as cptrain_train_io
import mlwc.cpmd.class_atoms_wan
import mlwc.ml.dataset.mldataset_descs
from mlwc.cmdline.cptrain_train.cptrain_train import (
    _load_itp_data,
    _load_trajectory_data,
    _log_dataset_summary,
    _validate_xyz_with_mol,
)
from mlwc.include.mlwc_logger import setup_library_logger
from mlwc.ml.dataset.mldataset_abstract import DataSetContext
from mlwc.ml.dataset.mldataset_xyz import ConcreteFactory_xyz, ConcreteFactory_xyz_coc
from mlwc.ml.model.mlmodel_basic import NET_withoutBN
from mlwc.ml.train.ml_train import Trainer

logger = setup_library_logger("MLWC." + __name__)


def _format_name_length(name: str, width: int) -> str:
    """Formats a string to a specified width.

    If the string's length is less than or equal to the width, it is right-aligned.
    If the string's length is greater than the width, it is truncated and a prefix is added.

    Parameters
    ----------
    name : str
        The string to format.
    width : int
        The specified width.

    Returns
    -------
    str
        The formatted string.

    Examples
    --------
    >>> _format_name_length("system", 42)
    '                                      system'
    >>> _format_name_length("a_very_long_system_name", 20)
    '-- g_system_name'
    """
    if len(name) <= width:
        return "{: >{}}".format(name, width)
    else:
        name = name[-(width - 3) :]
        name = "-- " + name
        return name


def mltrain(yaml_filename: str) -> None:
    """Trains a machine learning model using the parameters specified in a YAML file.

    This function reads the YAML file, loads the model and data, and then trains the model.

    Parameters
    ----------
    yaml_filename : str
        The path to the YAML file containing the training parameters.

    Returns
    -------
    None

    Examples
    --------
    >>> mltrain("input.yaml")
    """

    # parser, args = parse_cml_args(sys.argv[1:])

    # if hasattr(args, "handler"):
    #    args.handler(args)
    # else:
    #    parser.print_help()

    #

    # * 1 :: read input yaml file
    with open(yaml_filename, encoding="utf-8") as file:
        yml = yaml.safe_load(file)
        print(yml)
    model_cfg = cptrain_train_io.VariablesModel(yml)
    train_cfg = cptrain_train_io.VariablesTraining(yml)
    data_cfg = cptrain_train_io.VariablesData(yml)

    #
    # * 2 :: load models
    # TODO :: utilize other models than NET_withoutBN
    # !! モデルは何を使っても良いが，インスタンス変数として
    # !! self.modelname
    # !! だけは絶対に指定しないといけない．chやohなどを区別するためにTrainerクラスでこの変数を利用している
    # * Construct instance of NN model (NeuralNetwork class)
    torch.manual_seed(model_cfg.seed)
    np.random.seed(model_cfg.seed)
    # TODO モデルの生成をfactoryパターンで行うように変更する
    # https://www.smartbowwow.com/2019/04/pythonsignaturefactorytips.html
    model = NET_withoutBN(
        modelname=model_cfg.modelname,
        nfeatures=model_cfg.nfeature,
        M=model_cfg.M,
        Mb=model_cfg.Mb,
        bondtype=data_cfg.bond_name,
        hidden_layers_enet=model_cfg.hidden_layers_enet,
        hidden_layers_fnet=model_cfg.hidden_layers_fnet,
        list_atomim_number=model_cfg.list_atomim_number,
        list_descriptor_length=model_cfg.list_descriptor_length,
    )

    # from torchinfo import summary
    # summary(model=model_ring)

    # * 3:: load data (xyz or descriptor)
    logger.info(" -------------------------------------- ")
    if data_cfg.type == "xyz":
        logger.info("data type :: xyz")
        # * load itp/mol file
        itp_data = _load_itp_data(data_cfg.itp_file)

        # * load trajectories
        logger.info(" Loading xyz file :: %s", data_cfg.file_list)
        _validate_xyz_with_mol(
            data_cfg.file_list, itp_data
        )  # check atomic arrangement is consistent with itp/mol files
        atoms_list = _load_trajectory_data(data_cfg.file_list)
        _log_dataset_summary(atoms_list, data_cfg, train_cfg)

        # * convert xyz to atoms_wan
        logger.info(" splitting ase.atoms into atomic ase.atoms and WCs")
        atoms_wan_list: list = []
        # for atoms in atoms_list[0]:
        for traj in atoms_list:  # loop over trajectories
            print(f" NEW TRAJ :: {len(traj)}")
            for atoms in traj:  # loop over atoms (frames)
                atoms_wan_list.append(
                    mlwc.cpmd.class_atoms_wan.atoms_wan(
                        atoms, itp_data.num_atoms_per_mol, itp_data
                    )
                )

        #
        #
        # * Assign WCs
        # TODO :: joblibでの並列化を試したが失敗した．
        # TODO :: どうもjoblibだとインスタンス変数への代入はうまくいかないっぽい．
        # TODO :: 代替案としてpytorchによる高速割り当てアルゴリズムを実装中．
        logger.info(" Assigning Wannier Centers")
        for atoms_wan_fr in atoms_wan_list:

            def y(x):
                return x._calc_wcs()

            y(atoms_wan_fr)
        logger.info(" Finish Assigning Wannier Centers")

        # save assined results
        # TODO :: 割当後のデータをより洗練されたフォーマットで保存する．
        result_atoms = []
        for atoms_wan_fr in atoms_wan_list:
            result_atoms.append(atoms_wan_fr.make_atoms_with_wc())
        ase.io.write("mol_with_WC.xyz", result_atoms)
        # save total dipole moment

        # * dataset/dataloader
        # set dataset
        # https://yiskw713.hatenablog.com/entry/2023/01/22/151940
        strategy_map = {
            "CH": [ConcreteFactory_xyz(), itp_data.bond_index["CH_1_bond"], "bond"],
            "OH": [ConcreteFactory_xyz(), itp_data.bond_index["OH_1_bond"], "bond"],
            "CO": [ConcreteFactory_xyz(), itp_data.bond_index["CO_1_bond"], "bond"],
            "CC": [ConcreteFactory_xyz(), itp_data.bond_index["CC_1_bond"], "bond"],
            "O": [ConcreteFactory_xyz(), itp_data.o_list, "lonepair"],
            "COC": [ConcreteFactory_xyz_coc(), itp_data, "coc"],
            "COH": [ConcreteFactory_xyz_coc(), itp_data, "coh"],
        }

        # loop over bondtype
        for bond_name, modeldir, modelname in zip(
            data_cfg.bond_name, train_cfg.modeldir, model_cfg.modelname
        ):
            # save input file to output directory
            with open(modeldir + "/input.yaml", "w") as f:
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

            # TODO モデルの生成(NET_withoutBN)をfactoryパターンで行うように変更する
            # https://www.smartbowwow.com/2019/04/pythonsignaturefactorytips.html
            model = NET_withoutBN(
                modelname=modelname,  # loop variable
                nfeatures=model_cfg.nfeature,
                M=model_cfg.M,
                Mb=model_cfg.Mb,
                bondtype=bond_name,  # loop variable
                hidden_layers_enet=model_cfg.hidden_layers_enet,
                hidden_layers_fnet=model_cfg.hidden_layers_fnet,
                list_atomim_number=model_cfg.list_atomim_number,
                list_descriptor_length=model_cfg.list_descriptor_length,
            )

            # make dataset
            dataset = DataSetContext(strategy).create_dataset(
                atoms_wan_list,
                calculate_bond,
                "allinone",
                Rcs=model_cfg.Rcs,
                Rc=model_cfg.Rc,
                MaxAt=24,
                bondtype=bondtype,
            )
            #
            # ここからtraining
            Train = Trainer(
                model,  # model
                # Torch device(cpu/cuda/mps)
                device=torch.device(train_cfg.device),
                # batch size for training (recommend: 32)
                batch_size=train_cfg.batch_size,
                # batch size for validation (recommend: 32)
                validation_batch_size=train_cfg.validation_batch_size,
                max_epochs=train_cfg.max_epochs,
                learning_rate=train_cfg.learning_rate,  # dict of scheduler
                # num of data （xyz frame for xyz data type/ data number for descriptor data type)
                n_train=train_cfg.n_train,
                n_val=train_cfg.n_val,
                modeldir=modeldir,  # loop variable
                restart=train_cfg.restart,
            )
            #
            # * decompose dateset into train/valid
            # note :: the numbr of train/valid data is set by n_train/n_val
            Train.set_dataset(dataset)
            # training
            Train.train()
            Train.validate_model()
            # FINISH FUNCTION

    elif data_cfg.type == "descriptor":  # calculation from descriptor
        for filename in data_cfg.file_list:
            print(f"Reading input descriptor :: {filename}_descs.npy")
            print(f"Reading input truevalues :: {filename}_true.npy")
            descs_x = np.load(filename + "_descs.npy")
            descs_y = np.load(filename + "_true.npy")

            # !! 記述子の形は，(フレーム数*ボンド数，記述子の次元数)となっている．これが前提なので注意
            print(f"shape descs_x :: {np.shape(descs_x)}")
            print(f"shape descs_y :: {np.shape(descs_y)}")
            print("Finish reading desc and true_y")
            print(f"max descs_x   :: {np.max(descs_x)}")
            #
            # * dataset/dataloader

            # make dataset
            dataset = mlwc.ml.dataset.mldataset_descs.DataSet_descs(descs_x, descs_y)

        # ここからtraining
        #
        #
        Train = mlwc.ml.train.ml_train.Trainer(
            model,  # model
            # Torch device(cpu/cuda/mps)
            device=torch.device(train_cfg.device),
            # batch size for training (recommend: 32)
            batch_size=train_cfg.batch_size,
            # batch size for validation (recommend: 32)
            validation_batch_size=train_cfg.validation_batch_size,
            max_epochs=train_cfg.max_epochs,
            learning_rate=train_cfg.learning_rate,  # dict of scheduler
            # num of data （xyz frame for xyz data type/ data number for descriptor data type)
            n_train=train_cfg.n_train,
            n_val=train_cfg.n_val,
            modeldir=train_cfg.modeldir,
            restart=train_cfg.restart,
        )

        #
        # * decompose dateset into train/valid
        # note :: the numbr of train/valid data is set by n_train/n_val
        Train.set_dataset(dataset)
        # training
        Train.train()
        # FINISH FUNCTION


def command_cptrain_train(args) -> int:
    """Wrapper function for mltrain.

    This function serves as a command-line interface wrapper for the `mltrain` function.
    It takes command-line arguments, passes the input YAML file to `mltrain`, and returns 0 upon completion.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments. It must contain `input` attribute that specifies the path to the YAML file.

    Returns
    -------
    int
        0 upon successful completion.

    Examples
    --------
    >>> # Assuming args.input is "input.yaml"
    >>> command_cptrain_train(args)
    0
    """
    mltrain(args.input)
    return 0
