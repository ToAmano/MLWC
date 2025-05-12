# -*- coding: utf-8 -*-
# fugaku上のpython3.8で型指定をする方法（https://future-architect.github.io/articles/20201223/）
from __future__ import annotations

"""
This module implements the training process for the ML model.
It reads the input parameters from a YAML file, loads the data,
constructs the neural network model, and trains the model.

"""
import os

import ase
import ase.io
import numpy as np
import torch
import yaml

import mlwc.bond.atomtype
import mlwc.cmdline.cptrain_train.cptrain_train_io as cptrain_train_io
import mlwc.ml.dataset.mldataset_descs
from mlwc.cpmd.assign_wcs.assign_wcs_torch import (
    atoms_wan,
    calculate_molcoord,
    convert_atoms_to_bondwfc,
    extract_wcs,
)
from mlwc.cpmd.bondcenter.bondcenter import calc_bondcenter_dict
from mlwc.include.mlwc_logger import setup_library_logger
from mlwc.ml.dataset.mldataset_abstract import DataSetContext
from mlwc.ml.dataset.mldataset_atoms import ConcreteFactory_atoms, DataSet_atoms
from mlwc.ml.dataset.mldataset_xyz import ConcreteFactory_xyz, ConcreteFactory_xyz_coc
from mlwc.ml.model.mlmodel_basic import NET_withoutBN
from mlwc.ml.model.mlmodel_basic_descs import NET_withoutBN_descs
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
    with open(yaml_filename) as file:
        yml = yaml.safe_load(file)
        print(yml)
    input_model = cptrain_train_io.variables_model(yml)
    input_train = cptrain_train_io.variables_training(yml)
    input_data = cptrain_train_io.variables_data(yml)

    #
    # * 2 :: load models
    # TODO :: utilize other models than NET_withoutBN
    # !! モデルは何を使っても良いが，インスタンス変数として
    # !! self.modelname
    # !! だけは絶対に指定しないといけない．chやohなどを区別するためにTrainerクラスでこの変数を利用している
    # * Construct instance of NN model (NeuralNetwork class)
    torch.manual_seed(input_model.seed)
    np.random.seed(input_model.seed)

    # * 3:: load data (xyz or descriptor)
    logger.info(" -------------------------------------- ")
    if input_data.type == "xyz":
        print("data type :: xyz")
        # * itpデータの読み込み
        # note :: itpファイルは記述子からデータを読み込む場合は不要なのでコメントアウトしておく

        # 実際の読み込み
        if not os.path.isfile(input_data.itp_file):
            logger.error(f"ERROR :: itp file {input_data.itp_file} does not exist")
        if input_data.itp_file.endswith(".itp"):
            itp_data = mlwc.bond.atomtype.read_itp(input_data.itp_file)
        elif input_data.itp_file.endswith(".mol"):
            itp_data = mlwc.bond.atomtype.ReadMolFile(input_data.itp_file)
        else:
            raise ValueError(
                "ERROR :: itp_filename should end with .itp or .mol :: {input_data.itp_file}"
            )
        # bonds_list=itp_data.bonds_list
        # TODO :: ここで変数を定義してるのはあまりよろしくない．
        NUM_MOL_ATOMS: int = itp_data.num_atoms_per_mol
        logger.info(f" The number of atoms in a single molecule :: {NUM_MOL_ATOMS}")
        # atomic_type=itp_data.atomic_type

        # * load trajectories
        logger.info(f" Loading xyz file :: {input_data.file_list}")
        # check atomic arrangement is consistent with itp/mol files
        for xyz_filename in input_data.file_list:
            tmp_atoms = ase.io.read(xyz_filename, index="1")
            print(tmp_atoms.get_chemical_symbols()[:NUM_MOL_ATOMS])
            if tmp_atoms.get_chemical_symbols()[:NUM_MOL_ATOMS] != itp_data.atom_list:
                raise ValueError("configuration different for xyz and itp !!")

        atoms_list: list = []
        for xyz_filename in input_data.file_list:
            tmp_atoms = ase.io.read(xyz_filename, index=":")
            atoms_list.append(tmp_atoms)
            print(f" len xyz == {len(tmp_atoms)}")
        logger.info(" Finish loading xyz file...")
        logger.info(f" The number of trajectories are {len(atoms_list)}")
        logger.info("")
        logger.info(
            " ----------------------------------------------------------------------- "
        )
        logger.info(
            " -----------  Summary of training Data --------------------------------- "
        )
        logger.info("found %d system(s):" % len(input_data.file_list))
        logger.info(
            ("%s  " % _format_name_length("system", 42))
            + (
                "%6s  %6s  %6s %6s"
                % ("nun_frames", "batch_size", "num_batch", "natoms(include WC)")
            )
        )
        for xyz_filename, atoms in zip(input_data.file_list, atoms_list):
            logger.info(
                "%s  %6d  %6d  %6d %6d"
                % (
                    xyz_filename,
                    len(atoms),  # num of frames
                    input_train.batch_size,
                    int(len(atoms) / input_train.batch_size),
                    len(atoms[0].get_atomic_numbers()),
                )
            )
        logger.info(
            "--------------------------------------------------------------------------------------"
        )

        # * convert xyz to atoms_wan

        logger.info(" splitting ase.atoms into atomic ase.atoms and WCs")
        atoms_wan_list: list = []
        # for atoms in atoms_list[0]:
        for traj in atoms_list:  # loop over trajectories
            NUM_ALL_ATOM = len(traj[0]) - traj[0].get_chemical_symbols().count("X")
            NUM_MOL = int(NUM_ALL_ATOM / itp_data.num_atoms_per_mol)
            logger.info(f" NEW TRAJ :: len={len(traj)} :: NUM_MOL={NUM_MOL}")
            for atoms in traj:  # loop over atoms (frames)
                [atoms_nowan, wfc_list] = extract_wcs(atoms)  # atoms, X
                mol_coords = calculate_molcoord(
                    atoms_nowan, itp_data.bonds_list, itp_data.representative_atom_index
                )
                dict_bcs = calc_bondcenter_dict(mol_coords, itp_data.bonds)
                dict_mu = convert_atoms_to_bondwfc(
                    atoms_nowan,
                    wfc_list,
                    itp_data.bonds_type,
                    itp_data.bonds_list,
                    itp_data.bond_index,
                    itp_data.representative_atom_index,
                )
                data = atoms_wan()
                data.set_params(atoms_nowan, NUM_MOL, dict_mu, dict_bcs)
                atoms_wan_list.append(data)
        logger.info(" Finish Assigning Wannier Centers")

        # # save assined results
        # # TODO :: 割当後のデータをより洗練されたフォーマットで保存する．
        # result_atoms = []
        # for atoms_wan_fr in atoms_wan_list:
        #     result_atoms.append(atoms_wan_fr.make_atoms_with_wc())
        # ase.io.write("mol_with_WC.xyz", result_atoms)

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
            input_data.bond_name, input_train.modeldir, input_model.modelname
        ):
            # save input file to output directory
            with open(modeldir + "/input.yaml", "w") as f:
                yaml.dump(yml, f, default_flow_style=False, allow_unicode=True)

            # extract dataset parameters ftom input_data.bond_name(CH,OH,CO,CC,O,COC,COH)
            strategy = strategy_map.get(bond_name)[0]
            calculate_bond = strategy_map.get(bond_name)[1]
            bondtype = strategy_map.get(bond_name)[2]
            logger.info(" -----------  Summary of dataset ------------ ")
            logger.info("  bond_name         :: %s", bond_name)
            logger.info("  calculate_bond    :: %s", calculate_bond)
            logger.info("  bondtype          :: %s", bondtype)
            logger.info("  dataset_function  :: %s", strategy.__class__.__name__)
            logger.info(" -------------------------------------------- ")
            if strategy is None:
                raise ValueError(f"Unsupported bond_name: {bond_name}")

            dataset = DataSet_atoms(atoms_wan_list, "CH_1_bond")

            model = NET_withoutBN_descs(
                modelname=modelname,  # loop variable
                nfeatures=input_model.nfeature,
                M=input_model.M,
                Mb=input_model.Mb,
                Rc=input_model.Rc,
                Rcs=input_model.Rcs,
                bondtype=bond_name,  # loop variable
                hidden_layers_enet=input_model.hidden_layers_enet,
                hidden_layers_fnet=input_model.hidden_layers_fnet,
                list_atomim_number=input_model.list_atomim_number,
                list_maxat=input_model.list_descriptor_length,
            )

            #
            # ここからtraining

            train = Trainer(
                model,  # model
                # Torch device(cpu/cuda/mps)
                device=torch.device(input_train.device),
                # batch size for training (recommend: 32)
                batch_size=input_train.batch_size,
                # batch size for validation (recommend: 32)
                validation_batch_size=input_train.validation_batch_size,
                max_epochs=input_train.max_epochs,
                learning_rate=input_train.learning_rate,  # dict of scheduler
                # num of data （xyz frame for xyz data type/ data number for descriptor data type)
                n_train=input_train.n_train,
                n_val=input_train.n_val,
                modeldir=modeldir,  # loop variable
                restart=input_train.restart,
            )
            #
            # * decompose dateset into train/valid
            # note :: the numbr of train/valid data is set by n_train/n_val
            train.set_dataset(dataset)
            # training
            train.train()
            # FINISH FUNCTION

    elif input_data.type == "descriptor":  # calculation from descriptor
        for filename in input_data.file_list:
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
        train = Trainer(
            model,  # model
            # Torch device(cpu/cuda/mps)
            device=torch.device(input_train.device),
            # batch size for training (recommend: 32)
            batch_size=input_train.batch_size,
            # batch size for validation (recommend: 32)
            validation_batch_size=input_train.validation_batch_size,
            max_epochs=input_train.max_epochs,
            learning_rate=input_train.learning_rate,  # dict of scheduler
            # num of data （xyz frame for xyz data type/ data number for descriptor data type)
            n_train=input_train.n_train,
            n_val=input_train.n_val,
            modeldir=input_train.modeldir,
            restart=input_train.restart,
        )

        #
        # * decompose dateset into train/valid
        # note :: the numbr of train/valid data is set by n_train/n_val
        train.set_dataset(dataset)
        # training
        train.train()
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
