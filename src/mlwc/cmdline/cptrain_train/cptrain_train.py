# -*- coding: utf-8 -*-
"""
This module implements the training process for the ML model.
It reads the input parameters from a YAML file, loads the data,
constructs the neural network model, and trains the model.

"""

# fugaku上のpython3.8で型指定をする方法（https://future-architect.github.io/articles/20201223/）
from __future__ import annotations

import os
import time
from typing import List

import ase
import ase.io
import numpy as np
import torch
import yaml

import mlwc.bond.atomtype
import mlwc.ml.dataset.mldataset_descs
from mlwc.cmdline.cptrain_train import cptrain_train_io
from mlwc.cpmd.assign_wcs.assign_wcs_torch import atoms_wan
from mlwc.include.mlwc_logger import setup_library_logger
from mlwc.ml.dataset.mldataset_atoms import DatasetAtoms
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


def _set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _load_itp_data(filepath: str):
    if not os.path.isfile(filepath):
        logger.error("ITP file does not exist: %s", filepath)
        raise FileNotFoundError(f"Missing ITP file: {filepath}")

    if filepath.endswith(".itp"):
        itp_data = mlwc.bond.atomtype.read_itp(filepath)
    elif filepath.endswith(".mol"):
        itp_data = mlwc.bond.atomtype.ReadMolFile(filepath)
    else:
        raise ValueError(f"Unsupported file format for ITP: {filepath}")
    logger.info(
        " The number of atoms in a molecule :: %s",
        itp_data.num_atoms_per_mol,
    )
    return itp_data


def _validate_xyz_with_mol(xyz_files: List[str], itp_data) -> None:
    """Check consistency with mol and xyz"""
    for xyz_file in xyz_files:
        atoms = ase.io.read(xyz_file, index="1")
        if (
            atoms.get_chemical_symbols()[: itp_data.num_atoms_per_mol]
            != itp_data.atom_list
        ):
            raise ValueError(f"Atom mismatch in file: {xyz_file}")


def _load_trajectory_data(file_list: List[str]) -> List[List["ase.Atoms"]]:
    atoms_list = []
    for xyz_file in file_list:
        traj = ase.io.read(xyz_file, index=":")
        atoms_list.append(traj)
        logger.info("Loaded %s with %d frames.", xyz_file, len(traj))
    return atoms_list


def _log_dataset_summary(atoms_list, data_cfg, train_cfg) -> None:
    logger.info(
        " ----------------------------------------------------------------------- "
    )
    logger.info(
        " -----------  Summary of training Data --------------------------------- "
    )
    logger.info("found %d system(s):", len(data_cfg.file_list))
    logger.info(
        ("%s  ", _format_name_length("system", 42))
        + (
            "%6s  %6s  %6s %6s",
            "nun_frames",
            "batch_size",
            "num_batch",
            "natoms(include WC)",
        )
    )
    for xyz_filename, atoms in zip(data_cfg.file_list, atoms_list):
        logger.info(
            "%s  %6d  %6d  %6d %6d",
            xyz_filename,
            len(atoms),  # num of frames
            train_cfg.batch_size,
            int(len(atoms) / train_cfg.batch_size),
            len(atoms[0].get_atomic_numbers()),
        )
    logger.info(
        "--------------------------------------------------------------------------------------"
    )


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

    # * 1 :: read input yaml file
    with open(yaml_filename, encoding="utf-8") as file:
        yml = yaml.safe_load(file)
        print(yml)
    model_cfg = cptrain_train_io.VariablesModel(yml)
    train_cfg = cptrain_train_io.VariablesTraining(yml)
    data_cfg = cptrain_train_io.VariablesData(yml)
    # set pytorch/numpy seed
    _set_random_seed(model_cfg.seed)

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
        # TODO :: 割当後のデータをより洗練されたフォーマットで保存する．
        logger.info(" splitting ase.atoms into atomic ase.atoms and WCs")
        atoms_wan_list: list = []
        result_atoms_list: list = []
        for traj in atoms_list:  # loop over trajectories
            logger.info(" TRAJ :: len=%d ", len(traj))
            start_time = time.perf_counter()  # start time check
            for atoms in traj:  # loop over atoms (frames)
                data = atoms_wan()
                data.set_params_from_atoms(atoms, itp_data)
                atoms_wan_list.append(data)
                result_atoms_list.append(data.make_atoms_with_wc())
            end_time = time.perf_counter()  # timer stop
            logger.info(" ELAPSED TIME  {:.2f}".format((end_time - start_time)))
        ase.io.write("mol_with_WC.xyz", result_atoms_list)
        logger.info(" Finish Assigning Wannier Centers")

        # * dataset/dataloader
        # loop over bondtype
        for bondtype, modeldir, modelname in zip(
            data_cfg.bondtype, train_cfg.modeldir, model_cfg.modelname
        ):
            logger.info(
                " bondtype = %s :: modeldir = %s :: modelname = %s",
                bondtype,
                modeldir,
                modelname,
            )
            # save input file to output directory
            with open(modeldir + "/input.yaml", "w", encoding="utf-8") as f:
                yaml.dump(yml, f, default_flow_style=False, allow_unicode=True)

            dataset = DatasetAtoms(atoms_wan_list, bondtype)

            model = NET_withoutBN_descs(
                modelname=modelname,  # loop variable
                nfeatures=model_cfg.nfeature,
                M=model_cfg.M,
                Mb=model_cfg.Mb,
                Rc=model_cfg.Rc,
                Rcs=model_cfg.Rcs,
                bondtype=bondtype,  # loop variable
                hidden_layers_enet=model_cfg.hidden_layers_enet,
                hidden_layers_fnet=model_cfg.hidden_layers_fnet,
                list_atomim_number=model_cfg.list_atomim_number,
                list_maxat=model_cfg.list_descriptor_length,
            )

            # training
            train = Trainer(
                model,  # model
                device=torch.device(train_cfg.device),  # Torch device(cpu/cuda/mps)
                batch_size=train_cfg.batch_size,  # batch size for training (recommend: 32)
                validation_batch_size=train_cfg.validation_batch_size,  # batch size for validation (recommend: 32)
                max_epochs=train_cfg.max_epochs,
                learning_rate=train_cfg.learning_rate,  # dict of scheduler
                n_train=train_cfg.n_train,  # num of data （xyz frame for xyz data type/ data number for descriptor data type)
                n_val=train_cfg.n_val,
                modeldir=modeldir,  # loop variable
                restart=train_cfg.restart,
            )
            # * decompose dateset into train/valid. the number of train/valid data is set by n_train/n_val
            train.set_dataset(dataset)
            train.train()
            # train.validate_model()
            # FINISH FUNCTION

            dataset = DatasetAtoms(atoms_wan_list, bondtype)
            # FIXME :: hard code :: batch_size=32
            # FIXME :: num_worker = 0 for mps
            dataloader_valid = torch.utils.data.DataLoader(
                dataset,
                batch_size=32,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                num_workers=0,
            )

            # lists for results
            pred_list: list = []
            true_list: list = []

            # * Test models
            start_time = time.perf_counter()  # start time check
            model.eval()  # model to evaluation mode
            with torch.no_grad():  # https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
                for data in dataloader_valid:
                    if isinstance(data[0], dict):  # data[0]がdictの場合
                        for i in range(len(data[1])):
                            data_1 = [
                                {key: value[i] for key, value in data[0].items()},
                                data[1][i],
                            ]
                            x = data_1[0]
                            y = data_1[1]
                            y_pred = model(**x)
                            pred_list.append(y_pred.to("cpu").detach().numpy())
                            true_list.append(y.detach().numpy())
                    elif (
                        data[0].dim() == 3
                    ):  # 3次元の場合[NUM_BATCH,NUM_BOND,288]はデータを整形する
                        # TODO :: torch.reshape(data[0], (-1, 288)) does not work !!
                        for x, y in zip(data[0], data[1]):
                            y_pred = model(x.to("cpu"))
                            pred_list.append(y_pred.to("cpu").detach().numpy())
                            true_list.append(y.detach().numpy())
                    elif data[0].dim() == 2:  # 2次元の場合はそのまま
                        # self.batch_step(data,validation=True)
                        x = data[0]
                        y = data[1]
                        y_pred = model(x)
                        pred_list.append(y_pred.to("cpu").detach().numpy())
                        true_list.append(y.detach().numpy())
            #
            pred_list = np.array(pred_list).reshape(-1, 3)
            true_list = np.array(true_list).reshape(-1, 3)
            end_time = time.perf_counter()  # timer stop
            # calculate RSME
            rmse = np.sqrt(np.mean((true_list - pred_list) ** 2))
            # save results
            logger.info(" ======")
            logger.info("  Finish testing.")
            logger.info("  Save results as pred_true_list.txt")
            logger.info(" RSME_train = %s", rmse)
            logger.info(" ")
            logger.info(np.shape(pred_list))
            logger.info(np.shape(true_list))
            np.savetxt("pred_list.txt", pred_list)
            np.savetxt("true_list.txt", true_list)
            # make figures
            mlwc.ml.train.ml_train.make_figure(pred_list, true_list)
            mlwc.ml.train.ml_train.plot_residure_density(pred_list, true_list)

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

        # training
        train = Trainer(
            model,
            device=torch.device(train_cfg.device),
            batch_size=train_cfg.batch_size,  # recommend: 32
            validation_batch_size=train_cfg.validation_batch_size,  # recommend: 32
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
    """
    mltrain(args.input)
    return 0
