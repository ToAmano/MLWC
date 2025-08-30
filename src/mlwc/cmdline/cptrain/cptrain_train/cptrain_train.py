# -*- coding: utf-8 -*-
"""
This module implements the training process for the ML model.
It reads the input parameters from a YAML file, loads the data,
constructs the neural network model, and trains the model.

"""

# fugaku上のpython3.8で型指定をする方法（https://future-architect.github.io/articles/20201223/）
from __future__ import annotations

import pathlib
from typing import List

import ase
import ase.io
import numpy as np
import torch
import yaml
from sklearn.metrics import mean_squared_error

import mlwc.ml.dataset.mldataset_descs
from mlwc.cmdline.cptrain.cptrain_train import cptrain_train_io
from mlwc.cmdline.cptrain.cptrain_train.cptrain_core import (
    _evaluate_model_with_dataset,
    _generate_atomswan_from_atoms,
    _load_itp_data,
    _load_trajectory_data,
    _validate_xyz_with_mol,
)
from mlwc.include.mlwc_logger import setup_library_logger
from mlwc.include.utils import get_torch_device
from mlwc.ml.dataset.mldataset_atoms import DatasetAtoms
from mlwc.ml.model.mlmodel_basic_descs import NetWithoutBatchNormalizationDescs
from mlwc.ml.train.ml_train import Trainer

logger = setup_library_logger("MLWC." + __name__)


def _set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


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
    logger.info("")
    logger.info(" Load Setting File")
    logger.info("==================")
    logger.info("")
    with open(yaml_filename, encoding="utf-8") as file:
        yml = yaml.safe_load(file)
        # print(yml)
    model_cfg = cptrain_train_io.VariablesModel(yml)
    train_cfg = cptrain_train_io.VariablesTraining(yml)
    data_cfg = cptrain_train_io.VariablesData(yml)
    # * Get seed list
    seed_list = model_cfg.seed
    if not isinstance(seed_list, list):
        seed_list = [seed_list]
    logger.info("Target seeds: %s", seed_list)

    # * 3:: load data (xyz or descriptor)
    logger.info(" -------------------------------------- ")
    if data_cfg.type == "xyz":
        logger.info("data type :: xyz")
        # * load itp/mol file
        itp_data = _load_itp_data(data_cfg.itp_file)
        # TODO :: check consistency with list_atomic_number

        # * load trajectories
        _validate_xyz_with_mol(
            data_cfg.file_list, itp_data
        )  # check atomic arrangement is consistent with itp/mol files
        atoms_list: List[ase.Atoms] = _load_trajectory_data(data_cfg.file_list)
        # _log_dataset_summary(atoms_list, data_cfg, train_cfg)

        # * convert xyz to atoms_wan
        # TODO :: How to save assined WCs data ??
        atoms_wan_list, result_atoms_list = _generate_atomswan_from_atoms(
            atoms_list, itp_data
        )
        ase.io.write("mol_with_WC.xyz", result_atoms_list)

        # Training (loop over bondtype and initial seeds)
        logger.info("")
        logger.info(" Training Model(s)")
        logger.info("==================")
        logger.info("")

        # --- Main loop for seeds ---
        for seed in seed_list:
            logger.info("")
            logger.info(" Start Training for Seed: %s", seed)
            logger.info("===================================")
            # set pytorch/numpy seed
            _set_random_seed(seed)

            for bondtype, modeldir, modelname in zip(
                data_cfg.bondtype, train_cfg.modeldir, model_cfg.modelname
            ):
                logger.info(
                    " bondtype = %s :: modeldir = %s :: modelname = %s",
                    bondtype,
                    modeldir,
                    modelname,
                )
                # Create seed-specific directory
                seeded_modeldir = f"{modeldir}_seed_{seed}"
                pathlib.Path(seeded_modeldir).mkdir(parents=True, exist_ok=True)
                logger.info(" Results will be saved in %s", seeded_modeldir)

                # save input file to output directory
                with open(seeded_modeldir + "/input.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(yml, f, default_flow_style=False, allow_unicode=True)

                dataset = DatasetAtoms(atoms_wan_list, bondtype)

                model = NetWithoutBatchNormalizationDescs(
                    modelname=modelname,  # loop variable
                    nfeatures=model_cfg.nfeature,
                    m=model_cfg.M,
                    mb=model_cfg.Mb,
                    rc=model_cfg.Rc,
                    rcs=model_cfg.Rcs,
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
                    modeldir=seeded_modeldir,  # loop variable
                    restart=train_cfg.restart,
                )
                # * decompose dateset into train/valid. the number of train/valid data is set by n_train/n_val
                train.set_dataset(dataset)
                train.train()
                # train.validate_model()
                # FINISH FUNCTION

                logger.info("")
                logger.info(" Model Validation")
                logger.info("=================")
                logger.info("")
                dataset = DatasetAtoms(atoms_wan_list, bondtype)
                device: str = get_torch_device()
                # * get prediction/teacher data to evaluate the model
                true_list, pred_list = _evaluate_model_with_dataset(
                    model, dataset, device
                )

                # save results
                logger.info(" ======")
                logger.info("  Finish testing.")
                logger.info("  Save results to %s", seeded_modeldir)
                logger.info(" ")
                logger.info(
                    " RMSE_train = %s",
                    np.sqrt(mean_squared_error(true_list, pred_list)),
                )
                logger.info(" ")
                np.savetxt(f"{seeded_modeldir}/pred_list.txt", pred_list)
                np.savetxt(f"{seeded_modeldir}/true_list.txt", true_list)
                # make figures
                mlwc.ml.train.ml_train.make_figure(pred_list, true_list)
                mlwc.ml.train.ml_train.plot_residure_density(pred_list, true_list)
    # !! DEPRECATED
    # TODO:: remove below
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
