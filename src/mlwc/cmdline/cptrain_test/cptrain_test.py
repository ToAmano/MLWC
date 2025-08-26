#!/usr/bin/env python
# coding: utf-8
"""
This script provides a command-line interface for testing machine learning models
trained to predict properties of molecular systems, particularly focusing on
Wannier center (WC) based models. It loads a pre-trained model, molecular
structure data from an XYZ file, and atom type information from an ITP or MOL file.
The script then evaluates the model's performance by comparing its predictions
against the true values from the provided data, calculating metrics such as
Root Mean Squared Error (RMSE) and R-squared. The results are saved to files
and visualized using plots.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
from sklearn.metrics import r2_score

import mlwc.ml.train.ml_train  # for figures
from mlwc.cmdline.cptrain_train.cptrain_core import (
    _generate_atomswan_from_atoms,
    _load_itp_data,
    _load_trajectory_data,
)
from mlwc.include.mlwc_logger import setup_library_logger
from mlwc.include.utils import get_torch_device
from mlwc.ml.dataset.mldataset_atoms import DatasetAtoms

logger = setup_library_logger("MLWC." + __name__)


def mltest(
    model_filename: str, xyz_filename: str, itp_filename: str, bond_name: str
) -> None:
    """
    Tests a machine learning model for predicting molecular properties.

    This function loads a pre-trained PyTorch model, molecular structure data
    from an XYZ file, and atom type information from an ITP or MOL file.
    It then evaluates the model's performance by comparing its predictions
    against the true values from the provided data, calculating metrics such as
    Root Mean Squared Error (RMSE) and R-squared. The results are saved to files
    and visualized using plots.

    Parameters
    ----------
    model_filename : str
        Path to the pre-trained PyTorch model file (``.pth`` or ``.pt``).
    xyz_filename : str
        Path to the XYZ file containing the molecular structure data.
    itp_filename : str
        Path to the ITP or MOL file containing the atom type information.
    bond_name : str
        Name of the bond to be calculated.

    Returns
    -------
    None

    Examples
    --------
    >>> mltest('model.pth', 'traj.xyz', 'mol.itp', 'OH')
    """
    # load model
    device: str = get_torch_device()
    model = torch.jit.load(model_filename).to(device)
    model.eval()  # set model to evaluation mode

    #
    logger.info(" ==========  Model Parameter information  ============ ")
    if hasattr(model, "m"):
        logger.info(" m         = %s", model.m)
    else:
        logger.info("The model do not contain m")
    if hasattr(model, "mb"):
        logger.info(" mb        = %s", model.mb)
    else:
        logger.info("The model do not contain mb")
    if hasattr(model, "nfeatures"):
        logger.info(" nfeatures = %s", model.nfeatures)
        MaxAt: int = int(model.nfeatures / 4 / 3)
        logger.info(" MaxAt     = %s", MaxAt)
    else:
        logger.info("The model do not contain nfeatures")
    if hasattr(model, "rcs") and hasattr(model, "rc") and hasattr(model, "type"):
        logger.info(" rcs = %s", model.rcs)
        logger.info(" rc = %s", model.rc)
        logger.info(" type = %s", model.bondtype)
        bond_name: str = model.bondtype  # overwride
        # Rcs: float = model.Rcs
        # Rc: float = model.Rc
    else:
        logger.info(" WARNING :: model is old (not include Rc, Rcs, type)")
        # Rcs: float = 4.0  # default value
        # Rc: float = 6.0  # default value
    logger.info(" ====================== ")

    # * read itp
    itp_data = _load_itp_data(itp_filename)

    # * load trajectory
    atoms_list = _load_trajectory_data(xyz_filename)

    # * convert xyz to atoms_wan and assign WCs to BCs
    # TODO :: 割当後のデータをより洗練されたフォーマットで保存する．
    atoms_wan_list, _ = _generate_atomswan_from_atoms(atoms_list, itp_data)

    dataset = DatasetAtoms(atoms_wan_list, bond_name)
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
    with torch.no_grad():  # https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
        for data in dataloader_valid:
            if isinstance(data[0], dict):  # data[0]がdictの場合
                for i in range(len(data[1])):
                    data_1 = [
                        {key: value[i].to(device) for key, value in data[0].items()},
                        data[1][i],
                    ]
                    x = data_1[0]
                    y = data_1[1]
                    y_pred = model(**x, device=device)
                    pred_list.append(y_pred.to("cpu").detach().numpy())
                    true_list.append(y.detach().numpy())
            elif (
                data[0].dim() == 3
            ):  # 3次元の場合[NUM_BATCH,NUM_BOND,288]はデータを整形する
                # TODO :: torch.reshape(data[0], (-1, 288)) does not work !!
                for x, y in zip(data[0], data[1]):
                    y_pred = model(x.to(device))
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
    logger.info(" r^2        = %s", r2_score(true_list, pred_list))
    logger.info(" ")
    logger.info(" ELAPSED TIME  {:.2f}".format((end_time - start_time)))
    logger.info(np.shape(pred_list))
    logger.info(np.shape(true_list))
    # make figures
    model_dir: str = os.path.dirname(model_filename)
    np.savetxt(model_dir + "/pred_list.txt", pred_list)
    np.savetxt(model_dir + "/true_list.txt", true_list)
    mlwc.ml.train.ml_train.make_figure(pred_list, true_list, directory=model_dir)
    mlwc.ml.train.ml_train.plot_residure_density(
        pred_list, true_list, directory=model_dir
    )
    return 0


def command_cptrain_test(args) -> int:
    """
    Command-line interface for testing a machine learning model.

    This function serves as the entry point for testing a pre-trained machine
    learning model using the provided command-line arguments. It parses the
    arguments to extract the model file, XYZ file, MOL/ITP file, and bond type,
    and then calls the `mltest` function to perform the actual testing.

    Parameters
    ----------
    args : argparse.Namespace
        An object containing the command-line arguments.

    Returns
    -------
    int
        Returns 0 upon successful execution.

    Examples
    --------
    >>> args = argparse.Namespace(model='model.pth', xyz='traj.xyz', mol='mol.itp', bond='OH')
    >>> command_cptrain_test(args)
    0
    """
    logger.info(" ")
    logger.info(" CPtrain.py test :: validation for ML models")
    logger.info(" ")
    mltest(args.model, args.xyz, args.mol, args.bond)
    return 0
