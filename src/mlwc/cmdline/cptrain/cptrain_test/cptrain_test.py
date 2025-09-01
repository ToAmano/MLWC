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
from typing import List

import ase
import torch

from mlwc.cmdline.cptrain.cptrain_train.cptrain_core import (
    _calculate_and_show_metrics,
    _evaluate_model_with_dataset,
    _generate_atomswan_from_atoms,
    _load_itp_data,
    _load_trajectory_data,
    _make_and_save_accuracy_figures,
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
    try:
        model = torch.jit.load(model_filename).to(device)
        model.eval()  # set model to evaluation mode
    except FileNotFoundError:
        logger.error("Model file does not exist: %s", model_filename)
        raise
    except RuntimeError as e:
        logger.error("Error loading model: %s", e)
        raise

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
        bond_name: str = model.bondtype  # override
        # Rcs: float = model.Rcs
        # Rc: float = model.Rc
    else:
        logger.info(" WARNING :: model is old (not include Rc, Rcs, type)")
        # Rcs: float = 4.0  # default value
        # Rc: float = 6.0  # default value
    logger.info(" ====================== ")
    try:
        model.print_parameters()
    except:
        logger.info(" WARNING :: model is old (do not have print_parameters function)")

    # * read molecular bonds information
    itp_data = _load_itp_data(itp_filename)

    # * load trajectory
    atoms_list: List[ase.Atoms] = _load_trajectory_data(xyz_filename)

    # * convert xyz to atoms_wan and assign WCs to BCs
    # TODO :: 割当後のデータをより洗練されたフォーマットで保存する．
    atoms_wan_list, _ = _generate_atomswan_from_atoms(atoms_list, itp_data)
    dataset = DatasetAtoms(atoms_wan_list, bond_name)

    # * get prediction/teacher data to evaluate the model
    true_list, pred_list = _evaluate_model_with_dataset(model, dataset, device)

    # save results
    _calculate_and_show_metrics(true_list, pred_list)

    # make&save figures
    model_dir = os.path.dirname(model_filename)
    _make_and_save_accuracy_figures(true_list, pred_list, model_dir)

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
