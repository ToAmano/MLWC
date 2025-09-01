# -*- coding: utf-8 -*-
"""
This script predicts the dipole moment of molecules in a trajectory using a machine learning model.

It reads a trajectory file (XYZ format), calculates descriptors based on bond centers,
and predicts the dipole moment using a pre-trained PyTorch model.
The results are written to several output files, including total dipole moment,
molecular dipole moment, and bond dipole moments.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import ase
import ase.io
import numpy as np
import torch
import yaml

from mlwc.cmdline.cptrain.cptrain_pred import cptrain_pred_io
from mlwc.cmdline.cptrain.cptrain_train.cptrain_core import (
    _load_itp_data,
    _load_trajectory_file,
    _validate_xyz_with_mol,
)
from mlwc.cpmd.assign_wcs.assign_wcs_torch import atoms_wan

# physics constant
from mlwc.include.constants import Constant
from mlwc.include.mlwc_logger import setup_library_logger
from mlwc.ml.dataset.mldataset_atoms import prepare_input_tensor

# Debye   = 3.33564e-30
# charge  = 1.602176634e-019
# ang      = 1.0e-10
coef: float = Constant.Ang * Constant.Charge / Constant.Debye

logger = setup_library_logger("MLWC." + __name__)


def load_model(
    model_filename: str, device: Literal["cpu", "cuda", "mps"]
) -> torch.jit.ScriptModule:
    """Load a TorchScript model from a file.

    Parameters
    ----------
    model_filename : str
        The path to the TorchScript model file.
    device : str
        The device to load the model onto (e.g., "cpu", "cuda", "mps").

    Returns
    -------
    torch.jit.ScriptModule
        The loaded TorchScript model, or None if the file does not exist.

    Raises
    ------
    ValueError
        If the specified device is not "cpu", "cuda", or "mps".

    Examples
    --------
    >>> model = load_model("model.pt", "cpu")
    """
    if device not in ["cpu", "cuda", "mps"]:
        raise ValueError("ERROR :: device should be cpu or cuda or mps")
    model = None
    if os.path.isfile(model_filename):
        model = torch.jit.load(model_filename)
        model = model.to(device)
        logger.info("%s :: %s", model_filename, model)
        model.share_memory()  # https://knto-h.hatenablog.com/entry/2018/05/22/130745
        model.eval()
    else:
        logger.info("%s is not loaded", model_filename)
    return model


def make_atoms_wc(
    list_mol_coords: list,
    atomic_numbers: list[int],
    cell: np.ndarray,
    list_symbol: list[str],
    list_coord: list[np.ndarray],
) -> ase.Atoms:
    """Create an ASE Atoms object with Wannier centers.

    Parameters
    ----------
    list_mol_coords : numpy.ndarray
        Molecular coordinates. Shape: (mol, # of atoms, 3).
    atomic_numbers : list
        List of atomic numbers.
    cell : numpy.ndarray
        Unit cell of the system.
    list_symbol : list[str]
        List of symbols for Wannier centers.
    list_coord : list[numpy.ndarray]
        List of coordinates for Wannier centers. Shape: (mol, # of WCs, 3).

    Returns
    -------
    ase.Atoms
        An ASE Atoms object with the combined atomic coordinates and Wannier centers.

    Raises
    ------
    ValueError
        If the lengths of `list_coord` and `list_symbol` do not match.
    ValueError
        If the shape of `coords` in `list_coord` is not (mol, id, 3).
    ValueError
        If the first axis of `coords` and `list_mol_coords` is not the number of molecules.

    Examples
    --------
    >>> atoms = make_atoms_wc(list_mol_coords, atomic_numbers, cell, list_symbol, list_coord)
    """
    if len(list_coord) != len(list_symbol):
        raise ValueError(
            f"len(list_coord) should be len(list_symbol) :: {len(list_coord)} :: {len(list_symbol)}"
        )
    for coords in list_coord:
        if len(np.shape(coords)) != 3:
            raise ValueError(
                f"The shape of coords should be [mol,id,3] :: {np.shape(coords)} "
            )
        if np.shape(coords)[0] != np.shape(list_mol_coords)[0]:
            raise ValueError(
                f"The first axis of coords and list_mol_coords is the number of molecules :: {np.shape(coords)} :: {np.shape(list_mol_coords)}"
            )
        if np.shape(coords)[2] != 3:
            raise ValueError(
                f"The shape of coords should be [mol,id,3] :: {np.shape(coords)} "
            )
    NUM_MOL: int = len(list_mol_coords)
    # concatenate coordinate
    for symbol, coords in zip(list_symbol, list_coord):
        list_mol_coords = np.concatenate((list_mol_coords, coords), axis=1)
        atomic_numbers = atomic_numbers + [symbol] * np.shape(coords)[1]

    # repeat atomic_numbers for NUM_MOL tiles
    atomic_numbers = atomic_numbers * NUM_MOL

    # make ase atoms
    atoms = ase.Atoms(
        atomic_numbers,
        positions=list_mol_coords.reshape(-1, 3),
        cell=cell,
        pbc=[1, 1, 1],
    )
    return atoms


def load_all_models(model_dir: str, device: str, names: list[str]) -> dict[str, Any]:
    models = {
        name: load_model(f"{model_dir}/model_{name}.pt", device) for name in names
    }
    if all(model is None for model in models.values()):
        raise FileNotFoundError(
            "None of the models loaded. Please check the directory."
        )
    return models


def initialize_dipole_output_files(
    savedir: str,
    unitcell,
    temperature: float,
    timestep: float,
    save_bonddipole: bool,
):

    dipole_files = {
        "total": Path(savedir) / "total_dipole.txt",
        "molecule": Path(savedir) / "molecule_dipole.txt",
        "ch": Path(savedir) / "ch_dipole.txt",
        "co": Path(savedir) / "co_dipole.txt",
        "oh": Path(savedir) / "oh_dipole.txt",
        "cc": Path(savedir) / "cc_dipole.txt",
        "o": Path(savedir) / "o_dipole.txt",
        "coc": Path(savedir) / "coc_dipole.txt",
        "coh": Path(savedir) / "coh_dipole.txt",
    }

    opened_files = {
        key: open(path, "w", encoding="utf-8") for key, path in dipole_files.items()
    }

    def write_header(file, header_line: str):
        file.write(header_line + "\n")
        file.write("#UNITCELL[Ang] ")
        file.write(
            " ".join(str(unitcell[i, j]) for i in range(3) for j in range(3)) + "\n"
        )
        file.write(f"#TEMPERATURE[K] {temperature}\n")
        file.write(f"#TIMESTEP[fs] {timestep}\n")

    write_header(opened_files["total"], "# index dipole_x dipole_y dipole_z")
    write_header(
        opened_files["molecule"],
        "# frame_index molecule_dipole_x molecule_dipole_y molecule_dipole_z",
    )

    if save_bonddipole:
        for key in ["ch", "co", "oh", "cc", "o", "coc", "coh"]:
            write_header(
                opened_files[key], "# frame_index bond_index dipole_x dipole_y dipole_z"
            )

    return opened_files


def close_files(files: dict) -> None:
    for file in files.values():
        file.close()


def process_frame(
    fr_index: int,
    fr_atoms,
    models,
    itp_data,
    input_general,
    dipole_files,
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    """
    Process a single frame: generate descriptors, predict dipoles, accumulate results.
    Returns total dipole, list of virtual site coordinates, and their atomic numbers.
    """
    data = atoms_wan()
    data.set_params_from_atoms(fr_atoms, itp_data)
    NUM_MOL = data.NUM_MOL
    fr_atoms = data.atoms_nowan

    sum_dipole = np.zeros(3)
    wc_coords: list[np.ndarray] = []
    wc_symbols: list[int] = []

    def run_prediction_and_save(
        model, data: atoms_wan, bond_key: str, divide_coef: float, dipole_file
    ) -> np.ndarray:
        input_tensors = prepare_input_tensor(data, bond_key)  # 入力を処理
        y_pred = model(**input_tensors).to("cpu").detach().numpy().reshape(-1, 3)
        if input_general.save_bonddipole:
            save_dipole(dipole_file, fr_index, y_pred)
        centers = input_tensors["bond_centers"].detach().numpy()
        virtual_sites = (centers + y_pred / coef / divide_coef).reshape(NUM_MOL, -1, 3)
        return y_pred, virtual_sites

    # モデルごとに処理
    # TODO :: 全てのボンド種について計算する
    y_pred = {}
    for bond_type, model_key, symbol in [
        ("CH_1_bond", "ch", 100),
        ("CO_1_bond", "co", 101),
        ("OH_1_bond", "oh", 102),
        ("CC_1_bond", "cc", 103),
    ]:
        bond_idx = itp_data.bond_index.get(bond_type, [])
        model = models.get(model_key)
        dipole_file = dipole_files.get(model_key)
        if not bond_idx or model is None:
            continue
        y_pred[bond_type], virtual_site = run_prediction_and_save(
            model, data, bond_type, -2.0, dipole_file
        )
        wc_coords.append(data.dict_bcs[bond_type])
        wc_symbols.append(2)
        wc_coords.append(virtual_site)
        wc_symbols.append(symbol)
        sum_dipole += np.sum(y_pred[bond_type], axis=0)

    # coh, coc, o の特別処理
    # TODO :: 全てのlone pairに対応
    for bond_type, special_key, symbol in [
        ("COH_1_bond", "coh", 106),
        ("COC_1_bond", "coc", 105),
        ("Olp", "o", 10),
    ]:
        if special_key not in models or models[special_key] is None:
            continue
        model = models.get(special_key)
        dipole_file = dipole_files.get(special_key)
        atom_positions = fr_atoms.get_positions()
        if special_key == "coh":
            # TODO :: このままだと通常のo_listを使ってしまっていてまずい．
            # TODO :: ちゃんとcohに対応したo_listを作るようにする．
            positions = atom_positions[
                np.argwhere(fr_atoms.get_atomic_numbers() == 8).reshape(-1)
            ]
        elif special_key == "coc":
            positions = atom_positions[
                np.argwhere(fr_atoms.get_atomic_numbers() == 8).reshape(-1)
            ]
        elif special_key == "o":
            positions = atom_positions[
                np.argwhere(fr_atoms.get_atomic_numbers() == 8).reshape(-1)
            ]
        if len(positions) == 0:
            continue
        y_pred["special_key"], virtual_site = run_prediction_and_save(
            model, data, bond_type, -4.0, dipole_file
        )
        wc_coords.append(virtual_site)
        wc_symbols.append(symbol)
        sum_dipole += np.sum(y_pred["special_key"], axis=0)

    return data.mol_coords, sum_dipole, wc_coords, wc_symbols, y_pred, NUM_MOL


def save_dipole(dipole_file, fr_index, y_pred) -> None:
    np.savetxt(
        dipole_file,
        np.hstack(
            [
                np.ones((len(y_pred), 1)) * fr_index,
                np.arange(len(y_pred)).reshape(-1, 1),
                y_pred,
            ]
        ),
        fmt="%d %d %f %f %f",
    )


def mlpred(yaml_filename: str) -> None:
    """Predict dipole moments of molecules in a trajectory using a machine learning model.

    This function reads a YAML input file, loads a pre-trained PyTorch model,
    calculates descriptors based on bond centers, and predicts the dipole moment
    for each frame in the trajectory. The results are written to several output files.

    Parameters
    ----------
    yaml_filename : str
        The path to the YAML input file containing the simulation parameters,
        model paths, and other settings.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the specified YAML file or any of the model files do not exist.
    ValueError
        If the atomic arrangement in the XYZ trajectory file is inconsistent
        with the atomic types defined in the ITP/MOL file.

    Examples
    --------
    >>> mlpred("input.yaml")
    """
    # read input yaml file
    with open(yaml_filename, encoding="utf-8") as file:
        yml = yaml.safe_load(file)
        print(yml)
    input_general = cptrain_pred_io.variables_general(yml)
    input_descriptor = cptrain_pred_io.variables_descriptor(yml)
    input_predict = cptrain_pred_io.variables_predict(yml)

    # save input file to output directory
    with open(input_general.savedir + "/input.yaml", "w", encoding="utf-8") as f:
        yaml.dump(yml, f, default_flow_style=False, allow_unicode=True)

    # from torchinfo import summary
    # summary(model=model_ring)

    # * load data (xyz or descriptor)
    logger.info(" -------------------------------------- ")
    # * load itp
    itp_data = _load_itp_data(input_general.bondfilename)
    logger.info(
        " The number of atoms in a single molecule :: %d", itp_data.num_atoms_per_mol
    )

    # * load trajectories
    logger.info(" Loading xyz file :: %s", input_descriptor.xyzfilename)
    _validate_xyz_with_mol([input_descriptor.xyzfilename], itp_data)
    atoms_traj: list[ase.Atoms] = _load_trajectory_file(input_descriptor.xyzfilename)
    logger.info(" Finish loading xyz file...")
    logger.info(" The number of atoms in a single frame :: %d", len(atoms_traj[0]))

    model_names: list[str] = ["ch", "co", "oh", "cc", "o", "coc", "coh"]
    models = load_all_models(input_predict.model_dir, input_predict.device, model_names)

    # method to calculate bond centers
    dipole_files = initialize_dipole_output_files(
        input_general.savedir,
        atoms_traj[0].get_cell(),
        input_general.temperature,
        input_general.timestep,
        input_general.save_bonddipole,
    )

    # * loop over trajectories
    for fr_index, fr_atoms in enumerate(atoms_traj):
        # split atoms and wan
        list_mol_coords, sum_dipole, wc_coords, wc_symbols, y_pred, NUM_MOL = (
            process_frame(
                fr_index,
                fr_atoms,
                models,
                itp_data,
                input_general,
                dipole_files,
            )
        )
        # >>>>>>>  final process in the loop >>>>>>>
        # make atoms
        atoms_wc = make_atoms_wc(
            list_mol_coords,
            itp_data.atom_list,
            fr_atoms.cell,
            wc_symbols,
            wc_coords,
        )

        # write to files
        dipole_files["total"].write(
            f"{fr_index} {sum_dipole[0]} {sum_dipole[1]} {sum_dipole[2]}\n"
        )
        # calculate molecular dipole
        molecule_dipole = sum(
            value.reshape((NUM_MOL, -1, 3)).sum(axis=1) for key, value in y_pred.items()
        )
        np.savetxt(
            dipole_files["molecule"],
            np.hstack(
                [
                    np.ones((len(molecule_dipole), 1)) * fr_index,
                    np.arange(len(molecule_dipole)).reshape(-1, 1),
                    molecule_dipole,
                ]
            ),
            fmt="%d %d %f %f %f",
        )
        # append atoms to file
        ase.io.write(
            os.path.join(input_general.savedir, "/mol_wc.xyz"), atoms_wc, append=True
        )

    # finish writing
    close_files(dipole_files)
    return 0


def command_cptrain_pred(args) -> int:
    """Run the dipole prediction workflow.

    This function serves as a command-line interface to the `mlpred` function.
    It takes command-line arguments, passes the input YAML file to `mlpred`,
    and returns 0 upon completion.

    Parameters
    ----------
    args : argparse
    """
    mlpred(args.input)
    return 0
