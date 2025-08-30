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
import numpy.typing as npt  # for annotation
import torch
import torch.multiprocessing as mp
import yaml
from jaxtyping import Float

import mlwc.bond.atomtype
import mlwc.cpmd.asign_wcs
import mlwc.cpmd.class_atoms_wan
from mlwc.cmdline.cptrain.cptrain_pred import cptrain_pred_io
from mlwc.cmdline.cptrain.cptrain_train.cptrain_train import (
    _load_itp_data,
    _validate_xyz_with_mol,
)
from mlwc.cpmd.bondcenter.bondcenter import calc_bondcenter
from mlwc.cpmd.pbc.pbc import pbc
from mlwc.cpmd.pbc.pbc_mol import pbc_mol

# physics constant
from mlwc.include.constants import Constant
from mlwc.include.mlwc_logger import setup_cmdline_logger
from mlwc.ml.descriptor.descriptor_abstract import Descriptor
from mlwc.ml.descriptor.descriptor_torch import DescriptorTorchBondcenter

# Debye   = 3.33564e-30
# charge  = 1.602176634e-019
# ang      = 1.0e-10
coef: float = Constant.Ang * Constant.Charge / Constant.Debye


logger = setup_cmdline_logger("MLWC." + __name__)


def _format_name_length(name, width):
    """Format a string to a specified width.

    If the string is shorter than the width, it is right-aligned and padded with spaces.
    If the string is longer than the width, it is truncated and " -- " is prepended.

    Parameters
    ----------
    name : str
        The string to format.
    width : int
        The desired width of the formatted string.

    Returns
    -------
    str
        The formatted string.

    Examples
    --------
    >>> _format_name_length("name", 10)
    '      name'
    >>> _format_name_length("verylongname", 10)
    '-- gname'
    """
    if len(name) <= width:
        return "{: >{}}".format(name, width)
    else:
        name = name[-(width - 3) :]
        name = "-- " + name
        return name


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
    input_descriptor,
    input_predict,
    input_general,
    dipole_files,
    pbc_mol,
    DESC,
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    """
    Process a single frame: generate descriptors, predict dipoles, accumulate results.
    Returns total dipole, list of virtual site coordinates, and their atomic numbers.
    """
    # Extract bonded atoms and molecule-wise coordinates
    atoms_wan = mlwc.cpmd.class_atoms_wan.atoms_wan(
        fr_atoms, itp_data.num_atoms_per_mol, itp_data
    )
    NUM_MOL = atoms_wan.NUM_MOL

    mol_coords = pbc(pbc_mol).compute_pbc(
        vectors_array=atoms_wan.atoms_nowan.get_positions(),
        cell=atoms_wan.atoms_nowan.get_cell(),
        bonds_list=itp_data.bonds_list,
        NUM_ATOM_PAR_MOL=itp_data.num_atoms_per_mol,
        ref_atom_index=itp_data.representative_atom_index,
    )

    bond_centers = calc_bondcenter(mol_coords, itp_data.bonds_list)
    fr_atoms = atoms_wan.atoms_nowan
    fr_atoms.set_positions(mol_coords.reshape((-1, 3)))

    sum_dipole = np.zeros(3)
    wc_coords = [bond_centers]
    wc_symbols = [2]

    def run_prediction_and_save(model, centers, divide_coef, dipole_file) -> np.ndarray:
        desc = DESC.calc_descriptor(
            atoms=fr_atoms,
            bond_centers=centers,
            list_atomic_number=[6, 1, 8],
            list_maxat=[24, 24, 24],
            Rcs=input_descriptor.Rcs,
            Rc=input_descriptor.Rc,
            device=input_predict.device,
        )
        X = torch.from_numpy(desc.astype(np.float32)).clone()
        y_pred = (
            model(X.to(input_predict.device)).to("cpu").detach().numpy().reshape(-1, 3)
        )
        if input_general.save_bonddipole:
            save_dipole(dipole_file, fr_index, y_pred)
        virtual_sites = (centers + y_pred / coef / divide_coef).reshape(NUM_MOL, -1, 3)
        return y_pred, virtual_sites

    # モデルごとに処理
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
        bc = np.array(bond_centers)[:, bond_idx, :].reshape((-1, 3))
        y_pred[bond_type], virtual_site = run_prediction_and_save(
            model, bc, -2.0, dipole_file
        )
        wc_coords.append(virtual_site)
        wc_symbols.append(symbol)
        sum_dipole += np.sum(y_pred[bond_type], axis=0)

    # coh, coc, o の特別処理
    for special_key, symbol in [("coh", 106), ("coc", 105), ("o", 10)]:
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
            model, positions, -4.0, dipole_file
        )
        wc_coords.append(virtual_site)
        wc_symbols.append(symbol)
        sum_dipole += np.sum(y_pred["special_key"], axis=0)

    return mol_coords, sum_dipole, wc_coords, wc_symbols, y_pred, NUM_MOL


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
    atoms_traj: list[ase.Atoms] = ase.io.read(input_descriptor.xyzfilename, index=":")
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

    # ASSIGN=cpmd.asign_wcs.asign_wcs(NUM_MOL,NUM_MOL_ATOMS,atoms_traj[0].get_cell())
    DESC = Descriptor(DescriptorTorchBondcenter())  # set strategy

    # * loop over trajectories
    for fr_index, fr_atoms in enumerate(atoms_traj):
        # split atoms and wan
        list_mol_coords, sum_dipole, wc_coords, wc_symbols, y_pred, NUM_MOL = (
            process_frame(
                fr_index,
                fr_atoms,
                models,
                itp_data,
                input_descriptor,
                input_predict,
                input_general,
                dipole_files,
                pbc_mol,
                DESC,
            )
        )
        # # coordinates list for making ase.Atoms
        # list_wc_coords: str = [list_bond_centers]
        # list_wc_symbols: str[int] = [2]

        # if len(itp_data.bond_index["CH_1_bond"]) != 0 and models["ch"] is not None:
        #     # extract the coordinates of ch_bond
        #     bond_centers = np.array(list_bond_centers)[
        #         :, itp_data.bond_index["CH_1_bond"], :
        #     ].reshape((-1, 3))

        #     Descs_ch = DESC.calc_descriptor(
        #         atoms=fr_atoms,
        #         bond_centers=bond_centers,
        #         list_atomic_number=[6, 1, 8],
        #         list_maxat=[24, 24, 24],
        #         Rcs=input_descriptor.Rcs,  # 4
        #         Rc=input_descriptor.Rc,  # 6
        #         device=input_predict.device,
        #     )  # cuda or cpu
        #     X_ch = torch.from_numpy(Descs_ch.astype(np.float32)).clone()
        #     y_pred_ch = (
        #         models["ch"](X_ch.to(input_predict.device)).to("cpu").detach().numpy()
        #     )  # 予測 (NUM_MOL*len(bond_index),3)
        #     # !! ここは形としては(NUM_MOL*len(bond_index),3)となるが，予測だけする場合NUM_MOLの情報をgetできないのでreshape(-1,3)としてしまう．
        #     y_pred_ch = y_pred_ch.reshape((-1, 3))
        #     list_wc_coords.append(
        #         (
        #             bond_centers
        #             + y_pred_ch
        #             / (Constant.Ang * Constant.Charge / Constant.Debye)
        #             / (-2.0)
        #         ).reshape(NUM_MOL, -1, 3)
        #     )
        #     list_wc_symbols.append(100)
        #     del Descs_ch
        #     sum_dipole += np.sum(y_pred_ch, axis=0)  # 双極子に加算
        #     if input_general.save_bonddipole:
        #         np.savetxt(
        #             dipole_files["ch"],
        #             np.hstack(
        #                 [
        #                     np.ones((len(y_pred_ch), 1)) * fr_index,
        #                     np.arange(len(y_pred_ch)).reshape(-1, 1),
        #                     y_pred_ch,
        #                 ]
        #             ),
        #             fmt="%d %d %f %f %f",
        #         )

        # # co, oh, cc, o
        # if len(itp_data.bond_index["CO_1_bond"]) != 0 and models["co"] is not None:
        #     bond_centers = np.array(list_bond_centers)[
        #         :, itp_data.bond_index["CO_1_bond"], :
        #     ].reshape((-1, 3))
        #     Descs_co = DESC.calc_descriptor(
        #         atoms=fr_atoms,
        #         bond_centers=bond_centers,
        #         list_atomic_number=[6, 1, 8],
        #         list_maxat=[24, 24, 24],
        #         Rcs=input_descriptor.Rcs,
        #         Rc=input_descriptor.Rc,
        #         device=input_predict.device,
        #     )
        #     X_co = torch.from_numpy(
        #         Descs_co.astype(np.float32)
        #     ).clone()  # オリジナルの記述子を一旦tensorへ
        #     y_pred_co = (
        #         models["co"](X_co.to(input_predict.device)).to("cpu").detach().numpy()
        #     )
        #     list_wc_coords.append(
        #         (
        #             bond_centers
        #             + y_pred_co
        #             / (Constant.Ang * Constant.Charge / Constant.Debye)
        #             / (-2.0)
        #         ).reshape(NUM_MOL, -1, 3)
        #     )
        #     list_wc_symbols.append(101)
        #     y_pred_co = y_pred_co.reshape((-1, 3))
        #     del Descs_co
        #     sum_dipole += np.sum(y_pred_co, axis=0)  # 双極子に加算
        #     if input_general.save_bonddipole:
        #         np.savetxt(
        #             dipole_files["co"],
        #             np.hstack(
        #                 [
        #                     np.ones((len(y_pred_co), 1)) * fr_index,
        #                     np.arange(len(y_pred_co)).reshape(-1, 1),
        #                     y_pred_co,
        #                 ]
        #             ),
        #             fmt="%d %d %f %f %f",
        #         )

        # if len(itp_data.bond_index["CC_1_bond"]) != 0 and models["cc"] is not None:
        #     bond_centers = np.array(list_bond_centers)[
        #         :, itp_data.bond_index["CC_1_bond"], :
        #     ].reshape((-1, 3))
        #     Descs_cc = DESC.calc_descriptor(
        #         atoms=fr_atoms,
        #         bond_centers=bond_centers,
        #         list_atomic_number=[6, 1, 8],
        #         list_maxat=[24, 24, 24],
        #         Rcs=input_descriptor.Rcs,
        #         Rc=input_descriptor.Rc,
        #         device=input_predict.device,
        #     )
        #     X_cc = torch.from_numpy(
        #         Descs_cc.astype(np.float32)
        #     ).clone()  # オリジナルの記述子を一旦tensorへ
        #     y_pred_cc = (
        #         models["cc"](X_cc.to(input_predict.device)).to("cpu").detach().numpy()
        #     )
        #     list_wc_coords.append(
        #         (
        #             bond_centers
        #             + y_pred_cc
        #             / (Constant.Ang * Constant.Charge / Constant.Debye)
        #             / (-2.0)
        #         ).reshape(NUM_MOL, -1, 3)
        #     )
        #     list_wc_symbols.append(102)
        #     y_pred_cc = y_pred_cc.reshape((-1, 3))
        #     del Descs_cc
        #     sum_dipole += np.sum(y_pred_cc, axis=0)
        #     if input_general.save_bonddipole:
        #         np.savetxt(
        #             dipole_files["cc"],
        #             np.hstack(
        #                 [
        #                     np.ones((len(y_pred_cc), 1)) * fr_index,
        #                     np.arange(len(y_pred_cc)).reshape(-1, 1),
        #                     y_pred_cc,
        #                 ]
        #             ),
        #             fmt="%d %d %f %f %f",
        #         )

        # if len(itp_data.bond_index["OH_1_bond"]) != 0 and models["oh"] is not None:
        #     bond_centers = np.array(list_bond_centers)[
        #         :, itp_data.bond_index["OH_1_bond"], :
        #     ].reshape((-1, 3))
        #     Descs_oh = DESC.calc_descriptor(
        #         atoms=fr_atoms,
        #         bond_centers=bond_centers,
        #         list_atomic_number=[6, 1, 8],
        #         list_maxat=[24, 24, 24],
        #         Rcs=input_descriptor.Rcs,
        #         Rc=input_descriptor.Rc,
        #         device=input_predict.device,
        #     )
        #     X_oh = torch.from_numpy(
        #         Descs_oh.astype(np.float32)
        #     ).clone()  # オリジナルの記述子を一旦tensorへ
        #     y_pred_oh = (
        #         models["oh"](X_oh.to(input_predict.device)).to("cpu").detach().numpy()
        #     )
        #     y_pred_oh = y_pred_oh.reshape((-1, 3))
        #     list_wc_coords.append(
        #         (
        #             bond_centers
        #             + y_pred_oh
        #             / (Constant.Ang * Constant.Charge / Constant.Debye)
        #             / (-2.0)
        #         ).reshape(NUM_MOL, -1, 3)
        #     )
        #     list_wc_symbols.append(103)
        #     del Descs_oh
        #     sum_dipole += np.sum(y_pred_oh, axis=0)
        #     if input_general.save_bonddipole:
        #         np.savetxt(
        #             dipole_files["oh"],
        #             np.hstack(
        #                 [
        #                     np.ones((len(y_pred_oh), 1)) * fr_index,
        #                     np.arange(len(y_pred_oh)).reshape(-1, 1),
        #                     y_pred_oh,
        #                 ]
        #             ),
        #             fmt="%d %d %f %f %f",
        #         )

        # if len(itp_data.o_list) != 0 and models["o"] is not None:
        #     o_positions = fr_atoms.get_positions()[
        #         np.argwhere(fr_atoms.get_atomic_numbers() == 8).reshape(-1)
        #     ]  # o原子の座標
        #     # o_positions = o_positions.reshape((-1,3)) # これを入れないと，[*,1,3]の形になってしまう
        #     Descs_o = DESC.calc_descriptor(
        #         atoms=fr_atoms,
        #         bond_centers=o_positions,
        #         list_atomic_number=[6, 1, 8],
        #         list_maxat=[24, 24, 24],
        #         Rcs=input_descriptor.Rcs,
        #         Rc=input_descriptor.Rc,
        #         device=input_predict.device,
        #     )
        #     X_o = torch.from_numpy(
        #         Descs_o.astype(np.float32)
        #     ).clone()  # オリジナルの記述子を一旦tensorへ
        #     y_pred_o = (
        #         models["o"](X_o.to(input_predict.device)).to("cpu").detach().numpy()
        #     )
        #     y_pred_o = y_pred_o.reshape((-1, 3))
        #     list_wc_coords.append(
        #         (
        #             o_positions
        #             + y_pred_o
        #             / (Constant.Ang * Constant.Charge / Constant.Debye)
        #             / (-4.0)
        #         ).reshape(NUM_MOL, -1, 3)
        #     )
        #     list_wc_symbols.append(10)
        #     del Descs_o
        #     sum_dipole += np.sum(y_pred_o, axis=0)
        #     if input_general.save_bonddipole:
        #         np.savetxt(
        #             dipole_files["o"],
        #             np.hstack(
        #                 [
        #                     np.ones((len(y_pred_o), 1)) * fr_index,
        #                     np.arange(len(y_pred_o)).reshape(-1, 1),
        #                     y_pred_o,
        #                 ]
        #             ),
        #             fmt="%d %d %f %f %f",
        #         )

        # # !! >>>> ここからCOH/COC >>>
        # if len(itp_data.coc_index) != 0 and models["coc"] is not None:
        #     # TODO :: このままだと通常のo_listを使ってしまっていてまずい．
        #     # TODO :: ちゃんとcohに対応したo_listを作るようにする．
        #     o_positions = fr_atoms.get_positions()[
        #         np.argwhere(fr_atoms.get_atomic_numbers() == 8).reshape(-1)
        #     ]  # o原子の座標
        #     Descs_coc = DESC.calc_descriptor(
        #         atoms=fr_atoms,
        #         bond_centers=o_positions,
        #         list_atomic_number=[6, 1, 8],
        #         list_maxat=[24, 24, 24],
        #         Rcs=input_descriptor.Rcs,
        #         Rc=input_descriptor.Rc,
        #         device=input_predict.device,
        #     )
        #     X_coc = torch.from_numpy(
        #         Descs_coc.astype(np.float32)
        #     ).clone()  # オリジナルの記述子を一旦tensorへ
        #     y_pred_coc = (
        #         models["coc"](X_coc.to(input_predict.device)).to("cpu").detach().numpy()
        #     )
        #     y_pred_coc = y_pred_coc.reshape((-1, 3))
        #     del Descs_coc
        #     sum_dipole += np.sum(y_pred_coc, axis=0)
        #     if input_general.save_bonddipole:
        #         np.savetxt(
        #             dipole_files["coc"],
        #             np.hstack(
        #                 [
        #                     np.ones((len(y_pred_coc), 1)) * fr_index,
        #                     np.arange(len(y_pred_coc)).reshape(-1, 1),
        #                     y_pred_coc,
        #                 ]
        #             ),
        #             fmt="%d %d %f %f %f",
        #         )

        # if len(itp_data.coh_index) != 0 and models["coh"] is not None:
        #     # TODO :: このままだと通常のo_listを使ってしまっていてまずい．
        #     # TODO :: ちゃんとcohに対応したo_listを作るようにする．
        #     o_positions = fr_atoms.get_positions()[
        #         np.argwhere(fr_atoms.get_atomic_numbers() == 8).reshape(-1)
        #     ]  # o原子の座標
        #     Descs_coh = DESC.calc_descriptor(
        #         atoms=fr_atoms,
        #         bond_centers=o_positions,
        #         list_atomic_number=[6, 1, 8],
        #         list_maxat=[24, 24, 24],
        #         Rcs=input_descriptor.Rcs,
        #         Rc=input_descriptor.Rc,
        #         device=input_predict.device,
        #     )
        #     X_coh = torch.from_numpy(
        #         Descs_coh.astype(np.float32)
        #     ).clone()  # オリジナルの記述子を一旦tensorへ
        #     y_pred_coh = (
        #         models["coh"](X_coh.to(input_predict.device)).to("cpu").detach().numpy()
        #     )
        #     y_pred_coh = y_pred_coh.reshape((-1, 3))
        #     del Descs_coh
        #     sum_dipole += np.sum(y_pred_coh, axis=0)
        #     if input_general.save_bonddipole:
        #         np.savetxt(
        #             dipole_files["coh"],
        #             np.hstack(
        #                 [
        #                     np.ones((len(y_pred_coh), 1)) * fr_index,
        #                     np.arange(len(y_pred_coh)).reshape(-1, 1),
        #                     y_pred_coh,
        #                 ]
        #             ),
        #             fmt="%d %d %f %f %f",
        #         )

        # !! <<< ここまでCOH/COC <<<
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
        ase.io.write(input_general.savedir + "/mol_wc.xyz", atoms_wc, append=True)

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
