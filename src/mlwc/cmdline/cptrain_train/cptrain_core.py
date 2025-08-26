from __future__ import annotations

import os
from typing import List, Union

import ase
import ase.io
from ase.io.trajectory import Trajectory

from mlwc.bond.extractor_itp import ReadItpFile
from mlwc.bond.extractor_rdkit import create_molecular_info
from mlwc.cpmd.assign_wcs.assign_wcs_torch import atoms_wan
from mlwc.include.mlwc_logger import setup_library_logger, timer_dec

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


def _load_itp_data(filepath: str):
    """load bond data from a file"""
    if not os.path.isfile(filepath):
        logger.error("ITP file does not exist: %s", filepath)
        raise FileNotFoundError(f"Missing ITP file: {filepath}")

    if filepath.endswith(".itp"):
        itp_data = ReadItpFile(filepath)
    elif filepath.endswith(".mol"):
        itp_data = create_molecular_info(filepath)
    else:
        raise ValueError(f"Unsupported file format for ITP: {filepath}")
    return itp_data


def _validate_xyz_with_mol(file_list: Union[List[str], str], itp_data) -> None:
    """Check consistency with mol and xyz"""
    file_list: List[str] = _normalize_to_list(file_list)
    for xyz_file in file_list:
        atoms = ase.io.read(xyz_file, index="1")  # only read the second configuration
        if (
            atoms.get_chemical_symbols()[: itp_data.num_atoms_per_mol]
            != itp_data.atom_list
        ):
            raise ValueError(f"Atom mismatch in file: {xyz_file}")


def _load_trajectory_file(xyz_file: str) -> List["ase.Atoms"]:
    """Attempts to load a trajectory from a file using ASE.

    Tries to use `ase.io.read` first; falls back to `Trajectory` if needed.
    Raises informative errors if both methods fail.
    """
    if not os.path.isfile(xyz_file):
        raise FileNotFoundError(f"Trajectory file not found: {xyz_file}")

    try:  # it can also handle ase.db file
        traj = ase.io.read(xyz_file, index=":")
        return traj
    except Exception:
        logger.debug("ase.io.read failed for %s", xyz_file)

    try:
        traj = Trajectory(xyz_file)
        return traj
    except Exception as exc:
        logger.error("Failed to load trajectory from %s.", xyz_file)
        raise RuntimeError(f"Failed to load trajectory from {xyz_file}") from exc


@timer_dec(logger)
def _load_trajectory_data(
    file_list: Union[List[str], str],
) -> List[ase.Atoms]:
    """load trajectory data using ase"""
    logger.info("")
    logger.info(" Loading xyz file")
    logger.info("=================")
    logger.info("")
    file_list = _normalize_to_list(file_list)
    logger.info(" found %d system(s):", len(file_list))
    logger.info(
        "%s  %6s %6s",
        _format_name_length("system", 42),
        "num_frames",
        "natoms(include WC)",
    )  # TODO :: print batch size/ batch num??

    all_frames = []
    for filepath in file_list:
        traj = _load_trajectory_file(filepath)
        all_frames.extend(traj)
        logger.info(
            "%s  %10d %18d",
            _format_name_length(filepath, 42),
            len(traj),  # num of frames
            len(traj[0].get_atomic_numbers()),
        )
    logger.info(" Total frames loaded: %d", len(all_frames))
    return all_frames


def _normalize_to_list(file_input: Union[str, List[str]]) -> List[str]:
    """Ensure the input is always returned as a list of strings."""
    if isinstance(file_input, str):
        return [file_input]
    if not isinstance(file_input, list):
        raise TypeError(f"Expected str or list[str], got {type(file_input).__name__}")
    return file_input


@timer_dec(logger)
def _generate_atomswan_from_atoms(atoms_list: List[ase.Atoms], itp_data):
    logger.info("")
    logger.info(" Assign WCs to BCs")
    logger.info("==================")
    logger.info("")
    atoms_wan_list: List = []
    result_atoms_list: List[ase.Atoms] = []
    for atoms in atoms_list:  # loop over atoms (frames)
        data = atoms_wan()
        data.set_params_from_atoms(atoms, itp_data)
        atoms_wan_list.append(data)
        result_atoms_list.append(data.make_atoms_with_wc())
    return atoms_wan_list, result_atoms_list
