from __future__ import annotations

import os
from typing import List

import ase
import ase.io
from ase.io.trajectory import Trajectory

from mlwc.bond.extractor_itp import ReadItpFile
from mlwc.bond.extractor_rdkit import create_molecular_info
from mlwc.include.mlwc_logger import setup_library_logger, timer_dec

logger = setup_library_logger("MLWC." + __name__)


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
    logger.info(
        " The number of atoms in a molecule :: %s",
        itp_data.num_atoms_per_mol,
    )
    return itp_data


def _validate_xyz_with_mol(xyz_files: List[str], itp_data) -> None:
    """Check consistency with mol and xyz"""
    for xyz_file in xyz_files:
        atoms = ase.io.read(xyz_file, index="1")  # only read the second configuration
        if (
            atoms.get_chemical_symbols()[: itp_data.num_atoms_per_mol]
            != itp_data.atom_list
        ):
            raise ValueError(f"Atom mismatch in file: {xyz_file}")


@timer_dec(logger)
def _load_trajectory_file(xyz_file: str) -> List["ase.Atoms"]:
    """Attempts to load a trajectory from a file using ASE.

    Tries to use `ase.io.read` first; falls back to `Trajectory` if needed.
    Raises informative errors if both methods fail.
    """
    logger.info(" Loading xyz file :: %s", xyz_file)
    if not os.path.isfile(xyz_file):
        raise FileNotFoundError(f"Trajectory file not found: {xyz_file}")

    try:  # it can also handle ase.db file
        traj = ase.io.read(xyz_file, index=":")
        logger.info(" Finish loading xyz file. len(traj) = %d", len(traj))
        return traj
    except Exception:
        logger.debug("ase.io.read failed for %s", xyz_file)

    try:
        traj = Trajectory(xyz_file)
        logger.info(" Finish loading xyz file. len(traj) = %d", len(traj))
        return traj
    except Exception as exc:
        logger.error("Failed to load trajectory from %s.", xyz_file)
        raise RuntimeError(f"Failed to load trajectory from {xyz_file}") from exc


@timer_dec(logger)
def _load_trajectory_data(file_list: List[str]) -> List[List["ase.Atoms"]]:
    """load trajectory data using ase"""
    atoms_list = []
    for xyz_file in file_list:
        traj = _load_trajectory_file(xyz_file)
        atoms_list.append(traj)
        logger.info("Loaded %s with %d frames.", xyz_file, len(traj))
    return atoms_list
