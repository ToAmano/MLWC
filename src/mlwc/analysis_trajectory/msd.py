"""
This module provides functions for analyzing molecular dynamics trajectories,
including the calculation of Mean Square Displacement (MSD).
"""

import ase
import numpy as np
import pandas as pd

from mlwc.cpmd.pbc.pbc_numpy import pbc_3d
from mlwc.include.mlwc_logger import setup_cmdline_logger

logger = setup_cmdline_logger(__name__)


def calculate_msd(
    list_atoms: list[ase.Atoms], initial_step: int = 0, timestep_fs: float | None = None
) -> list[float]:
    """
    Calculate the Mean Square Displacement (MSD) of atoms in a list of frames.

    The MSD is a measure of the average distance that atoms travel over time.
    It can be used to calculate the diffusion coefficient of a material.

    Parameters
    ----------
    list_atoms : list[ase.Atoms]
        A list of ASE Atoms objects representing the trajectory.
    initial_step : int, optional
        The initial step to consider for MSD calculation. Defaults to 0.
    timestep_fs : float | None, optional
        The time step in femtoseconds. If provided, the diffusion coefficient is also calculated. Defaults to None.

    Returns
    -------
    list[float] or tuple[pd.DataFrame, float]
        If timestep_fs is None, returns a list of MSD values.
        If timestep_fs is provided, returns a tuple containing a Pandas DataFrame with time and MSD values, and the diffusion coefficient.

    Examples
    --------
    Example with timestep:
    >>> import ase.io
    >>> list_atoms = ase.io.read('trajectory.xyz', index=':')
    >>> df, diffusion_coefficient = calculate_msd(list_atoms, initial_step=10, timestep_fs=1.0)
    >>> print(df.head())
           time       msd
    0    10.0  0.000000
    1    11.0  0.001234
    2    12.0  0.002468
    3    13.0  0.003702
    4    14.0  0.004936
    >>> print(diffusion_coefficient)
    1.23456789e-05

    Example without timestep:
    >>> import ase.io
    >>> list_atoms = ase.io.read('trajectory.xyz', index=':')
    >>> msd = calculate_msd(list_atoms, initial_step=10)
    >>> print(msd[:5])
    [0.0, 0.001234, 0.002468, 0.003702, 0.004936]
    """
    cell = list_atoms[0].get_cell()
    # extract all the atomic position (n_frames, n_atoms, 3)
    all_positions = np.array([atoms.positions for atoms in list_atoms])
    # extract not X atoms
    if_notx = np.array(list_atoms[0].get_chemical_symbols()) != "X"
    selected_positions = all_positions[initial_step:, if_notx, :]
    # calculate the difference from the first frame
    diff_positions = selected_positions - selected_positions[0]
    diff_positions_pbc = pbc_3d.compute_pbc(diff_positions, cell)
    diff_distance = np.linalg.norm(diff_positions_pbc, axis=2) ** 2
    msd = np.mean(diff_distance, axis=1)
    if timestep_fs:
        # diffusion coefficient in m^2/s
        # Ang = 10e-10, fs = 10e-15s
        diffusion_coefficient = msd[-1] * 10e-5 / (6 * len(msd) * timestep_fs)
        df = pd.DataFrame()
        df["time"] = np.arange(initial_step, len(list_atoms)) * timestep_fs
        df["msd"] = msd
        return df, diffusion_coefficient
    return msd
