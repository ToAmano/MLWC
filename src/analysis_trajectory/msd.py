

import numpy as np
import ase
from cpmd.pbc.pbc_numpy import pbc_3d
from include.mlwc_logger import setup_cmdline_logger
logger = setup_cmdline_logger(__name__)


def calculate_msd(list_atoms: list[ase.Atoms], initial_step: int = 0) -> list[float]:
    # get atomic positions
    len = len(list_atoms) - initial_step
    # extract all the atomic position (n_frames, n_atoms, 3)
    all_positions = np.array([atoms.positions for atoms in list_atoms])
    # extract not X atoms
    if_notx = (np.array(list_atoms[0].get_chemical_symbols()) != "X")
    selected_positions = all_positions[initial_step:, if_notx, :]
    #
    diff_positions = selected_positions-selected_positions[0]
    # apply PBC
    diff_positions_pbc = pbc_3d.compute_pbc(diff_positions)
    #
    diff_distance = np.linalg.norm(diff_positions_pbc, axis=2)**2
    msd = np.mean(diff_distance, axis=1)
    return msd

    # msd = []
    # L = list_atoms[initial_step].get_cell()[0][0]  # get cell
    # logger.info(f"Lattice constant (a[0][0]): {L}")
    # for i in range(initial_step, len(list_atoms)):  # loop over MD step
    #     msd.append(0.0)
    #     X_counter = 0
    #     for j in range(len(list_atoms[i])):  # loop over atom
    #         if list_atoms[i][j].symbol == "X":  # skip WC
    #             X_counter += 1
    #             continue
    #         # treat the periodic boundary condition
    #         drs = list_atoms[i][j].position - \
    #             list_atoms[initial_step][j].position
    #         tmp = np.where(drs > L/2, drs-L, drs)
    #         msd[-1] += np.linalg.norm(tmp)**2  # こういう書き方ができるのか．．．
    #     msd[-1] /= (len(list_atoms[i])-X_counter)
    # return msd
