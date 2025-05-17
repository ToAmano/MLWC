import numpy as np

from mlwc.include.mlwc_logger import setup_cmdline_logger

logger = setup_cmdline_logger("MLWC." + __name__)


def calc_bondcenter(atomic_positions: np.ndarray, bond_list: list[list]) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    atomic_positions : np.ndarray
        num_mol*num_atoms*3 shape
    bond_list : list[list]
        _description_

    Returns
    -------
    np.ndarray
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    # check the shape of atomic_positions
    if atomic_positions.ndim != 3 or atomic_positions.shape[2] != 3:
        raise ValueError(
            f"Invalid shape for vectors_array. Expected shape [a, b, 3], but got {np.shape(atomic_positions)}."
        )
    # atomic coords of both end of bonds
    bond_list = np.array(bond_list)
    start_points = atomic_positions[:, bond_list[:, 0], :]
    end_points = atomic_positions[:, bond_list[:, 1], :]

    # calculate BCs
    bond_centers = (start_points + end_points) / 2.0

    return bond_centers


def calc_bondcenter_dict(atomic_positions: np.ndarray, dict_bonds: dict):
    dict_bcs = {
        key: (
            (
                atomic_positions[:, np.array(item)[:, 0], :]
                + atomic_positions[:, np.array(item)[:, 1], :]
            )
            / 2
            if item
            else np.array([])
        )
        for key, item in dict_bonds.items()
    }
    return dict_bcs
