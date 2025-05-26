"""
totaldipole_io.py

read total_dipole.txt file and return totaldipole instance

"""

import numpy as np

from mlwc.fourier.dipolebase.dipolebase_io import DipoleParser
from mlwc.fourier.totaldipole.totaldipole import TotalDipole
from mlwc.include.mlwc_logger import setup_library_logger

logger = setup_library_logger(__name__)


def read_file(
    totaldipole_filename: str, start: int | None = None, end: int | None = None
) -> TotalDipole:
    """read total_dipole.txt file and return totaldipole instance
    Numpyのnp.readtxtやnp.readの実装を参考にしている．
    すなわち，create_totaldipoleの部分はクラスにしておいて，呼び出し自体はメソッドとして定義しない．


    Args:
        totaldipole_filename (str): total_dipole.txt file

    Returns:
        _type_: totaldipole instance
    """
    try:
        logger.debug("Reading data from: %s", totaldipole_filename)
        data = np.loadtxt(totaldipole_filename, comments="#")[
            start:end, 1:
        ]  # Drop time column
        time = DipoleParser.get_timestep(totaldipole_filename)
        temp = DipoleParser.get_temperature(totaldipole_filename)
        cell = DipoleParser.get_unitcell(totaldipole_filename)

        totaldipole_instance = TotalDipole()
        totaldipole_instance.set_params(data, cell, time, temp)
        return totaldipole_instance

    except Exception as e:
        logger.error("Failed to read file {totaldipole_filename}: {e}")
        raise
