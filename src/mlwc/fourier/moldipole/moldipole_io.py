import numpy as np

from mlwc.fourier.dipolebase.dipolebase_io import DipoleParser
from mlwc.fourier.moldipole.moldipole import moldipole
from mlwc.include.mlwc_logger import setup_cmdline_logger

logger = setup_cmdline_logger("MLWC." + __name__)


def read_file(moldipole_filename: str):
    """read moldipole type file"""
    moldipole_instance = moldipole()
    # load txt in numpy ndarray
    data = np.loadtxt(moldipole_filename, comments="#")
    NUM_MOL = int(np.max(data[:, 1])) + 1
    # データ形状を変更[frame,mol_id,3dvector]
    data = data[:, 2:].reshape(-1, NUM_MOL, 3)
    time = DipoleParser.get_timestep(moldipole_filename)
    temp = DipoleParser.get_temperature(moldipole_filename)
    cell = DipoleParser.get_unitcell(moldipole_filename)
    moldipole_instance.set_params(data, cell, time, temp)
    return moldipole_instance
