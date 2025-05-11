"""
totaldipole_io.py

read total_dipole.txt file and return totaldipole instance

"""

import numpy as np

from mlwc.fourier.totaldipole.totaldipole import TotalDipole
from mlwc.include.mlwc_logger import setup_library_logger

logger = setup_library_logger(__name__)


class CreateTotalDipole:
    """make totaldipole instance from total_dipole.txt file"""

    @classmethod
    def get_timestep(cls, filename: str) -> int:
        """extract timestep from total_dipole.txt"""
        with open(filename, mode="r", encoding="utf-8") as f:
            line = f.readline()
            while line:
                line = f.readline()
                if line.startswith("#TIMESTEP"):
                    time = float(line.split(" ")[1])
                    break
        return time

    @classmethod
    def get_unitcell(cls, filename: str):
        """extract unitcell from total_dipole.txt"""
        with open(filename, mode="r", encoding="utf-8") as f:
            line = f.readline()
            while line:
                line = f.readline()
                if line.startswith("#UNITCELL"):
                    unitcell = line.strip("\n").strip().split(" ")[1:]
                    break
        unitcell = np.array([float(i) for i in unitcell]).reshape([3, 3])
        return unitcell

    @classmethod
    def get_temperature(cls, filename: str) -> float:
        """extract unitcell from total_dipole.txt"""
        with open(filename, mode="r", encoding="utf-8") as f:
            line = f.readline()
            while line:
                line = f.readline()
                if line.startswith("#TEMPERATURE"):
                    temperature: float = float(line.split(" ")[1])
                    break
        return temperature


def read_file(
    totaldipole_filename: str, start: int | None = None, end: int | None = None
):
    """read total_dipole.txt file and return totaldipole instance
    Numpyのnp.readtxtやnp.readの実装を参考にしている．
    すなわち，create_totaldipoleの部分はクラスにしておいて，呼び出し自体はメソッドとして定義しない．


    Args:
        totaldipole_filename (str): total_dipole.txt file

    Returns:
        _type_: totaldipole instance
    """
    totaldipole_instance = TotalDipole()
    # load txt in numpy ndarray
    data = np.loadtxt(totaldipole_filename, comments="#")
    data = data[start:end, 1:]  # only retain 3D dipole vector
    time = CreateTotalDipole.get_timestep(totaldipole_filename)
    temp = CreateTotalDipole.get_temperature(totaldipole_filename)
    cell = CreateTotalDipole.get_unitcell(totaldipole_filename)
    totaldipole_instance.set_params(data, cell, time, temp)
    return totaldipole_instance
