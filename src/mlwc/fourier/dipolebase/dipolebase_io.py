import numpy as np

from mlwc.include.mlwc_logger import setup_library_logger

logger = setup_library_logger(__name__)


class DipoleParser:
    """make totaldipole instance from total_dipole.txt file"""

    @classmethod
    def get_timestep(cls, filename: str) -> int:
        """extract timestep from total_dipole.txt"""
        with open(filename, mode="r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#TIMESTEP"):
                    return float(line.split()[1])
        raise ValueError("Missing #TIMESTEP in file.")

    @classmethod
    def get_unitcell(cls, filename: str):
        """extract unitcell from total_dipole.txt"""
        with open(filename, mode="r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#UNITCELL"):
                    unitcell = line.strip("\n").strip().split(" ")[1:]
                    unitcell = np.array([float(i) for i in unitcell]).reshape([3, 3])
                    # values = list(map(float, line.strip().split()[1:]))
                    return unitcell
        raise ValueError("Missing #UNITCELL in file.")

    @classmethod
    def get_temperature(cls, filename: str) -> float:
        """extract unitcell from total_dipole.txt"""
        with open(filename, mode="r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#TEMPERATURE"):
                    return float(line.split()[1])
        raise ValueError("Missing #TEMPERATURE in file.")
