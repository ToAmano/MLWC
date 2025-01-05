"""
totaldipole_io.py

read total_dipole.txt file and return totaldipole instance

"""

import numpy as np
from diel.totaldipole.totaldipole import totaldipole
from include.mlwc_logger import root_logger
logger = root_logger(__name__)

class create_totaldipole:
    @classmethod
    def get_timestep(cls,filename)->int:
        """extract timestep from total_dipole.txt
        """
        with open(filename) as f:
            line = f.readline()
            while line:
                line = f.readline()
                if line.startswith("#TIMESTEP"):
                    time = float(line.split(" ")[1]) 
                    break
        return time
    
    
    @classmethod
    def get_unitcell(cls,filename):
        """extract unitcell from total_dipole.txt
        """
        with open(filename) as f:
            line = f.readline()
            while line:
                line = f.readline()
                if line.startswith("#UNITCELL"):
                    unitcell = line.strip("\n").strip().split(" ")[1:]
                    break
        unitcell = np.array([float(i) for i in unitcell]).reshape([3,3]) 
        return unitcell
    
    @classmethod
    def get_temperature(cls,filename)->float:
        """extract unitcell from total_dipole.txt
        """
        with open(filename) as f:
            line = f.readline()
            while line:
                line = f.readline()
                if line.startswith("#TEMPERATURE"):
                    temp = float(line.split(" ")[1]) 
                    break
        temperature = temp
        return temp

def read_file(totaldipole_filename:str):
    """read total_dipole.txt file and return totaldipole instance

    Args:
        totaldipole_filename (str): total_dipole.txt file

    Returns:
        _type_: totaldipole instance
    """
    totaldipole_instance = totaldipole()    
    data = np.loadtxt(totaldipole_filename,comments='#') # load txt in numpy ndarray
    time = create_totaldipole.get_timestep(totaldipole_filename)
    temp = create_totaldipole.get_temperature(totaldipole_filename)
    cell = create_totaldipole.get_unitcell(totaldipole_filename)
    totaldipole_instance.set_params(data,cell,time,temp) 
    return totaldipole_instance