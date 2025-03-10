import sys
import ase.units
import ase.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cpmd.read_core
import cpmd.read_traj
from diel.moldipole.moldipole import moldipole
from include.mlwc_logger import root_logger
logger = root_logger("MLWC."+__name__)

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
                    unitcell = line.strip("\n").split(" ")[1:]
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

def read_file(moldipole_filename:str):
    moldipole_instance = moldipole()    
    data = np.loadtxt(moldipole_filename,comments='#') # load txt in numpy ndarray
    NUM_MOL = int(np.max(data[:,1]))+1    
    # データ形状を変更[frame,mol_id,3dvector]
    data = data[:,2:].reshape(-1, NUM_MOL,3)    
    time = create_totaldipole.get_timestep(moldipole_filename)
    temp = create_totaldipole.get_temperature(moldipole_filename)
    cell = create_totaldipole.get_unitcell(moldipole_filename)
    moldipole_instance.set_params(data,cell,time,temp) 
    return moldipole_instance