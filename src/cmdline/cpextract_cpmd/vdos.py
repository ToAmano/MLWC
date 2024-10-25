import sys
import ase
import ase.io 
import numpy as np
import pandas as pd
import scipy
import argparse
import matplotlib.pyplot as plt
import diel.vdos
from include.mlwc_logger import root_logger
logger = root_logger(__name__)



class VDOS:
    """ class to calculate mean-square displacement
        See 
    Returns:
        _type_: _description_
    """
    def __init__(self,filename:str,timestep:float,NUM_ATOM_PER_MOL:int,initial_step:int=1):
        self.__filename = filename # xyz
        self.__initial_step = initial_step # initial step to calculate msd
        import os
        if not os.path.isfile(self.__filename):
            raise FileNotFoundError(" ERROR :: "+str(self.__filename)+" does not exist !!")
        
        if self.__initial_step < 1:
            raise ValueError("ERROR: initial_step must be larger than 1")
        
        # read xyz
        logger.info(" READING TRAJECTORY... This may take a while, be patient.")
        self._traj = ase.io.read(self.__filename,index=":")
        
        # timestep in [fs]
        self._timestep = timestep
        self._NUM_ATOM_PER_MOL = NUM_ATOM_PER_MOL
        
    def calc_com_vdos(self) -> pd.DataFrame:
        # molecular center of mass velosity
        com_velocity = diel.vdos.calc_com_velocity(self._traj,self._NUM_ATOM_PER_MOL, self._timestep)
        com_acf  = diel.vdos.calc_vel_acf(com_velocity)
        # com vdos of molecule
        com_vdos = diel.vdos.calc_vdos(np.mean(com_acf,axis=0), self._timestep)
        com_vdos.to_csv("com_vdos.csv",index=False)
        return com_vdos
    
    def calc_all_vdos(self,NUM_ATOM:int) -> pd.DataFrame:
        # molecular center of mass velosity
        all_velocity = diel.vdos.calc_all_velocity(self._traj,NUM_ATOM, self._timestep)
        all_acf  = diel.vdos.calc_vel_acf(all_velocity)
        # com vdos of molecule
        all_vdos = diel.vdos.calc_vdos(np.mean(all_acf,axis=0), self._timestep)
        all_vdos.to_csv("all_vdos.csv",index=False)
        return all_vdos
    
    def calc_atom_vdos(self,atomic_number:int)-> pd.DataFrame:
        atom_velocity = diel.vdos.calc_atom_velocity(self._traj,atomic_number,self._timestep)
        # calculate acf
        atom_acf = diel.vdos.calc_vel_acf(atom_velocity)
        np.savetxt(f"atom_atomicnumber{atomic_number}_acf.txt", atom_acf) 
        acf_mean = np.mean(atom_acf,axis=0)
        atom_vdos   = diel.vdos.calc_vdos(acf_mean, self._timestep)
        # atom_vdos.to_csv(f"atom_atomicnumber{atomic_number}_vdos.csv",index=False)
        return atom_vdos
    
    def calc_vdos(self):
        """calculate vdos

        Returns:
            _type_: _description_
        """
        # 原子種ごとのvdos
        C_vdos = self.calc_atom_vdos(6)
        C_vdos.to_csv("C_vdos.csv", index=False)
        O_vdos = self.calc_atom_vdos(8)        
        O_vdos.to_csv("O_vdos.csv", index=False)
        H_vdos = self.calc_atom_vdos(1)
        H_vdos.to_csv("H_vdos.csv", index=False)
        
        # WCs
        WO_vdos = self.calc_atom_vdos(10) # dieltoolsではOlpはNe(10)に対応
        WCH_vdos = self.calc_atom_vdos(0)
        WCO_vdos = self.calc_atom_vdos(101)
        WCC_vdos = self.calc_atom_vdos(102)
        WOH_vdos = self.calc_atom_vdos(103)
        WO_vdos.to_csv("WO_vdos.csv", index=False)
        WCH_vdos.to_csv("WCH_vdos.csv", index=False)
        WCO_vdos.to_csv("WCO_vdos.csv", index=False)
        WCC_vdos.to_csv("WCC_vdos.csv", index=False)
        WOH_vdos.to_csv("WOH_vdos.csv", index=False)
        return 0
    

    def calc_vdos_all(self):
        """calculate vdos
        
        indexごとに計算するので，まさにすべての原子種について計算する．

        Returns:
            _type_: _description_
        """
        # atomic velocity
        atom_velocity = diel.vdos.calc_velocity(self._traj,self._timestep)
        # calculate acf
        atom_acf = diel.vdos.calc_vel_acf(atom_velocity)
        np.savetxt("atom_acf.txt", atom_acf) 

        # 原子種ごとvdos
        H_vdos   = diel.vdos.calc_vdos(diel.vdos.average_vdos_atomic_species(atom_acf, self._traj[0], 1), self._timestep)
        C_vdos   = diel.vdos.calc_vdos(diel.vdos.average_vdos_atomic_species(atom_acf, self._traj[0], 6), self._timestep)
        O_vdos   = diel.vdos.calc_vdos(diel.vdos.average_vdos_atomic_species(atom_acf, self._traj[0], 8), self._timestep)
        H_vdos.to_csv("H_vdos.csv", index=False)
        C_vdos.to_csv("C_vdos.csv", index=False)
        O_vdos.to_csv("O_vdos.csv", index=False)
        # WCs
        WO_vdos = diel.vdos.calc_vdos(diel.vdos.average_vdos_atomic_species(atom_acf, self._traj[0], 10), self._timestep) # dieltoolsではOlpはNe(10)に対応
        WCH_vdos = diel.vdos.calc_vdos(diel.vdos.average_vdos_atomic_species(atom_acf, self._traj[0], 0), self._timestep) 
        WCO_vdos = diel.vdos.calc_vdos(diel.vdos.average_vdos_atomic_species(atom_acf, self._traj[0], 101), self._timestep)
        WCC_vdos = diel.vdos.calc_vdos(diel.vdos.average_vdos_atomic_species(atom_acf, self._traj[0], 102), self._timestep) 
        WOH_vdos = diel.vdos.calc_vdos(diel.vdos.average_vdos_atomic_species(atom_acf, self._traj[0], 103), self._timestep) 
        WO_vdos.to_csv("WO_vdos.csv", index=False)
        WCH_vdos.to_csv("WCH_vdos.csv", index=False)
        WCO_vdos.to_csv("WCO_vdos.csv", index=False)
        WCC_vdos.to_csv("WCC_vdos.csv", index=False)
        WOH_vdos.to_csv("WOH_vdos.csv", index=False)
        # TODO: H(CH),H(OH)
        logger.info(" Calculate index base VDOS...")
        print(self._NUM_ATOM_PER_MOL)
        for atomic_index in range(self._NUM_ATOM_PER_MOL): #vdos for all index
            vdos = diel.vdos.calc_vdos(diel.vdos.average_vdos_specify_index(atom_acf,[atomic_index], self._NUM_ATOM_PER_MOL),self._timestep)
            vdos.to_csv(f"Index_{atomic_index}_vdos.csv", index=False)
        # average_vdos_specify_index(acf, atoms, index:list[int], num_atoms_per_mol:int)
        return 0
    
    
def command_cpmd_vdos(args): # 原子ごとのvdos
    vdos = VDOS(args.Filename,float(args.timestep),int(args.numatom),int(args.initial))
    if args.mode == "all":
        vdos.calc_com_vdos()
        vdos.calc_vdos()
    if args.mode == "com":
        vdos.calc_com_vdos()
    if args.mode == "H":
        H_vdos = vdos.calc_atom_vdos(1)
        H_vdos.to_csv("H_vdos.csv", index=False)
    if args.mode == "C":
        C_vdos = vdos.calc_atom_vdos(6)
        C_vdos.to_csv("C_vdos.csv", index=False)
    if args.mode == "O":
        O_vdos = vdos.calc_atom_vdos(8)
        O_vdos.to_csv("O_vdos.csv", index=False)
    if args.mode == "total":
        if args.numatom == None:
            raise ValueError("ERROR: numatom must be specified for total mode")
        vdos.calc_all_vdos(int(args.numatom))
    return 0
