
import sys
import ase
import ase.io 
import numpy as np
import pandas as pd
import scipy
import argparse
import matplotlib.pyplot as plt
from include.mlwc_logger import root_logger
logger = root_logger(__name__)


class MSD:
    """ class to calculate mean-square displacement
        See 
    Returns:
        _type_: _description_
    """
    def __init__(self,filename:str,initial_step:int=1):
        self.__filename = filename # xyz
        self.__initial_step = initial_step # initial step to calculate msd
        import os
        if not os.path.isfile(self.__filename):
            raise FileNotFoundError(" ERROR :: "+str(self.__filename)+" does not exist !!")
        
        if self.__initial_step < 1:
            raise ValueError("ERROR: initial_step must be larger than 1")
        
        # read xyz
        logger.info(" READING TRAJECTORY... This may take a while, be patient.")
        self.__traj = ase.io.read(self.__filename,index=":")
        
    def calc_msd(self):
        """calculate msd

        Returns:
            _type_: _description_
        """
        import numpy as np
        msd = []
        L = self.__traj[self.__initial_step].get_cell()[0][0] # get cell
        logger.info(f"Lattice constant (a[0][0]): {L}")
        for i in range(self.__initial_step,len(self.__traj)): # loop over MD step
            msd.append(0.0)
            X_counter=0
            for j in range(len(self.__traj[i])): # loop over atom
                if self.__traj[i][j].symbol == "X": # skip WC
                    X_counter += 1
                    continue
                # treat the periodic boundary condition
                drs = self.__traj[i][j].position - self.__traj[self.__initial_step][j].position
                tmp = np.where(drs>L/2,drs-L,drs)
                msd[-1] += np.linalg.norm(tmp)**2 #こういう書き方ができるのか．．．
            msd[-1] /= (len(self.__traj[i])-X_counter)
        # 計算されたmsdを保存する．
        df = pd.DataFrame()
        df["msd"] = msd
        df["step"] = np.arange(self.__initial_step,len(self.__traj))
        df.to_csv(self.__filename+"_msd.csv")
        return msd
    
    
def command_cpmd_msd(args): #平均移動距離
    msd = MSD(args.Filename,args.initial)
    msd.calc_msd()
    return 0