

import numpy as np
import pandas as pd
import ase
from include.mlwc_logger import root_logger
logger = root_logger(__name__)

def calculate_msd(atoms:ase.Atoms,initial_step:int=0)->list[float]:
    # FIXME: loop over atomsを廃止し，distance関数を使う．
    msd = []
    L = atoms[initial_step].get_cell()[0][0] # get cell
    logger.info(f"Lattice constant (a[0][0]): {L}")
    for i in range(initial_step,len(atoms)): # loop over MD step
        msd.append(0.0)
        X_counter=0
        for j in range(len(atoms[i])): # loop over atom
            if atoms[i][j].symbol == "X": # skip WC
                X_counter += 1
                continue
            # treat the periodic boundary condition
            drs = atoms[i][j].position - atoms[initial_step][j].position
            tmp = np.where(drs>L/2,drs-L,drs)
            msd[-1] += np.linalg.norm(tmp)**2 #こういう書き方ができるのか．．．
        msd[-1] /= (len(atoms[i])-X_counter)
    return msd