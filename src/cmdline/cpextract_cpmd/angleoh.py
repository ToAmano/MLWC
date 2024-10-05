
import sys
import numpy as np
import pandas as pd
import scipy
import argparse
import matplotlib.pyplot as plt
import cpmd.read_core
import cpmd.read_traj
from include.mlwc_logger import root_logger

try:
    import ase.units
except ImportError:
    sys.exit("Error: ase not installed")

logger = root_logger(__name__)


class ANGLEOH:
    """ class to calculate mean-square displacement
        See 
    Returns:
        _type_: _description_
    """
    def __init__(self,filename:str,molfile:str, timestep:float,NUM_ATOM_PER_MOL:int,initial_step:int=1):
        self.__filename = filename # xyz
        self.__initial_step = initial_step # initial step to calculate msd
        self.__molfile = molfile # .mol
        # timestep in [fs]
        self._timestep = timestep # timestep in [fs]
        self._NUM_ATOM_PER_MOL = NUM_ATOM_PER_MOL # including WCs & BCs
        
        import os
        if not os.path.isfile(self.__filename):
            raise FileNotFoundError(" ERROR :: "+str(self.__filename)+" does not exist !!")
        
        if self.__initial_step < 1:
            raise ValueError("ERROR: initial_step must be larger than 1")

        if not os.path.isfile(self.__molfile):
            raise FileNotFoundError(" ERROR :: "+str(self.__molfile)+" does not exist !!")
        
        
        # * itpデータの読み込み
        # note :: itpファイルは記述子からデータを読み込む場合は不要なのでコメントアウトしておく
        import ml.atomtype
        # 実際の読み込み
        if self.__molfile.endswith(".itp"):
            self.itp_data=ml.atomtype.read_itp(self.__molfile)
        elif self.__molfile.endswith(".mol"):
            self.itp_data=ml.atomtype.read_mol(self.__molfile)
        else:
            raise ValueError("ERROR :: itp_filename should end with .itp or .mol")
        # bonds_list=itp_data.bonds_list
        # NUM_MOL_ATOMS=self.itp_data.num_atoms_per_mol
        
        # read xyz
        import ase
        import ase.io 
        logger.info(" READING TRAJECTORY... This may take a while, be patient.")
        self._traj = ase.io.read(self.__filename,index=":")
        logger.info(f" FINISH READING TRAJECTORY... :: len(traj) = {len(self._traj)}")
        # 
        self.NUM_MOL = len(self._traj[0])//self._NUM_ATOM_PER_MOL
        assert len(self._traj[0])%self._NUM_ATOM_PER_MOL == 0, "ERROR: Number of atoms in the first step is not divisible by the number of atoms per molecule"
        logger.info(f" NUM_MOL == {self.NUM_MOL}")

        
    def calc_angleoh(self):
        """calculate vdos

        Returns:
            _type_: _description_
        """
        import numpy as np
        import diel.hydrogenbond
        # Oリスト
        o_list = []
        h_list = []
        for [a,b] in self.itp_data.oh_bond:
            print(a,b)
            if a in self.itp_data.o_list:
                o_list.append(a)
                h_list.append(b)
            elif a in self.itp_data.h_list:
                h_list.append(a)
                o_list.append(b)
        print(o_list)
        print(h_list)
        
        # OHボンドのH原子のリスト
        hydrogen_list:list = [self._NUM_ATOM_PER_MOL*mol_id+atom_id for mol_id in range(self.NUM_MOL) for atom_id in h_list ]
        # OHボンドのO原子のリスト
        oxygen_list:list   = [self._NUM_ATOM_PER_MOL*mol_id+atom_id for mol_id in range(self.NUM_MOL) for atom_id in o_list ]
        # calculate OH vector
        bond_vectors = diel.hydrogenbond.calc_oh(self._traj,oxygen_list,hydrogen_list)
        np.save(self.__filename+"_oh_angle_list.npy",bond_vectors)        
        # 全ての時系列に対して自己相関を計算 (axis=1で各行に対して自己相関を計算)
        # 'same' モードで時系列の長さを維持
        # !! numpy correlate does not support FFT
        correlations = np.apply_along_axis(lambda x: scipy.signal.correlate(x, x, mode='full'), axis=0, arr=bond_vectors)
        correlations = np.sum(correlations,axis=2) # inner dot
        
        # 自己相関の平均化 (axis=1で全ての時系列に対する平均を取る)
        mean_correlation = np.mean(correlations, axis=1)[len(bond_vectors)-1:] # acf
        df_acf = diel.hydrogenbond.make_df_acf(mean_correlation,self._timestep)
        df_acf.to_csv(self.__filename+"_oh_acf.csv",index=False)
        logger.info(" acf is saved as "+self.__filename+"_oh_acf.csv")
                
        # Fourier Transform
        df_roo:pd.DataFrame = diel.hydrogenbond.calc_lengthcorr(mean_correlation, self._timestep)
        df_roo.to_csv(self.__filename+"_oh_ft.csv",index=False)
        logger.info(" ft is saved as "+self.__filename+"_oh_ft.csv")
        return 0
    

def command_cpmd_angleoh(args): # 水素結合解析のための角度分布
    angleoh = ANGLEOH(args.filename,args.molfile, float(args.timestep),int(args.numatom),int(args.initial))
    angleoh.calc_angleoh()
    return 0