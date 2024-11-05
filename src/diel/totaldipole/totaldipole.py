# 発想を転換して，total_dipole.txtに対するクラスをここで実装する．
# pandas dataframeを継承するかどうかは難しいところ．

# 設計として，インスタンスがtimestep, temperature, unitcell, dataを持つ．

import sys
import ase.units
import ase.io
import numpy as np
# 誘電関数の計算まで
import statsmodels.api as sm 
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import cpmd.read_core
import cpmd.read_traj
from diel.acf_fourier import raw_calc_eps0_dielconst # for calculation of dielectric constant
from include.mlwc_logger import root_logger
logger = root_logger(__name__)

class totaldipole:
    """plot time vs dipole figure for total_dipole
    
    Returns:
        _type_: _description_
    """
    def __init__(self):
        self.timestep:float = None
        self.temperature:float = None
        self.unitcell:np.ndarray = None
        self.data:np.ndarray = None

    def set_params(self,data:np.ndarray,unitcell:np.ndarray,timestep:float,temperature:float):
        if not isinstance(data,np.ndarray):
            raise ValueError(" ERROR :: data is not numpy array")
        if not isinstance(unitcell,np.ndarray):
            raise ValueError(" ERROR :: unitcell is not numpy array")
        if not isinstance(timestep,float):
            raise ValueError(" ERROR :: timestep is not float")
        if not isinstance(temperature,float):
            raise ValueError(" ERROR :: temperature is not float")
        if np.shape(data)[1] != 4:
            raise ValueError(" ERROR :: data shape is not correct")
        self.data = data
        self.unitcell = unitcell
        self.timestep = timestep
        self.temperature = temperature
        return 0

    def print_info(self):
        logger.info(" ============================ ")
        logger.info(f" number of data :: {np.shape(self.data)}")
        logger.info(f" timestep [fs] :: {self.timestep}")
        logger.info(f" temperature [K] :: {self.temperature}")
        logger.info(f" unitcell [Ang] :: {self.unitcell}")
        logger.info(" ============================ ")
        return 0

    def get_volume(self):
        A3 = 1.0e-30
        return np.abs(np.dot(np.cross(self.unitcell[:,0],self.unitcell[:,1]),self.unitcell[:,2])) * A3

    def get_mean_dipole(self):
        dMx=self.data[:,0]#-np.mean(self.data[:,0])
        dMy=self.data[:,1]#-np.mean(self.data[:,1])
        dMz=self.data[:,2]#-np.mean(self.data[:,2])
        # mean_M=np.mean(dMx)**2+np.mean(dMy)**2+np.mean(dMz)**2    # <M>^2
        return np.array([np.mean(dMx),np.mean(dMy),np.mean(dMz)])

    def get_mean_dipolesquare(self):
        dMx=self.data[:,0]#-np.mean(self.data[:,0])
        dMy=self.data[:,1]#-np.mean(self.data[:,1])
        dMz=self.data[:,2]#-np.mean(self.data[:,2])
        mean_M2=np.mean(dMx**2)+np.mean(dMy**2)+np.mean(dMz**2) # <M^2>
        return mean_M2

    def calc_dielconst(self, eps_inf:float=1.0) -> float:
        '''
        eps0だけ計算する．    
        '''
        # cell_dipoles_pred = np.load(filename)
        
        # N=int(np.shape(cell_dipoles_pred)[0]/2)
        # N=int(np.shape(secell_dipoles_pred)[0])
        # N=99001
        # print("nlag :: ", N)

        # >>>>>>>>>>>
        eps0 = 8.8541878128e-12
        debye = 3.33564e-30
        # nm3 = 1.0e-27
        # nm = 1.0e-9
        # A3 = 1.0e-30
        kb = 1.38064852e-23
        # volume of the cell in m^3
        volume:float = self.get_volume()
        # 平均値計算
        mean_M2:float= self.get_mean_dipolesquare() # <M^2>  # (np.mean(dMx_pred**2)+np.mean(dMy_pred**2)+np.mean(dMz_pred**2)) # <M^2>
        mean_M:float = np.sum(np.square(self.get_mean_dipole())) # <M>^2  # np.mean(dMx_pred)**2+np.mean(dMy_pred)**2+np.mean(dMz_pred)**2    # <M>^2
        # 比誘電率
        eps_0:float = eps_inf + ((mean_M2-mean_M)*debye**2)/(3.0*volume*kb*self.temperature*eps0)
        return [eps_0, mean_M2, mean_M]

    def calc_time_vs_dielconst(self,start:int,end:int,eps_inf:float = 1) -> pd.DataFrame:
        """時間 vs 誘電定数の計算を行う

        Returns:
            _type_: _description_
        """
        eps0_list=[]
        mean_M2_list=[]
        mean_M_list=[]
        time_list = []

        if start > len(self.data):
            raise ValueError(" ERROR :: start is larger than data length")
        if end > len(self.data):
            raise ValueError(" ERROR :: end is larger than data length")

        if end == -1:
            calc_data = self.data[start:,1:]
        else:
            calc_data = self.data[start:end,1:]
        logger.info(f"length calc_data :: {len(calc_data)}")

        SAMPLE=100 #!! hard code
        for index in range(len(calc_data)):
            if index == 0:
                continue
            if index %SAMPLE == 0:
                logger.debug(index)
                # TODO :: replace with newly implemented function
                [eps_0_tmp, M2_tmp, M_tmp] = raw_calc_eps0_dielconst(calc_data[:index,:],self.unitcell,self.temperature,eps_inf)
                eps0_list.append(eps_0_tmp)
                mean_M2_list.append(M2_tmp)
                mean_M_list.append(M_tmp)
                time_list.append(index*self.timestep)
        # データの保存
        df = pd.DataFrame()
        df["time_fs"] = time_list # in fs
        df["eps0"] = eps0_list
        df["mean_M2"] = mean_M2_list
        df["mean_M"] = mean_M_list
        df.to_csv("eps0_vs_time.csv",index=False)
        return df
    
    @classmethod
    def plot_time_vs_dielconst(cls,df:pd.DataFrame):
        if "time_fs" not in df.columns:
            raise ValueError(" ERROR :: time column is not found in DataFrame")
        if "eps0" not in df.columns:
            raise ValueError(" ERROR :: eps0 column is not found in DataFrame")
        # 時間 vs 誘電定数のプロットを行う
        fig, ax = plt.subplots(figsize=(8,5),tight_layout=True) # figure, axesオブジェクトを作成
        ax.plot(df["time_fs"]/1000/1000, df["eps0"] , label = "dielconst") # time in ns
        
        # 各要素で設定したい文字列の取得
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        xlabel="Time [ns]" #"Time $\mathrm{ps}$"
        ylabel="Dielconst"
        
        # 各要素の設定を行うsetコマンド
        ax.set_xlabel(xlabel,fontsize=22)
        ax.set_ylabel(ylabel,fontsize=22)
        
        # https://www.delftstack.com/ja/howto/matplotlib/how-to-set-tick-labels-font-size-in-matplotlib/#ax.tick_paramsaxis-xlabelsize-%25E3%2581%25A7%25E7%259B%25AE%25E7%259B%259B%25E3%2582%258A%25E3%2583%25A9%25E3%2583%2599%25E3%2583%25AB%25E3%2581%25AE%25E3%2583%2595%25E3%2582%25A9%25E3%2583%25B3%25E3%2583%2588%25E3%2582%25B5%25E3%2582%25A4%25E3%2582%25BA%25E3%2582%2592%25E8%25A8%25AD%25E5%25AE%259A%25E3%2581%2599%25E3%2582%258B
        ax.tick_params(axis='x', labelsize=15 )
        ax.tick_params(axis='y', labelsize=15 )
        
        ax.legend(loc="upper right",fontsize=15 )
        
        #pyplot.savefig("eps_real2.pdf",transparent=True) 
        # plt.show()
        fig.savefig("time_dielconst.pdf")
        fig.delaxes(ax)
        return 0
    
    def plot_total_dipole(self):
        """make histgram&plot histgram

        Returns:
            _type_: _description_
        """
        fig, ax = plt.subplots(figsize=(8,5),tight_layout=True) # figure, axesオブジェクトを作成
        ax.plot(self.data[:,0]*self.timestep/1000, self.data[:,1], label = "x")     # 描画
        ax.plot(self.data[:,0]*self.timestep/1000, self.data[:,2], label = "y")     # 描画
        ax.plot(self.data[:,0]*self.timestep/1000, self.data[:,3], label = "z")     # 描画
        
        # 各要素で設定したい文字列の取得
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        xlabel="Time [ps]" #"Time $\mathrm{ps}$"
        ylabel="Dipole [D]"
        
        # 各要素の設定を行うsetコマンド
        ax.set_xlabel(xlabel,fontsize=22)
        ax.set_ylabel(ylabel,fontsize=22)
        
        # https://www.delftstack.com/ja/howto/matplotlib/how-to-set-tick-labels-font-size-in-matplotlib/#ax.tick_paramsaxis-xlabelsize-%25E3%2581%25A7%25E7%259B%25AE%25E7%259B%259B%25E3%2582%258A%25E3%2583%25A9%25E3%2583%2599%25E3%2583%25AB%25E3%2581%25AE%25E3%2583%2595%25E3%2582%25A9%25E3%2583%25B3%25E3%2583%2588%25E3%2582%25B5%25E3%2582%25A4%25E3%2582%25BA%25E3%2582%2592%25E8%25A8%25AD%25E5%25AE%259A%25E3%2581%2599%25E3%2582%258B
        ax.tick_params(axis='x', labelsize=15 )
        ax.tick_params(axis='y', labelsize=15 )
        
        ax.legend(loc="upper right",fontsize=15 )
        
        #pyplot.savefig("eps_real2.pdf",transparent=True) 
        # plt.show()
        fig.savefig("time_totaldipole.pdf")
        fig.delaxes(ax)
        return 0