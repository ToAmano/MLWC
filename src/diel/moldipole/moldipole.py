# 発想を転換して，total_dipole.txtに対するクラスをここで実装する．
# pandas dataframeを継承するかどうかは難しいところ．

# 設計として，インスタンスがtimestep, temperature, unitcell, dataを持つ．

import sys
import ase.units
import ase.io
import numpy as np
import statsmodels.api as sm 
import pandas as pd
import matplotlib.pyplot as plt
import cpmd.read_core
import cpmd.read_traj
from diel.acf_fourier import raw_calc_eps0_dielconst # for calculation of dielectric constant
from include.mlwc_logger import root_logger
logger = root_logger("MLWC."+__name__)

class moldipole:
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
        if len(np.shape(data)) != 3:
            raise ValueError(f" ERROR :: data shape is not correct :: len(np.shape(data)) == {len(np.shape(data))}")
        if np.shape(data)[2] != 3:
            raise ValueError(f" ERROR :: data shape is not correct :: np.shape(data)[2] == {np.shape(data)[2]}")
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

    def get_num_mol(self):
        return np.shape(self.data)[1]

    def get_totaldipole(self,max_length:int=-1):
        return np.sum(self.data[:max_length],axis=1)
        
    def get_mean_moldipole(self,max_length:int=-1):
        abs_moldipole = np.linalg.norm(self.data[:max_length],axis=2) # absoulte value of moldipole
        abs_moldipole = abs_moldipole.reshape(-1) # reshape to 2D ( reshape try to return "view")
        return np.mean(abs_moldipole,axis=0)

    def get_mean_dipolesquare(self,max_length:int=-1):
        """_summary_

        Args:
            max_length (int, optional): use up to max_length data of the first row. Defaults to -1.

        Returns:
            _type_: _description_
        """
        data = self.data[:max_length]        
        dMx=data[:,0]-np.mean(data[:,0])
        dMy=data[:,1]-np.mean(data[:,1])
        dMz=data[:,2]-np.mean(data[:,2])
        mean_M2=np.mean(dMx**2)+np.mean(dMy**2)+np.mean(dMz**2) # <M^2>
        return mean_M2

    def calc_gfactor(self,max_length:int=-1)->float:                
        """calculate kirkwood g-factor
        
        The definition of the Kirkwood g-factor is given by the following equation:
        g = <M^2> / (N * <mu>^2)
        where M is the the total dipole moment, N is the number of molecules, and <mu> is the mean molecular dipole moment.
        For detail, see the following references:
        - handgraaf2004Densityfunctional
        - pagliai2003Hydrogen
        - sharma2007Dipolar
        
        
        Returns:
            float: g_factor
        """
        
        total_dipole = self.get_totaldipole(max_length)
        dMx = total_dipole[:,0] - np.mean(total_dipole[:,0])
        dMy = total_dipole[:,1] - np.mean(total_dipole[:,1])
        dMz = total_dipole[:,2] - np.mean(total_dipole[:,2])
        mean_M2=np.mean(dMx**2)+np.mean(dMy**2)+np.mean(dMz**2) # <M^2>
        mean_moldipole =self.get_mean_moldipole(max_length)
        num_mol = self.get_num_mol()
        g_factor = mean_M2/num_mol/(mean_moldipole**2)
        return g_factor

    def calc_dielectric_spectrum(self,eps_n2:float, start:int, end:int, step:int):
        from diel.acf_fourier import dielec
        from diel.acf_fourier import calc_total_mol_acf_self
        from diel.acf_fourier import calc_total_mol_acf_cross
        from diel.dipole_core import diel_function
        logger.info(" ==================== ")
        logger.info(f"  start index :: {start}")
        logger.info(f"  end   index :: {end}")
        logger.info(f" moving average step :: {step}")
        logger.info(" ==================== ")
        process = dielec(self.unitcell, self.temperature, self.timestep)
        if end == -1:
            calc_data = self.data[start:,:,:]
        else:
            calc_data = self.data[start:end,:,:]
        logger.info(" ====================== ")
        logger.info(f"  len(data)    :: {len(calc_data)}")
        logger.info(" ====================== ")
        # まずはACFの計算
        self_data  = calc_total_mol_acf_self(calc_data,engine="tsa")
        cross_data = calc_total_mol_acf_cross(calc_data,engine="tsa")
        # rfreq_self = rfreq_cross
        rfreq_self, ffteps1_self, ffteps2_self   = process.calc_fourier_only_with_window(self_data,eps_n2,window="hann")
        rfreq_cross, ffteps1_cross, ffteps2_cross = process.calc_fourier_only_with_window(cross_data,eps_n2,window="hann")
        rfreq_total, ffteps1_total, ffteps2_total = process.calc_fourier_only_with_window(self_data+cross_data,eps_n2,window="hann")

        # here, we introduce moving-average for both dielectric-function and refractive-index
        diel_self = diel_function(rfreq_self, ffteps1_self, ffteps2_self,step)
        diel_self.diel_df.to_csv(self._filename+"_self_diel.csv",index=False)
        diel_self.refractive_df.to_csv(self._filename+"_self_refractive.csv",index=False)
        # cross
        diel_cross = diel_function(rfreq_cross, ffteps1_cross, ffteps2_cross,step)
        diel_cross.diel_df.to_csv(self._filename+"_cross_diel.csv",index=False)
        diel_cross.refractive_df.to_csv(self._filename+"_cross_refractive.csv",index=False)
        # total
        diel_total = diel_function(rfreq_total, ffteps1_total, ffteps2_total,step)
        diel_total.diel_df.to_csv(self._filename+"_total_diel.csv",index=False)
        diel_total.refractive_df.to_csv(self._filename+"_total_refractive.csv",index=False)
        return 0


    def calc_time_vs_gfactor(self,start:int,end:int) -> pd.DataFrame:
        """時間 vs 誘電定数の計算を行う

        Returns:
            _type_: _description_
        """
        from diel.acf_fourier import raw_calc_gfactor
        gfactor_list=[]
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
                g_factor = raw_calc_gfactor(calc_data[:index])
                gfactor_list.append(g_factor)
                time_list.append(index*self.timestep)
        # データの保存
        df = pd.DataFrame()
        df["time_fs"] = time_list # in fs
        df["gfactor"] = gfactor_list
        df.to_csv("gfactor_vs_time.csv",index=False)
        return df
    
    @classmethod
    def plot_time_vs_gfactor(cls,df:pd.DataFrame):
        if "time_fs" not in df.columns:
            raise ValueError(" ERROR :: time column is not found in DataFrame")
        if "gfactor" not in df.columns:
            raise ValueError(" ERROR :: gfactor column is not found in DataFrame")
        # 時間 vs 誘電定数のプロットを行う
        fig, ax = plt.subplots(figsize=(8,5),tight_layout=True) # figure, axesオブジェクトを作成
        ax.plot(df["time_fs"]/1000/1000, df["gfactor"] , label = "gfactor") # time in ns
        
        # 各要素で設定したい文字列の取得
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        xlabel="Time [ns]" #"Time $\mathrm{ps}$"
        ylabel="G-factor"
        
        # 各要素の設定を行うsetコマンド
        ax.set_xlabel(xlabel,fontsize=22)
        ax.set_ylabel(ylabel,fontsize=22)
        
        # https://www.delftstack.com/ja/howto/matplotlib/how-to-set-tick-labels-font-size-in-matplotlib/#ax.tick_paramsaxis-xlabelsize-%25E3%2581%25A7%25E7%259B%25AE%25E7%259B%259B%25E3%2582%258A%25E3%2583%25A9%25E3%2583%2599%25E3%2583%25AB%25E3%2581%25AE%25E3%2583%2595%25E3%2582%25A9%25E3%2583%25B3%25E3%2583%2588%25E3%2582%25B5%25E3%2582%25A4%25E3%2582%25BA%25E3%2582%2592%25E8%25A8%25AD%25E5%25AE%259A%25E3%2581%2599%25E3%2582%258B
        ax.tick_params(axis='x', labelsize=15 )
        ax.tick_params(axis='y', labelsize=15 )
        
        ax.legend(loc="upper right",fontsize=15 )
        
        fig.savefig("time_gfactor.pdf")
        fig.delaxes(ax)
        return 0