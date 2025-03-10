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
logger = root_logger("MLWC."+__name__)

class totaldipole:
    """plot time vs dipole figure for total_dipole
    
    Returns:
        _type_: _description_
    """
    def __init__(self):
        """Initialize totaldipole class.

        Attributes:
            timestep (float): time step of MD simulation.
            temperature (float): temperature of MD simulation.
            unitcell (np.ndarray): unit cell of MD simulation.
            data (np.ndarray): dipole data of MD simulation.
        """
        self.timestep:float = None
        self.temperature:float = None
        self.unitcell:np.ndarray = None
        self.data:np.ndarray = None

    def set_params(self,data:np.ndarray,unitcell:np.ndarray,timestep:float,temperature:float):
        """Set parameters for totaldipole class.

        Args:
            data (np.ndarray): dipole data of MD simulation. shape=(time, 4).
            unitcell (np.ndarray): unit cell of MD simulation. shape=(3, 3).
            timestep (float): time step of MD simulation [fs].
            temperature (float): temperature of MD simulation [K].

        Returns:
            int: 0 if successful.

        Raises:
            ValueError: if data, unitcell, timestep, or temperature is not the correct type.
            ValueError: if data shape is not correct.

        Examples:
            >>> data = np.random.rand(100, 4)
            >>> unitcell = np.random.rand(3, 3)
            >>> timestep = 1.0
            >>> temperature = 300.0
            >>> total_dipole = totaldipole()
            >>> total_dipole.set_params(data, unitcell, timestep, temperature)
            0
        """
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
        """Print information of totaldipole class.

        Returns:
            int: 0 if successful.

        Examples:
            >>> data = np.random.rand(100, 4)
            >>> unitcell = np.random.rand(3, 3)
            >>> timestep = 1.0
            >>> temperature = 300.0
            >>> total_dipole = totaldipole()
            >>> total_dipole.set_params(data, unitcell, timestep, temperature)
            0
            >>> total_dipole.print_info()
            0
        """
        logger.info(" ============================ ")
        logger.info(f" number of data :: {np.shape(self.data)}")
        logger.info(f" timestep [fs] :: {self.timestep}")
        logger.info(f" temperature [K] :: {self.temperature}")
        logger.info(f" unitcell [Ang] :: {self.unitcell}")
        logger.info(" ============================ ")
        return 0

    def get_volume(self):
        """Calculate the volume of the unit cell.

        Returns:
            float: Volume of the unit cell in m^3.

        Examples:
            >>> unitcell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            >>> total_dipole = totaldipole()
            >>> total_dipole.unitcell = unitcell
            >>> total_dipole.get_volume()
            1e-30
        """
        A3 = 1.0e-30
        return np.abs(np.dot(np.cross(self.unitcell[:,0],self.unitcell[:,1]),self.unitcell[:,2])) * A3

    def get_mean_dipole(self):
        """Calculate the mean dipole moment.

        Returns:
            np.ndarray: Mean dipole moment in x, y, and z directions.

        Examples:
            >>> data = np.random.rand(100, 4)
            >>> total_dipole = totaldipole()
            >>> total_dipole.data = data
            >>> total_dipole.get_mean_dipole()
            array([..., ..., ...])
        """
        dMx=self.data[:,0]#-np.mean(self.data[:,0])
        dMy=self.data[:,1]#-np.mean(self.data[:,1])
        dMz=self.data[:,2]#-np.mean(self.data[:,2])
        # mean_M=np.mean(dMx)**2+np.mean(dMy)**2+np.mean(dMz)**2    # <M>^2
        return np.array([np.mean(dMx),np.mean(dMy),np.mean(dMz)])

    def get_mean_dipolesquare(self):
        """Calculate the mean square dipole moment.

        Returns:
            float: Mean square dipole moment.

        Examples:
            >>> data = np.random.rand(100, 4)
            >>> total_dipole = totaldipole()
            >>> total_dipole.data = data
            >>> total_dipole.get_mean_dipolesquare()
            ...
        """
        dMx=self.data[:,0]#-np.mean(self.data[:,0])
        dMy=self.data[:,1]#-np.mean(self.data[:,1])
        dMz=self.data[:,2]#-np.mean(self.data[:,2])
        mean_M2=np.mean(dMx**2)+np.mean(dMy**2)+np.mean(dMz**2) # <M^2>
        return mean_M2

    def calc_dielconst(self, eps_inf:float=1.0) -> float:
        """Calculate the dielectric constant.

        Calculates only eps0.

        Args:
            eps_inf (float, optional): Dielectric constant at infinite frequency. Defaults to 1.0.

        Returns:
            list[float]: A list containing the dielectric constant, mean square dipole moment, and mean dipole moment.

        Examples:
            >>> data = np.random.rand(100, 4)
            >>> unitcell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            >>> timestep = 1.0
            >>> temperature = 300.0
            >>> total_dipole = totaldipole()
            >>> total_dipole.set_params(data, unitcell, timestep, temperature)
            0
            >>> total_dipole.calc_dielconst()
            [..., ..., ...]
        """

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
        """
        Calculate the dielectric constant as a function of time.

        This method calculates the dielectric constant at various time steps
        within a specified range, providing insights into the time evolution
        of the dielectric properties of the simulated system.

        Parameters
        ----------
        start : int
            The starting index for the time range.
        end : int
            The ending index for the time range. Use -1 to include all data points to the end.
        eps_inf : float, optional
            The dielectric constant at infinite frequency. Defaults to 1.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the time, dielectric constant,
            mean square dipole moment, and mean dipole moment.
            The DataFrame is also saved to "eps0_vs_time.csv".

        Raises
        ------
        ValueError
            If `start` or `end` is larger than the data length.

        Examples
        --------
        >>> data = np.random.rand(100, 4)
        >>> unitcell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> timestep = 1.0
        >>> temperature = 300.0
        >>> total_dipole = totaldipole()
        >>> total_dipole.set_params(data, unitcell, timestep, temperature)
        0
        >>> df = total_dipole.calc_time_vs_dielconst(start=10, end=50, eps_inf=1.0)
        >>> print(df.head())
           time_fs      eps0   mean_M2    mean_M
        0     10.0  1.000000  0.083333  0.000000
        1     20.0  1.000000  0.083333  0.000000
        2     30.0  1.000000  0.083333  0.000000
        3     40.0  1.000000  0.083333  0.000000
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
        # save data to csv
        df = pd.DataFrame()
        df["time_fs"] = time_list # in fs
        df["eps0"] = eps0_list
        df["mean_M2"] = mean_M2_list
        df["mean_M"] = mean_M_list
        df.to_csv("eps0_vs_time.csv",index=False)
        return df
    
    @classmethod
    def plot_time_vs_dielconst(cls,df:pd.DataFrame):
        """Plot time vs dielectric constant.

        This method plots the dielectric constant as a function of time,
        visualizing the time evolution of the dielectric properties.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame containing the time and dielectric constant data.
            It must have columns named "time_fs" and "eps0".

        Raises
        ------
        ValueError
            If "time_fs" or "eps0" column is not found in DataFrame.

        Returns
        -------
        int
            0 if successful. The plot is saved to "time_dielconst.pdf".

        Examples
        --------
        >>> data = {'time_fs': [10, 20, 30, 40], 'eps0': [1.0, 1.1, 1.2, 1.3]}
        >>> df = pd.DataFrame(data)
        >>> totaldipole.plot_time_vs_dielconst(df)
        0
        """
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
        """Plot total dipole moment as a function of time.

        This method plots the total dipole moment in x, y, and z directions
        as a function of time, providing insights into the dipole moment
        behavior of the simulated system.

        Returns
        -------
        int
            0 if successful. The plot is saved to "time_totaldipole.pdf".

        Examples
        --------
        >>> data = np.random.rand(100, 4)
        >>> unitcell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> timestep = 1.0
        >>> temperature = 300.0
        >>> total_dipole = totaldipole()
        >>> total_dipole.set_params(data, unitcell, timestep, temperature)
        0
        >>> total_dipole.plot_total_dipole()
        0
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