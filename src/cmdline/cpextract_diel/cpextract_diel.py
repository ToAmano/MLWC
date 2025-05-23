#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""cpextract_diel.py
CPextract.py diel subcommand for processing dipole.txt files generated by dieltools

- histgram :: Plot histgram of molecule dipole/bond dipole
"""


#
# simple code to extract data from CP.x outputs
# define sub command of CPextract.py
#

import sys
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import cpmd.read_core
import cpmd.read_traj
import ase.units
from include.mlwc_logger import setup_library_logger
logger = setup_library_logger("MLWC."+__name__)



class Plot_histgram:
    """plot histgram of molecule/bond dipole
    

    Returns:
        _type_: _description_
    """
    def __init__(self,dipole_filename,max=None):
        self._filename = dipole_filename
        import os
        if not os.path.isfile(self._filename):
            raise FileNotFoundError(" ERROR (Plot_histgram) :: "+str(self._filename)+" does not exist !!")
        self.data = np.loadtxt(self._filename) # load txt in numpy ndarray
        self.max  = max
        logger.info(" --------- ")
        logger.info(f" number of data :: {np.shape(self.data)}")
        logger.info(f" max value [D]  :: {self.max}")
        logger.info(" --------- ")
    
    def get_histgram(self):
        """ヒストグラムのデータを保存
        
        """
        import pandas as pd
        # 先にデータの数と値域から最適なヒストグラム構成を考える
        # TODO :: bins = 1000で固定しているので修正
        _length = len(self.data)
        
        # 最大値を計算する
        plot_data = np.linalg.norm(self.data[:,2:].reshape(-1,3),axis=1)
        _max_val = np.max(plot_data)
        # 最大値が4以下なら5で固定する
        if self.max != None:
            _hist_max_val = float(self.max)
        elif _max_val < 4:
            _hist_max_val = 5
        else:
            _hist_max_val = _max_val+2
        
        
        # https://qiita.com/nkay/items/56bda7143981e3d5303f
        df =pd.DataFrame()
        hist = np.histogram(plot_data, bins = 1000, range = [0,_hist_max_val], density=True )
        df["dipole"] = (hist[1][1:] + hist[1][:-1]) / 2
        df["density"] = hist[0] 
        df.to_csv(self._filename+"_hist.txt")
        return df


    
    def plot_dipole_histgram(self):
        """make histgram&plot histgram

        Returns:
            _type_: _description_
        """
        logger.info(" ---------- ")
        logger.info(" dipole histgram plot ")
        logger.info(" ---------- ")
        
        # 最大値を計算する
        plot_data = np.linalg.norm(self.data[:,2:].reshape(-1,3),axis=1)
        _max_val = np.max(plot_data)
        # 最大値が4以下なら5で固定する
        if self.max != None:
            _hist_max_val = float(self.max)
        elif _max_val < 4:
            _hist_max_val = 5
        else:
            _hist_max_val = _max_val+2
        
        plot_data = np.linalg.norm(self.data[:,2:].reshape(-1,3),axis=1)
        fig, ax = plt.subplots(figsize=(8,5),tight_layout=True) # figure, axesオブジェクトを作成
        ax.hist(plot_data, bins = 1000, range = [0,_hist_max_val], density=True)     # 描画
        
        # 各要素で設定したい文字列の取得
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        xlabel="Dipole [D]" #"Time $\mathrm{ps}$"
        ylabel="Density"
        
        # 各要素の設定を行うsetコマンド
        ax.set_xlabel(xlabel,fontsize=22)
        ax.set_ylabel(ylabel,fontsize=22)
        
        # https://www.delftstack.com/ja/howto/matplotlib/how-to-set-tick-labels-font-size-in-matplotlib/#ax.tick_paramsaxis-xlabelsize-%25E3%2581%25A7%25E7%259B%25AE%25E7%259B%259B%25E3%2582%258A%25E3%2583%25A9%25E3%2583%2599%25E3%2583%25AB%25E3%2581%25AE%25E3%2583%2595%25E3%2582%25A9%25E3%2583%25B3%25E3%2583%2588%25E3%2582%25B5%25E3%2582%25A4%25E3%2582%25BA%25E3%2582%2592%25E8%25A8%25AD%25E5%25AE%259A%25E3%2581%2599%25E3%2582%258B
        ax.tick_params(axis='x', labelsize=15 )
        ax.tick_params(axis='y', labelsize=15 )
        
        ax.legend(loc="upper right",fontsize=15 )
        
        #pyplot.savefig("eps_real2.pdf",transparent=True) 
        # plt.show()
        fig.savefig(self._filename+"_dipolehist.pdf")
        fig.delaxes(ax)
        return 0

class Plot_totaldipole:
    """plot time vs dipole figure for total_dipole
    
    Returns:
        _type_: _description_
    """
    def __init__(self,dipole_filename):
        self._filename = dipole_filename
        import os
        if not os.path.isfile(self._filename):
            logger.error(" ERROR (Plot_totaldipole) :: "+str(self._filename)+" does not exist !!")
            logger.error(" ")
            return 1
        logger.info(" ============================ ")
        logger.info(f" filename  :: {self._filename}")
        self.data = np.loadtxt(self._filename,comments='#') # load txt in numpy ndarray
        logger.info(f" number of data :: {np.shape(self.data)}")
        self.__get_timestep()
        logger.info(f" timestep [fs] :: {self.timestep}")
        self.__get_temperature()
        logger.info(f" temperature [K] :: {self.temperature}")
        self.__get_unitcell()
        logger.info(f" unitcell [Ang] :: {self.unitcell}")
        logger.info(" ============================ ")

    def __get_timestep(self)->int:
        """extract timestep [fs] from total_dipole.txt
        """
        with open(self._filename) as f:
            line = f.readline()
            while line:
                line = f.readline()
                if line.startswith("#TIMESTEP"):
                    time = float(line.split(" ")[1]) 
                    break
        self.timestep = time
        return 0
    
    def __get_unitcell(self):
        """extract unitcell from total_dipole.txt
        """
        with open(self._filename) as f:
            line = f.readline()
            while line:
                line = f.readline()
                if line.startswith("#UNITCELL"):
                    unitcell = line.strip("\n").strip().split(" ")[1:]
                    break
        self.unitcell = np.array([float(i) for i in unitcell]).reshape([3,3]) 
        return 0
    
    def __get_temperature(self):
        """extract unitcell from total_dipole.txt
        """
        with open(self._filename) as f:
            line = f.readline()
            while line:
                line = f.readline()
                if line.startswith("#TEMPERATURE"):
                    temp = float(line.split(" ")[1]) 
                    break
        self.temperature = temp
        return 0
    
    def calc_dielectric_spectrum(self,eps_n2:float, start:int, end:int, step:int,window:str="hann",if_fft:bool=True):
        """total dipole.txtから全スペクトルを計算する

        Args:
            eps_n2 (float): _description_
            start (int): _description_
            end (int): _description_
            step (int): _description_

        Returns:
            _type_: _description_
        """
        from diel.acf_fourier import dielec
        from diel.dipole_core import diel_function
        logger.info(" ==================== ")
        logger.info(f"  start index :: {start}")
        logger.info(f"  end   index :: {end}")
        logger.info(f" moving average step :: {step}")
        logger.info(" ==================== ")
        process = dielec(self.unitcell, self.temperature, self.timestep)
        if end == -1:
            calc_data = self.data[start:,1:] # dipoleのみを抽出
        else:
            calc_data = self.data[start:end,1:] # dipoleのみを抽出
        logger.info(" ====================== ")
        logger.info(f"  len(data)          :: {len(calc_data)}")
        logger.info(f"  total MD time [ps] :: {len(calc_data)*self.timestep/1000}")
        logger.info(" ====================== ")
        if (start >= end) and (end != -1):
            raise ValueError("end must be larger than start")
        
        # replace abnormally large dipole (10*average dipole) with previous values
        ave_dipole = np.mean(np.linalg.norm(calc_data,axis=1))
        calc_data = np.where(np.abs(calc_data)>10*ave_dipole, 0, calc_data) # 双極子のズレが大きい場合は0で置き換え
        # here, we do not include moving-average
        rfreq, ffteps1, ffteps2 = process.calc_fourier(calc_data, eps_n2, window,if_fft) # calc dielectric function
        # here, we introduce moving-average for both dielectric-function and refractive-index
        diel = diel_function(rfreq, ffteps1, ffteps2,step)
        diel.diel_df.to_csv(self._filename+"_diel.csv")
        diel.refractive_df.to_csv(self._filename+"_refractive.csv")
        return 0
    
    def calc_dielectric_derivative_spectrum(self, start:int, end:int, step:int):
        """微分スペクトルから計算する．公式はzhang2020Deepを利用

        Args:
            eps_n2 (float): _description_
            start (int): _description_
            end (int): _description_
            step (int): _description_
        """
        from diel.acf_fourier import dielec
        from diel.dipole_core import diel_function
        process = dielec(self.unitcell, self.temperature, self.timestep)
        if end == -1:
            calc_data = self.data[start:,1:]
        else:
            calc_data = self.data[start:end,1:]
        # calculate alphan
        rfreq, alphan = process.calc_derivative_spectrum(calc_data,window="hann")
        import pandas as pd
        df = pd.DataFrame()    
        df["freq_thz"] = rfreq
        df["freq_kayser"] = rfreq*33.3
        window = np.ones(step)/step 
        df["alphan"] = np.convolve(alphan,window,mode="same")
        df.to_csv(self._filename+"_alphan.csv")
        return 0

        
    
    def plot_total_dipole(self):
        """make histgram&plot histgram

        Returns:
            _type_: _description_
        """
        logger.info(" ---------- ")
        logger.info(" dipole histgram plot ")
        logger.info(" ---------- ")
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
        fig.savefig(self._filename+"_time_dipole.pdf")
        fig.delaxes(ax)
        return 0

def fit_diel(freq:np.ndarray, imag_diel:np.ndarray,num_hn_functions:int=1, lower_bound:float=0.1,upper_bound:float=1.0):
    import diel.fit_diel
    # keep initial freq
    init_freq = freq
    
    # フィッティング範囲の設定
    if lower_bound is not None:
        freq = freq[freq >= lower_bound]
        epsilon_imag = imag_diel[-len(freq):]  # 周波数に対応する範囲で誘電率を切り取る

    if upper_bound is not None:
        freq = freq[freq <= upper_bound]
        epsilon_imag = imag_diel[:len(freq)]  # 周波数に対応する範囲で誘電率を切り取る
        
    from scipy.optimize import least_squares

    # 初期推定値の設定
    initial_guess = [1, 1, 1e-3, 0.5, 0.5] * args.num_hn_functions


    # 制約条件の設定 (alphaとbetaは0から1の間)
    bounds_lower = [0, 0, 0, 0] * num_hn_functions
    bounds_upper = [np.inf, np.inf, 1, 1] * num_hn_functions
    # 最小二乗法によるフィッティングを実行
    result = least_squares(diel.fit_diel.residuals, initial_guess, bounds=(bounds_lower, bounds_upper), args=(freq, imag_diel))
    logger.info(" ====================== ")
    logger.info("   fitting result       ")
    logger.info(f" {result.x}            ")
    logger.info(" ====================== ")
    
    # フィッティング結果
    epsilon_fit = havriliak_negami_sum(freq,result.x)
    
    # save to pd.dataframe
    df = pd.DataFrame()
    df["freq_kayser"] = init_freq
    df["fit_imag_diel"] = havriliak_negami_sum(init_freq,result.x)
    df.to_csv("fit_hn_diel_imag.csv")
    return df

class Plot_moleculedipole(Plot_totaldipole):
    """plot time vs dipole figure for total_dipole
    
    Returns:
        _type_: _description_
    """
    def __init__(self,dipole_filename):
        # 継承元から初期化
        super().__init__(dipole_filename)
        self.__get_num_mol()
        logger.info(" --------- ")
        logger.info(f" number of mol :: {self.__NUM_MOL}")
        logger.info(" --------- ")
        # データ形状を変更[frame,mol_id,3dvector]
        self.data = self.data[:,2:].reshape(-1,self.__NUM_MOL,3)
        
    def __get_num_mol(self):
        """extract num_mol from molecule_dipole.txt
        """
        # 1行目の最大値が分子数
        self.__NUM_MOL = int(np.max(self.data[:,1]))+1
        return 0
    
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
        # self term ACF
        self_data  = calc_total_mol_acf_self(calc_data,engine="tsa")
        rfreq_self, ffteps1_self, ffteps2_self    = process.calc_fourier_only_with_window(self_data,eps_n2,window="hann")
        # here, we introduce moving-average for both dielectric-function and refractive-index
        diel_self = diel_function(rfreq_self, ffteps1_self, ffteps2_self,step)
        diel_self.diel_df.to_csv(self._filename+"_self_diel.csv")
        diel_self.refractive_df.to_csv(self._filename+"_self_refractive.csv")
        logger.info(" finish self terms")
        # cross term ACF
        cross_data = calc_total_mol_acf_cross(calc_data,engine="tsa")
        logger.info(" finish cross terms")
        # rfreq_self = rfreq_cross
        rfreq_cross, ffteps1_cross, ffteps2_cross = process.calc_fourier_only_with_window(cross_data,eps_n2,window="hann")
        rfreq_total, ffteps1_total, ffteps2_total = process.calc_fourier_only_with_window(self_data+cross_data,eps_n2,window="hann")
        # cross
        diel_cross = diel_function(rfreq_cross, ffteps1_cross, ffteps2_cross,step)
        diel_cross.diel_df.to_csv(self._filename+"_cross_diel.csv")
        diel_cross.refractive_df.to_csv(self._filename+"_cross_refractive.csv")
        # total
        diel_total = diel_function(rfreq_total, ffteps1_total, ffteps2_total,step)
        diel_total.diel_df.to_csv(self._filename+"_total_diel.csv")
        diel_total.refractive_df.to_csv(self._filename+"_total_refractive.csv")
        return 0
        




# --------------------------------
# 以下CPextract.pyからロードする関数たち
# --------------------------------

def command_diel_histgram(args):
    EVP=Plot_histgram(args.Filename,args.max)
    EVP.get_histgram()
    EVP.plot_dipole_histgram()
    return 0

def command_diel_total(args):
    EVP=Plot_totaldipole(args.Filename)
    EVP.plot_total_dipole()
    return 0

def command_diel_spectra(args):
    EVP=Plot_totaldipole(args.Filename)
    # moving average:: https://chaos-kiyono.hatenablog.com/entry/2022/07/25/212843
    # https://qiita.com/FallnJumper/items/e0afa1fb05ea448caae1
    if args.fft == "True":
        if_fft = True
    elif args.fft == "False":
        if_fft = False
    else:
        raise ValueError("fft should be True or False")  # 他の値に応じて処理
    EVP.calc_dielectric_spectrum(float(args.eps),int(args.start),int(args.end),int(args.step),args.window,if_fft) # epsを受け取ってfloat変換
    EVP.calc_dielectric_derivative_spectrum(int(args.start), int(args.end), int(args.step)) # 微分公式のテスト
    return 0

def command_diel_mol(args):
    EVP=Plot_moleculedipole(args.Filename)
    EVP.calc_dielectric_spectrum(float(args.eps),int(args.start),int(args.end),int(args.step)) # epsを受け取ってfloat変換
    return 0

def command_diel_fit(args):
    df = pd.read_csv(args.Filename)
    fit_diel(df["freq_kayser"].values,df["imag_diel"].values,args.num_hn_functions,args.lower_bound,args.upper_bound)
    return 0
