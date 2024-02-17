#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# simple code to extract data from CP.x outputs
# define sub command of CPextract.py
#

import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cpmd.read_core
import cpmd.read_traj

try:
    import ase.units
except ImportError:
    sys.exit("Error: ase not installed")


class Plot_energies:
    '''
   Short Legend and Physical Units in the Output
   ---------------------------------------------
   NFI    [int]          - step index
   EKINC  [HARTREE A.U.] - kinetic energy of the fictitious electronic dynamics
   TEMPH  [K]            - Temperature of the fictitious cell dynamics
   TEMP   [K]            - Ionic temperature
   ETOT   [HARTREE A.U.] - Scf total energy (Kohn-Sham hamiltonian)
   ENTHAL [HARTREE A.U.] - Enthalpy ( ETOT + P * V )
   ECONS  [HARTREE A.U.] - Enthalpy + kinetic energy of ions and cell
   ECONT  [HARTREE A.U.] - Constant of motion for the CP lagrangian    
    '''
    def __init__(self,energies_filename):
        self.__filename = energies_filename
        self.data = np.loadtxt(self.__filename)

        import os
        if not os.path.isfile(self.__filename):
            print(" ERROR :: "+str(self.__filename)+" does not exist !!")
            print(" ")
            return 1

    def plot_Energy(self):
        print(" ---------- ")
        print(" energy plot :: column 0 & 4(ECLASSICAL) ")
        print(" ---------- ")
        fig, ax = plt.subplots(figsize=(8,5),tight_layout=True) # figure, axesオブジェクトを作成
        ax.plot(self.data[:,0], self.data[:,4]/ase.units.Hartree, label=self.__filename, lw=3)     # 描画

        # 各要素で設定したい文字列の取得
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        xlabel="Timestep" #"Time $\mathrm{ps}$"
        ylabel="Energy[eV]"

        # 各要素の設定を行うsetコマンド
        ax.set_xlabel(xlabel,fontsize=22)
        ax.set_ylabel(ylabel,fontsize=22)
        
        # https://www.delftstack.com/ja/howto/matplotlib/how-to-set-tick-labels-font-size-in-matplotlib/#ax.tick_paramsaxis-xlabelsize-%25E3%2581%25A7%25E7%259B%25AE%25E7%259B%259B%25E3%2582%258A%25E3%2583%25A9%25E3%2583%2599%25E3%2583%25AB%25E3%2581%25AE%25E3%2583%2595%25E3%2582%25A9%25E3%2583%25B3%25E3%2583%2588%25E3%2582%25B5%25E3%2582%25A4%25E3%2582%25BA%25E3%2582%2592%25E8%25A8%25AD%25E5%25AE%259A%25E3%2581%2599%25E3%2582%258B
        ax.tick_params(axis='x', labelsize=15 )
        ax.tick_params(axis='y', labelsize=15 )
        
        ax.legend(loc="upper right",fontsize=15 )
        
        #pyplot.savefig("eps_real2.pdf",transparent=True) 
        # plt.show()
        fig.savefig(self.__filename+"_E.pdf")
        fig.delaxes(ax)
        return 0

    def plot_energy_histgram(self):
         print(" ---------- ")
         print(" energy plot of histgram :: column 0 & 4(ECLASSICAL) ")
         print(" ---------- ")
         fig, ax = plt.subplots(figsize=(8,5),tight_layout=True) # figure, axesオブジェクトを作成
         ax.hist((self.data[:,4]-np.average(self.data[:,4]))/ase.units.Hartree*1000, bins=100, label=self.__filename+"average={}".format(np.average(self.data[:,4]))+"eV")     # 描画

         # 各要素で設定したい文字列の取得
         xticklabels = ax.get_xticklabels()
         yticklabels = ax.get_yticklabels()
         xlabel="Energy[meV]" #"Time $\mathrm{ps}$"
         ylabel="number"

         # 各要素の設定を行うsetコマンド
         ax.set_xlabel(xlabel,fontsize=22)
         ax.set_ylabel(ylabel,fontsize=22)

         ax.tick_params(axis='x', labelsize=15 )
         ax.tick_params(axis='y', labelsize=15 )

         ax.legend(loc="upper right",fontsize=15 )

         #pyplot.savefig("eps_real2.pdf",transparent=True)
         # plt.show()
         fig.savefig(self.__filename+"_Ehist.pdf")
         fig.delaxes(ax)
         return 0

        
    
    def plot_Temperature(self):
        fig, ax = plt.subplots(figsize=(8,5),tight_layout=True) # figure, axesオブジェクトを作成
        ax.plot(self.data[:,0], self.data[:,2], label=self.__filename, lw=3)     # 描画

        # 各要素で設定したい文字列の取得
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        xlabel="Timesteps"       #"Time $\mathrm{ps}$"
        ylabel="Temperature [K]"
        
        # 各要素の設定を行うsetコマンド
        ax.set_xlabel(xlabel,fontsize=22)
        ax.set_ylabel(ylabel,fontsize=22)
        
        # https://www.delftstack.com/ja/howto/matplotlib/how-to-set-tick-labels-font-size-in-matplotlib/#ax.tick_paramsaxis-xlabelsize-%25E3%2581%25A7%25E7%259B%25AE%25E7%259B%259B%25E3%2582%258A%25E3%2583%25A9%25E3%2583%2599%25E3%2583%25AB%25E3%2581%25AE%25E3%2583%2595%25E3%2582%25A9%25E3%2583%25B3%25E3%2583%2588%25E3%2582%25B5%25E3%2582%25A4%25E3%2582%25BA%25E3%2582%2592%25E8%25A8%25AD%25E5%25AE%259A%25E3%2581%2599%25E3%2582%258B
        ax.tick_params(axis='x', labelsize=15 )
        ax.tick_params(axis='y', labelsize=15 )
        
        ax.legend(loc="upper right",fontsize=15 )
        
        fig.savefig(self.__filename+"_T.pdf")
        fig.delaxes(ax)
        return 0

        
    
    def process(self):
        print(" ==========================")
        print(" Reading {:<20}   :: making Temperature & Energy plots ".format(self.__filename))
        print("")
        self.plot_Energy()
        self.plot_Temperature()
        self.plot_energy_histgram()

class Plot_forces:
    '''
   Short Legend and Physical Units in the Output
   ---------------------------------------------
   NFI    [int]          - step index
   EKINC  [HARTREE A.U.] - kinetic energy of the fictitious electronic dynamics
   TEMPH  [K]            - Temperature of the fictitious cell dynamics
   TEMP   [K]            - Ionic temperature
   ETOT   [HARTREE A.U.] - Scf total energy (Kohn-Sham hamiltonian)
   ENTHAL [HARTREE A.U.] - Enthalpy ( ETOT + P * V )
   ECONS  [HARTREE A.U.] - Enthalpy + kinetic energy of ions and cell
   ECONT  [HARTREE A.U.] - Constant of motion for the CP lagrangian    
    '''
    def __init__(self,ftrajectory_filename):
        self.__filename = ftrajectory_filename
        self.data = np.loadtxt(self.__filename)

        import os
        if not os.path.isfile(self.__filename):
            print(" ERROR :: "+str(self.__filename)+" does not exist !!")
            print(" ")
            return 1

    def plot_Force(self):
        print(" ---------- ")
        print(" Force histgram plot :: column 7-9 ")
        print(" ---------- ")
        fig, ax = plt.subplots(figsize=(8,5),tight_layout=True) # figure, axesオブジェクトを作成
        HaBohr_to_eV_Ang=51.42208619083232 
        ax.hist(self.data[:,7]*HaBohr_to_eV_Ang, bins=100, label=self.__filename+"_x", alpha=0.5)
        ax.hist(self.data[:,8]*HaBohr_to_eV_Ang, bins=100, label=self.__filename+"_y",  alpha=0.5)
        ax.hist(self.data[:,9]*HaBohr_to_eV_Ang, bins=100, label=self.__filename+"_z", alpha=0.5)

        # 各要素で設定したい文字列の取得
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        xlabel="Force [eV/Ang]" #"Time $\mathrm{ps}$"
        ylabel="number"

        # 各要素の設定を行うsetコマンド
        ax.set_xlabel(xlabel,fontsize=22)
        ax.set_ylabel(ylabel,fontsize=22)
        
        # https://www.delftstack.com/ja/howto/matplotlib/how-to-set-tick-labels-font-size-in-matplotlib/#ax.tick_paramsaxis-xlabelsize-%25E3%2581%25A7%25E7%259B%25AE%25E7%259B%259B%25E3%2582%258A%25E3%2583%25A9%25E3%2583%2599%25E3%2583%25AB%25E3%2581%25AE%25E3%2583%2595%25E3%2582%25A9%25E3%2583%25B3%25E3%2583%2588%25E3%2582%25B5%25E3%2582%25A4%25E3%2582%25BA%25E3%2582%2592%25E8%25A8%25AD%25E5%25AE%259A%25E3%2581%2599%25E3%2582%258B
        ax.tick_params(axis='x', labelsize=15 )
        ax.tick_params(axis='y', labelsize=15 )
        
        ax.legend(loc="upper right",fontsize=15 )
        
        #pyplot.savefig("eps_real2.pdf",transparent=True) 
        # plt.show()
        fig.savefig(self.__filename+"_F.pdf")
        fig.delaxes(ax)
        return 0

    
    def process(self):
        print(" ==========================")
        print(" Reading {:<20}   :: making Temperature & Energy plots ".format(self.__filename))
        print("")
        self.plot_Force()


def dfset(filename,cpmdout,interval_step:int,start_step:int=0):
    '''
    Trajectoryとforceを読み込んで，DFSET_exportを作る（CPMD.x用）
    '''
    traj =  cpmd.read_traj_cpmd.CPMD_ReadPOS(filename=filename, cpmdout=cpmdout)
    # import forces
    traj.set_force_from_file(filename)
    traj.export_dfset_pwin(interval_step,start_step)
    print(" ")
    print(" make DFSET_export...")
    print(" ")
    return 0


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
            print(" ERROR :: "+str(self.__filename)+" does not exist !!")
            print(" ")
            return 1
        
        if self.__initial_step < 1:
            print("ERROR: initial_step must be larger than 1")
            return 1
        
        # read xyz
        import ase
        import ase.io 
        print(" READING TRAJECTORY... This may take a while, be patient.")
        self.__traj = ase.io.read(self.__filename)
        
    def calc_msd(self):
        """calculate msd

        Returns:
            _type_: _description_
        """
        import numpy as np
        msd = []
        L = self.__traj[self.__initial_step].get_cell()[0][0] # get cell
        print(f"Lattice constant (a[0][0]): {L}")
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
        import pandas as pd
        df = pd.DataFrame()
        df["msd"] = msd
        df["step"] = np.arange(self.__initial_step,len(self.__traj))
        df.to_csv(self.__filename+"_msd.txt")
        return msd
        
class DIPOLE:
    """ class to calculate total dipole moment using classical charge
        See 
    Returns:
        _type_: _description_
    """
    def __init__(self,filename:str,charge_filename:str):
        self._filename = filename # xyz
        self._charge_filename = charge_filename # charge
        import os
        if not os.path.isfile(self._filename):
            print(" ERROR :: "+str(self._filename)+" does not exist !!")
            print(" ")
            return 1
        
        # read xyz
        import ase
        import ase.io 
        import cpmd.read_traj_cpmd
        print(" READING TRAJECTORY... This may take a while, be patient.")
        self._traj, wannier_list=cpmd.read_traj_cpmd.raw_xyz_divide_aseatoms_list(self._filename)
        print(f"FINISH READING TRAJECTORY... {len(self._traj)} steps")
        
        # read charge
        self._charge = np.loadtxt(self._charge_filename)
        print(f"FINISH READING CHARGE... {len(self._charge)} atoms")
        print(self._charge)        
        print(" ==========================")
        
        self._NUM_ATOM_PER_MOL:int = len(self._charge)
        if len(self._traj[0]) % self._NUM_ATOM_PER_MOL != 0:
            print("ERROR: Number of atoms in the first step is not divisible by the number of atoms per molecule")
            return 1
        self._NUM_MOL:int = int(self._traj[0].get_number_of_atoms()/self._NUM_ATOM_PER_MOL)
        print(f"NUM_MOL :: {self._NUM_MOL}")
        self._charge_system = np.tile(self._charge, self._NUM_MOL) # NUM_MOL回繰り返し
        print(self._charge_system)
        
    def calc_dipole(self):
        """calculate msd

        Returns:
            _type_: _description_
        """
        
        import numpy as np
        dipole_list = []
        for counter,atoms in enumerate(self._traj): # loop over MD step
            # self._charge_systemからsystem dipoleを計算
            tmp_dipole = np.einsum("i,ij->j",self._charge_system, atoms.get_positions())
            dipole_list.append([counter,tmp_dipole[0],tmp_dipole[1],tmp_dipole[2]])
        # 計算されたdipoleを保存する．
        np.savetxt("classical_dipole.txt",np.array(dipole_list),header=" index dipole_x dipole_y dipole_z")
        return dipole_list
        
        

class Plot_dipole:
    
    '''
    DIPOLEファイルの双極子モーメントと，それを変換した誘電関数のデータをプロット．
    誘電関数を計算するにはタイムステップdtが必要であり，結局outputファイルをみることになりそう．．

    DIPOLEに関しては，CPMDのマニュアルP192で以下のようになっている．
    Columns 2 to 4 in the DIPOLE file are the electronic contribution to the dipole moment,
    columns 5 to 7 are the total (electronic + ionic) dipole moment.
    All dipole moments are divided by the volume of the box.

    従って，5-7列めをプロットする必要がある．
    '''
    
    def __init__(self,evp_filename,stdout):
        self.__filename = evp_filename
        self.data = np.loadtxt("DIPOLE") # 読み込むのはdipoleファイル
        import os
        if not os.path.isfile("DIPOLE"):
            print(" ERROR :: "+str("DIPOLE")+" does not exist !!")
            print(" ")
            return 1
        if stdout != "":
            # from ase.io import read
            from cpmd.read_traj_cpmd import raw_cpmd_get_timestep
            self.timestep=raw_cpmd_get_timestep(stdout)/1000 # fs単位で読み込むので，psへ変換
            print(" timestep [ps] :: {}".format(self.timestep))
        else:
            self.timestep=0.001 # ps単位で，defaultを1fs=0.001psにしておく

    def plot_dipole(self):
        fig, ax = plt.subplots(figsize=(8,5),tight_layout=True) # figure, axesオブジェクトを作成
        ax.plot(self.data[:,0]*self.timestep, self.data[:,4], label=self.__filename+"_x", lw=3)  # 描画
        ax.plot(self.data[:,0]*self.timestep, self.data[:,5], label=self.__filename+"_y", lw=3)  # 描画
        ax.plot(self.data[:,0]*self.timestep, self.data[:,6], label=self.__filename+"_z", lw=3)  # 描画

        # 各要素で設定したい文字列の取得
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        xlabel="Time $\mathrm{ps}$"
        ylabel="Dipole [D/Volume?]"

        # 各要素の設定を行うsetコマンド
        ax.set_xlabel(xlabel,fontsize=22)
        ax.set_ylabel(ylabel,fontsize=22)
        
        # https://www.delftstack.com/ja/howto/matplotlib/how-to-set-tick-labels-font-size-in-matplotlib/#ax.tick_paramsaxis-xlabelsize-%25E3%2581%25A7%25E7%259B%25AE%25E7%259B%259B%25E3%2582%258A%25E3%2583%25A9%25E3%2583%2599%25E3%2583%25AB%25E3%2581%25AE%25E3%2583%2595%25E3%2582%25A9%25E3%2583%25B3%25E3%2583%2588%25E3%2582%25B5%25E3%2582%25A4%25E3%2582%25BA%25E3%2582%2592%25E8%25A8%25AD%25E5%25AE%259A%25E3%2581%2599%25E3%2582%258B
        ax.tick_params(axis='x', labelsize=15 )
        ax.tick_params(axis='y', labelsize=15 )
        
        ax.legend(loc="upper right",fontsize=15 )
        
        #pyplot.savefig("eps_real2.pdf",transparent=True) 
        # plt.show()
        fig.savefig(self.__filename+"_Dipole.pdf")
        fig.delaxes(ax)
        return 0

    def plot_dielec(self):
        '''
        誘電関数の計算，及びそのプロットを行う．
        体積による規格化や，前にかかる係数などは何も処理しない．

        ---------
        TODO :: ちゃんとDIPOLEファイルでの係数の定義を突き止める．
        '''
        import statsmodels.api as sm 
        # eps0 = 8.8541878128e-12
        # debye = 3.33564e-30
        # nm3 = 1.0e-27
        # nm = 1.0e-9
        # A3 = 1.0e-30
        # kb = 1.38064852e-23
        # T =400 

        # # time =ms["time"].to_numpy()
        # V = np.abs(np.dot(np.cross(UNITCELL_VECTORS[:,0],UNITCELL_VECTORS[:,1]),UNITCELL_VECTORS[:,2])) * A3
        # ## V = np.abs(np.dot(np.cross(traj[0].UNITCELL_VECTOR[:,0],traj[0].UNITCELL_VECTOR[:,1]),traj[0].UNITCELL_VECTOR[:,2])) * A3
        # print("SUPERCELL VOLUME (m^3) :: ", V )
        # # V=   11.1923*11.1923*11.1923 * A3
        # kbT = kb * T 

        # dMx=cell_dipoles_pred[:,0]
        # dMy=cell_dipoles_pred[:,1]
        # dMz=cell_dipoles_pred[:,2]

        N=int(np.shape(self.data[:,0])[0]/2)
        print("nlag :: ", N)
        
        # 自己相関関数を求める
        acf_x = sm.tsa.stattools.acf(self.data[:,4],nlags=N,fft=False)
        acf_y = sm.tsa.stattools.acf(self.data[:,5],nlags=N,fft=False)
        acf_z = sm.tsa.stattools.acf(self.data[:,6],nlags=N,fft=False)

        # time in ps
        time=self.data[:,0]*self.timestep #(in ps)

        #天野さんコード(numpy)
        from quadrupole.calc_fourier import calc_fourier

        # eps_n2 = 1.333**2
        eps_0 = 1.0269255134097743
        eps_n2 = 3.1**2   # eps_n2=eps_inf^2 ?
        eps_inf = 1.0     # should be fixed 
        #eps_0 = pred_eps
        #data=acfs["acf"].to_numpy()
        fft_data =(acf_x+acf_y+acf_z)/3
        
        TIMESTEP =(time[1]-time[0])  # psec.
        print("TIMESTEP [fs] :: ", TIMESTEP*1000)

        rfreq, ffteps1, ffteps2=calc_fourier(fft_data, eps_0, eps_n2, TIMESTEP)

        # convert THz to cm-1
        kayser = rfreq * 33.3 

        # make a plot
        fig, ax = plt.subplots(figsize=(8,5),tight_layout=True) # figure, axesオブジェクトを作成
        ax.plot(kayser, ffteps2, label="DIPOLES", lw=3)  # 描画
        # 各要素で設定したい文字列の取得
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        xlabel="Frequency $\mathrm{cm}^{-1}$"
        ylabel="Dielec [arb.-unit]"

        # 各要素の設定を行うsetコマンド
        ax.set_xlabel(xlabel,fontsize=22)
        ax.set_ylabel(ylabel,fontsize=22)
        # 描画するのは0以上でok!
        ax.set_xlim([0,max(kayser)])

        ax.tick_params(axis='x', labelsize=15 )
        ax.tick_params(axis='y', labelsize=15 )

        # https://www.delftstack.com/ja/howto/matplotlib/how-to-set-tick-labels-font-size-in-matplotlib/#ax.tick_paramsaxis-xlabelsize-%25E3%2581%25A7%25E7%259B%25AE%25E7%259B%259B%25E3%2582%258A%25E3%2583%25A9%25E3%2583%2599%25E3%2583%25AB%25E3%2581%25AE%25E3%2583%2595%25E3%2582%25A9%25E3%2583%25B3%25E3%2583%2588%25E3%2582%25B5%25E3%2582%25A4%25E3%2582%25BA%25E3%2582%2592%25E8%25A8%25AD%25E5%25AE%259A%25E3%2581%2599%25E3%2582%258B
        ax.tick_params(axis='x', labelsize=15 )
        ax.tick_params(axis='y', labelsize=15 )
        
        ax.legend(loc="upper right",fontsize=15 )
        
        #pyplot.savefig("eps_real2.pdf",transparent=True) 
        # plt.show()
        fig.savefig(self.__filename+"_Dielec.pdf")
        fig.delaxes(ax)

        return 0

    
    def process(self):
        print(" ==========================")
        print(" Reading {:<20}   :: making Dipole plots ".format(self.__filename))
        print("")
        self.plot_dipole()
        self.plot_dielec()


# def plot_dipole(filename):
    
#     '''
#     DIPOLEファイルの双極子モーメントと，それを変換した誘電関数のデータをプロット．
#     誘電関数を計算するにはタイムステップdtが必要であり，結局outputファイルをみることになりそう．．

#     DIPOLEに関しては，CPMDのマニュアルP192で以下のようになっている．
#     Columns 2 to 4 in the DIPOLE file are the electronic contribution to the dipole moment,
#     columns 5 to 7 are the total (electronic + ionic) dipole moment.
#     All dipole moments are divided by the volume of the box.

#     従って，5-7列めをプロットする必要がある．
#     '''

#     import os
#     if not os.path.isfile(filename):
#         print(" ERROR :: "+str(filename)+" does not exist !!")
#         print(" ")
#         return 1
#     data = np.loadtxt(filename)
#     print(" --------- ")
#     print(" plot DIPOLE column 4,5 and 6")
#     print(" --------- ")
#     fig, ax = plt.subplots(figsize=(8,5),tight_layout=True) # figure, axesオブジェクトを作成
#     ax.plot(data[:,0], data[:,4], label="x", lw=3)     # 描画
#     ax.plot(data[:,0], data[:,5], label="y", lw=3)     # 描画
#     ax.plot(data[:,0], data[:,6], label="z", lw=3)     # 描画
    
    
#     # 各要素で設定したい文字列の取得
#     xticklabels = ax.get_xticklabels()
#     yticklabels = ax.get_yticklabels()
#     xlabel="Timesteps"       #"Time $\mathrm{ps}$"
#     ylabel="Dipole/Volume [D/Ang^3]"
    
#     # 各要素の設定を行うsetコマンド
#     ax.set_xlabel(xlabel,fontsize=22)
#     ax.set_ylabel(ylabel,fontsize=22)
    
#     # https://www.delftstack.com/ja/howto/matplotlib/how-to-set-tick-labels-font-size-in-matplotlib/#ax.tick_paramsaxis-xlabelsize-%25E3%2581%25A7%25E7%259B%25AE%25E7%259B%259B%25E3%2582%258A%25E3%2583%25A9%25E3%2583%2599%25E3%2583%25AB%25E3%2581%25AE%25E3%2583%2595%25E3%2582%25A9%25E3%2583%25B3%25E3%2583%2588%25E3%2582%25B5%25E3%2582%25A4%25E3%2582%25BA%25E3%2582%2592%25E8%25A8%25AD%25E5%25AE%259A%25E3%2581%2599%25E3%2582%258B
#     ax.tick_params(axis='x', labelsize=15 )
#     ax.tick_params(axis='y', labelsize=15 )
    
#     ax.legend(loc="upper right",fontsize=15 )
    
#     fig.savefig("DIPOLE_D.pdf")
#     fig.delaxes(ax)
#     return 0


def delete_wfcs_from_ionscenter(filename:str="IONS+CENTERS.xyz",stdout:str="bomd-wan.out",output:str="IONS_only.xyz"):
    '''
    XYZからions_centers.xyzを削除して，さらにsupercell情報を付与する．
    '''
    

    import cpmd.read_traj_cpmd
    # トラジェクトリを読み込む
    test_read_trajecxyz=ase.io.read(filename,index=":")

    # もしsupercell情報を持っていればそれを採用する．
    if test_read_trajecxyz[0].get_cell() != "":
        UNITCELL_VECTORS = test_read_trajecxyz[0].get_cell()
    else:    
        # supercellを読み込み
        UNITCELL_VECTORS = cpmd.read_traj_cpmd.raw_cpmd_read_unitcell_vector(stdout)

    # 出力するase.atomsのリスト
    answer_atomslist=[]

    # ワニエの座標を廃棄する．
    for config_num, atom in enumerate(test_read_trajecxyz):    
        # for debug
        # 配列の原子種&座標を取得
        atom_list=test_read_trajecxyz[config_num].get_chemical_symbols()
        coord_list=test_read_trajecxyz[config_num].get_positions()
        
        atom_list_tmp=[]
        coord_list_tmp=[]
        for i,j in enumerate(atom_list):
            if j != "X": # 原子がXだったらappendしない
                atom_list_tmp.append(atom_list[i])
                coord_list_tmp.append(coord_list[i])
    
        CM = ase.Atoms(atom_list_tmp,
                       positions=coord_list_tmp,    
                       cell= UNITCELL_VECTORS,   
                       pbc=[1, 1, 1]) 
        answer_atomslist.append(CM)

    # 保存
    ase.io.write(output,answer_atomslist)
    print("==========")
    print(" a trajectory is saved to IONS_only.xyz")
    print(" ")

    return 0



def add_supercellinfo(filename:str="IONS+CENTERS.xyz",stdout:str="bomd-wan.out",output:str="IONS+CENTERS+cell.xyz"):
    '''
    XYZにstdoutから読み込んだsupercell情報を付与する．

    notes
    --------
    XYZではなく，場合によってはTRAJECTORYを読み込みたい場合があるのでその場合に対応している．
    '''

    import cpmd.read_traj_cpmd

    if filename == "TRAJECTORY":
        print(" warning :: file name is TRAJECTORY. ")
        answer_atomslist = cpmd.read_traj_cpmd.CPMD_ReadPOS(filename,cpmdout)

    else:
        # トラジェクトリを読み込む
        test_read_trajecxyz=ase.io.read(filename,index=":")
        
        # supercellを読み込み
        # TODO :: stdout以外からも読み込めると良い．
        UNITCELL_VECTORS = cpmd.read_traj_cpmd.raw_cpmd_read_unitcell_vector(stdout)
        
        # 出力するase.atomsのリスト
        answer_atomslist=[]
        
        # trajectoryを読み込んでaseへ変換
        for config_num, atom in enumerate(test_read_trajecxyz):    
            # for debug
            # 配列の原子種&座標を取得
            atom_list=test_read_trajecxyz[config_num].get_chemical_symbols()
            coord_list=test_read_trajecxyz[config_num].get_positions()
        
            CM = ase.Atoms(atom_list,
                           positions=coord_list,    
                           cell= UNITCELL_VECTORS,   
                           pbc=[1, 1, 1]) 
            answer_atomslist.append(CM)

    # 保存
    ase.io.write(output,answer_atomslist)    
    print("==========")
    print(" a trajectory is saved to ", output)
    print(" ")

    return 0



# --------------------------------
# 以下CPextract.pyからロードする関数たち
# --------------------------------

        
def command_cpmd_energy(args):
    EVP=Plot_energies(args.Filename)
    EVP.process()
    return 0


def command_cpmd_force(args):
    EVP=Plot_forces(args.Filename)
    EVP.process()
    return 0 

def command_cpmd_dfset(args):
    dfset(args.Filename,args.cpmdout,args.interval,args.start)
    return 0

def command_cpmd_dipole(args):
    '''
    plot DIPOLE file
    '''
    Dipole=Plot_dipole(args.Filename, args.stdout)
    Dipole.process()
    return 0


def command_cpmd_xyz(args):
    '''
    make IONS_only.xyz from IONS+CENTERS.xyz
    '''
    delete_wfcs_from_ionscenter(args.Filename, args.stdout,args.output)
    return 0

def command_cpmd_xyzsort(args):
    """ cpmdのsortされたIONS+CENTERS.xyzを処理する．


    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    import cpmd.converter_cpmd
    cpmd.converter_cpmd.back_convert_cpmd(args.input,args.output,args.sortfile)
    return 0

def command_cpmd_addlattice(args):
    """cpmdで得られたxyzにstdoutの格子定数情報を付加する．


    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    add_supercellinfo(args.input,args.stdout,args.output)
    return 0

def command_cpmd_msd(args):
    msd = MSD(args.Filename,args.initial)
    msd.calc_msd()
    return 0

def command_cpmd_charge(args):
    dipole = DIPOLE(args.Filename,args.charge)
    dipole.calc_dipole()
    return 0