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


    def plot_Energy(self):
        fig, ax = plt.subplots(figsize=(8,5),tight_layout=True) # figure, axesオブジェクトを作成
        ax.plot(self.data[:,0], self.data[:,5]/ase.units.Hartree, label=self.__filename, lw=3)     # 描画

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

        
def command_cpmd_energy(args):
    EVP=Plot_energies(args.Filename)
    EVP.process()
    return 0

        
def command_cpmd_dfset(args):
    dfset(args.Filename,args.cpmdout,args.interval,args.start)
    return 0
