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
            print(" ERROR :: "+str(filename)+" does not exist !!")
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


def plot_dipole(filename):
    import os
    if not os.path.isfile(filename):
        print(" ERROR :: "+str(filename)+" does not exist !!")
        print(" ")
        return 1
    data = np.loadtxt(filename)
    print(" --------- ")
    print(" plot DIPOLE column 4,5 and 6")
    print(" --------- ")
    fig, ax = plt.subplots(figsize=(8,5),tight_layout=True) # figure, axesオブジェクトを作成
    ax.plot(data[:,0], data[:,4], label="x", lw=3)     # 描画
    ax.plot(data[:,0], data[:,5], label="y", lw=3)     # 描画
    ax.plot(data[:,0], data[:,6], label="z", lw=3)     # 描画
    
    
    # 各要素で設定したい文字列の取得
    xticklabels = ax.get_xticklabels()
    yticklabels = ax.get_yticklabels()
    xlabel="Timesteps"       #"Time $\mathrm{ps}$"
    ylabel="Dipole/Volume [D/Ang^3]"
    
    # 各要素の設定を行うsetコマンド
    ax.set_xlabel(xlabel,fontsize=22)
    ax.set_ylabel(ylabel,fontsize=22)
    
    # https://www.delftstack.com/ja/howto/matplotlib/how-to-set-tick-labels-font-size-in-matplotlib/#ax.tick_paramsaxis-xlabelsize-%25E3%2581%25A7%25E7%259B%25AE%25E7%259B%259B%25E3%2582%258A%25E3%2583%25A9%25E3%2583%2599%25E3%2583%25AB%25E3%2581%25AE%25E3%2583%2595%25E3%2582%25A9%25E3%2583%25B3%25E3%2583%2588%25E3%2582%25B5%25E3%2582%25A4%25E3%2582%25BA%25E3%2582%2592%25E8%25A8%25AD%25E5%25AE%259A%25E3%2581%2599%25E3%2582%258B
    ax.tick_params(axis='x', labelsize=15 )
    ax.tick_params(axis='y', labelsize=15 )
    
    ax.legend(loc="upper right",fontsize=15 )
    
    fig.savefig("DIPOLE_D.pdf")
    fig.delaxes(ax)
    return 0


def delete_wfcs_from_ionscenter(filename:str="IONS+CENTERS.xyz",stdout:str="bomd-wan.out",output:str="IONS_only.xyz"):
    '''
    XYZからions_centers.xyzを削除して，さらにsupercell情報を付与する．
    '''
    

    import cpmd.read_traj_cpmd
    # トラジェクトリを読み込む
    test_read_trajecxyz=ase.io.read(filename,index=":")

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



def add_supercellinfo(filename:str="IONS+CENTERS.xyz",stdout:str="bomd-wan.out",output:str="IONS_only.xyz"):
    '''
    XYZにstdoutから読み込んだsupercell情報を付与する．
    '''
    

    import cpmd.read_traj_cpmd
    # トラジェクトリを読み込む
    test_read_trajecxyz=ase.io.read(filename,index=":")

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

        
def command_cpmd_dfset(args):
    dfset(args.Filename,args.cpmdout,args.interval,args.start)
    return 0

def command_cpmd_dipole(args):
    '''
    plot DIPOLE file
    '''
    plot_dipole(args.Filename)
    return 0 


def command_cpmd_xyz(args):
    '''
    make IONS_only.xyz from IONS+CENTERS.xyz
    '''
    delete_wfcs_from_ionscenter(args.Filename, args.stdout,args.output)
    return 0

def command_cpmd_xyzsort(args):
    '''
    cpmdのsortされたIONS+CENTERS.xyzを処理する．
    '''
    import cpmd.converter_cpmd
    cpmd.converter_cpmd.back_convert_cpmd(args.input,args.output,args.sortfile)
    return 0

