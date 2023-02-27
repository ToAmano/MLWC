
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# simple code to extract data from CP.x outputs
# define sub-command of CPextract.py
# 
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




def parse_cml_args(cml):

    description='''
    Simple script for plotting CP.x output.
    At the moment, only read *.evp files and plot t vs Energy and t vs Temperature.
    Usage:
    $ python CPextract.py file
    
    For details of available options, please type
    $ python CPextract.py -h
    '''
    
    
    
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     # epilog=CMD_EXAMPLE
                                     )
    parser.add_argument("mode", \
                        default="cp",\
                        help="code name. cp or cpmd.\n"
                        )
    
    parser.add_argument("Filename", \
                        help='CP.x *.evp file, CP.x std output file, CPMD std output file.\n'
                        )

    parser.add_argument("--postime", \
                        help='if True, analyze CP.x *.pos file. \n'
                        )

    parser.add_argument("--dfset", \
                        help='if True, analyze CP.x *.pos, *.for, and input file to generate DFSET. \n'
                        )
    parser.add_argument("--interval", \
                        help='dfsetの場合のinterval\n'
                        )
    parser.add_argument("--start", \
                        help='dfsetの場合のstart_step\n'
                        )
    parser.add_argument("--TRAJ", \
                        help='CPMDのTRAJECTORYファイル\n'
                        )
    parser.add_argument("--force", \
                        help='cp.xのforceファイル\n'
                        )
    parser.add_argument("--pos", \
                        help='cp.xのposファイル\n'
                        )
    parser.add_argument("--pwin", \
                        help='cp.xのpwinファイル\n'
                        )
    
    
    # parser.add_argument(
    #          '--jump',
    #          nargs='?',
    #          default=False,
    #          help=
    #          'how to treat periodic boundary condition. If true, atoms stay in the cell, \n'
    #          'while atoms move across the cell if False. \n'
    #          'Recommend True for liquid, False for crystal. \n'
    #          ' Currently only available in .xyz. '
    # )
    return parser.parse_args(cml)    
    
    

class Plot_evp:
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
    def __init__(self,evp_filename):
        self.__filename = evp_filename
        self.data = np.loadtxt(self.__filename)


    def plot_Energy(self):
        fig, ax = plt.subplots(figsize=(8,5),tight_layout=True) # figure, axesオブジェクトを作成
        ax.plot(self.data[:,1], self.data[:,5]/ase.units.Hartree, label=self.__filename, lw=3)     # 描画

        # 各要素で設定したい文字列の取得
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        xlabel="Time $\mathrm{ps}$"
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
        ax.plot(self.data[:,1], self.data[:,4], label=self.__filename, lw=3)     # 描画

        # 各要素で設定したい文字列の取得
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        xlabel="Time $\mathrm{ps}$"
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

def extract_pos(pos_filename):
    '''
    Extract timesteps and time[ps] from *.pos file.
    Useful to check the procedure of simulations.

    input
    -------------
    pos_filename :: string
        *.pos filename
    '''
    
    f   = open(pos_filename, 'r') # read SPOSCAR

    while True:
        data = f.readline()
        if data.split() == []:
            break
        if len(data.split())==2:
            print(data.split()) # 単に画面に出力するだけ
    return 0


class Plot_mlwf:
    '''
    Extract MLWF steps in std-output, save in MLWF_converge.txt, and making plots.
    '''
    def __init__(self,stdout_filename):

        self.__filename  = stdout_filename
        f   = open(stdout_filename, 'r') # read SPOSCAR
        self.mlwf_data = []
        counter   = 0
        
        while True:
            data = f.readline()
            if not data:
                break
            # debug:: print(data.split())
            #
            #
            # print(data)
            if data.startswith("   MD Simulation time step"):
                self.__timestep = float(data.split()[5])
                print("TIME STEP[a.u.] is ...  ", self.__timestep)
            
            if data.startswith("   MLWF step"):
                counter = counter+1
                data_split=data.split()
                if data_split[6] == "not":
                    flag = 1
                else:
                    flag = 0
                #
                self.mlwf_data.append([int(counter), int(data_split[2]), float(data_split[5]), flag])
        # save in txt
        fwrite   = open("CPextract_mlwf.txt", 'w')
        fwrite.write("# steps nstep mlwf converge? (if converge, flag ==0)\n")
        for i in range(len(self.mlwf_data)):
            line= '{:6} {:6} {} {:2} \n'.format(self.mlwf_data[i][0],self.mlwf_data[i][1],self.mlwf_data[i][2],self.mlwf_data[i][3])
            fwrite.write(line)
        fwrite.close()

        # convert to np.array (for making plot)
        self.mlwf_data=np.array(self.mlwf_data)
    
    def plot_mlwf_converge(self):
        fig, ax = plt.subplots(figsize=(8,5),tight_layout=True) # figure, axesオブジェクトを作成
        ax.plot(self.mlwf_data[:,0]*self.__timestep*2.4189*1e-5, self.mlwf_data[:,2], label=self.__filename, lw=3)     # 描画

        # 各要素で設定したい文字列の取得
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        xlabel="Time $\mathrm{ps}$"
        ylabel="MLWF spread"

        # 各要素の設定を行うsetコマンド
        ax.set_xlabel(xlabel,fontsize=22)
        ax.set_ylabel(ylabel,fontsize=22)
        
        # https://www.delftstack.com/ja/howto/matplotlib/how-to-set-tick-labels-font-size-in-matplotlib/#ax.tick_paramsaxis-xlabelsize-%25E3%2581%25A7%25E7%259B%25AE%25E7%259B%259B%25E3%2582%258A%25E3%2583%25A9%25E3%2583%2599%25E3%2583%25AB%25E3%2581%25AE%25E3%2583%2595%25E3%2582%25A9%25E3%2583%25B3%25E3%2583%2588%25E3%2582%25B5%25E3%2582%25A4%25E3%2582%25BA%25E3%2582%2592%25E8%25A8%25AD%25E5%25AE%259A%25E3%2581%2599%25E3%2582%258B
        ax.tick_params(axis='x', labelsize=15 )
        ax.tick_params(axis='y', labelsize=15 )

        ax.set_yscale('log')
        
        ax.legend(loc="upper right",fontsize=15 )
        
        #pyplot.savefig("eps_real2.pdf",transparent=True) 
        # plt.show()
        fig.savefig(self.__filename+"_mlwf.pdf")
        fig.delaxes(ax)
        return 0

    def process(self):
        print(" ==========================")
        print(" Reading {:<20}   :: making MLWF convergence plots ".format(self.__filename))
        print("")
        self.plot_mlwf_converge()


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
    
    def __init__(self,evp_filename):
        self.__filename = evp_filename
        self.data = np.loadtxt(self.__filename)


    def plot_dipole(self):
        fig, ax = plt.subplots(figsize=(8,5),tight_layout=True) # figure, axesオブジェクトを作成
        ax.plot(self.data[:,0], self.data[:,4], label=self.__filename+"_x", lw=3)  # 描画
        ax.plot(self.data[:,0], self.data[:,5], label=self.__filename+"_y", lw=3)  # 描画
        ax.plot(self.data[:,0], self.data[:,6], label=self.__filename+"_z", lw=3)  # 描画

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

        def process(self):
        print(" ==========================")
        print(" Reading {:<20}   :: making Dipole plots ".format(self.__filename))
        print("")
        self.plot_Energy()
        self.plot_Temperature()

        

def dfset(filename,pwin,file_force,interval_step:int,start_step:int=0):
    '''
    Trajectoryとforceを読み込んで，DFSET_exportを作る（CP.x用）
    '''
    traj =  cpmd.read_traj.ReadPOS(filename=filename, pwin=pwin)
    # import forces 
    traj.set_force_from_file(file_force)
    traj.export_dfset_pwin(interval_step,start_step)
    print(" ")
    print(" make DFSET_export...")
    print(" ")
    return 0



# def cpextract_main():
#     '''
#          Simple script for plotting CP.x output
#         Usage:
#         $ python CPextract.py file

#         For details of available options, please type
#         $ python CPextract.py -h
#     '''
#     ARGS = parse_cml_args(sys.argv[1:])
#     # FCS_FILENAME = args.Filename
#     print("MODE     :: ", ARGS.mode)
#     print("Filename :: ", ARGS.Filename)
#     print("postime  :: ", ARGS.postime)
#     print("dfset    :: ", ARGS.dfset)
    
#     if ARGS.postime=="True":
#         extract_pos(ARGS.Filename)
#         return 0
#     elif ARGS.Filename == "ENERGIES":
#         EVP=Plot_energies(ARGS.Filename)
#         EVP.process()
#     else: # cp.xのstd-outputファイルの場合(wannier解析)
#         print("test")
        


def command_cp_evp(args):
    EVP=Plot_evp(args.Filename)
    EVP.process()
    return 0

def command_cp_dipole(args):
    Dipole=Plot_evp(args.Filename)
    Dipole.process()
    return 0

def command_cp_dfset(args):
    dfset(args.Filename, args.pwin, args.force, args.interval, args.start)
    return 0

def command_cp_wf(args):
    EVP=Plot_mlwf(args.Filename)
    EVP.process()
    return 0

