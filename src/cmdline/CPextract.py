#!/usr/bin/env python
# coding: utf-8

"""_summary_

!! cpextract.py

このファイルは単にparserを定義している．実行するメインの関数は他のファイルで定義されている．

cpextract cp コマンド (cp.x用のparser)
    - cpextract cp evp  (*.evpをparseする)
    - cpextract cp dfset (dfsetファイルを作成する)
    - cpextract cp wan   (- stdout+wanをparseする )
    - stdoutをparseする (収束を見る？)

cpextract cpmd コマンド (cpmd.x用のparser)
    - cpextract cpmd energy ( ENERGIESをparseする )
    - cpextract cpmd dfset (dfsetファイルを作成する)
    - cpextract cpmd dipole (DIPOLEをparseする)
    - cpextract cpmd xyz    (IONS+CENTERS.xyzをparseしてワニエなしのものを作成する．)
    - cpextract cpmd sort   (IONS+CENTERS.xyzをparseしてsortしなおす．)
    - cpextract cpmd addlattice (IONS+CENTERS.xyzをparseしてsupercell情報を加える) 

cpextract diel コマンド（dieltools用のparser）
    - cpextract diel 
    - cpextract diel histgram  (molecular_dipoleとbond_dipoleのヒストグラムを描く(ためのデータ生成))
    - 
"""

from __future__ import annotations # fugaku上のpython3.8で型指定をする方法（https://future-architect.github.io/articles/20201223/）


import argparse
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt

if sys.version_info.minor < 9: # versionによる分岐 https://www.lifewithpython.com/2015/06/python-check-python-version.html
    print("WARNING :: recommended python version is 3.9 or above. Your version is :: {}".format(sys.version_info.major))
elif sys.version_info.minor < 7:
    print("ERROR !! python is too old. Please use 3.7 or above. Your version is :: {}".format(sys.version_info.major))
    

import cpmd.read_core
import cpmd.read_traj

# cmdlines
import cmdline.cpextract_cp as cpextract_cp
import cmdline.cpextract_cpmd as cpextract_cpmd
import cmdline.cpextract_diel as cpextract_diel


try:
    import ase.units
except ImportError:
    sys.exit("Error: ase not installed")

# * --------------------------------

#def command_cp(args):
#    print("Hello, cp!")

# def command_cp_evp(args):
#    print("Hello, cp_evp!")

# def command_cp_dfset(args):
#     print("Hello, cp_dfset!")

# def command_cp_wf(args):
#     print("Hello, cp_wf!")

# def command_cpmd_energy(args):
#    print("Hello, cpmd_energy!")
# * --------------------------------


def command_help(args):
    print(parser.parse_args([args.command, "--help"]))


def parse_cml_args(cml):
    parser = argparse.ArgumentParser(description="CPextract.py")
    subparsers = parser.add_subparsers()
    
    # * ------------
    # cpextract cp 
    parser_cp = subparsers.add_parser("cp", help="cp sub-command for CP.x")
    # parser_cp.set_defaults(handler=command_cp)
    
    # create sub-parser for sub-command cool
    cp_sub_parsers = parser_cp.add_subparsers(help='sub-sub-command help')
    
    # cpextract cp evp 
    parser_cp_evp = cp_sub_parsers.add_parser('evp', help='cp.x evp parser')
    parser_cp_evp.add_argument("Filename", \
                        help='CP.x *.evp file to be parsed.\n'
                        )
    parser_cp_evp.set_defaults(handler=cpextract_cp.command_cp_evp)

    # cpextract cp dfset
    parser_cp_dfset = cp_sub_parsers.add_parser('dfset', help='cp.x to dfset converter')
    parser_cp_dfset.add_argument("Filename", \
                        help='CP.x *.pos file to be parsed.\n'
                        )
    parser_cp_dfset.add_argument("for", \
                        help='CP.x *.for file to be parsed.\n'
                        )
    parser_cp_dfset.add_argument("-i","--interval", \
                                 help='dfsetの場合のinterval\n',\
                                 default=10,
                        )
    parser_cp_dfset.add_argument("-s","--start", \
                                 help='dfsetの場合のstart_step\n',\
                                 default=0,
                        )
    parser_cp_dfset.set_defaults(handler=cpextract_cp.command_cp_dfset)

    # cpextract cp wf
    parser_cp_wf = cp_sub_parsers.add_parser('wan', help='cp.x wf stdoutput parser')
    parser_cp_wf.add_argument("Filename", \
                        help='CP.x (cp-wf) stdout file to be parsed.\n'
                        )
    parser_cp_wf.set_defaults(handler=cpextract_cp.command_cp_wf)

    # * ------------
    # cpextract cpmd
    parser_cpmd = subparsers.add_parser("cpmd", help="cpmd sub-command for CPMD.x")
    # parser_cpmd.set_defaults(handler=command_cpmd)

    # create sub-parser for sub-command cool
    cpmd_sub_parsers = parser_cpmd.add_subparsers(help='sub-sub-command help')

    # cpextract cpmd energy
    parser_cpmd_energy = cpmd_sub_parsers.add_parser('energy', help='cpmd.x ENERGIES parser')
    parser_cpmd_energy.add_argument("-F", "--Filename", \
                        help='CPMD.x ENERGIES file to be parsed.\n', \
                        default="ENERGIES"
                        )
    parser_cpmd_energy.set_defaults(handler=cpextract_cpmd.command_cpmd_energy)
    
    # cpextract cpmd force
    parser_cpmd_force = cpmd_sub_parsers.add_parser('force', help='cpmd.x FTRAJECTORY parser')
    parser_cpmd_force.add_argument("-F", "--Filename", \
                        help='CPMD.x FTRAJECTORY file to be parsed.\n', \
                        default="FTRAJECTORY"
                        )
    parser_cpmd_force.set_defaults(handler=cpextract_cpmd.command_cpmd_force)
    
    # cpextract cpmd dipole
    parser_cpmd_dipole = cpmd_sub_parsers.add_parser('dipole', help='cpmd.x DIPOLE parser')
    parser_cpmd_dipole.add_argument("-F", "--Filename", \
                        help='CPMD.x DIPOLE file to be parsed.\n', \
                        default="DIPOLE"
                        )
    parser_cpmd_dipole.add_argument("-s", "--stdout", \
                         help='CPMD.x stdout file to be parsed for a system volume and a timestep.\n', \
                        default="", \
                        )
    parser_cpmd_dipole.set_defaults(handler=cpextract_cpmd.command_cpmd_dipole)
    
    # cpextract cpmd dfset
    parser_cpmd_dfset = cpmd_sub_parsers.add_parser('dfset', help='cpmd.x to DFSET converter')
    parser_cpmd_dfset.add_argument("-F", "--Filename", \
                                   help='CPMD.x ENERGIES file to be parsed.\n', \
                                   default="ENERGIES"
                                   )
    parser_cpmd_dfset.add_argument("-o", "--cpmdout", \
                                   help='CPMD.x std output file to be parsed.\n', \
                                   )
    parser_cpmd_dfset.add_argument("-i", "--interval", \
                                   help='interval to extract structures.\n',\
                                   default=10,\
                                   )
    parser_cpmd_dfset.add_argument("-s", "--start", \
                                   help='start step to extract structures.\n',\
                                   default=0
                                   )
    parser_cpmd_dfset.set_defaults(handler=cpextract_cpmd.command_cpmd_dfset)


    # cpextract cpmd xyz
    parser_cpmd_xyz = cpmd_sub_parsers.add_parser('xyz', help='cpmd.x xyz parser')
    parser_cpmd_xyz.add_argument("-F", "--Filename", \
                        help='CPMD.x IONS+CENTERS.xyz file to be parsed.\n', \
                        default="IONS+CENTERS.xyz"
                        )
    parser_cpmd_xyz.add_argument("-s", "--stdout", \
                        help='CPMD.x stdout file to be parsed for unitcell vectors.\n', \
                        default="bomd-wan.out"
                        )
    parser_cpmd_xyz.add_argument("-o", "--output", \
                        help='resultant xyz filename.\n', \
                        default="IONS+CENTERS.xyz"
                        )
    parser_cpmd_xyz.set_defaults(handler=cpextract_cpmd.command_cpmd_xyz)
    

    # cpextract cpmd sort
    parser_cpmd_sort = cpmd_sub_parsers.add_parser('sort', help='cpmd.x xyz parser to sort')
    parser_cpmd_sort.add_argument("-i", "--input", \
                        help='CPMD.x IONS+CENTERS.xyz file to be parsed.\n', \
                        default="IONS+CENTERS.xyz"
                        )
    parser_cpmd_sort.add_argument("-o", "--output", \
                        help='resultant xyz filename.\n', \
                        default="IONS+CENTERS_sorted.xyz"
                        )
    parser_cpmd_sort.add_argument("-s", "--sortfile", \
                        help='sort file by CPmake.py.\n', \
                        default="sort_index.txt"
                        )
    parser_cpmd_sort.set_defaults(handler=cpextract_cpmd.command_cpmd_xyzsort)


  # cpextract cpmd addlattice
    parser_cpmd_addlattice = cpmd_sub_parsers.add_parser('addlattice', help='cpmd.x addlattice parser to sort')
    parser_cpmd_addlattice.add_argument("-i", "--input", \
                        help='CPMD.x IONS+CENTERS.xyz/FTRAJ.xyz file to be parsed.\n', \
                        default="IONS+CENTERS.xyz"
                        )
    parser_cpmd_addlattice.add_argument("-o", "--output", \
                        help='resultant xyz filename.\n', \
                        default="IONS+CENTERS_addlattice.xyz"
                        )
    parser_cpmd_addlattice.add_argument("-s", "--stdout", \
                        help='CPMD.x stdout file including lattice information.\n', \
                        )
    parser_cpmd_addlattice.set_defaults(handler=cpextract_cpmd.command_cpmd_addlattice)

    # cpextract cpmd msd
    parser_cpmd_msd = cpmd_sub_parsers.add_parser('msd', help='cpmd.x xyz parser to calculate msd')
    parser_cpmd_msd.add_argument("-F", "--Filename", \
                        help='CPMD.x xyz file to be parsed. IONS+CENTERS.xyz or TRAJEC.xyz \n', \
                        default="IONS+CENTERS.xyz"
                        )
    parser_cpmd_msd.add_argument("-i", "--initial", \
                        help='initial step to start msd calcuCPMD.x xyz file to be parsed. IONS+CENTERS.xyz or TRAJEC.xyz \n', \
                        default="IONS+CENTERS.xyz"
                        )
    parser_cpmd_msd.set_defaults(handler=cpextract_cpmd.command_cpmd_msd)

    # cpextract cpmd vdos
    parser_cpmd_vdos = cpmd_sub_parsers.add_parser('vdos', \
                        help='cpmd.x xyz parser to calculate VDOS', \
                        description="cpmd.x xyz parser to calculate VDOS")
    parser_cpmd_vdos.add_argument("-F", "--Filename", \
                        help='CPMD.x xyz file to be parsed. IONS+CENTERS.xyz or TRAJEC.xyz \n', \
                        default="IONS+CENTERS.xyz"
                        )
    parser_cpmd_vdos.add_argument("-t", "--timestep", \
                        help='timestep in fs. Default value is 0.484 fs (20a.u.). \n', \
                        default="0.484" #20 a.u.
                        )
    parser_cpmd_vdos.add_argument("-n", "--numatom", \
                        help='number of atoms in a molecule \n', \
                        default="6"
                        )    
    parser_cpmd_vdos.add_argument("-i", "--initial", \
                        help='initial step to start msd calcuCPMD.x xyz file to be parsed. IONS+CENTERS.xyz or TRAJEC.xyz \n', \
                        default="1"
                        )
    parser_cpmd_vdos.set_defaults(handler=cpextract_cpmd.command_cpmd_vdos)


    # cpextract cpmd roo
    parser_cpmd_roo = cpmd_sub_parsers.add_parser('roo', \
                        help='cpmd.x xyz parser to calculate rOO correlation function', \
                        description="cpmd.x xyz parser to calculate VDOS"
                        )
    parser_cpmd_roo.add_argument("-F", "--filename", \
                        help='CPMD.x xyz file to be parsed. It must include lattice information. \n', \
                        default="IONS+CENTERS.xyz"
                        )
    parser_cpmd_roo.add_argument("-t", "--timestep", \
                        help='timestep in fs. Default value is 0.484 fs (20a.u.). \n', \
                        default="0.484" #20 a.u.
                        )
    parser_cpmd_roo.add_argument("-n", "--numatom", \
                        help='number of atoms in a molecule, including WCs and BCs. \n', \
                        default="6"
                        )    
    parser_cpmd_roo.add_argument("-m", "--molfile", \
                        help='mol file for bonding information. \n', \
                        default="input_GMX.mol"
                        )    
    parser_cpmd_roo.add_argument("-i", "--initial", \
                        help='initial step to start msd calcuCPMD.x xyz file to be parsed. IONS+CENTERS.xyz or TRAJEC.xyz \n', \
                        default="1"
                        )
    parser_cpmd_roo.set_defaults(handler=cpextract_cpmd.command_cpmd_roo)



    # cpextract cpmd charge
    # !! 古典電荷によるtotal dipoleの計算
    parser_cpmd_msd = cpmd_sub_parsers.add_parser('charge', \
                        help='cpmd.x xyz parser to calculate total dipole',\
                        description='cpmd.x xyz parser to calculate total dipole'
                        )
    parser_cpmd_msd.add_argument("-F", "--Filename", \
                        help='CPMD.x xyz file to be parsed. IONS+CENTERS.xyz or TRAJEC.xyz \n', \
                        default="IONS+CENTERS.xyz"
                        )
    # TODO charge fileとして.molを受け付けたい
    parser_cpmd_msd.add_argument("-c", "--charge", \
                        help='charge file to be parsed. \n', \
                        default="charge.txt"
                        )
    parser_cpmd_msd.set_defaults(handler=cpextract_cpmd.command_cpmd_charge)


    # * ------------
    # cpextract diel
    parser_diel = subparsers.add_parser("diel", 
                        help="diel sub-command for post process",\
                        description="diel sub-command for post process")
    # create sub-parser for sub-command cool
    diel_sub_parsers = parser_diel.add_subparsers(help='sub-sub-command help')
    
    # CPextract.py diel histgram
    parser_diel_histgram = diel_sub_parsers.add_parser('histgram', 
                        help='post-process molecule_dipole.txt parser to plot histgram')
    parser_diel_histgram.add_argument("-F", "--Filename", \
                        help='filename of dipole.txt. Currently, total_dipole.txt is not supported.\n', \
                        default="molecule_dipole.txt"
                        )
    parser_diel_histgram.add_argument("-m", "--max", \
                        help='max value of histgram.\n', \
                        default=None
                        )
    parser_diel_histgram.set_defaults(handler=cpextract_diel.command_diel_histgram)
    
    # CPextract.py diel total
    parser_diel_total = diel_sub_parsers.add_parser('total', 
                        help='post-process total_dipole.txt, plotting time vs dipole figures',\
                        description='post-process total_dipole.txt, plotting time vs dipole figures')
    parser_diel_total.add_argument("-F", "--Filename", \
                        help='filename of total_dipole.txt. Currently, only total_dipole.txt is supported.\n', \
                        default="total_dipole.txt"
                        )
    parser_diel_total.set_defaults(handler=cpextract_diel.command_diel_total)
    
    # CPextract.py diel spectra
    parser_diel_spectra = diel_sub_parsers.add_parser('spectra',
                        help='post-process total_dipole.txt, calculating dielectric function.',\
                        description='post-process total_dipole.txt, calculating dielectric function')
    parser_diel_spectra.add_argument("-F", "--Filename", \
                        help='filename of total_dipole.txt. Currently, only total_dipole.txt is supported.\n', \
                        default="total_dipole.txt"
                        )
    parser_diel_spectra.add_argument("-E", "--eps", \
                        help='eps_inf (eps_n2), usually use experimental value.\n', \
                        )
    parser_diel_spectra.add_argument("-s", "--start", \
                        help='start step. default is 0.\n', \
                        default="0"
                        )
    parser_diel_spectra.add_argument("-e", "--end", \
                        help='end step. default is -1 (include all data).\n', \
                        default="-1"
                        )
    parser_diel_spectra.add_argument("-w", "--step", \
                        help='# of steps to use for moving average of alpha. default is 1 (no moving average).\n', \
                        default="1"
                        )
    parser_diel_spectra.add_argument("-W", "--window", \
                        help='method to smooth the spectra. default is hann (hanning window).\n', \
                        default="hann"
                        )
    parser_diel_spectra.add_argument("-f", "--fft", \
                        help='If use FFT for acf. default is True.\n', \
                        default="True"
                        )
    
    parser_diel_spectra.set_defaults(handler=cpextract_diel.command_diel_spectra)
    
    # CPextract.py diel const    
    parser_diel_dielconst = diel_sub_parsers.add_parser('dielconst', 
                        help='post-process total_dipole.txt parser. calculate dielectric constant.',\
                        description='post-process total_dipole.txt parser. calculate dielectric constant.'
                        )
    parser_diel_dielconst.add_argument("-F", "--Filename", \
                        help='filename of total_dipole.txt. Currently, only total_dipole.txt is supported.\n', \
                        default="total_dipole.txt"
                        )
    parser_diel_dielconst.add_argument("-s", "--start", \
                        help='start step. default is 0.\n', \
                        default="0"
                        )
    parser_diel_dielconst.add_argument("-e", "--end", \
                        help='end step. default is -1 (include all data).\n', \
                        default="-1"
                        )
    parser_diel_dielconst.set_defaults(handler=cpextract_diel.command_diel_dielconst)
    
    
    # CPextract.py diel mol
    parser_diel_mol = diel_sub_parsers.add_parser('mol', 
                        help='post-process molecule_dipole.txt parser. calculate dielectric function.',\
                        description='post-process molecule_dipole.txt parser. calculate dielectric function.'
                        )
    parser_diel_mol.add_argument("-F", "--Filename", \
                        help='filename of molecule_dipole.txt. Currently, only molecule_dipole.txt is supported.\n', \
                        default="molecule_dipole.txt"
                        )
    parser_diel_mol.add_argument("-E", "--eps", \
                        help='eps_inf (eps_n2), usually use experimental value.\n', \
                        )
    parser_diel_mol.add_argument("-s", "--start", \
                        help='start step. default is 0.\n', \
                        default="0"
                        )
    parser_diel_mol.add_argument("-e", "--end", \
                        help='end step. default is -1 (include all data).\n', \
                        default="-1"
                        )
    parser_diel_mol.add_argument("-w", "--step", \
                        help='# of steps to use for moving average of alpha. default is 1 (no moving average).\n', \
                        default="1"
                        )
    parser_diel_mol.set_defaults(handler=cpextract_diel.command_diel_mol)
    
    # CPextract.py diel fit   
    parser_diel_fit = diel_sub_parsers.add_parser('fit', 
                        help='post-process *.csv parser. Fit dielectric function with HN function.',\
                        description='post-process *.csv parser. Fit dielectric function with HN function.'
                        )
    parser_diel_fit.add_argument("-F", "--Filename", \
                        help='filename of total_dipole.txt. Currently, only total_dipole.txt is supported.\n', \
                        default="total_dipole.txt"
                        )
    
    parser_diel_fit.add_argument("-n", "--num_hn_functions", \
                        help='The number of HN functions used for fitting.\n', \
                        default="1"
                        )
    parser_diel_fit.add_argument("-l", "--lower_bound", \
                        help='The lower bound used for fitting in cm-1\n', \
                        default=None
                        )

    parser_diel_fit.add_argument("-u", "--upper_bound", \
                        help='The upper bound used for fitting in cm-1\n', \
                        default=None
                        )
    
    parser_diel_fit.set_defaults(handler=cpextract_diel.command_diel_fit)
    
    # args = parser.parse_args()
    
    return parser, parser.parse_args(cml)   



def main():
    '''
         Simple script for plotting CP.x output
        Usage:
        $ python CPextract.py file

        For details of available options, please type
        $ python CPextract.py -h
    '''
    print(" ")
    print(" *****************************************************************")
    print("                       CPextract.py                               ")
    print("                       Version. 0.1.1                             ")
    print(" *****************************************************************")
    print(" ")

    parser, args = parse_cml_args(sys.argv[1:])

    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
