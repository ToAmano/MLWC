#!/usr/bin/env python
# coding: utf-8


'''
!! CPmake.py :: tools to make various input files...

このファイルは単にparserを定義している．実行するメインの関数は他のファイルで定義されている．

CPmake smile コマンド（csvからitpファイルを作成）
    - CPmake smile inputfilename

CPmake nose コマンド（VASP用に最適なnose-massをsuggestする．）
    - CPmake nose POSCAR TEMP --unit --frequency

CPmake cpmd コマンド (cpmd.x用の入力ファイル作成)
    - cpextract cpmd georelax (geometry relaxation計算用の入力を作成)
    - cpextract cpmd bomdrelax (bomd relaxation計算用の入力を作成)
    - cpmake cpmd workflow_cp
    - cpmake cpmd workflow
いずれも入力作成の元となる座標情報が必要．
TODO :: 現状はgromacsのgroファイルのみ対応しているのを増やしたい．
'''

from __future__ import annotations # fugaku上のpython3.8で型指定をする方法（https://future-architect.github.io/articles/20201223/）


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
import cmdline.cpmake_cpmd as cpmake_cpmd
import cmdline.cpmake_smile as cpmake_smile
import cmdline.cpmake_nose as cpmake_nose
import cmdline.cpmake_diel as cpmake_diel

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
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    
    # # * ------------
    # # cpextract cp 
    # parser_cp = subparsers.add_parser("cp", help="cp sub-command for CP.x")
    # # parser_cp.set_defaults(handler=command_cp)
    
    # # create sub-parser for sub-command cool
    # cp_sub_parsers = parser_cp.add_subparsers(help='sub-sub-command help')
    
    # # cpextract cp evp 
    # parser_cp_evp = cp_sub_parsers.add_parser('evp', help='cp.x evp parser')
    # parser_cp_evp.add_argument("Filename", \
    #                     help='CP.x *.evp file to be parsed.\n'
    #                     )
    # parser_cp_evp.set_defaults(handler=cpextract_cp.command_cp_evp)

    # # cpextract cp dfset
    # parser_cp_dfset = cp_sub_parsers.add_parser('dfset', help='cp.x to dfset converter')
    # parser_cp_dfset.add_argument("Filename", \
    #                     help='CP.x *.pos file to be parsed.\n'
    #                     )
    # parser_cp_dfset.add_argument("for", \
    #                     help='CP.x *.for file to be parsed.\n'
    #                     )
    # parser_cp_dfset.add_argument("-i","--interval", \
    #                              help='dfsetの場合のinterval\n',\
    #                              default=10,
    #                     )
    # parser_cp_dfset.add_argument("-s","--start", \
    #                              help='dfsetの場合のstart_step\n',\
    #                              default=0,
    #                     )
    # parser_cp_dfset.set_defaults(handler=cpextract_cp.command_cp_dfset)

    # # cpextract cp wf
    # parser_cp_wf = cp_sub_parsers.add_parser('wan', help='cp.x wf stdoutput parser')
    # parser_cp_wf.add_argument("Filename", \
    #                     help='CP.x (cp-wf) stdout file to be parsed.\n'
    #                     )
    # parser_cp_wf.set_defaults(handler=cpextract_cp.command_cp_wf)

    # * ------------
    # cpmake cpmd
    parser_cpmd = subparsers.add_parser("cpmd", help="cpmd sub-command for CPMD.x")
    # parser_cpmd.set_defaults(handler=command_cpmd)

    # create sub-parser for sub-command cool
    cpmd_sub_parsers = parser_cpmd.add_subparsers(help='sub-sub-command help')

    # cpmake cpmd georelax
    parser_cpmd_georelax = cpmd_sub_parsers.add_parser('georelax', \
                                                       help='cpmd.x geoetry relaxation calculation.')
    parser_cpmd_georelax.add_argument("-i", "--input", \
                        help='gromacs coordinates file (gro).\n', \
                        default="eq.pdb"
                        )
    parser_cpmd_georelax.add_argument("--type", \
                        help='determine if atoms are rearranged with atomic speicies or not.\n', \
                        default="default"
                        )
    
    parser_cpmd_georelax.set_defaults(handler=cpmake_cpmd.command_cpmd_georelax)


    # cpmake cpmd bomdrelax
    parser_cpmd_bomdrelax = cpmd_sub_parsers.add_parser('bomdrelax', \
                                                       help='cpmd.x bomd relaxation calculation.')
    parser_cpmd_bomdrelax.add_argument("-i", "--input", \
                        help='gromacs coordinates file (gro).\n', \
                        default="eq.pdb"
                        )
    parser_cpmd_bomdrelax.add_argument("--type", \
                        help='determine if atoms are rearranged with atomic speicies or not.\n', \
                        default="default"
                        )

    parser_cpmd_bomdrelax.set_defaults(handler=cpmake_cpmd.command_cpmd_bomdrelax)


    # cpmake cpmd bomd
    parser_cpmd_bomd = cpmd_sub_parsers.add_parser('bomd', \
                                                       help='cpmd.x bomd restart calculation.')
    parser_cpmd_bomd.add_argument("-i", "--input", \
                        help='gromacs coordinates file (gro).\n', \
                        default="eq.pdb"
                        )
    parser_cpmd_bomd.add_argument("-n", "--step", \
                        help='# of steps.\n', \
                        default="10000"
                        )
    parser_cpmd_bomd.add_argument("-t", "--time", \
                        help='time in a.u.\n', \
                        default="40"
                        )
    parser_cpmd_bomd.add_argument("--type", \
                        help='determine if atoms are rearranged with atomic speicies or not.\n', \
                        default="default"
                        )
    
    parser_cpmd_bomd.set_defaults(handler=cpmake_cpmd.command_cpmd_bomd)

    
    # cpmake cpmd bomdrestart
    parser_cpmd_bomdrestart = cpmd_sub_parsers.add_parser('bomdrestart', \
                                                       help='cpmd.x bomd restart calculation.')
    parser_cpmd_bomdrestart.add_argument("-i", "--input", \
                        help='gromacs coordinates file (gro).\n', \
                        default="eq.pdb"
                        )
    parser_cpmd_bomdrestart.add_argument("-n", "--step", \
                        help='# of steps.\n', \
                        default="10000"
                        )
    parser_cpmd_bomdrestart.add_argument("-t", "--time", \
                        help='time in a.u.\n', \
                        default="40"
                        )
    parser_cpmd_bomdrestart.add_argument("--type", \
                        help='determine if atoms are rearranged with atomic speicies or not.\n', \
                        default="default"
                        )
    
    parser_cpmd_bomdrestart.set_defaults(handler=cpmake_cpmd.command_cpmd_bomdrestart)


    # cpmake cpmd oneshot
    parser_cpmd_oneshot = cpmd_sub_parsers.add_parser('oneshot', \
                                                       help='cpmd.x bomd+wf oneshot calculation.')
    parser_cpmd_oneshot.add_argument("-i", "--input", \
                        help='gromacs coordinates file (gro).\n', \
                        default="eq.pdb"
                        )
    parser_cpmd_oneshot.add_argument("--type", \
                        help='determine if atoms are rearranged with atomic speicies or not.\n', \
                        default="default"
                        )

    parser_cpmd_oneshot.set_defaults(handler=cpmake_cpmd.command_cpmd_bomdoneshot)


    # cpmake cpmd cpmd
    parser_cpmd_cpmd = cpmd_sub_parsers.add_parser('cpmd', \
                                                       help='cpmd.x cpmd restart calculation.')
    parser_cpmd_cpmd.add_argument("-i", "--input", \
                        help='gromacs coordinates file (gro).\n', \
                        default="eq.pdb"
                        )
    parser_cpmd_cpmd.add_argument("--type", \
                        help='determine if atoms are rearranged with atomic speicies or not.\n', \
                        default="default"
                        )

    parser_cpmd_cpmd.set_defaults(handler=cpmake_cpmd.command_cpmd_cpmd)


    # cpmake cpmd cpmdwan
    parser_cpmd_cpmdwan = cpmd_sub_parsers.add_parser('cpmdwan', \
                                                       help='cpmd.x cpmd+wf restart calculation.')
    parser_cpmd_cpmdwan.add_argument("-i", "--input", \
                        help='gromacs coordinates file (gro).\n', \
                        default="eq.pdb"
                        )
    parser_cpmd_cpmdwan.add_argument("--type", \
                        help='determine if atoms are rearranged with atomic speicies or not.\n', \
                        default="default"
                        )

    parser_cpmd_cpmdwan.set_defaults(handler=cpmake_cpmd.command_cpmd_cpmdwan)
    

    
    
    # cpmake cpmd workflow
    parser_cpmd_workflow = cpmd_sub_parsers.add_parser('workflow', \
                                                       help='cpmd.x bomd workflow.')
    parser_cpmd_workflow.add_argument("-i", "--input", \
                        help='gromacs coordinates file (gro).\n', \
                        default="eq.pdb"
                        )
    parser_cpmd_workflow.add_argument("-n", "--step", \
                        help='# of steps.\n', \
                        default="10000"
                        )
    parser_cpmd_workflow.add_argument("-t", "--time", \
                        help='time in a.u.\n', \
                        default="40"
                        )
    parser_cpmd_workflow.add_argument("--type", \
                        help='determine if atoms are rearranged with atomic speicies or not.\n', \
                        default="default"
                        )
    parser_cpmd_workflow.set_defaults(handler=cpmake_cpmd.command_cpmd_workflow)


    
    # cpmake cpmd workflow
    parser_cpmd_workflow_cp = cpmd_sub_parsers.add_parser('workflow_cp', \
                                                       help='cpmd.x cpmd(not bomd) workflow.')
    parser_cpmd_workflow_cp.add_argument("-i", "--input", \
                        help='gromacs coordinates file (gro).\n', \
                        default="eq.pdb"
                        )
    parser_cpmd_workflow_cp.add_argument("-n", "--step", \
                        help='# of steps.\n', \
                        default="10000"
                        )
    parser_cpmd_workflow_cp.add_argument("-m", "--emass", \
                        help='fictious mass in [a.u.]. Usually for liquid molecules, 100~400 is recommended.\n', \
                        default="300"
                        )
    parser_cpmd_workflow_cp.add_argument("--type", \
                        help='determine if atoms are rearranged with atomic speicies or not.\n', \
                        default="default"
                        )
    parser_cpmd_workflow_cp.set_defaults(handler=cpmake_cpmd.command_cpmd_workflow_cp)

    

    # * ------------
    # cpmake smile
    parser_smile = subparsers.add_parser("smile", \
                                         help="convert csv including smiles to *.itp file (gromacs input)")
    parser_smile.add_argument("input", \
                         help='csv filename including smiles. It must contain SMILES and NAME.\n', \
                         )
    parser_smile.set_defaults(handler=cpmake_smile.command_smile)

    # * ------------
    # cpmake diel
    parser_diel = subparsers.add_parser("diel", \
                                         help="output reference input files for dieltools")
    parser_diel.add_argument("--type", \
                         help='input type. yaml or python.\n', \
                         )
    parser_diel.set_defaults(handler=cpmake_diel.command_diel)

    
    # * ------------
    # cpmake nose
    parser_nose = subparsers.add_parser("nose", \
                                         help="suggest appropriate nose-mass for VASP.")
    parser_nose.add_argument("input", \
                             help='VASP POSCAR file.Real POSCAR / QE.in to calculate the No. of Degrees of freedom. Default is POSCAR\n', \
                             default="POSCAR",\
                         )
    parser_nose.add_argument("temp", \
                              help='temperature [K].\n', \
                              default="300",\
                         )
    parser_nose.add_argument("-u","--unit", \
                              help='unit for frequency. cm-1 or THz.\n', \
                              default="cm-1",\
                         )
    parser_nose.add_argument("-f","--frequency", \
                              help='tipical frequency (phonon frequency, e.t.c.) of your system. \n', \
                             )

    
    parser_nose.set_defaults(handler=cpmake_nose.command_nose)

    
    return parser, parser.parse_args(cml)   



def main():
    '''
         Simple script to make various input
        Usage:
        $ python CPmake.py file

        For details of available options, please type
        $ python CPmake.py -h
    '''
    print(" ")
    print(" *****************************************************************")
    print("                         CPmake.py                                ")
    print("                       Version. 0.0.1                             ")
    print(" *****************************************************************")
    print(" ")

    parser, args = parse_cml_args(sys.argv[1:])

    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
