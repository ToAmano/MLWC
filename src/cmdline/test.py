#!/usr/bin/env python
# coding: utf-8

'''
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
    - 

'''


import argparse
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cpmd.read_core
import cpmd.read_traj

# cmdlines
import cpextract_cp
import cmdline.cpextract_cpmd.cpextract_cpmd as cpextract_cpmd


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
    parser_cp_dfset.add_argument("--interval", \
                        help='dfsetの場合のinterval\n'
                        )
    parser_cp_dfset.add_argument("--start", \
                        help='dfsetの場合のstart_step\n'
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
    parser_cpmd_evp = cpmd_sub_parsers.add_parser('energy', help='cpmd.x ENERGIES parser')
    parser_cpmd_evp.add_argument("-F", "--Filename", \
                        help='CPMD.x ENERGIES file to be parsed.\n', \
                        default="ENERGIES"
                        )
    parser_cpmd_evp.set_defaults(handler=cpextract_cpmd.command_cpmd_energy)
    
    
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
    print("                       Version. 0.0.2                             ")
    print(" *****************************************************************")
    print(" ")

    parser, args = parse_cml_args(sys.argv[1:])

    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
