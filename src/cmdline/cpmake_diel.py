#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""cpmake_diel.py
CPmake.py diel subcommand for output example input files for dieltools 

- yaml:   C++ yaml type input example
- python: python type input example
"""



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


def output_yaml()->int:
    # yamlを出力する．
    # import yaml
    
    # yml = {'member': ['name:Taro Yamada address:Hokkaido', 
    #                   'name:Ichiro Tanaka address:Tokyo', 
    #                   'name:Jiro Sato address:Okinawa']}
    # with open('test_out.yaml', 'w') as file:
    #     yaml.dump(yml, file, default_flow_style=False)
    config_yaml="""\
general:
  itpfilename: methanol.acpype/input_GMX.mol
  bondfilename: MET_bondlist_OPLS.txt
  savedir: dipole_50ns/
  temperature: 298
descriptor:
  calc: 1
  directory: ./
  xyzfilename: gromacs_trajectory_cell.xyz
  savedir: dipole_50ns/
  descmode: 2
  desctype: allinone
  haswannier: 1
  interval: 1
  desc_coh: 0
predict:
  calc: 1
  desc_dir: dipole_50ns/
  model_dir: /home/k0151/k015124/c++/20231025_model_rotate_methanol/
  modelmode: rotate
  bondspecies: 4
  save_truey: 0 """
    print(config_yaml)
    return 0

def output_python():
    config_inp="""\
&general
	itpfilename=gromacs_input/input1.itp
&descriptor
    calc=1
	directory=outputdata/
	xyzfilename=IONS+CENTERS_merge_cell.xyz
	savedir=test_true/
	descmode=3  
	haswannier=1
	interval=1
	desctype=allinone
	trueonly=1
&predict
        calc=0
	desc_dir=test_true/
	model_dir=20230523_model_rotate/
	itpfilename=gromacs_input/input_GMX.itp
	modelmode=rotate
	save_truey=1
    return 0"""
    print(config_inp)
    return 0


# --------------------------------
# 以下CPmake.pyからロードする関数たち
# --------------------------------

def command_diel(args):
    if args.type == "yaml":
        output_yaml()
    elif args.type == "python":
        output_python()
    return 0
