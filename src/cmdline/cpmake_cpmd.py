OB#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# simple code to extract data from CP.x outputs
# define sub command of CPextract.py
#

import cpmd
import ase
import ase.io
import cpmd.converter_cpmd


# --------------------------------
# 以下CPextract.pyからロードする関数たち
# --------------------------------


def command_cpmd_georelax(args):
    print(" ")
    print(" --------- ")
    print(" input geometry file :: ", args.input )
    print(" output geometry relaxation calculation :: georelax.inp")
    print(" ")
    ase_atoms=ase.io.read(args.input)
    test=cpmd.converter_cpmd.make_cpmdinput(ase_atoms)
    test.make_georelax()
    return 0

def command_cpmd_bomdrelax(args):
    print(" ")
    print(" --------- ")
    print(" input geometry file :: ", args.input )
    print(" output bomd relaxation calculation :: bomdrelax.inp")
    print(" ")
    ase_atoms=ase.io.read(args.input)
    test=cpmd.converter_cpmd.make_cpmdinput(ase_atoms)
    test.make_bomd_relax()
    return 0

def command_cpmd_bomdrestart(args):
    print(" ")
    print(" --------- ")
    print(" input geometry file :: ", args.input )
    print(" output bomd restart+wf calculation :: bomd-wan-restart.inp")
    print(" # of steps :: ", args.step)
    print(" timestep [a.u.] :: ", args.time)
    print(" ") 
    ase_atoms=ase.io.read(args.input)
    test=cpmd.converter_cpmd.make_cpmdinput(ase_atoms)
    test.make_bomd_restart(max_step=args.step,timestep=args.time)
    return 0

def command_cpmd_bomd(args):
    print(" ")
    print(" --------- ")
    print(" input geometry file :: ", args.input )
    print(" output bomd restart calculation :: bomd-restart.inp")
    print(" # of steps :: ", args.step)
    print(" timestep [a.u.] :: ", args.time)
    print(" ") 
    ase_atoms=ase.io.read(args.input)
    test=cpmd.converter_cpmd.make_cpmdinput(ase_atoms)
    test.make_bomd(max_step=args.step,timestep=args.time)
    return 0

def command_cpmd_workflow(args):
    print(" ")
    print(" --------- ")
    print(" input geometry file :: ", args.input )
    print(" output georelax calculation        :: georelax.inp")
    print(" output bomdrelax calculation       :: bomdrelax.inp")
    print(" output bomd restart+wf calculation :: bomd-wan-restart.inp")
    print(" # of steps for restart      :: ", args.step)
    print(" timestep [a.u.] for restart :: ", args.time)
    print(" ") 
    ase_atoms=ase.io.read(args.input)
    test=cpmd.converter_cpmd.make_cpmdinput(ase_atoms)
    test.make_georelax()
    test.make_bomd_relax()
    test.make_bomd_restart(max_step=args.step,timestep=args.time)
    return 0



    
