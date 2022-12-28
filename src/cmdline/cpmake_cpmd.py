#!/usr/bin/env python3
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
    test.make_georelax(args.type)
    return 0

def command_cpmd_bomdrelax(args):
    print(" ")
    print(" --------- ")
    print(" input geometry file :: ", args.input )
    print(" output bomd relaxation calculation :: bomdrelax.inp")
    print(" ")
    ase_atoms=ase.io.read(args.input)
    test=cpmd.converter_cpmd.make_cpmdinput(ase_atoms)
    test.make_bomd_relax(args.type)
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
    test.make_bomd_restart(max_step=args.step,timestep=args.time,type=args.type)
    return 0

def command_cpmd_bomdoneshot(args):
    print(" ")
    print(" --------- ")
    print(" input geometry file :: ", args.input )
    print(" output bomd wf oneshot calculation :: bomd-oneshot.inp")
    print(" ") 
    ase_atoms=ase.io.read(args.input)
    test=cpmd.converter_cpmd.make_cpmdinput(ase_atoms)
    test.make_bomd_oneshot(type=args.type)
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
    test.make_bomd(max_step=args.step,timestep=args.time,type=args.type)
    return 0

def command_cpmd_cpmd(args):
    print(" ")
    print(" --------- ")
    print(" input geometry file :: ", args.input )
    print(" output cpmd restart calculation :: cpmd-restart.inp")
    print(" # of steps :: ", args.step)
    print(" ") 
    ase_atoms=ase.io.read(args.input)
    test=cpmd.converter_cpmd.make_cpmdinput(ase_atoms)
    test.make_cpmd(max_step=args.step,type=args.type)
    return 0

def command_cpmd_cpmdwan(args):
    print(" ")
    print(" --------- ")
    print(" input geometry file :: ", args.input )
    print(" output cpmd+wf restart calculation :: cpmd-restart.inp")
    print(" # of steps :: ", args.step)
    print(" ") 
    ase_atoms=ase.io.read(args.input)
    test=cpmd.converter_cpmd.make_cpmdinput(ase_atoms)
    test.make_cpmd_wan(max_step=args.step,type=args.type)
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
    test.make_georelax(type=args.type)
    test.make_bomd_relax(type=args.type)
    test.make_bomd_restart(max_step=args.step,timestep=args.time,type=args.type)
    return 0

def command_cpmd_workflow_cp(args):
    print(" ")
    print(" --------- ")
    print(" input geometry file :: ", args.input )
    print(" output georelax calculation        :: georelax.inp")
    print(" output bomdrelax calculation       :: bomdrelax.inp")
    print(" output cpmd restart calculation    :: cpmd-restart.inp")
    print(" # of steps for restart      :: ", args.step)
    print(" timestep [a.u.] for restart :: 0.1[fs] (fix)")
    print(" ") 
    ase_atoms=ase.io.read(args.input)
    test=cpmd.converter_cpmd.make_cpmdinput(ase_atoms)
    test.make_georelax(type=args.type)
    test.make_cpmd(max_step="30000", type=args.type) #cpmdでのrelax計算を3psやる．
    test.make_cpmd(max_step=args.step,type=args.type)
    return 0



    
