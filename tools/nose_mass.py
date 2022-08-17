#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# simple code to suggest Nose-mass in VASP or QE
#

import argparse
import ase
import sys
import numpy as np
from ase.io import read, write

def nose_mass_VASP(temperature, ndof, t0, L):
    '''
    Suggested Q:
        Shuichi Nosé, J. Chem. Phys., 81, 511(1984).
    input:
    temperaute: in unit of Kelvin           :: by hand
    ndof: No. of degrees of freedom         :: from inputfile
    t0: The oscillation time in fs          :: by hand
    L: the length of the first basis vector :: from inputfile
    '''

    # Q in Energy[eV] * Times[s]**2
    qtmp = (t0 * 1E-15 / np.pi / 2)**2 * \
        2 * ndof * ase.units.kB * temperature \
        * ase.units._e

    # Q in AMU * Angstrom**2
    Q = qtmp / ase.units._amu / (L * 1E-10)**2

    return Q



def nose_mass_QE(temperature, ndof, t0, L):
    '''
    Suggested Q:
        Shuichi Nosé, J. Chem. Phys., 81, 511(1984).
    input:
    temperaute: in unit of Kelvin                  :: by hand
    ndof: No. of degrees of freedom                :: from inputfile
    t0: The oscillation time in fs                 :: by hand
    L: the length of the first basis vector in Ang :: from inputfile
    caution on ase.units:
    Time is given in units of Å*sqrt(u/eV) 
    
    '''

    # Q in Energy * Times**2
    qtmp = (t0 * 1E-15 / np.pi / 2)**2 * \
        2 * ndof * ase.units.kB * temperature \
        * ase.units._e

    # Q in AMU * Angstrom**2
    Q = qtmp / ase.units._amu / (L * 1E-10)**2

    return Q



def cnt_dof(atoms):
    '''
    Count No. of Degrees of Freedom
    atoms :: ase.atoms object
    
    caution!! :: we suppose there is no constraints
    '''
    # normal case :: return 3N-3
    return len(atoms) * 3 - 3

def parse_cml_args(cml):
    '''
    CML parser.
    '''
    arg = argparse.ArgumentParser(add_help=True)

    arg.add_argument('-i', dest='poscar', action='store', type=str,
                     default='POSCAR',
                     help='Real POSCAR / QE.in to calculate the No. of Degrees of freedom. Default is POSCAR ')
    arg.add_argument('-u', dest='unit', action='store', type=str,
                     default='fs',
                     choices=['cm-1', 'fs'],
                     help='Default unit of input frequency')
    arg.add_argument('-T', '--temperature', dest='temperature',
                     action='store', type=float,
                     default=300,
                     help='The temperature.')
    arg.add_argument('-f', '--frequency', dest='frequency',
                     action='store', type=float,
                     default=40,
                     help='The frequency of the temperature oscillation')

    return arg.parse_args(cml)


if __name__ == '__main__':
    arg = parse_cml_args(sys.argv[1:])

    if arg.unit == 'cm-1':
        THzToCm = 33.3564095198152
        t0 = 1000 * THzToCm / arg.frequency
    else:
        t0 = arg.frequency

    # load L and ndof from poscar
    pos  = read(arg.poscar)
    L    = np.linalg.norm(pos.cell, axis=1)[0]
    NDOF = cnt_dof(pos)

    # 
    
    # calculate nose-mass
    Q    = nose_mass(arg.temperature, NDOF, t0, L)

    print("SMASS = {}".format(Q))
