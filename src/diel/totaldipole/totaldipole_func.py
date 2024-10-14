import sys
import ase.units
import ase.io
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import cpmd.read_core
import cpmd.read_traj
from diel.totaldipole.totaldipole import totaldipole
from include.mlwc_logger import root_logger
logger = root_logger(__name__)

def calc_dielconst(totaldipole:totaldipole, eps_inf:float=1.0):
    return totaldipole.calc_dielconst(eps_inf)


    
    