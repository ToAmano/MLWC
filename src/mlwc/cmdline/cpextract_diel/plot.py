#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from mlwc.fourier.totaldipole.totaldipole import TotalDipole
from mlwc.fourier.totaldipole.totaldipole_io import read_file
from mlwc.include.mlwc_logger import setup_library_logger

logger = setup_library_logger("MLWC." + __name__)


def command_diel_plot(args):
    """Calculate dielectric constant and check convergence"""
    logger.info(" Cpextract.py diel plot")
    totaldipole_instance: TotalDipole = read_file(args.Filename)
    totaldipole_instance.print_info()
    totaldipole_instance.plot_total_dipole()
    return 0
