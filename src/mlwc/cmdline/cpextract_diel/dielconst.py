"""Code for CPextract.py diel dielconst"""

import argparse

import pandas as pd

from mlwc.fourier.totaldipole.totaldipole import TotalDipole
from mlwc.fourier.totaldipole.totaldipole_io import read_file
from mlwc.include.mlwc_logger import setup_library_logger

logger = setup_library_logger("MLWC." + __name__)


def command_diel_dielconst(args: argparse.Namespace) -> None:
    """Entry point of CPextract.py diel dielconst
    Calculate dielectric constant and check convergence
    """
    logger.info(" Cpextract.py diel dielconst")
    totaldipole_instance: TotalDipole = read_file(args.Filename)
    totaldipole_instance.print_info()
    df: pd.DataFrame = totaldipole_instance.calc_time_vs_dielconst(
        int(args.start), int(args.end)
    )
    totaldipole_instance.plot_time_vs_dielconst(df)
    return 0
