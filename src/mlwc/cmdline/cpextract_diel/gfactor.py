"""Code for CPextract.py diel gfactor"""

import pandas as pd

from mlwc.fourier.moldipole.moldipole import moldipole
from mlwc.fourier.moldipole.moldipole_io import read_file
from mlwc.include.mlwc_logger import setup_library_logger

logger = setup_library_logger("MLWC." + __name__)


def command_diel_gfactor(args):
    """Calculate G-factor and check convergence"""
    logger.info(" cpextract.py diel gfactor")
    moldipole_instance: moldipole = read_file(args.Filename)
    moldipole_instance.print_info()
    df: pd.DataFrame = moldipole_instance.calc_time_vs_gfactor(
        int(args.start), int(args.end)
    )
    moldipole_instance.plot_time_vs_gfactor(df)
    return 0
