from diel.moldipole.moldipole import moldipole
from diel.moldipole.moldipole_io import read_file
from include.mlwc_logger import root_logger
logger = root_logger(__name__)


def command_diel_gfactor(args):
    logger.info(" cpextract.py diel gfactor")
    moldipole_instance:moldipole = read_file(args.Filename)
    moldipole_instance.print_info()
    df = moldipole_instance.calc_time_vs_gfactor(int(args.start),int(args.end))
    moldipole_instance.plot_time_vs_gfactor(df)
    return 0
