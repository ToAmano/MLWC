from diel.totaldipole.totaldipole import totaldipole
from diel.totaldipole.totaldipole_io import read_file
from diel.acf_fourier import raw_calc_eps0_dielconst # for calculation of dielectric constant
from include.mlwc_logger import setup_library_logger
logger = setup_library_logger("MLWC."+__name__)



def command_diel_dielconst(args):
    logger.info(" cpextract.py diel dielconst")
    totaldipole_instance:totaldipole = read_file(args.Filename)
    totaldipole_instance.print_info()
    df = totaldipole_instance.calc_time_vs_dielconst(int(args.start),int(args.end))
    totaldipole_instance.plot_time_vs_dielconst(df)
    return 0
