"""difine functions which are also defined in the class."""

from diel.totaldipole.totaldipole import totaldipole
from include.mlwc_logger import root_logger
logger = root_logger(__name__)


def calc_dielconst(totaldipole: totaldipole, eps_inf: float = 1.0):
    return totaldipole.calc_dielconst(eps_inf)
