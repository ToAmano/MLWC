import os
import sys
from include.mlwc_logger import setup_cmdline_logger

logger = setup_cmdline_logger("MLWC."+__name__)


def if_file_exist(filename: str):
    is_file = os.path.isfile(filename)
    if not is_file:
        logger.error(f"ERROR not found the file :: {filename}")
        sys.exit("1")
    return 0


def python_version_check():

    python_major_ver = sys.version_info.major
    python_minor_ver = sys.version_info.minor

    logger.info(
        f" your python version is ... {python_major_ver}.{python_minor_ver}")

    if sys.version_info.minor < 9:  # https://www.lifewithpython.com/2015/06/python-check-python-version.html
        logger.warning(
            f"WARNING :: recommended python version is 3.9 or above. Your version is :: {sys.version_info.major}")
    elif sys.version_info.minor < 7:
        logger.error(
            f"ERROR !! python is too old. Please use 3.7 or above. Your version is :: {sys.version_info.major}")
    return 0
