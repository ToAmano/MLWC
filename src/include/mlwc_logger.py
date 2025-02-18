# -*- coding:utf-8 -*-
"""logger for mlwc
"""
from logging import Formatter, handlers, StreamHandler, getLogger, DEBUG, INFO
import logging
import os
from datetime import datetime



# https://qiita.com/Esfahan/items/275b0f124369ccf8cf18
def root_logger(logger_name: str, log_file: str = None, level: int = logging.INFO):
    """
    Set up a logger with a specific name and optional log file.

    Args:
        logger_name (str): Name of the logger.
        log_file (str, optional): Path to the log file. If None, logs are printed only to stdout.
        level (int, optional): Logging level (default is logging.INFO).
    
    Returns:
        logger: Configured logger object.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Prevent the logger from adding duplicate handlers if it has already been initialized
    if not logger.hasHandlers():
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Optional file handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
    
    return logger

def get_default_log_file_name() -> str:
    """
    Get a default log file name based on the current date and time.
    
    Returns:
        str: A log file name.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"log_{current_time}.log")

# Example usage in other parts of the library:
# logger = setup_logger(__name__, log_file=get_default_log_file_name())




# def set_up_script_logger(logfile: str, verbose: str = "CRITICAL"):
#     """_summary_
#     No 
#     -----
#     Logging levels:

#     +---------+--------------+----------------+----------------+----------------+
#     |         | our notation | python logging | tensorflow cpp | OpenMP         |
#     +=========+==============+================+================+================+
#     | debug   | 10           | 10             | 0              | 1/on/true/yes  |
#     +---------+--------------+----------------+----------------+----------------+
#     | info    | 20           | 20             | 1              | 0/off/false/no |
#     +---------+--------------+----------------+----------------+----------------+
#     | warning | 30           | 30             | 2              | 0/off/false/no |
#     +---------+--------------+----------------+----------------+----------------+
#     | error   | 40           | 40             | 3              | 0/off/false/no |
#     +---------+--------------+----------------+----------------+----------------+
#     Args:
#         logfile (str): _description_
#         verbose (str, optional): _description_. Defaults to "CRITICAL".

#     Returns:
#         _type_: _description_
#     """
#     import logging
#     formatter = logging.Formatter('%(asctime)s %(name)s %(funcName)s [%(levelname)s]: %(message)s')
#     # Configure the root logger so stuff gets printed
#     root_logger = logging.getLogger() # root logger
#     root_logger.setLevel(logging.DEBUG) # default level is INFO
#     level = getattr(logging, verbose.upper())  # convert string to log level (default INFO)
    
#     # setup stdout logger
#     # INFO以下のログを標準出力する
#     stdout_handler = logging.StreamHandler(stream=sys.stdout)
#     stdout_handler.setLevel(logging.INFO)
#     stdout_handler.setFormatter(formatter)
#     root_logger.addHandler(stdout_handler)
    
        
#     # root_logger.handlers = [
#     #     logging.StreamHandler(sys.stderr),
#     #     logging.StreamHandler(sys.stdout),
#     # ]
#     # root_logger.handlers[0].setLevel(level)        # stderr
#     # root_logger.handlers[1].setLevel(logging.INFO) # stdout
#     if logfile is not None: # add log file
#         root_logger.addHandler(logging.FileHandler(logfile, mode="w"))
#         root_logger.handlers[-1].setLevel(level)
#     return root_logger