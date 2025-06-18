# -*- coding:utf-8 -*-
"""
This module provides logging functionalities for the MLWC (Machine Learning for Wavefunction Correction) project.
It includes functions to set up root loggers, library loggers, and to generate default log file names.
The module supports both console and file logging, with customizable logging levels and formatting.

Examples:
    >>> import mlwc_logger
    >>> logger = mlwc_logger.root_logger("my_app", "my_app.log")
    >>> logger.info("This is an info message.")
"""
import logging
import os
import sys
import time
from datetime import datetime
from functools import wraps
from typing import Callable, Optional


def get_log_level():
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, log_level_str, logging.INFO)


# https://qiita.com/Esfahan/items/275b0f124369ccf8cf18
def setup_cmdline_logger(
    logger_name: str, log_file: str = None, level: int = logging.INFO
):
    """
    Set up a logger with a specific name and optional log file.

    Parameters
    ----------
    logger_name : str
        Name of the logger.
    log_file : str, optional
        Path to the log file. If None, logs are printed only to stdout.
    level : int, optional
        Logging level (default is logging.INFO).

    Returns
    -------
    logger
        Configured logger object.

    Examples
    --------
    >>> import logging
    >>> logger = root_logger(logger_name="my_app", log_file="my_app.log", level=logging.DEBUG)
    >>> logger.debug("This is a debug message")
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Prevent the logger from adding duplicate handlers if it has already been initialized
    if not logger.hasHandlers():
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %I:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        # Optional file handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %I:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    return logger


def setup_library_logger(
    logger_name: str, log_file: str = None, level: int = logging.INFO
):
    """
    Set up a logger for library files.

    This function configures a logger to be used within library files.
    It checks if the script is executed from the command line (e.g., CPtrain.py, CPextract.py)
    and adjusts the logger configuration accordingly. If executed from the command line,
    it assumes that the root logger has already been set up and avoids creating duplicate handlers.
    Otherwise, it calls the `root_logger` function to set up a new logger.

    Parameters
    ----------
    logger_name : str
        Name of the logger.
    log_file : str, optional
        Path to the log file. If None, logs are printed only to stdout.
    level : int, optional
        Logging level (default is logging.INFO).

    Returns
    -------
    logger
        Configured logger object.

    Examples
    --------
    >>> import logging
    >>> logger = setup_library_logger(logger_name="my_lib", log_file="my_lib.log", level=logging.DEBUG)
    >>> logger.debug("This is a debug message from the library")
    """
    if_commandline: bool = False
    command_list: list[str] = ["CPtrain.py", "CPextract.py", "CPmake.py"]
    for command in command_list:
        if command in sys.argv[0]:
            # if execute from command line (CPtrain.py CPextract.py, etc...)
            if_commandline = True
    if if_commandline:
        # we do not make handler as it is already made in the root logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.propagate = True
        return logger
    # if execute not from the command line
    return setup_cmdline_logger(logger_name, log_file, level)


def get_default_log_file_name() -> str:
    """
    Get a default log file name based on the current date and time.

    This function generates a default log file name based on the current date and time.
    The log file is created in the "logs" directory. If the directory does not exist, it will be created.

    Returns
    -------
    str
        A log file name in the format "logs/log_YYYYMMDD_HHMMSS.log".

    Examples
    --------
    >>> log_file = get_default_log_file_name()
    >>> print(log_file)
    'logs/log_20231026_153000.log'
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"log_{current_time}.log")


timer_logger = setup_library_logger("MLWC.Timer")


class Timer:
    """
    Context manager for measuring and logging the elapsed time of a code block.

    Parameters:
        name (str): A label for the timed block, used in the log message.
        logger (Optional[logging.Logger]): Logger instance to use. Defaults to MLWC.Timer logger.

    Usage:
        with Timer("my_task"):
            # code to measure
    """

    def __init__(
        self, log_prefix: str, logger: Optional[logging.Logger] = None
    ) -> None:
        self.log_prefix: str = log_prefix
        self._time_start_sec: float = 0.0
        self.logger = logger or timer_logger

    def __enter__(self):
        self._time_start_sec = time.perf_counter()

    def __exit__(self, ex_type, ex_value, trace):
        _time_end_sec = time.perf_counter() - self._time_start_sec
        self.logger.info(
            "%s ELAPSED TIME (s):: %.2f",
            self.log_prefix,
            _time_end_sec,
        )


def timer_dec(logger: Optional[logging.Logger] = None) -> Callable:
    """
    Decorator for measuring and logging the execution time of a function.

    Parameters:
        logger (Optional[logging.Logger]): Logger instance to use. Defaults to MLWC.Timer logger.

    Returns:
        Callable: Wrapped function with timing and logging enabled.

    Usage:
        @timer_dec()
        def my_function():
            # function body
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            log_prefix: str = getattr(func, "__name__", "anonymous_function")
            with Timer(log_prefix, logger=logger):
                return func(*args, **kwargs)

        return wrapper

    return decorator
