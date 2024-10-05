# -*- coding:utf-8 -*-
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