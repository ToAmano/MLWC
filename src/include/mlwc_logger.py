# -*- coding:utf-8 -*-
from logging import Formatter, handlers, StreamHandler, getLogger, DEBUG, INFO

# https://qiita.com/Esfahan/items/275b0f124369ccf8cf18
def root_logger(IF_DEBUG:bool=False):
    # set root logger
    logger = getLogger()

    # set formatter
    formatter = Formatter('%(asctime)s %(name)s %(funcName)s [%(levelname)s]: %(message)s')

    # set handler&formatter
    handler = StreamHandler()
    handler.setFormatter(formatter)

    # add handler to logger
    logger.addHandler(handler)
    # set log level
    if IF_DEBUG:
        logger.setLevel(DEBUG)
    else:
        logger.setLevel(INFO)

    return logger