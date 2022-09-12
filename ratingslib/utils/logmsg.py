"""
Module for logging
"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

import logging

# Define MESSAGE log level
MESSAGE = 11  # for a simple message
EXAMPLE = 12  # when examples running
MATRIX = 14  # for numpy matrices of rating methods
FILENAME = 29  # print filenames
PANDAS = 90  # for pandas DataFrame

NOMESSAGE = 100  # silent mode


def set_logger(level: int):
    """Set the level of log

    Parameters
    ----------
    level : int
        Level of logging
    """
    logging.addLevelName(MESSAGE, 'MESSAGE')
    logging.addLevelName(EXAMPLE, 'EXAMPLE')
    logging.addLevelName(MATRIX, 'MATRIX')
    logging.addLevelName(PANDAS, 'PANDAS DATAFRAME')
    logging.addLevelName(FILENAME, 'FILENAME DETAILS')
    logging.addLevelName(NOMESSAGE, 'NOMESSAGE')
    logging.basicConfig(format='%(levelname)s: %(message)s')  # , level=level)
    logging.getLogger().setLevel(level)


def log(level: int, *args, sep: str = ' ', new_line: bool = False):
    """Log a msg

    Parameters
    ----------
    level : int
        level for logging

    *args : sequence
        Messages to log

    sep : str, default = ' '
        Separator for messages

    new_line: bool, default = False
        Writes a new line at the beginning of the output
    """
    nl = '\n' if new_line else ''
    logging.log(level, nl + sep.join("{}".format(a) for a in args))
