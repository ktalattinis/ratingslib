"""
Helper functions for the project
"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

import errno
import inspect
import os
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ratingslib.datasets.parameters import COLUMNS_DICT
from ratingslib.utils import logmsg
from ratingslib.utils.logmsg import log
from ratingslib.utils.validation import validate_not_none_and_type


def parse_columns(
        columns_dict: Optional[Dict[str, Any]] = None) -> SimpleNamespace:
    """Return SimpleNamespace for the dictionary of columns

    Parameters
    ----------
    columns_dict : Optional[Dict[str, str]], default=None
        A dictionary mapping the column names of the dataset.
        If None is given, ``COLUMNS_DICT`` will be used
        See the module :mod:`ratingslib.datasets.parameters` for more details.

    Returns
    -------
    n : SimpleNamespace
        A simple object subclass that provides attribute access to its
        namespace. Attributes are the keys of ``columns_dict``.
    """
    if columns_dict is None:
        columns_dict = COLUMNS_DICT
    n = SimpleNamespace(**columns_dict)
    return n


def create_items_dict(items_df: pd.DataFrame) -> Dict[Any, int]:
    """Create dictionary containing all items

    Parameters
    ----------
    items_df : pandas.DataFrame
        Set of items (e.g. teams)

    Returns
    -------
    items_dict : Dict[Any, int]
        Dictionary that maps items' names to integer value.
        For instance in the case of soccer teams ::

            items_dict = {'Arsenal': 0,
                          'Aston Villa': 1,
                          'Birmingham': 2,
                          'Blackburn': 3
                          }

    """
    items_df.reset_index(inplace=True, drop=True)
    dictionary = pd.Series(items_df["Item"]).to_dict()
    items_dict = {v: k for k, v in dictionary.items()}
    return items_dict


def get_indices(*args, data=None, data_columns_dict=None) -> Tuple[int, ...]:
    """Return indices for variable length arguments"""
    if data_columns_dict is None:
        data_columns_dict = {col: i for i, col in enumerate(data.columns)}
    return tuple(data_columns_dict[arg] for arg in args)


def points(row,
           home_points_col_index: int,
           away_points_col_index: int) -> Tuple[int, int]:
    """Return points ij, ji for the given pair.
    The term points depends on the application type. In soccer indicates goals.

    Parameters
    ----------
    items_df : pandas.DataFrame
        Set of items (e.g. teams)

    home_points_col_index : int
        Column index of home item scores (points)

    away_points_col_index : int
        Column index of away item scores (points)

    Returns
    -------
    points_ij : int
        The number of points that item i scores against item j

    points_ji : int
        The number of points that item j scores against item i
    """
    points_ij = row[home_points_col_index]
    points_ji = row[away_points_col_index]
    return points_ij, points_ji


def indices(row: np.ndarray, items_dict: Dict[Any, int], home_col_index: int,
            away_col_index: int) -> Tuple[int, int]:
    """Return indices i,j for the given pair. Indices are the keys of items
    contained in items_dict

    Parameters
    ----------
    row : numpy.ndarray
        Details (names, scores, etc) of the pair of items.

    items_df : pandas.DataFrame
        Set of items (e.g. teams)

    home_col_index : int
        Column index of home item name

    away_points_col_index : int
        Column index of away item name

    Returns
    -------
    i : int
        Index number of home team

    j : int
        Index number of away team
    """
    i = items_dict[row[home_col_index]]
    j = items_dict[row[away_col_index]]
    return i, j


def indices_and_points(row,
                       items_dict: Dict[Any, int],
                       home_col_index: int,
                       away_col_index: int,
                       home_points_col_index: int,
                       away_points_col_index: int) -> Tuple[int, int,
                                                            int, int]:
    """Return indices i,j and points ij,ji for the given pair"""
    i, j = indices(row, items_dict, home_col_index, away_col_index)
    points_ij, points_ji = points(
        row, home_points_col_index, away_points_col_index)
    return i, j, points_ij, points_ji


def clean_args(parameters: dict, name: Callable) -> dict:
    """
    Select parameters that are valid for a function

    Parameters
    ----------
    parameters: dict
        Dictionary of parameters.

    name: Callable
        Name of the function.

    Returns
    -------
    fparams : dict
        Dictionary of valid parameters
    """
    fparams = {
        k: v for k, v in parameters.items() if k in [
            p.name for p in inspect.signature(name).parameters.values()]}
    return fparams


def set_options_pandas():
    """Set option for the presentation of pandas output """
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)


def log_pandas(pandas_df: pd.DataFrame, msg: str = ''):
    """Print (by logging function) a pandas dataframe with options
    for better presentation.
    Warning: set ``logging.level`` <= ``logmsg.PANDAS`` to show logs.

    Parameters
    ----------
    pandas_df : DataFrame
        A dataframe for logging.
    """
    set_options_pandas()
    log(logmsg.PANDAS, msg, pandas_df, sep='\n', new_line=True)


def print_pandas(pandas_df: pd.DataFrame, msg: str = ''):
    """Print (in console) a pandas dataframe with options
    for better presentation.

    Parameters
    ----------
    pandas_df : pandas.DataFrame
        A dataframe for printing.
    """
    set_options_pandas()
    print(msg, pandas_df, sep='\n')


def set_options_numpy(decimals: int):
    """Set display options for numpy matrices."""
    np.set_printoptions(precision=decimals,
                        linewidth=np.inf,
                        threshold=np.inf,
                        suppress=False)


def log_numpy_matrix(matrix: np.ndarray,
                     msg: str = '',
                     decimals: int = 2):
    """
    Print (by logging function) a numpy array with options
    for better presentation.
    Warning: set ``logging.level`` <= ``logmsg.MATRIX`` to show
    logs."""
    set_options_numpy(decimals)
    log(logmsg.MATRIX, msg, matrix, sep='\n', new_line=True)


def str_info(name: str) -> str:
    """Return pretty format of a string"""
    return "\n\n=====%s=====" % (name)


def print_info(name: str):
    """Pretty print a string"""
    print(str_info(name))


def print_loading(loading_percentage: float, demicals: int = 1, end: str = ""):
    """Print the loading percentage of progress bar in a formated string"""
    "{:.1%}"
    print(
        "{:.{demicals}%}".format(
            loading_percentage,
            demicals=demicals),
        end=end)


def get_filename(filename: str,
                 directory_path: str = None,
                 current_dirname: str = None,
                 check_if_not_exists: bool = True) -> str:
    """Return the filename path.

    Parameters
    ----------
    filename : str,
        The name of file or file path+filename

    directory_path : str, default=None
        The path where the file is stored

    current_dirname : str, default=None
        The path of current directory

    check_if_not_exists : bool, default=True
        If file not exists and check_if_not_exists is True don't raise error

    Returns
    -------
    str : the absolute path of filename.
    """
    validate_not_none_and_type(filename, str)
    path = None
    if current_dirname is None:  # there is an absolute path
        if directory_path is None:  # directory path and filename are together
            path_and_filename = os.path.abspath(filename)
        else:
            path = os.path.join(directory_path)
            path_and_filename = os.path.abspath(path + "/" + filename)
    else:  # directory path and filename are separate
        if directory_path is not None:
            path = os.path.join(current_dirname, directory_path)
        else:
            path = os.path.join(current_dirname)
        path_and_filename = os.path.join(path + filename)
    if path is not None and not os.path.isdir(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
    if not os.path.isfile(path_and_filename) and check_if_not_exists is False:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(
                errno.ENOENT), path_and_filename)
    return os.path.abspath(path_and_filename)


def get_filenames(*filenames,
                  directory_path: str,
                  current_dirname: str) -> Tuple[str, ...]:
    """Return the filename path.
    If any of files not exists then raise an error.

    Parameters
    ----------
    filenames : variable length list of str
        The name of files or file path+filename

    directory_path : str, default=None
        The path where the files are stored

    current_dirname : str, default=None
        The path of current directory

    Returns
    -------
    paths_and_filenames_tuple : tuple
        The absolute paths of filenames
    """
    paths_and_filenames_tuple: Tuple[str, ...] = ()
    for filename in filenames:
        path_and_filename = os.path.join(
            current_dirname, directory_path + filename)
        if not os.path.isfile(path_and_filename):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(
                    errno.ENOENT), path_and_filename)
        else:
            paths_and_filenames_tuple += (os.path.abspath(path_and_filename),)
    return paths_and_filenames_tuple


def concat_csv_files_without_header(filenames: List[str], outputfilename: str):
    """ Concatenation of csv files into new one without headers
    (keeps only headers in the beginning of the csv)

    Parameters
    ----------
    filename_list : List[str]
        List of names of files.

    outputfilename : str
        Filename of new file.
    """
    header = False
    with open(outputfilename, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                if not header:
                    outfile.write(infile.read())
                    header = True
                else:
                    lineNo = 0
                    for line in infile:
                        if lineNo != 0:
                            outfile.write(line)
                        else:
                            lineNo = 1
    outfile.close()
