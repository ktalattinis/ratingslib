"""
Module for parsing a dataset
"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT


import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

import pandas as pd
from ratingslib.application import Outcome
from ratingslib.utils.methods import (concat_csv_files_without_header,
                                      get_indices, parse_columns, print_pandas)


def _parse(pairs_data_df,
           col_names,
           parse_dates=None,
           frequency=None,
           start_period_from: int = 1,
           outcome: Outcome = None) -> pd.DataFrame:
    if parse_dates is not None and frequency is not None:
        pairs_data_df[col_names.period] = \
            pairs_data_df[str(parse_dates[0])].dt.to_period(frequency)
        pairs_data_df[col_names.period_number] = \
            pairs_data_df[col_names.period].dt.week
        period_number_index = get_indices(col_names.period_number,
                                          data=pairs_data_df)
        oldrow = 0
    data_np = pairs_data_df.to_numpy()
    if outcome is not None:
        outcome.set_col_indices(pairs_data_df)
    for index, row in enumerate(data_np):
        # If outcome is not None then map
        if outcome is not None:
            pairs_data_df.at[index, outcome.name] = outcome.outcome_value(row)
        # Set week numbers
        if parse_dates is not None:
            if index == 0:
                pairs_data_df.at[index,
                                 col_names.period_number] = start_period_from  # 1
                newrow = start_period_from  # 1
            elif oldrow == row[period_number_index]:
                pairs_data_df.at[index, col_names.period_number] = newrow
            else:
                newrow += 1
                pairs_data_df.at[index, col_names.period_number] = newrow
            oldrow = row[period_number_index]
    return pairs_data_df


def parse_pairs_data(filename_or_data: Union[str, pd.DataFrame],
                     columns_dict: dict = None,
                     parse_dates=None,
                     date_parser: Callable = None,
                     dayfirst: bool = True,
                     frequency: Optional[str] = None,
                     start_period_from: int = 1,
                     outcome: Outcome = None) -> Tuple[pd.DataFrame,
                                                       pd.DataFrame]:
    """Parse data from filename or from pandas DataFrame. 
    The data must be in pairs.
    For example in the case of soccer teams:

        ========== ========== ======== =========
        HomeTeam   AwayTeam   FTHG     FTAG
        ========== ========== ======== =========
        Team1      Team2       1        5
        Team2      Team3       5        4
        ========== ========== ======== =========

    The purpose of this function is to read a .csv file and store it to a
    pandas.DataFrame structure. 
    If the .csv contains dates then ``parse_dates``
    and ``date_parser`` parameters must set.
    In a case of competitions games if there are weekly games then
    frequency parameter must be defined.

    Parameters
    ----------

    filename_or_data : Union[str, pd.DataFrame]
        Filename to parse or a dataframe that contains pairwise scores

    columns_dict : dict, default=None,
        A dictionary mapping the column names of the dataset.
        If None is given, ``COLUMNS_DICT`` will be used
        See the module :mod:`ratingslib.datasets.parameters` for more details.

    parse_dates: Optional[Union[bool, List[int], List[str], List[list], dict]], default = None,
        Which column has the dates.
        The behavior is as follows:

        * boolean. If True -> try parsing the index.

        * list of int or names. e.g. If [1, 2, 3] -> then try parsing columns
          1, 2, 3 each as a separate date column.

        * list of lists. e.g. If [[1, 3]] -> combine columns 1 and 3 and parse
          as a single date column.
          dict, e.g. {'foo' : [1, 3]} -> parse columns 1, 3 as date and call
          result 'foo'

    date_parser : Callable,  default = None
        Function to use for converting a sequence of string columns to an array
        of datetime instances. For example in football-data.co.uk data the
        function is ::

            def parser_date_func(x): return datetime.strptime(x, '%d/%m/%y')

    dayfirst : bool, default=False
        DD/MM format dates, international and European format.

    frequency : Optional[str], default = None
        If is not None then specifies the frequency of the PeriodIndex.
        For example, in soccer competitions if we want each match-week to
        start from Thursday to Wednesday then we set the value "W-THU".
        For more information visit the site of pandas:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

    start_period_from : int, default = 1
        Defines the number that the period starts. Only valid if frequency set.
        For example if ``start_period_from=3`` then period starts from 3.

    outcome : SportOutcome = None
        The ``outcome`` parameter is related with
        application type. For sports application it must be an instance
        of subclass of SportOutcome class.
        e.g. for soccer the type of outcome is
        :class:`ratingslib.application.SoccerOutcome`.
        For more details see :mod:`ratingslib.application` module.

    Returns
    -------
    pairs_data_df : pandas.DataFrame
        A DataFrame of pairwise data after parsing.

    items_df : pandas.DataFrame
        Set of items (e.g. teams)

    """

    col_names = parse_columns(columns_dict)
    pairs_data_df = filename_or_data
    if isinstance(filename_or_data, str):
        pairs_data_df = pd.read_csv(filename_or_data, parse_dates=parse_dates,
                                    date_parser=date_parser, dayfirst=dayfirst,
                                    index_col=False)
        pairs_data_df.dropna(axis=0, how='all', inplace=True)
    if outcome is not None or (frequency is not None and
                               parse_dates is not None):
        pairs_data_df = _parse(pairs_data_df, col_names, parse_dates,
                               frequency, start_period_from, outcome)
    # in a case of reversing dataframe e.g. nfl
    # pairs_data_df=pairs_data_df.iloc[::-1]
    items_h = pairs_data_df[[col_names.item_i]]
    items_a = pairs_data_df[[col_names.item_j]]

    items_df = items_a.rename(
        columns={col_names.item_j: col_names.item_i})
    items_df = pd.concat([items_h, items_df], ignore_index=True)
    items_df = items_df.drop_duplicates(subset=col_names.item_i)
    items_df = items_df.rename(columns={col_names.item_i: 'Item'})
    items_df = items_df.sort_values(by=['Item'])
    items_df.reset_index(inplace=True, drop=True)

    # Inverse key: convert items to dictionary,
    # key is the team name and value is the index value
    # example: 'Arsenal':0,'Aston Villa':1

    return pairs_data_df, items_df


def parse_data(filename_or_data: Union[str, pd.DataFrame],
               reverse_attributes_cols: Optional[List[str]] = None,
               columns_dict: dict = None) -> Tuple[pd.DataFrame,
                                                   pd.DataFrame]:
    """Parse data (not in pairs form) from filename or from pandas.DataFrame

    Parameters
    ----------
    filename_or_data : Union[str, pd.DataFrame]
        Filename to parse or a dataframe

    reverse_attributes_cols : Optional[List[str]], optional
        If not None then columns will be multiplied by -1, by default None

    columns_dict : dict, optional
        The column names of data file.
        See ``ratingslib.datasets.parameters.COLUMNS_DICT`` for more
        details.

    Returns
    -------
    pairs_data_df : pandas.DataFrame
        A DataFrame of data after parsing.

    items_df : pandas.DataFrame
        Set of items (e.g. teams)
    """
    col_names = parse_columns(columns_dict)
    data_df = filename_or_data
    if isinstance(filename_or_data, str):
        data_df = pd.read_csv(filename_or_data)
    if reverse_attributes_cols is not None:
        data_df[reverse_attributes_cols] = \
            -data_df[reverse_attributes_cols]
    items_df = data_df[[col_names.item]]
    items_df = items_df.rename(columns={col_names.item: 'Item'})
    return data_df, items_df


def create_pairs_data(data_df: pd.DataFrame,
                      columns_dict: Optional[Dict[str, Any]] = None):
    """Convert dataset to pairs.
    For example from User-Movie (Rating-Item):

        =====  =======  ======= =======
        User   Movie1   Movie2  Movie3
        =====  =======  ======= =======
        u1      1        5       4
        u2      1        3       0
        =====  =======  ======= =======

    to MovieI-MovieJ (Item-Item):

        =======  ======== ======== =========
        MovieI   MovieJ   point_i  point_j
        =======  ======== ======== =========
        Movie1   Movie2    1         5
        Movie2   Movie3    5         4
        Movie1   Movie3    1         4
        Movie1   Movie2    1         3
        =======  ======== ======== =========

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame of rating-items form

    columns_dict : Optional[Dict[str, str]], default=None
        The column names of data file.
        See ``ratingslib.datasets.parameters.COLUMNS_DICT`` for more
        details.

    Returns
    -------
     item_item_df : pd.DataFrame
        DataFrame of item-item form

    """
    col_names = parse_columns(columns_dict)
    data_df.replace(0, np.nan, inplace=True)
    item_item_df = pd.DataFrame(columns=list(col_names.__dict__.keys()))

    data_values = data_df.values
    columns = {i: val for i, val in enumerate(data_df.columns)}
    pairs = list(itertools.combinations(columns.keys(), 2))
    # print(pairs)
    n = len(data_df.index)
    for i, pair in enumerate(pairs):
        col_i = pair[0]
        col_j = pair[1]
        cij = np.array([[col_i, col_j] for _ in range(n)])
        if i == 0:
            data = np.concatenate((cij, data_values[:, [
                col_i, col_j]]), axis=1)
        else:
            data = np.concatenate((data,
                                   np.concatenate((cij,  data_values[:, [col_i, col_j]]),
                                                  axis=1)), axis=0)
    # print(data)
    item_item_df = pd.DataFrame(
        data=data, columns=list(col_names.__dict__.values()))
    item_item_df.dropna(inplace=True)
    item_item_df[col_names.item_i].replace(columns, inplace=True)
    item_item_df[col_names.item_j].replace(columns, inplace=True)

    # pandas dataframes solution is slow
    # for pair in pairs:
    #     col_name_i = pair[0]
    #     col_name_j = pair[1]
    #     item_i = data_df[[col_name_i]].rename(
    #         columns={col_name_i: col_names.points_i})
    #     item_i[col_names.item_i] = col_name_i
    #     item_j = data_df[[col_name_j]].rename(
    #         columns={col_name_j: col_names.points_j})
    #     item_j[col_names.item_j] = col_name_j
    #     pair_df = pd.concat([item_i, item_j], axis=1)
    #     pair_df.dropna(inplace=True)
    #     item_item_df = pd.concat(
    #         [item_item_df, pair_df], axis=0, ignore_index=True)
    return item_item_df


def create_data_from(path: str,
                     year_min: int = 2005,
                     year_max: int = 2018):
    """Create csv files from given seasons and then write a concatenated
    csv file for those csv files.
    This function has column names for data files of footballdata.co.uk

    Parameters
    ----------
    path : str
        Path of files

    year_min : int, default = 2005
        Starting season

    year_max : int, default = 2018
        Ending season
    """
    from datetime import datetime
    import numpy as np
    from ratingslib.application import SoccerOutcome
    outcome = SoccerOutcome()
    filename_list = []
    teams = set()
    for i in range(year_min, year_max):
        season = i
        filename = ("./sports/soccer/footballdata-co-uk/" + str(season) + "-" +
                    str(season + 1) + "EPLfootballdata.csv")

        def parser(x): return datetime.strptime(x, "%d/%m/%y")
        data_season, teams_df = parse_pairs_data(filename,
                                                 parse_dates=['Date'],
                                                 date_parser=parser,
                                                 frequency='W-THU',
                                                 outcome=outcome)
        columns_soccer = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
                          'HST', 'AST', 'HS', 'AS',
                          'B365H', 'BWH', 'IWH',
                          'B365A', 'BWA', 'IWA',
                          'B365D', 'BWD', 'IWD',
                          'Week_Number', 'Period', 'FT']
        data_season = data_season[columns_soccer]
        data_season['Season'] = str(season) + "-" + str(season + 1)
        print(teams_df)
        teams.update(set(np.unique(teams_df.Team.unique()).flatten()))
        print_pandas(data_season.head())
        # ,data_season.HomeTeam.unique())
        to_csv_file = (path+"EPLfootballdata-" +
                       str(season) + "-" + str(season + 1) + ".csv")
        data_season.to_csv(to_csv_file, index=False)
        filename_list.append(to_csv_file)
    sorted_teams = sorted(teams)
    print(sorted_teams)
    print(len(teams))
    concat_csv_files_without_header(filename_list,
                                    (path+"EPLfootballdata-" +
                                     str(year_min) + "-" +
                                     str(year_max) + ".csv"))
