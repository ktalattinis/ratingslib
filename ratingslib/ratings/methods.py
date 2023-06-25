"""
This module gathers helper functions for ratings,
including calculation of statistics,
rating values normalization, outcomes counting.
"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT


from collections import Counter, OrderedDict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ratingslib.ratings.rating import RatingSystem
from ratingslib.utils.methods import create_items_dict, get_indices, parse_columns
from ratingslib.utils.validation import (ValidationError, validate_from_set,
                                         validate_type_of_elements)


def calc_items_stats(data_df,
                     items_df: pd.DataFrame,
                     items_dict: Optional[Dict[Any, int]] = None,
                     normalization: bool = False,
                     stats_columns_dict: Optional[Dict[str,
                                                       Dict[Any, Any]]] = None,
                     columns_dict: Optional[Dict[str, Any]] = None
                     ) -> pd.DataFrame:
    """
    Calculation of the items' statistics

    Parameters
    ----------
    data_df : pandas.DataFrame
        Items data

    items_df : pandas.DataFrame
        The name of items

    items_dict : Optional[Dict[Any, int]], default=None
            Dictionary with teams, where the key is

    normalization : bool, default=True
        If ```True`` values are divided to the number of times
        an item appeared in the dataset.

    stats_columns_dict: Optional[Dict[str, Dict[Any, Any]]]
        Dictionary that maps the statistic names to column names.
        Below is the explanation of dictionary:

            `H`: column name of statistic for home team
            e.g. in football-data.co.uk the column for goals is FTHG.

            `A`: column name of statistic for away team
            e.g. in football-data.co.uk the column for goals is FTAG.

            TYPE: {WIN, POINTS}

                #. if type is WIN then compare if H > A or A < H

                #. if type is POINTS then count points

    columns_dict : Optional[Dict[str, str]]
        The column names of data file.
        See ``ratingslib.datasets.parameters.COLUMNS_DICT`` for more
        details.

    Returns
    -------
    items_df : pandas.DataFrame
        DataFrame of items and their computed statistics

    """

    col_names = parse_columns(columns_dict)
    if stats_columns_dict is None:
        stats_columns_dict = {}
        # stats_columns_dict = stats.STATS_COLUMNS_DICT
    # compute one time, and then pass to get_indices method
    data_columns_dict = {col: i for i, col in enumerate(data_df.columns)}
    home_col_ind, away_col_ind = get_indices(
        col_names.item_i, col_names.item_j,
        data_columns_dict=data_columns_dict)
    data_np = data_df.to_numpy()
    if items_dict is None:
        items_dict = create_items_dict(items_df)
    for key in stats_columns_dict.keys():
        items_df[key] = 0.0
    items_df['NG'] = 0  # number of games played for each team
    for row in data_np:
        i = items_dict[row[home_col_ind]]
        j = items_dict[row[away_col_ind]]
        for key, value in stats_columns_dict.items():
            home_points_col_ind, away_points_col_ind = get_indices(
                value.get('ITEM_I'), value.get('ITEM_J'),
                data_columns_dict=data_columns_dict)
            item_i_points = row[home_points_col_ind]
            item_j_points = row[away_points_col_ind]
            statType = value.get('TYPE')
            if statType == 'WIN':
                if item_i_points > item_j_points:
                    items_df.at[i, key] += 1
                elif item_i_points < item_j_points:
                    items_df.at[j, key] += 1
            elif statType == 'POINTS':
                items_df.at[i, key] += item_i_points
                items_df.at[j, key] += item_j_points

        items_df.at[i, 'NG'] += 1
        items_df.at[j, 'NG'] += 1
    ng = items_df[['NG']].copy(deep=True)
    ng.loc[ng['NG'] == 0] = 1
    if normalization:
        for key in stats_columns_dict.keys():
            items_df[key] /= ng['NG']
    return items_df
    # w /= ng


def normalization_rating(
        items_df: pd.DataFrame,
        col_name: str) -> pd.Series:
    """
    Normalize rating column from all items such
    that minimum is 0 and maximum is 1.

    Parameters
    ----------
    items_df : pandas.DataFrame
        The items with their names and rating values

    col_name : str
        The name of column with rating values

    Returns
    -------
    normalized : pandas.Series
        Series with the normalized rating values

    """
    normalized = (items_df[col_name] - min(items_df[col_name])) / \
        (max(items_df[col_name]) - min(items_df[col_name]))
    return normalized


def count_classes_outcome_and_perc(
        outcomes_array: np.ndarray) -> Tuple[OrderedDict, OrderedDict]:
    """
    Count the number of outcomes and their percentages.
    Create an ordered dictionary with the total number of each outcome
    and another one dictionary with percentages.

    Parameters
    ----------
    outcomes_array : numpy.ndarray

    Returns
    -------
    Tuple[OrderedDict, OrderedDict]

    """
    size = len(outcomes_array)
    sorted_counter_dict: OrderedDict = OrderedDict(
        sorted(Counter(outcomes_array).items()))
    percentage_dict = {k: v / size for k, v in sorted_counter_dict.items()}
    sorted_perc_dict = OrderedDict(sorted(percentage_dict.items()))

    return sorted_counter_dict, sorted_perc_dict


def rating_systems_to_dict(
        rating_systems: Union[Dict[str, RatingSystem],
                              List[RatingSystem], RatingSystem],
        key_based_on: Literal['params_key', 'version'] = 'params_key') -> Dict[str, RatingSystem]:
    """Create a dictionary that maps the given rating systems.

    Parameters
    ----------
    rating_systems : Dict[str, RatingSystem] or List[RatingSystem] or RatingSystem
        A list that contains rating systems instances or a dictionary or 
        just a RatingSystem instance. 
        In the case of dictionary the function only validates the values.

    Returns
    -------
    rating_sys_dict : Dict[str, RatingSystem]
        Dictionary of rating systems.
    """
    # Input validation
    validate_from_set(key_based_on, {'params_key', 'version'}, key_based_on)
    if isinstance(rating_systems, RatingSystem):
        rating_list = [rating_systems]
    elif isinstance(rating_systems, dict):
        rating_list = list(rating_systems.values())
        rating_sys_dict = rating_systems
    elif isinstance(rating_systems, list):
        rating_list = rating_systems
    else:
        raise ValidationError('Wrong type of rating_systems.')
    validate_type_of_elements(rating_list,
                              RatingSystem)
    if isinstance(rating_systems, list) or isinstance(rating_systems, RatingSystem):  # convert to dict
        if key_based_on == 'params_key':
            rating_sys_dict = {value.params_key: value
                               for value in rating_list}
        elif key_based_on == 'version':
            rating_sys_dict = {value.version: value
                               for value in rating_list}
    return rating_sys_dict


def plot_ratings(data_df: pd.DataFrame, item_i: str, item_j: str, ratings_i: str, ratings_j: str,
                 starting_game=2,
                 items_list: Optional[List] = None):
    """Plots the ratings based on games ratings data

    Parameters
    ----------
    data_df : pd.DataFrame
        Games dataframe (item_i vs item_j)
        e.g.     
        HomeTeam     AwayTeam  HEloWin[HA=0_K=40_ks=400]  AEloWin[HA=0_K=40_ks=400]
         Wolves     West Ham                  1500.0000                  1500.0000


    item_i : str
        Item i column name e.g. HomeTeam

    item_j : str
        Item j column name e.g. AwayTeam

    ratings_i : str
        ratings i column e.g. HEloWin[HA=0_K=40_ks=400]

    ratings_j : str
        ratings j column e.g. AEloWin[HA=0_K=40_ks=400]

    starting_game : int, default=2
        starts from the game number in the plot (axis x).

    items_list : Optional[List], default=None
        Items to be included in the plot. If None all items will be plot.
    """
    import matplotlib.pyplot as plt
    if items_list is None:
        items_list = data_df[item_i].unique()
    for team in items_list:
        games_df = data_df[(data_df[item_i] == team) |
                           (data_df[item_j] == team)]
        games_df = games_df.sort_index()
        # print(games_df[[item_i, item_j, ratings_i, ratings_j]])
        indices = range(starting_game, len(games_df)+starting_game)
        plot_df = games_df[ratings_i].where(
            games_df[item_i] == team, games_df[ratings_j])
        plt.plot(indices, plot_df, label=team)
    plt.xlabel('Games')
    plt.xticks(indices)
    plt.ylabel('Ratings')
    plt.title('Ratings of Teams Over Time')
    plt.legend(loc='upper left')
    plt.show()
