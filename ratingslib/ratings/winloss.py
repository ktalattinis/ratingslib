"""
WinLoss Rating System
"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT


from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from ratingslib.ratings.rating import RatingSystem
from ratingslib.utils.enums import ratings
from ratingslib.utils.methods import (create_items_dict, get_indices,
                                      indices_and_points, log_numpy_matrix,
                                      parse_columns)


class Winloss(RatingSystem):
    """The traditional rating method which is popular in the field of sports.
    In the case of sports teams the method takes into account
    the total wins of each team. The first-ranked team is the team
    with the most wins.
    Note that for any kind of items, there are many ways to define the
    notion of a hypothetical matchup and then to determine scores and winners.

    Parameters
    ----------
    version : str, default=ratings.WINLOSS
        A string that shows the version of rating system. The available
        versions can be found in :class:`ratingslib.utils.enums.ratings` class.

    normalization : bool, default = True
        If ``True`` then the result will be normalized according to the total
        times each item occurs in the dataset.
        For example in sport teams set normalization = ``True`` if the teams
        haven't played same number of games. This means that each element of
        W vector is divided by the total number of games played by the
        respective team.

    Attributes
    ----------
    W : numpy.ndarray
        The WinLoss vector for items of shape (n,)
        where n = the total number of items. Each element of vector represents
        the total wins of the respective item.

    Examples
    --------
    The following example demonstrates Winloss rating system,
    for the 20 first soccer matches that took place during the 2018-2019
    season of English Premier League.

    >>> from ratingslib.datasets.filenames import dataset_path, FILENAME_EPL_2018_2019_20_GAMES
    >>> from ratingslib.ratings.winloss import Winloss
    >>> filename = dataset_path(FILENAME_EPL_2018_2019_20_GAMES)
    >>> Winloss(normalization=False).rate_from_file(filename)
                  Item  rating  ranking
    0          Arsenal     0.0        3
    1      Bournemouth     2.0        1
    2         Brighton     1.0        2
    3          Burnley     0.0        3
    4          Cardiff     0.0        3
    5          Chelsea     2.0        1
    6   Crystal Palace     1.0        2
    7          Everton     1.0        2
    8           Fulham     0.0        3
    9     Huddersfield     0.0        3
    10       Leicester     1.0        2
    11       Liverpool     2.0        1
    12        Man City     2.0        1
    13      Man United     1.0        2
    14       Newcastle     0.0        3
    15     Southampton     0.0        3
    16       Tottenham     2.0        1
    17         Watford     2.0        1
    18        West Ham     0.0        3
    19          Wolves     0.0        3
    """

    def __init__(self, version=ratings.WINLOSS, normalization=True):
        super().__init__(version)
        self.normalization = normalization
        self.create_ordered_dict(normalization=self.normalization)
        self.W: np.ndarray

    def computation_phase(self):
        """All the calculations are made in
        :meth:`ratingslib.ratings.winloss.Winloss.create_win_loss_vector` method.
        Winloss vector is the rating vector.
        """
        self.rating = self.W

    def create_win_loss_vector(self, data_df: pd.DataFrame,
                               items_df: pd.DataFrame,
                               columns_dict: Optional[Dict[str, Any]] = None
                               ) -> np.ndarray:
        """Construction of WinLoss vector."""
        col_names = parse_columns(columns_dict)
        home_col_ind, away_col_ind, home_points_col_ind, away_points_col_ind =\
            get_indices(col_names.item_i, col_names.item_j,
                        col_names.points_i, col_names.points_j, data=data_df)
        data_np = data_df.to_numpy()
        teams_dict = create_items_dict(items_df)
        n = len(teams_dict)
        W = np.zeros(n)
        ng = np.zeros(n)  # number of games
        for row in data_np:
            i, j, points_ij, points_ji = indices_and_points(
                row, teams_dict, home_col_ind, away_col_ind,
                home_points_col_ind, away_points_col_ind)
            if points_ij > points_ji:
                W[i] += 1
            elif points_ij < points_ji:
                W[j] += 1
            ng[i] += 1
            ng[j] += 1
        if self.normalization:
            ng[ng == 0] = 1
            W /= ng
        return W

    def preparation_phase(self, data_df: pd.DataFrame, items_df: pd.DataFrame,
                          columns_dict: Optional[Dict[str, Any]] = None):
        self.W = self.create_win_loss_vector(data_df,
                                             items_df,
                                             columns_dict=columns_dict)

    def rate(self, data_df: pd.DataFrame, items_df: pd.DataFrame,
             sort: bool = False,
             columns_dict: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        items_df = items_df.copy(deep=True)
        self.preparation_phase(data_df, items_df, columns_dict)
        self.computation_phase()
        log_numpy_matrix(self.W, 'W')
        items_df = self.set_rating(items_df, sort=sort)
        return items_df
