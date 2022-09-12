"""
Massey Rating System
"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT


import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from ratingslib.ratings.rating import RatingSystem
from ratingslib.utils.enums import ratings
from ratingslib.utils.methods import (create_items_dict, get_indices,
                                      indices_and_points, log_numpy_matrix,
                                      parse_columns)
from ratingslib.utils.validation import validate_type
from numpy import linalg


class Massey(RatingSystem):
    """This method was proposed by Kenneth Massey in 1997 for ranking college
    football teams [1]_.
    The Massey method apart from numbers of wins and losses, it also considers
    the point score data to rate items via a system of linear equations.
    It uses a linear least squares regression to solve a system of
    linear equations.
    Note that point score data depends on the application, for instance in
    soccer teams the points are the number of goals of each team.

    Parameters
    ----------
    version : str, default=ratings.MASSEY
        A string that shows the version of rating system. The available
        versions can be found in :class:`ratingslib.utils.enums.ratings` class.

    data_limit : int, default=0
        The parameter data_limit specifies the minimum number of observations
        in the dataset. Default is set ``0`` and indicates no limit.

    Attributes
    ----------
    Madj : numpy.ndarray
        The adjusted Massey matrix. The last row of this matrix is replaced
        with vector of all ones.

    d_adj : numpy.ndarray
        The adjusted point differentials vector.
        The last item of this vector is replaced zero.

    References
    ----------
    .. [1] Massey, K. (1997). Statistical models applied to the rating of sports teams. 
           Statistical models applied to the rating of sports teams.

    Examples
    --------
    The following example demonstrates Massey rating system,
    for the 20 first soccer matches that took place during the 2018-2019
    season of English Premier League.

    >>> from ratingslib.datasets.filenames import dataset_path, FILENAME_EPL_2018_2019_20_GAMES
    >>> from ratingslib.ratings.massey import Massey
    >>> filename = dataset_path(FILENAME_EPL_2018_2019_20_GAMES)
    >>> Massey().rate_from_file(filename)
                  Item        rating  ranking
    0          Arsenal  2.500000e+00       11
    1      Bournemouth  4.781250e+00        3
    2         Brighton -4.781250e+00       14
    3          Burnley -6.031250e+00       17
    4          Cardiff  3.031250e+00        9
    5          Chelsea  3.250000e+00        8
    6   Crystal Palace  5.031250e+00        2
    7          Everton -6.281250e+00       18
    8           Fulham  2.781250e+00       10
    9     Huddersfield  2.220446e-15       12
    10       Leicester -5.531250e+00       16
    11       Liverpool  7.281250e+00        1
    12        Man City  4.750000e+00        4
    13      Man United -5.156250e+00       15
    14       Newcastle  3.281250e+00        7
    15     Southampton -6.656250e+00       19
    16       Tottenham  4.531250e+00        5
    17         Watford -3.406250e+00       13
    18        West Ham  3.531250e+00        6
    19          Wolves -6.906250e+00       20
    """

    def __init__(self, version=ratings.MASSEY, data_limit=0):
        validate_type(data_limit, int, 'data_limit')
        if data_limit < 0:
            raise ValueError('data_limit should not be negative')
        super().__init__(version)
        self.data_limit = data_limit
        self.create_ordered_dict(data_limit=self.data_limit)
        self.Madj: np.ndarray
        self.d_adj: np.ndarray

    def computation_phase(self):
        try:
            self.rating = linalg.solve(self.Madj, self.d_adj)
        except linalg.LinAlgError:
            warnings.warn(
                "Singular matrix in Massey, all ratings will be set to 0")
            self.rating = np.zeros(len(self.d_adj))

    def create_massey_matrix(self, data_df: pd.DataFrame,
                             items_df: pd.DataFrame,
                             columns_dict: Optional[Dict[str, Any]] = None
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """Construction of adjusted Massey matrix (``M_adj``) and adjusted
        point differential vector (``d_adj``)"""
        col_names = parse_columns(columns_dict)
        home_col_ind, away_col_ind, home_points_col_ind, away_points_col_ind =\
            get_indices(col_names.item_i, col_names.item_j,
                        col_names.points_i, col_names.points_j, data=data_df)
        data_np = data_df.to_numpy()
        teams_dict = create_items_dict(items_df)

        n = len(teams_dict)
        Madj = np.zeros((n, n))
        d_adj = np.zeros(n)
        games = np.zeros(n)

        for row in data_np:
            i, j, points_ij, points_ji = indices_and_points(
                row, teams_dict, home_col_ind, away_col_ind,
                home_points_col_ind, away_points_col_ind)
            d_adj[i] += points_ij - points_ji
            games[i] += 1
            d_adj[j] += points_ji - points_ij
            games[j] += 1

            Madj[i][j] -= 1
            Madj[j][i] -= 1

        for i in range(n):
            Madj[i][i] = games[i]

        Madj[n - 1] = [1 for i in range(n)]
        d_adj[n - 1] = 0

        return Madj, d_adj

    def preparation_phase(self, data_df: pd.DataFrame, items_df: pd.DataFrame,
                          columns_dict: Optional[Dict[str, Any]] = None):
        self.Madj, self.d_adj = self.create_massey_matrix(
            data_df, items_df, columns_dict=columns_dict)

        np.set_printoptions(precision=2, linewidth=1000, threshold=np.inf)
        log_numpy_matrix(self.Madj, 'Massey adjusted')
        log_numpy_matrix(self.d_adj, 'd adjusted')

    def rate(self, data_df: pd.DataFrame, items_df: pd.DataFrame,
             sort: bool = False,
             columns_dict: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        items_df = items_df.copy(deep=True)
        if len(data_df.index) > self.data_limit:
            self.preparation_phase(data_df, items_df, columns_dict)
            self.computation_phase()
        else:
            items_dict = create_items_dict(items_df)
            self.rating = np.zeros(len(items_dict))  # [0.0] * len(items_dict)
        items_df = self.set_rating(items_df, sort=sort)
        return items_df
