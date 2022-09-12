"""
Colley rating system
"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT


from ratingslib.ratings.rating import RatingSystem
from ratingslib.utils.enums import ratings
from ratingslib.utils.methods import parse_columns, get_indices,\
    indices_and_points, create_items_dict, log_numpy_matrix
import numpy as np
import pandas as pd
from typing import Any, Optional, Dict, Tuple


class Colley(RatingSystem):
    """This class implements the Colley rating system.
    This system was proposed by astrophysicist Dr. Wesley Colley in 2001 for
    ranking sports teams. Colley’s method [1]_ makes use of an idea from
    probability theory, known as Laplace’s ‘‘rule of succession’’.
    In fact, it is a modified form of the win-loss method, which uses the
    percentage of wins of each team.

    Parameters
    ----------
    version : str, default=ratings.COLLEY
        A string that shows the version of rating system. The available
        versions can be found in :class:`ratingslib.utils.enums.ratings` class.

    Attributes
    ----------
    C : numpy.ndarray
        The Colley matrix of shape (n,n) where n = the total number of
        items.

    b : numpy.ndarray
        The right-hand side vector ``b`` of shape (n,)
        where n = the total number of items.

    References
    ----------
    .. [1] Colley, W. (2002). Colley’s bias free college football ranking method: The Colley Matrix Explained.

    Examples
    --------
    The following example demonstrates Colley rating system,
    for the 20 first soccer matches that took place during the 2018-2019
    season of English Premier League.

    >>> from ratingslib.datasets.filenames import dataset_path, FILENAME_EPL_2018_2019_20_GAMES
    >>> from ratingslib.ratings.colley import Colley
    >>> filename = dataset_path(FILENAME_EPL_2018_2019_20_GAMES)
    >>> Colley().rate_from_file(filename)
                  Item    rating  ranking
    0          Arsenal  0.333333       16
    1      Bournemouth  0.686012        3
    2         Brighton  0.562500        6
    3          Burnley  0.401786       10
    4          Cardiff  0.394345       11
    5          Chelsea  0.666667        5
    6   Crystal Palace  0.501488        8
    7          Everton  0.562500        6
    8           Fulham  0.293155       17
    9     Huddersfield  0.333333       16
    10       Leicester  0.473214        9
    11       Liverpool  0.712798        2
    12        Man City  0.666667        5
    13      Man United  0.508929        7
    14       Newcastle  0.391369       12
    15     Southampton  0.366071       14
    16       Tottenham  0.671131        4
    17         Watford  0.741071        1
    18        West Ham  0.349702       15
    19          Wolves  0.383929       13
    """

    def __init__(self, version=ratings.COLLEY):
        super().__init__(version)
        self.create_ordered_dict()
        self.C: np.ndarray
        self.b: np.ndarray

    def computation_phase(self):
        """Solve the system Cr=b to obtain the Colley rating vector r."""
        self.rating = np.linalg.solve(self.C, self.b)

    def create_colley_matrix(self, data_df: pd.DataFrame,
                             items_df: pd.DataFrame,
                             columns_dict: Optional[Dict[str, Any]] = None
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """Construction of Colley coefficient matrix ``C`` and right-hand
        side vector ``b``."""
        col_names = parse_columns(columns_dict)
        home_col_ind, away_col_ind, home_points_col_ind, away_points_col_ind =\
            get_indices(col_names.item_i,
                        col_names.item_j, col_names.points_i, col_names.points_j,
                        data=data_df)
        data_np = data_df.to_numpy()
        items_dict = create_items_dict(items_df)
        n = len(items_dict)
        C = np.zeros((n, n))
        b = np.zeros(n)
        games = np.zeros(n)
        TW = np.zeros(n)  # counts the total wins for each team
        TL = np.zeros(n)  # counts the total losses for each team
        for row in data_np:
            i, j, points_ij, points_ji = indices_and_points(
                row, items_dict, home_col_ind, away_col_ind,
                home_points_col_ind, away_points_col_ind)
            if points_ij > points_ji:
                TW[i] += 1
                TL[j] += 1
            elif points_ij < points_ji:
                TL[i] += 1
                TW[j] += 1
            # in the case of tie we just decrease C[i][j], C[j][i] and increase
            # games[i], games[j]
            C[i][j] -= 1
            C[j][i] -= 1
            games[i] += 1
            games[j] += 1
        for i in range(n):
            C[i][i] = 2 + games[i]
            b[i] = 1 + 0.5 * (TW[i] - TL[i])
        return C, b

    def preparation_phase(self, data_df: pd.DataFrame, items_df: pd.DataFrame,
                          columns_dict: Optional[Dict[str, Any]] = None):
        self.C, self.b = self.create_colley_matrix(data_df, items_df,
                                                   columns_dict=columns_dict)
        log_numpy_matrix(self.C, 'C')
        log_numpy_matrix(self.b, 'b')

    def rate(self, data_df: pd.DataFrame, items_df: pd.DataFrame,
             sort: bool = False,
             columns_dict: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        items_df = items_df.copy(deep=True)
        self.preparation_phase(data_df, items_df, columns_dict)
        self.computation_phase()
        items_df = self.set_rating(items_df, sort=sort)
        return items_df
