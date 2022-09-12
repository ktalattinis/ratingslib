"""
AccuRate Rating System
"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from ratingslib.ratings.rating import RatingSystem
from ratingslib.utils.enums import ratings
from ratingslib.utils.logmsg import set_logger
from ratingslib.utils.methods import (create_items_dict, get_indices,
                                      indices_and_points, parse_columns, points,
                                      print_info, print_pandas)


class AccuRate(RatingSystem):
    """
    This class implements the :class:`ratingslib.ratings.RatingSystem` abstract
    class using an approach called AccuRate for the computation of
    rating values as described in the paper [1]_


    Parameters
    ----------
    version : str, default=ratings.ACCURATE
        A string that shows the version of rating system. The available
        versions can be found in :class:`ratingslib.utils.enums.ratings` class.

    starting_point : int
        the value where the initial rating starts

    References
    ----------
    .. [1] Kyriakides, G., Talattinis, K., & Stephanides, G. (2017).
           A Hybrid Approach to Predicting Sports Results and an AccuRATE Rating System.
           International Journal of Applied and Computational Mathematics, 3(1), 239–254.

    Examples
    --------
    The following example demonstrates Accurate rating system
    for a simple soccer competition where only two teams participate,
    team “Good” and team “Better”.

    >>> from ratingslib.datasets.filenames import dataset_path, FILENAME_ACCURATE_PAPER_EXAMPLE
    >>> from ratingslib.ratings.accurate import AccuRate
    >>> filename = dataset_path(FILENAME_ACCURATE_PAPER_EXAMPLE)
    >>> AccuRate().rate_from_file(filename)
         Team    rating  ranking
    0  Better  1.681793        1
    1    Good -1.587401        2
    """

    def __init__(self, version: str = ratings.ACCURATE,
                 starting_point: float = 0):
        super().__init__(version)
        self.create_ordered_dict()
        self.starting_point = float(starting_point)

    def create_rating_vector(self, data: pd.DataFrame, items_df: pd.DataFrame,
                             columns_dict: Optional[Dict[str, Any]] = None
                             ) -> np.ndarray:
        """Calculates ratings according to pairs of items data."""
        col_names = parse_columns(columns_dict)
        home_col_ind, away_col_ind, home_points_col_ind, away_points_col_ind,\
            home_shots_col_ind, away_shots_col_ind,\
            home_shotstarget_col_index,\
            away_shotstarget_col_index = get_indices(
                col_names.item_i, col_names.item_j,
                col_names.points_i, col_names.points_j, col_names.ts_i,
                col_names.ts_j, col_names.tst_i, col_names.tst_j,
                data=data)
        data_np = data.to_numpy()
        teams_dict = create_items_dict(items_df)
        rating_old = np.array([float(self.starting_point)
                               for _ in range(len(items_df))])
        rating_new = np.array([float(self.starting_point)
                               for _ in range(len(items_df))])
        for row in data_np:
            i, j, points_ij, points_ji = indices_and_points(
                row, teams_dict, home_col_ind, away_col_ind,
                home_points_col_ind, away_points_col_ind)
            ts_ij, ts_ji = points(row,
                                  home_shots_col_ind, away_shots_col_ind)
            tst_ij, tst_ji = points(row,
                                    home_shotstarget_col_index,
                                    away_shotstarget_col_index)
            S_ij = points_ij
            S_ji = points_ji
            d_i = abs(S_ij - S_ji)
            d_j = d_i
            if tst_ij == 0:
                k_i = 0.0
            else:
                k_i = tst_ij / ts_ij
            if tst_ji == 0:
                k_j = 0.0
            else:
                k_j = tst_ji / ts_ji
            if S_ij > S_ji:  # team i is the winner
                rating_new[i] = rating_old[i] + (pow(d_i, k_i))
                rating_new[j] = rating_old[j] - (pow(d_j, (1-k_j)))
            elif S_ij < S_ji:  # team j is the winner
                rating_new[i] = rating_old[i] - (pow(d_i, (1-k_i)))
                rating_new[j] = rating_old[j] + (pow(d_j, k_j))
            else:
                rating_new[i] = rating_old[i]
                rating_new[j] = rating_old[j]
            rating_old[i] = rating_new[i]
            rating_old[j] = rating_new[j]
        return rating_new

    def computation_phase(self):
        """Nothing to compute, all the calculations are made in
        :meth:`ratingslib.ratings.accurate.AccuRate.create_rating_vector` method
        """

    def preparation_phase(self, data_df: pd.DataFrame, items_df: pd.DataFrame,
                          columns_dict: Optional[Dict[str, Any]] = None):
        self.rating = self.create_rating_vector(data_df, items_df,
                                                columns_dict=columns_dict)

    def rate(self, data_df: pd.DataFrame, items_df: pd.DataFrame,
             sort: bool = False,
             columns_dict: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        items_df = items_df.copy(deep=True)
        self.rating = self.create_rating_vector(data_df, items_df,
                                                columns_dict=columns_dict)
        self.preparation_phase(data_df, items_df, columns_dict)
        items_df = self.set_rating(items_df, sort=sort)
        return items_df
