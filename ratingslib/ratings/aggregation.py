"""
Module that implements rating and ranking aggregation methods
"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from ratingslib.ratings.keener import Keener
from ratingslib.ratings.markov import Markov
from ratingslib.ratings.rating import RatingSystem
from ratingslib.utils.enums import ratings
from ratingslib.utils.methods import (create_items_dict,
                                      parse_columns)
from ratingslib.utils.validation import validate_not_none_and_type, validate_type


# RATING AGGREGATION

class RatingAggregation(RatingSystem):
    """
    Class for Rating aggregation

    Parameters
    ----------
    version : str, default=ratings.AGGREGATIONMARKOV
        A string that shows the version of rating system. The available
        versions can be found in :class:`ratingslib.utils.enums.ratings` class.

    votes_or_weights : Optional[Dict[str, float]]
            Votes or weigths for matrices

    b : float, optional
        Valid if aggregation method = ratings.AGGREGATIONMARKOV, by default 0.9


    """

    def __init__(self, version=ratings.AGGREGATIONMARKOV,
                 votes_or_weights: Optional[Dict[str, float]] = None,
                 b: float = 0.9):
        super().__init__(version)
        self.votes_or_weights = votes_or_weights
        self.b = b

    def calc_rating_distances(self,
                              data_df: pd.DataFrame,
                              rating_column_name: str) -> np.ndarray:
        """Calculate and create pairwise matrix by taking into account
        the rating differences (as distances)

        Parameters
        ----------
        data_df : pd.DataFrame
            dataset of ratings

        rating_column_name : str
            which is the rating column of dataset

        Returns
        -------
        matrix : np.ndarray
            rating distances matrix
        """
        validate_type(data_df, pd.DataFrame, 'data_df')
        validate_type(rating_column_name, str, 'column_name')
        x = data_df[rating_column_name].values
        n = len(x)
        mat = np.zeros((n, n))
        for i, row_i in enumerate(x):
            for j, row_j in enumerate(x):
                sum_ratings = row_i - row_j
                # print(i, j, row_i, row_j, sum_ratings)
                if sum_ratings <= 0:
                    mat[i][j] = 0
                else:
                    mat[i][j] = sum_ratings
        mat = mat / mat.sum()
        # print(mat)
        return mat

    def calc_dict_rating_distances(
            self,
            data_df: pd.DataFrame,
            rating_columns: List[str]) -> Dict[str, np.ndarray]:
        """Calculate and create dictionary of pairwise matrices by
        taking into account the rating differences (as distances).
        Each column represents the rating method name.


        Parameters
        ----------
        data_df : pd.DataFrame
            dataset of ratings

        rating_columns : List[str]
            list of columns that refers to ratings

        Returns
        -------
        matrices_dict : Dict[str, np.ndarray]
            dictionary that maps column to rating distance matrix
        """
        validate_type(data_df, pd.DataFrame, 'data_df')
        validate_type(rating_columns, list, 'columns')
        matrices_dict: Dict[str, np.ndarray] = {}
        for col in rating_columns:
            matrices_dict[col] = self.calc_rating_distances(
                data_df, col)
        return matrices_dict

    @staticmethod
    def rating_aggregation(matrices_dict: Dict[str, np.ndarray],
                           votes_or_weights: Optional[Dict[str, float]] = None,
                           aggregation_method: str = ratings.AGGREGATIONMARKOV,
                           b: float = 0.9) -> np.ndarray:
        """Rating aggregation from rating lists

        Parameters
        ----------
        matrices_dict : Dict[str, np.ndarray]
            Dictionary that maps name to rating distance matrix

        votes_or_weights : Optional[Dict[str, float]]
            Votes or weigths for matrices

        aggregation_method : str, default=ratings.AGGREGATIONMARKOV
            Name of aggregation method

        b : float, optional
            Valid if aggregation method = ratings.AGGREGATIONMARKOV, by default 0.9

        Returns
        -------
        rating : numpy.ndarray
            Aggregated rating vector

        Raises
        ------
        ValueError
            If matrices_dict and votes_or_weights parameters
            don't have the same size
        """
        # np.set_printoptions(precision=5, suppress=True)
        validate_not_none_and_type(matrices_dict, dict, "matrices_dict")
        if votes_or_weights is not None:
            if len(matrices_dict) != len(votes_or_weights):
                raise ValueError(
                    'matrices_dict and votes_or_weights parameters ' +
                    'must have the same size')
        else:
            votes_or_weights = {}
            for k, v in matrices_dict.items():
                votes_or_weights[k] = 10
        total = sum(votes_or_weights.values())
        n = len(next(iter(matrices_dict.values())))  # get the first matrix
        raverage = np.zeros((n, n))
        for k, v in matrices_dict.items():
            raverage += (votes_or_weights[k] / total) * v

        if aggregation_method == ratings.AGGREGATIONMARKOV:
            raverage = Markov.do_stochastic(raverage.T)
            _, _, rating = Markov.compute(raverage, b)
        elif aggregation_method == ratings.AGGREGATIONOD:
            o = raverage @ np.ones(n)
            d = np.ones(n).T @ raverage
            rating = o / d
        elif aggregation_method == ratings.AGGREGATIONPERRON:
            e = np.ones(n)
            epsilon = sys.float_info.epsilon
            val = epsilon * (e @ e.T)
            A = raverage + val
            rating = Keener.compute(A)

        # print(rating)
        return rating

    def computation_phase(self):
        self.rating = RatingAggregation.rating_aggregation(
            matrices_dict=self.matrices_dict,
            aggregation_method=self.version,
            votes_or_weights=self.votes_or_weights,
            b=self.b
        )

    def preparation_phase(self, data_df: pd.DataFrame, items_df: pd.DataFrame,
                          columns_dict: Optional[Dict[str, Any]] = None):
        col_names = parse_columns(columns_dict)
        self.matrices_dict = self.calc_dict_rating_distances(
            data_df, col_names.ratings)

    def rate(self, data_df: pd.DataFrame, items_df: pd.DataFrame,
             sort: bool = False,
             columns_dict: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        items_df = items_df.copy(deep=True)
        items_dict = create_items_dict(items_df)
        self.preparation_phase(data_df, items_df, columns_dict=columns_dict)
        self.computation_phase()
        items_df = self.set_rating(items_df, sort=sort)
        return items_df


# RANKING AGGREGATION

class RankingAggregation(RatingSystem):
    """
    Class for Ranking Aggregation

    Parameters
    ----------
    version : str, default=ratings.RANKINGAVG
        A string that shows the version of rating system. The available
        versions can be found in :class:`ratingslib.utils.enums.ratings` class.


    """

    def __init__(self, version=ratings.RANKINGAVG):
        super().__init__(version)

    @staticmethod
    def ranking_aggregation(data_df: pd.DataFrame,
                            rating_columns: List[str],
                            aggregation_method: str = ratings.RANKINGAVG
                            ) -> np.ndarray:
        """Ranking aggregation from ranking lists

        Parameters
        ----------
        data_df : pd.DataFrame
            Rating values are the columns of DataFrame

        aggregation_method : str, default=ratings.AGGREGATIONAVG
            Name of aggregation method

        Returns
        -------
        rating : numpy.ndarray
            Aggregated rating vector

        Raises
        ------
        ValueError
            If matrices_dict and votes_or_weights parameters
            don't have the same size
        """
        ranking_columns = []
        if aggregation_method == ratings.RANKINGAVG:
            rating_lower_best = True
            len_items = len(data_df.index)+1
        else:
            rating_lower_best = False
            len_items = len(data_df.index)
        for rc in rating_columns:
            rank_column = rc+"_rank"
            data_df[rank_column] = -data_df[rc].rank(
                method='dense', ascending=rating_lower_best).astype(int) + len_items
            ranking_columns.append(rc+"_rank")
        data = data_df[ranking_columns].values
        if aggregation_method == ratings.RANKINGBORDA:
            ratings_val = np.sum(data, axis=1)
        elif aggregation_method == ratings.RANKINGAVG:
            ratings_val = np.mean(data, axis=1)
        return ratings_val

    def computation_phase(self):
        """All the calculations are made in
        :meth:`ratingslib.ratings.aggregations.RankingAggregation.ranking_aggregation` method.
        """

    def preparation_phase(self, data_df: pd.DataFrame, items_df: pd.DataFrame,
                          columns_dict: Optional[Dict[str, Any]] = None):
        col_names = parse_columns(columns_dict)
        self.rating = RankingAggregation.ranking_aggregation(
            data_df, col_names.ratings, self.version)

    def rate(self, data_df: pd.DataFrame, items_df: pd.DataFrame,
             sort: bool = False,
             columns_dict: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        items_df = items_df.copy(deep=True)
        items_dict = create_items_dict(items_df)
        self.preparation_phase(data_df, items_df, columns_dict=columns_dict)
        self.computation_phase()
        rating_lower_best = True if self.version == ratings.RANKINGAVG else False
        items_df = self.set_rating(
            items_df, rating_lower_best=rating_lower_best, sort=sort)
        return items_df
