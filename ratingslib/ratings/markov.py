"""
Markov (GeM) Rating System
"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT


import logging
from typing import Any, Dict, Literal, Optional, Set, Union

import numpy as np
import pandas as pd
from ratingslib.ratings.rating import RatingSystem
from ratingslib.utils.enums import ratings
from ratingslib.utils.methods import (create_items_dict, get_indices,
                                      indices_and_points, log_numpy_matrix,
                                      parse_columns)
from ratingslib.utils.validation import (ValidationError, is_number,
                                         validate_from_set,
                                         validate_not_none_and_type)


class Markov(RatingSystem):
    """This class implements the Markov (GeM - Generalized Markov Method)
    rating system.
    GeM was first used by graduate students, Angela Govan [1]_ and
    Luke Ingram [2]_ to successfully rank NFL football and NCAA basketball
    teams respectively.
    The Markov (GeM) method is related to the famous PageRank method [3]_ and
    it uses parts of finite Markov chains and graph theory in order to
    generate ratings of n objects in a finite set.
    Not only sports but also any problem that can be represented as a weighted
    directed graph can be solved using GeM model.

    Parameters
    ----------
    version : str, default=ratings.MARKOV
        A string that shows the version of rating system. The available
        versions can be found in :class:`ratingslib.utils.enums.ratings` class.

    b : float, default=1
        The damping factor. Valid numbers are in the range [0,1]

    stats_markov_dict : Optional[Dict[str, Dict[Any, Any]]], default=None
        A dictionary containing statistics details for the method. For instance
        for soccer teams rating, the following dictionary
        ``stats_markov_dict``::

            stats_markov_dict = {
            'TotalWins': {'VOTE': 10, 'ITEM_I': 'FTHG', 'ITEM_J': 'FTAG',
                           'METHOD': 'VotingWithLosses'},
            'TotalGoals': {'VOTE': 10, 'ITEM_I': 'FTHG', 'ITEM_J': 'FTAG',
                            'METHOD': 'WinnersAndLosersVotePoint'}
            }

        specifies the following details:

         * ``TotalGoals`` and ``TotalWins`` are the names of two statistics

         * ``'VOTE' : 10`` means that the vote is 10. Those votes will be
           converted as weights. The statistics in this example are equally
           weighted

         * ``'ITEM_I': 'FTHG'`` and ``'ITEM_J': 'FTAG'`` are the column names for home
           and away team respectively

         * The key ``'METHOD'`` specifies which method constructs the
           voting matrix. The available methods are:

           #. ``'VotingWithLosses'`` when the losing team casts a
              number of votes equal to the margin of victory in its matchup
              with a stronger opponent.

           #. ``'WinnersAndLosersVotePoint'`` when both the winning and losing
              teams vote with the number of points given up.

           #. ``'LosersVotePointDiff'`` when the losing team cast a number of
              votes

        See also the implementation of the method
            :meth:`create_voting_matrix`


    Attributes
    ----------
    stats : Dict[str, np.ndarray]
        Dictionary that maps voting and stochastic arrays.
        The keys that starts with V map the voting matrices and with S map
        the stochastic matrices

    params : Dict[str, Optional[Dict[str, Dict[Any, Any]]]]
        Dictionary that maps parameters to their values.

    stochastic_matrix : np.ndarray
        A Stochastic Markov  matrix is a square matrix where each entry
        describes the probability that the item will vote for the respective
        item.

    stochastic_matrix_asch : np.ndarray
        A Stochastic Markov matrix that is irreducible

    pi_steady : np.ndarray
        The stationary vector or dominant eigenvector of the
        :attr:`stochastic_matrix`.

    group : Set[str]
        Set of statistics names

    Raises
    ------
    ValueError 
        Value of `b` ∈ [0, 1]


    Examples
    --------
    The following example demonstrate GeM rating system,
    for the 20 first soccer matches that took place during the 2018-2019
    season of English Premier League.

    >>> from ratingslib.datasets.filenames import dataset_path, FILENAME_EPL_2018_2019_20_GAMES
    >>> from ratingslib.ratings.markov import Markov
    >>> filename = dataset_path(FILENAME_EPL_2018_2019_20_GAMES)
    >>> votes = {
            'TW': {
                'VOTE': 10,
                'ITEM_I': 'FTHG',
                'ITEM_J': 'FTAG',
                'METHOD': 'VotingWithLosses'},
            'TG': {
                'VOTE': 10,
                'ITEM_I': 'FTHG',
                'ITEM_J': 'FTAG',
                'METHOD': 'WinnersAndLosersVotePoint'},
            'TST': {
                'VOTE': 10,
                'ITEM_I': 'HST',
                'ITEM_J': 'AST',
                'METHOD': 'WinnersAndLosersVotePoint'},
            'TS': {
                'VOTE': 10,
                'ITEM_I': 'HS',
                'ITEM_J': 'AS',
                'METHOD': 'WinnersAndLosersVotePoint'},
        }
    >>> Markov(b=0.85, stats_markov_dict=votes).rate_from_file(filename)
                    Item    rating  ranking
        0          Arsenal  0.050470       11
        1      Bournemouth  0.039076       15
        2         Brighton  0.051460       10
        3          Burnley  0.071596        2
        4          Cardiff  0.024085       20
        5          Chelsea  0.045033       13
        6   Crystal Palace  0.037678       16
        7          Everton  0.066307        3
        8           Fulham  0.036356       17
        9     Huddersfield  0.032164       19
        10       Leicester  0.055491        7
        11       Liverpool  0.056879        6
        12        Man City  0.048325       12
        13      Man United  0.061052        4
        14       Newcastle  0.035814       18
        15     Southampton  0.051716        9
        16       Tottenham  0.053079        8
        17         Watford  0.082788        1
        18        West Ham  0.041824       14
        19          Wolves  0.058807        5

    References
    ----------
    .. [1] Govan, A. Y. (2008). Ranking Theory with Application to Popular Sports. 
           Ph.D. dissertation, North Carolina State University.

    .. [2] Ingram, L. C. (2007). Ranking NCAA sports teams with Linear algebra.
           Ranking NCAA sports teams with Linear algebra. Charleston

    .. [3] Sergey Brin and Lawrence Page. The Anatomy of a Large-Scale
           Hypertextual Web Search Engine. Computer Networks and
           ISDN Systems, 33:107-17, 1998.

    """

    def __init__(self, *,
                 version=ratings.MARKOV,
                 b: float = 1,
                 stats_markov_dict: Optional[Union[Dict[str, dict],
                                                   Set[str]]] = None):

        if not is_number(b) or b < 0 or b > 1:
            raise ValueError(
                'The value of b must be a number in the range of [0,1]')
        if stats_markov_dict is None:
            raise ValueError('stats_markov_dict must not be None')
        RatingSystem.__init__(self, version=version)
        self.b = b
        self.stats: Dict[str, np.ndarray] = {}  # dictionary with stats array
        self.stats_markov_dict = None
        self.group: Set[str]
        if isinstance(stats_markov_dict, dict):
            Markov.validate_stats_markov_dict(stats_markov_dict)
            self.stats_markov_dict = stats_markov_dict
            self.votes_comb = ''.join(
                str(stat['VOTE'])
                for stat in self.stats_markov_dict.values())
        elif isinstance(stats_markov_dict, set):
            self.group = stats_markov_dict
        else:
            raise TypeError('Wrong type for stats_markov_dict parameter. ' +
                            'Valid types are dict or set')
        self.stochastic_matrix: np.ndarray
        self.stochastic_matrix_asch: np.ndarray
        self.pi_steady: np.ndarray
        self.set_group()
        self.create_ordered_dict(b=self.b)

    def set_group(self):
        """Set the group of statistics."""
        if self.stats_markov_dict is not None:
            group_list = []
            for key, stat in self.stats_markov_dict.items():
                if stat.get('VOTE') != 0:
                    group_list.append(key)
            self.group = set(group_list)

    @staticmethod
    def do_stochastic(voting_matrix: np.ndarray):
        """Normalize the rows of the voting matrix to develop a
        stochastic transition probability matrix.

        Parameters
        ----------
        voting_matrix : List[list]

        Returns
        -------
        stochastic_matrix : numpy.ndarray
            Stochastic matrix built from the corresponding voting
        """
        # voting_matrix_np = np.asarray(voting_matrix)
        n = len(voting_matrix)
        stochastic_matrix = np.zeros((n, n))
        for ii in range(n):
            tmp = sum(voting_matrix[ii])
            for i in range(n):
                if tmp == 0:
                    stochastic_matrix[ii, i] = 1.0 / n
                else:
                    stochastic_matrix[ii, i] = voting_matrix[ii, i] / tmp
        return stochastic_matrix

    @staticmethod
    def compute(stochastic_matrix, b):
        n = len(stochastic_matrix)
        stochastic_matrix_asch = (b * stochastic_matrix
                                  + ((1 - b) / n) * np.ones((n, n)))
        eigenvalues, eigenvectors = np.linalg.eig(
            stochastic_matrix_asch.T)
        # Find index of eigenvalue = 1
        idx = np.argmin(np.abs(eigenvalues - 1))
        w = np.real(eigenvectors[:, idx]).T
        # Normalize eigenvector to get a probability distribution
        pi_steady = w / np.sum(w)
        rating = pi_steady
        return stochastic_matrix_asch, pi_steady, rating

    def computation_phase(self):
        """Compute the stationary vector or dominant
        eigenvector of the transpose of irreducible matrix. Stationary vector
        is the rating vector.
        Note:  irreducible matrix is the :attr:`stochastic_matrix_asch` and
        stationary vector is the :attr:`pi_steady`."""
        self.stochastic_matrix_asch, self.pi_steady, self.rating = \
            Markov.compute(self.stochastic_matrix, self.b)
        if np.any((self.rating < 0)):
            print(np.sum(self.rating), self.rating)
        if np.any((self.rating > 1)):
            print(np.sum(self.rating), self.rating)

    def create_voting_matrix(
        self, *,
        voting_method: Literal['VotingWithLosses',
                               'WinnersAndLosersVotePoint',
                               'LosersVotePointDiff'],
        data_df: pd.DataFrame,
        items_df: pd.DataFrame,
        col_name_home: str, col_name_away: str,
        columns_dict: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Selection of method for developing voting matrix.
        The available methods are:

         #. ``'VotingWithLosses'`` when the losing team casts a
            number of votes equal to the margin of victory in its matchup
            with a stronger opponent.

         #. ``'WinnersAndLosersVotePoint'`` when both the winning and losing
            teams vote with the number of points given up.

         #. ``'LosersVotePointDiff'`` when the losing team cast a number of
            votes.

        """
        col_names = parse_columns(columns_dict)
        home_col_ind, away_col_ind, home_points_col_ind, away_points_col_ind =\
            get_indices(col_names.item_i, col_names.item_j,
                        col_name_home, col_name_away, data=data_df)
        data_np = data_df.to_numpy()
        teams_dict = create_items_dict(items_df)
        n = len(teams_dict)
        voting_array = np.zeros((n, n))
        for row in data_np:
            i, j, points_ij, points_ji = indices_and_points(
                row, teams_dict, home_col_ind, away_col_ind,
                home_points_col_ind, away_points_col_ind)
            if voting_method == 'VotingWithLosses':
                if points_ij > points_ji:
                    voting_array[j][i] += 1
                elif points_ij < points_ji:
                    voting_array[i][j] += 1
                else:
                    voting_array[i][j] += 0.5
                    voting_array[j][i] += 0.5
            elif voting_method == 'WinnersAndLosersVotePoint':
                voting_array[i][j] += points_ji
                voting_array[j][i] += points_ij
            elif voting_method == 'LosersVotePointDiff':
                sump = points_ij - points_ji
                if sump < 0:
                    voting_array[i][j] += (-sump)
                else:
                    voting_array[j][i] += sump
        return voting_array

    def preparation_phase(self, data_df: pd.DataFrame, items_df: pd.DataFrame,
                          columns_dict: Optional[Dict[str, Any]] = None):
        """During preparation phase, voting and stochastic matrices are
        constructed for each statistic according to the method specified
        in the dictionary of attr:`stats_markov_dict`."""

        for stat_name, value in self.stats_markov_dict.items():
            voting_method = value.get('METHOD')
            home_column = value.get('ITEM_I')
            away_column = value.get('ITEM_J')

            self.stats["V" + stat_name] = self.create_voting_matrix(
                voting_method=voting_method,
                data_df=data_df,
                items_df=items_df,
                col_name_home=home_column,
                col_name_away=away_column,
                columns_dict=columns_dict)

            log_numpy_matrix(self.stats["V" + stat_name],
                             "Voting " + stat_name)
            # dictionary key start with S indicates the stochastic matrix
            self.stats["S" + stat_name] = Markov.do_stochastic(
                self.stats["V" + stat_name])

            log_numpy_matrix(self.stats["S" + stat_name],
                             "Stochastic " + stat_name)

    def rate(self, data_df: pd.DataFrame, items_df: pd.DataFrame,
             sort: bool = False,
             columns_dict: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        items_df = items_df.copy(deep=True)
        n = len(items_df)
        s = np.zeros((n, n))
        self.set_group()
        if self.stats_markov_dict is None:
            raise ValueError("stats_markov_dict must not be None")

        self.preparation_phase(data_df, items_df, columns_dict)

        # Sum votes
        total = sum(values['VOTE']
                    for values in self.stats_markov_dict.values())
        # Calculate weights and sum stochastic matrices
        for key, value in self.stats_markov_dict.items():
            logging.info("weight: " + str(value.get('VOTE') / total))
            s += (value.get('VOTE') / total) * self.stats["S" + key]
        # Set stochastic matrix
        self.stochastic_matrix = s
        log_numpy_matrix(self.stochastic_matrix, "Stochastic Matrix")
        # Find stationary vector
        self.computation_phase()
        # Set rating for items
        items_df = self.set_rating(items_df, sort=sort)

        return items_df

    @staticmethod
    def validate_stats_markov_dict(stats_markov_dict: dict):
        available_voting_methods = {'VotingWithLosses',
                                    'WinnersAndLosersVotePoint',
                                    'LosersVotePointDiff'}
        stat_details_keys = {'VOTE', 'ITEM_I', 'ITEM_J', 'METHOD'}
        if not all(isinstance(n, str) for n in stats_markov_dict.keys()):
            raise ValidationError('All keys of stats_markov_dict must be' +
                                  ' string. Check name of statistics.')
        for stat in stats_markov_dict.values():
            for k, v in stat.items():
                validate_from_set(k, stat_details_keys, 'statistic keys')
                if k == 'VOTE':
                    if not is_number(k):
                        ValidationError('VOTE value must be number, ' +
                                        'int or float')
                elif k == 'ITEM_I' or k == 'ITEM_J':
                    validate_not_none_and_type(v, str, k)
                elif k == 'METHOD':
                    validate_from_set(v, available_voting_methods, k)
