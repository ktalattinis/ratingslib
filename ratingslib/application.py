"""
Module for the outcome of application
"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

import math
from abc import ABC, abstractmethod
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ratingslib.utils.methods import get_indices, parse_columns, points


class Outcome(ABC):
    """A base class for the outcome of application"""

    def __init__(self, name: str, outcomes_list: list, columns_dict=None):
        super(Outcome, self).__init__()
        self.col_names = parse_columns(columns_dict)
        self.name = name
        self.outcomes_list = outcomes_list
        self.outcomes_dict = self.get_outcomes()
        self.outcomes_values_list = list(self.outcomes_dict.values())

    def get_outcomes(self) -> dict:
        """Creates and returns a dictionary with outcomes.
        The keys are the possible outcomes that defined by the property of
        ``outcomes_list`` and the values of the dictionary are
        positive integers. For example for three possible outcomes then the
        values are 1,2,3. """
        outcomes_dict = {o: i + 1 for i, o in enumerate(self.outcomes_list)}
        return outcomes_dict

    @abstractmethod
    def set_col_indices(self, pairs_data_df: pd.DataFrame):
        """To be overridden in subclasses."""
        raise NotImplementedError(self.__class__.__name__ + " is abstract")


class SportOutcome(Outcome):
    """An abstract class for Outcome. The term of outcome is related with
    application type. For example in sports like soccer for every matchup
    between two teams there are three possible final outcomes:
    Home Win, Away Win and Draw.
    In other applications like a backgammon game, there are two possible
    outcomes. Another example with two possible outcomes is the under 2.5 goals
    or over 2.5 goals in soccer. Under 2.5 means that the total number of
    goals scored during the match will be under or 2 and over means 3 or above.
    Also in soccer if the outcome is defined by the final score this
    means that there are many possible results e.g. (0-0, 0-1, 1-0, 1-1, etc.).

    Extend this class and override abstract methods to define the application
    outcomes.

    Parameters
    ----------
    name : str
        The name of outcome. In soccer possible outcomes names are
        'FT' (from Full Time) or 'FTR' (from Full Time Result).

    outcomes_list : list
        The list of outcome names.

    columns_dict : dict, default=None
        A dictionary mapping the column names of the dataset.
        See the module :mod:`ratingslib.datasets.parameters` for more details.

    Attributes
    ----------
    outcomes_dict : dict
        Outcome names mapped to their values

    outcomes_values_list : list
        list of outcome values
    """

    @abstractmethod
    def outcome_value(self, row):
        """To be overridden in subclasses."""
        raise NotImplementedError(self.__class__.__name__ + " is abstract")

    @abstractmethod
    def fit_and_predict(self, train_X, train_Y, test_X, method: Literal['MLE', 'RANK'] = 'RANK') -> tuple:
        """To be overridden in subclasses."""
        raise NotImplementedError(self.__class__.__name__ + " is abstract")


class SoccerOutcome(SportOutcome):
    """
    In soccer the final outcome during a matchup between two teams there
    are three possible outcomes:

        - H : Home Win     (1)
        - A : Away Win     (2)
        - D : Draw         (3)

    Attributes
    ----------
    name : str
        Name of outcome. In soccer possible outcome names can be set to
        'FT' (from Full Time) or 'FTR' (from Full Time Result)

    columns_odds_dict : dict
        dictionary mapping the odd columns from dataset
            - dictionary keys:
                - `H` for home odds,
                - `D` for draw odds,
                - `A` for away odds
            - dictionary values:
                list of column names of betting odds from selected bookmakers

    home_points_col_index : int
        the index number of column for the goals of home team

    away_points_col_index : int
        the index number of column for the goals of away team

    Notes
    -----
    The odds of the final outcome are offered from the following bookmaker
    companies:

        - Bet365
        - Bet&Win
        - Interwetten

    """
    outcomes_list = ["H", "A", "D"]

    def __init__(self, columns_dict=None):
        name = 'FT'
        super().__init__(name, self.outcomes_list, columns_dict)
        for k, v in self.outcomes_dict.items():
            setattr(self, k, v)
        self.columns_odds_dict = {
            'H': [self.col_names.B365H,
                  self.col_names.BWH,
                  self.col_names.IWH],
            'A': [self.col_names.B365A,
                  self.col_names.BWA,
                  self.col_names.IWA],
            'D': [self.col_names.B365D,
                  self.col_names.BWD,
                  self.col_names.IWD]
        }
        self.home_points_col_index = None
        self.away_points_col_index = None

    def set_col_indices(self, pairs_data_df: pd.DataFrame):
        """Set the indices for home and away goals

        Parameters
        ----------
        pairs_data_df : pd.DataFrame
            the games data of teams
        """
        self.home_points_col_index, self.away_points_col_index = get_indices(
            self.col_names.points_i, self.col_names.points_j, data=pairs_data_df)

    def outcome_value(self, row) -> int:
        """This method returns the outcome by given match. The decision is
        based on the goals scored by each team.

        Returns
        -------
        int
            integer value of outcome (1 = 'H', 2 = 'A', 3 = 'D')
        """
        points_ij, points_ji = points(
            row, self.home_points_col_index, self.away_points_col_index)
        if points_ij > points_ji:
            return self.H
        elif points_ij < points_ji:
            return self.A
        else:
            return self.D

    @staticmethod
    def _fit(train_X: np.ndarray, train_Y: np.ndarray, outcomes_dict: dict):
        """Find a, h parameters"""
        def mle(parameters):
            a, h = parameters
            total = 0
            for i, row in enumerate(train_X):
                rating_home = row[0]
                rating_away = row[1]
                d = math.exp(a * (rating_home-rating_away) + h)
                result = train_Y[i]  # [0]
                if result == outcomes_dict['H']:
                    total += np.log((d)/(1+d+d**0.5))
                elif result == outcomes_dict['A']:
                    total += np.log(1/(1+d+d**0.5))
                elif result == outcomes_dict['D']:
                    total += np.log((d**0.5)/(1+d+d**0.5))

            neg_LL = -1*total
            # print(neg_LL)
            return neg_LL
        bounds = [(0, None), (0, None)]
        # , method='L-BFGS-B')
        mle_model = minimize(mle, [1, 1], bounds=bounds)
        # print(mle_model)
        a = mle_model.x[0]
        h = mle_model.x[1]
        # print(a, h)
        return a, h

    @staticmethod
    def _fit_and_predict(train_X: np.ndarray,
                         train_Y: np.ndarray,
                         test_X: np.ndarray,
                         outcomes_dict: dict,
                         method: Literal['MLE', 'RANK'] = 'RANK'):
        if method == 'MLE' and len(train_X) != 0:
            a, h = SoccerOutcome._fit(
                train_X, train_Y, outcomes_dict)
        predictions = []
        predictions_prob = []
        for index in range(len(test_X)):
            rating_home = test_X[index][0]
            rating_away = test_X[index][1]
            if method == 'RANK':
                result, rating_prob = SoccerOutcome.predict_from_ratings_logic(
                    rating_home, rating_away, outcomes_dict)
            elif method == 'MLE':
                if len(train_X) != 0:
                    result, rating_prob = SoccerOutcome.predict_from_ratings_mle(
                        rating_home, rating_away, a, h, outcomes_dict)
                else:
                    break

            predictions.append(result)
            predictions_prob.append(rating_prob)
        return predictions, predictions_prob

    def fit_and_predict(self, train_X, train_Y, test_X, method: Literal['MLE', 'RANK'] = 'RANK'):
        """Fit and predict the final outcome

        Parameters
        ----------
        train_X : numpy.ndarray
            The training set that includes only the features

        train_Y : numpy.ndarray
            The outcome labels of training set

        test_X : numpy.ndarray
            The outcome labels of test set

        method : Literal['MLE', 'RANK'], default='RANK'
            Two available methods for predictions: 'RANK' or 'MLE'

        Returns
        -------
        tuple
            The predictions for the target outcome and the
            predictions distribution
        """
        return SoccerOutcome._fit_and_predict(train_X, train_Y, test_X, self.outcomes_dict, method)

    @staticmethod
    def predict_from_ratings_mle(rating_home: float,
                                 rating_away: float,
                                 a: float,
                                 h: float,
                                 outcomes_dict: dict) -> Tuple[int, list]:
        """The ratings can be turned into predictions by selecting the highest
        probability from outcomes.
        The probabilities can be computed by applying the modified logistic
        function proposed by [1]_

        Parameters
        ----------
        rating_home : float
            the rating value of home-team

        rating_away : float
            the rating value of away-team

        a : float
            parameter in probability equation

        h : float
            home advantage parameter

        Returns
        -------
        result : int
            The value of predicted result

        rating_prob : list
            Prediction probabilities for each outcome

        References
        ----------
        .. [1] Lasek, J., Szlávik, Z., & Bhulai, S. (2013).
               The predictive power of ranking systems in association football. 
               International Journal of Applied Pattern Recognition, 1, 27–46.

        """
        d = math.exp(a * (rating_home-rating_away) + h)
        d_sqrt = d ** 0.5
        prob_h = (d/(1+d+d_sqrt))
        prob_a = (1/(1+d+d_sqrt))
        prob_d = (d_sqrt)/(1+d+d_sqrt)
        # print(prob_h, prob_a, prob_d)
        rating_prob = [prob_h, prob_a, prob_d]
        max_prob = max(rating_prob)
        if max_prob == prob_h:
            result = outcomes_dict['H']
        elif max_prob == prob_a:
            result = outcomes_dict['A']
        elif max_prob == prob_d:
            result = outcomes_dict['D']
        # print(sum(rating_prob))
        return result, rating_prob

    @staticmethod
    def predict_from_ratings_logic(rating_home: float, rating_away: float, outcomes_dict: dict,
                                   calc_prob: bool = True) -> Tuple[int, list]:
        """Prediction of the final outcome in a game between two teams is based
        on their ratings. The logic of prediction is that a higher rating is
        preferred over the lower rating.
        For example in a match between teamA and teamB
        with ratingA and ratingB respectively,

            * if ratingA > ratingB then prediction of the winner is teamA
            * if ratingA < ratingB then prediction of the winner is teamB
            * if ratingA = ratingB then prediction is the Draw.


        Parameters
        ----------
        rating_home : float
            the rating value of home-team

        rating_away : float
            the rating value of away-team

        calc_prob : bool, default=True
            If ``True`` calculate the probability of winning.
            Probabilities are calculated after ratings normalization.

        Returns
        -------
        result : int
            The value of predicted result

        rating_prob : list
            Prediction probabilities for each outcome

        Examples
        --------
        """
        if rating_home > rating_away:
            result = outcomes_dict['H']
            # iff we use normalize ratings
            index_win = 0
            rating_win = rating_home
        elif rating_home < rating_away:
            result = outcomes_dict['A']
            # iff we use normalize ratings
            index_win = 1
            rating_win = rating_away
        else:
            result = outcomes_dict['D']
            # iff we use normalize ratings
            index_win = 2
            rating_win = rating_home + rating_away

        rating_prob = [0.0, 0.0, 0.0]
        if calc_prob:
            sumr = rating_home + rating_away
            if sumr != 0:
                rating_prob[index_win] = 1 if index_win == 2 else rating_win / sumr
            else:
                rating_prob[2] = 1

        return result, rating_prob

    def __repr__(self):
        attrs = vars(self)
        include_keys = {"outcomes_dict", "columns_odds_dict", "name"}
        return '['+', '.join("%s: %s" % (k, v) for (k, v) in attrs.items()
                             if k in include_keys)+']'
