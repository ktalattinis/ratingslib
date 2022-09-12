"""
Elo Rating System
"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT


from typing import Any, List, Optional, Dict
from ratingslib.ratings.methods import rating_systems_to_dict

from ratingslib.ratings.rating import RatingSystem
from ratingslib.utils.enums import ratings
from ratingslib.utils.methods import parse_columns, get_indices, \
    indices_and_points, create_items_dict
import numpy as np
import pandas as pd

from ratingslib.utils.validation import validate_from_set


class Elo(RatingSystem):
    """Elo ranking system developed by Arpad Elo [1]_ in order to
    rank chess players, this system has been adopted by quite a lot of sports
    and organizations.

    This implementation includes two basic versions of Elo:

        * The first is called EloWin and takes into account total wins of
          items. In soccer teams the final outcome determines the
          winner.
        * The second is called EloPoint and takes into account items scores.
          In soccer the points are the goals scored be each team.

    Note that for any kind of items, there are many ways to define the
    notion of a hypothetical matchup and then to determine scores and winners.

    Parameters
    ----------
    version : str, default=ratings.ELOWIN
        a string that shows version of rating system

    K : int, default=40
        K-factor is the maximum possible adjustment per pair of items.
        For soccer, K–factor plays an important role because it balances the
        deviation for the goal difference in the game against prior ratings.

    HA : int, default=0
        The home advantage factor is an adjustment that is used due to the fact
        that home teams tend to score more goals. Elo system applies the
        home-field advantage factor, by adding it to the rating of home team.
        Many implementations of Elo model for soccer, set the home-field
        advantage to 100. The default value ``0`` means that method does not 
        take into account home advantage factor.

    ks : float, default=400
        Parameter ξ (``ks``) affects the spread of ratings and comes from
        logistic function. For chess and soccer games usually, ξ is set to 400.

    starting_point : float, default = 1500
        The value where the initial rating starts

    Notes
    -----
    Soccer application and Elo:
    According to the type of soccer tournament the following values represents
    the K-Factor value suggested by several internet sites [2]_:

        * World Cup Finals = 60
        * Continental Championship Finals and Major Intercontinental
          tournaments = 50
        * World Cup Qualifiers and Major Tournaments = 40
        * All other tournaments = 30
        * Friendly matches = 20

    References
    ----------
    .. [1] Elo, A. E. (1978). The rating of chessplayers, past and present. Arco Pub.
    .. [2] http://www.eloratings.net/about

    Examples
    --------
    The following examples demonstrates the EloWin and the EloPoint version,
    for the 20 first soccer matches that took place during the 2018-2019
    season of English Premier League.

    >>> from ratingslib.datasets.filenames import dataset_path, FILENAME_EPL_2018_2019_20_GAMES
    >>> from ratingslib.ratings.elo import Elo
    >>> from ratingslib.utils.enums import ratings
    >>> filename = dataset_path(FILENAME_EPL_2018_2019_20_GAMES)
    >>> Elo(version=ratings.ELOWIN, starting_point=0).rate_from_file(filename)
                  Item     rating  ranking
    0          Arsenal -37.707535       12
    1      Bournemouth  37.707535        3
    2         Brighton   2.292465        5
    3          Burnley -18.849977        9
    4          Cardiff -20.000000       10
    5          Chelsea  37.707535        3
    6   Crystal Palace   0.000000        7
    7          Everton  20.000000        4
    8           Fulham -37.707535       12
    9     Huddersfield -37.707535       12
    10       Leicester   1.150023        6
    11       Liverpool  40.000000        1
    12        Man City  37.707535        3
    13      Man United  -2.292465        8
    14       Newcastle -20.000000       10
    15     Southampton -20.000000       10
    16       Tottenham  37.707535        3
    17         Watford  38.849977        2
    18        West Ham -37.707535       12
    19          Wolves -21.150023       11
    >>> Elo(version=ratings.ELOPOINT, starting_point=0).rate_from_file(filename)
                  Item     rating  ranking
    0          Arsenal -11.592411       17
    1      Bournemouth  12.658841        5
    2         Brighton  -6.337388       14
    3          Burnley  -6.091179       13
    4          Cardiff  -9.654647       15
    5          Chelsea  13.592411        4
    6   Crystal Palace   0.191876       10
    7          Everton   4.000000        8
    8           Fulham -15.861198       18
    9     Huddersfield -21.846379       20
    10       Leicester   6.230248        7
    11       Liverpool  23.141457        1
    12        Man City  19.846379        2
    13      Man United   0.337388        9
    14       Newcastle  -4.345353       12
    15     Southampton  -4.000000       11
    16       Tottenham   9.861198        6
    17         Watford  16.091179        3
    18        West Ham -15.992174       19
    19          Wolves -10.230248       16
    """

    def __init__(self, version=ratings.ELOWIN,
                 K: int = 40, HA=0, ks=400,
                 starting_point: float = 1500):
        super().__init__(version)
        self.K = K
        self.HA = HA
        self.ks = ks
        self.create_ordered_dict(K=self.K, HA=self.HA, ks=self.ks)
        self.starting_point = float(starting_point)

    def create_rating_vector(self, data_df: pd.DataFrame,
                             items_df: pd.DataFrame,
                             columns_dict: Optional[Dict[str, Any]] = None
                             ) -> np.ndarray:
        """Calculates Elo ratings according to pairs of items data."""
        col_names = parse_columns(columns_dict)
        home_col_ind, away_col_ind, home_points_col_ind, away_points_col_ind =\
            get_indices(col_names.item_i, col_names.item_j,
                        col_names.points_i, col_names.points_j, data=data_df)
        data_np = data_df.to_numpy()
        items_dict = create_items_dict(items_df)
        rating_old = np.array([float(self.starting_point)
                               for _ in range(len(items_df))])
        rating_new = np.array([float(self.starting_point)
                               for _ in range(len(items_df))])
        for row in data_np:
            if self.version == ratings.ELOWIN:
                i, j, points_home, points_away = indices_and_points(
                    row, items_dict, home_col_ind, away_col_ind,
                    home_points_col_ind, away_points_col_ind)
                if points_home > points_away:
                    points_ij = 1.0
                    points_ji = 0.0
                elif points_home < points_away:
                    points_ij = 0.0
                    points_ji = 1.0
                elif points_home == points_away:
                    points_ij = 0.5
                    points_ji = 0.5
                S_ij = points_ij
                S_ji = points_ji
            elif self.version == ratings.ELOPOINT:
                i, j, points_ij, points_ji = indices_and_points(
                    row, items_dict, home_col_ind, away_col_ind,
                    home_points_col_ind, away_points_col_ind)
                S_ij = (points_ij + 1) / (points_ij + points_ji + 2)
                S_ji = (points_ji + 1) / (points_ji + points_ij + 2)

            d_ij = rating_old[i] - rating_old[j] + self.HA
            d_ji = rating_old[j] - rating_old[i]

            m_ij = 1 / (1 + np.power(10, (-d_ij / self.ks)))
            m_ji = 1 / (1 + np.power(10, (-d_ji / self.ks)))
            # print("old",rating_old)
            rating_new[i] = rating_old[i] + self.K * (S_ij - m_ij)
            rating_new[j] = rating_old[j] + self.K * (S_ji - m_ji)
            rating_old[i] = rating_new[i]
            rating_old[j] = rating_new[j]

            # print(index,"new",rating_new)
        return rating_new

    def computation_phase(self):
        """Nothing to compute, all computations are made in
        :meth:`ratingslib.ratings.elo.Elo.create_rating_vector` method"""

    def preparation_phase(self, data_df: pd.DataFrame, items_df: pd.DataFrame,
                          columns_dict: Optional[Dict[str, Any]] = None):
        self.rating = self.create_rating_vector(data_df, items_df,
                                                columns_dict=columns_dict)

    def rate(self, data_df: pd.DataFrame, items_df: pd.DataFrame,
             sort: bool = False,
             columns_dict: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        items_df = items_df.copy(deep=True)
        self.preparation_phase(data_df, items_df, columns_dict)
        self.computation_phase()
        items_df = self.set_rating(items_df, sort=sort)
        return items_df

    @staticmethod
    def prepare_for_gridsearch_tuning(
            *, version_list: List[str] = None,
            k_range: Optional[List[float]] = None,
            ks_range: Optional[List[float]] = None,
            HA_range: Optional[List[float]] = None,
    ) -> Dict[str, RatingSystem]:
        """Create instances that are intended for tuning parameters.

        Parameters
        ----------
        version_list : List[str]
            List of Elo versions

        k_range : Optional[List[float]], default=None
            List of k values. If ``None`` then
            parameter is not intended for tuning

        ks_range : Optional[List[float]], default=None
            List of ks values. If ``None`` then
            parameter is not intended for tuning

        HA_range : Optional[List[float]], default=None
            List of HA values. If ``None`` then
            parameter is not intended for tuning

        Returns
        -------
        rating_systems_dict : dict
            Dictionary that contains Elo instances with the
            parameters we want for tuning.

        """
        if version_list is None:
            version_list = [ratings.ELOWIN, ratings.ELOPOINT]
        for v in version_list:
            validate_from_set(
                v, {ratings.ELOWIN, ratings.ELOPOINT}, 'version_list')
        if k_range is None:
            k_range = list(np.arange(5, 41, step=5))
        if ks_range is None:
            ks_range = list(np.arange(50, 401, step=50))
        if HA_range is None:
            HA_range = list(np.arange(50, 101, step=10))

        rating_systems_list = [Elo(version=v,
                                   K=k,
                                   ks=ks,
                                   HA=ha,
                                   starting_point=1500)
                               for k in k_range
                               for ks in ks_range
                               for ha in HA_range for v in version_list
                               ]
        rating_systems_dict = rating_systems_to_dict(rating_systems_list)
        return rating_systems_dict
