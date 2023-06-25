"""
Tests for rating methods, metrics and helper functions
"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

import unittest

import numpy as np
import pandas as pd
from ratingslib.datasets.filenames import (
    FILENAME_ACCURATE_PAPER_EXAMPLE,
    FILENAME_NCAA_2005_ATLANTIC, FILENAME_CHARTIER_PAPER_MOVIES, FILENAME_EPL_2018_2019_20_GAMES,
    FILENAME_GOVAN_THESIS, FILENAME_NFL_2009_2010_FULLNAMES_GAMES_AND_PLAYOFFS,
    FILENAME_NFL_2009_2010_SHORTNAMES_NO_PLAYOFF, FILENAME_OOSA,
    datasets_paths)
from ratingslib.datasets.parse import create_pairs_data, parse_pairs_data
from ratingslib.datasets.soccer import COLUMNS_DICT, stats
from ratingslib.ratings.accurate import AccuRate
from ratingslib.ratings.aggregation import RankingAggregation, RatingAggregation
from ratingslib.ratings.colley import Colley
from ratingslib.ratings.elo import Elo
from ratingslib.ratings.keener import Keener
from ratingslib.ratings.markov import Markov
from ratingslib.ratings.massey import Massey
from ratingslib.ratings.methods import calc_items_stats
from ratingslib.ratings.od import OffenseDefense
from ratingslib.ratings.winloss import Winloss
from ratingslib.tests.test_all import printdetails
from ratingslib.utils.enums import ratings
from numpy.ma.testutils import assert_array_almost_equal
from pandas.testing import assert_frame_equal
from scipy.stats import stats as statsscipy

# get the full absolute path
FP_FILENAME_NCAA_2005_ATLANTIC, FP_FILENAME_EPL_2018_2019_20_GAMES, FP_FILENAME_GOVAN_THESIS,\
    FP_FILENAME_ACCURATE_PAPER_EXAMPLE, \
    FP_FILENAME_NFL_2009_2010_FULLNAMES_GAMES_AND_PLAYOFFS, \
    FP_FILENAME_NFL_2009_2010_SHORTNAMES_NO_PLAYOFF, \
    FP_FILENAME_OOSA, FP_FILENAME_BOOK_MOVIES = datasets_paths(
        FILENAME_NCAA_2005_ATLANTIC, FILENAME_EPL_2018_2019_20_GAMES,
        FILENAME_GOVAN_THESIS,
        FILENAME_ACCURATE_PAPER_EXAMPLE,
        FILENAME_NFL_2009_2010_FULLNAMES_GAMES_AND_PLAYOFFS,
        FILENAME_NFL_2009_2010_SHORTNAMES_NO_PLAYOFF,
        FILENAME_OOSA, FILENAME_CHARTIER_PAPER_MOVIES)


class TestRatingSystems(unittest.TestCase):
    """
    Test for rating methods.
    The following tests: test_colley, test_massey, test_winloss, test_markov,
    test_offense_defense, test_rating_movies, and test_aggregation
    are based on NCAA 2005 data of an isolated group of Atlantic Coast.
    Conference teams. This example is provided in the book
    "Who's #1? The Science of Rating and Ranking" [1]_.
    The test_accurate is based on the paper [2]_.
    The test for rating_movies is based on the paper [3]_.


    References
    ----------
    .. [1] Langville, A. N., & Meyer, C. D. (2012). Who's# 1?: the science of rating and ranking.
           Princeton University Press.

    .. [2] Kyriakides, G., Talattinis, K., & Stephanides, G. (2017).
           A Hybrid Approach to Predicting Sports Results and an AccuRATE Rating System.
           International Journal of Applied and Computational Mathematics, 3(1), 239–254.

    .. [3] Chartier, T., Langville, A., & Simov, P. (2010). March Madness to Movies. Math Horizons, 17, 16–19.

    """
    @printdetails
    def test_colley(self):
        """Test for Colley rating system"""
        c = Colley().rate_from_file(FP_FILENAME_NCAA_2005_ATLANTIC, sort=True)
        assert_array_almost_equal(c.rating.values,
                                  np.array([0.79, 0.65, 0.50, 0.36, 0.21]),
                                  decimal=2)
        self.assertAlmostEqual(c.rating.values.sum(), 2.5, 4)
        c1 = Colley().rate_from_file(FP_FILENAME_OOSA, sort=True)
        assert_array_almost_equal(
            c1.rating.values, np.array([0.756, 0.524, 0.503, 0.378, 0.339]),
            decimal=2)
        self.assertAlmostEqual(c1.rating.values.sum(), 2.5, 4)

    @printdetails
    def test_massey(self):
        """Test for Massey rating system"""
        m = Massey().rate_from_file(FP_FILENAME_NCAA_2005_ATLANTIC, sort=True)
        assert_array_almost_equal(m.rating.values,
                                  np.array([18.2, 18.0, -3.4, -8.0, -24.8]),
                                  decimal=2)
        self.assertAlmostEqual(m.rating.values.sum(), 0, 5)

    @printdetails
    def test_winloss(self):
        """Test for Winloss method"""
        w = Winloss(normalization=False).rate_from_file(FP_FILENAME_NCAA_2005_ATLANTIC,
                                                        sort=True)
        assert_array_almost_equal(w.rating.values,
                                  np.array([4, 3, 2, 1, 0]),
                                  decimal=2)

        w = Winloss(normalization=True).rate_from_file(FP_FILENAME_NCAA_2005_ATLANTIC,
                                                       sort=True)
        assert_array_almost_equal(w.rating.values,
                                  np.array([1, 0.75, 0.5, 0.25, 0]), decimal=2)

    @printdetails
    def test_keener(self):
        """Test for Keener rating system"""
        k = Keener(normalization=False).rate_from_file(
            FP_FILENAME_NFL_2009_2010_SHORTNAMES_NO_PLAYOFF, sort=True)
        assert_array_almost_equal(k.rating.values,
                                  np.array([0.036139, 0.035722, 0.035051,
                                            0.035026, 0.034817, 0.034783,
                                            0.03471, 0.034683, 0.033883,
                                            0.033821, 0.033529, 0.033415,
                                            0.03269, 0.032346, 0.031876,
                                            0.031789, 0.031483, 0.030785,
                                            0.030538, 0.03048, 0.029805,
                                            0.02941, 0.029107, 0.029066,
                                            0.028962, 0.028006, 0.027923,
                                            0.027262, 0.026222, 0.026194,
                                            0.025595, 0.024881]),
                                  decimal=4)

    @printdetails
    def test_elo(self):
        """Test for Elo rating system"""
        # Elo based on wins
        elw = Elo(version=ratings.ELOWIN,
                  K=32, ks=1000,
                  HA=0,
                  starting_point=0).rate_from_file(
                      FP_FILENAME_NFL_2009_2010_FULLNAMES_GAMES_AND_PLAYOFFS,
                      sort=True)
        assert_array_almost_equal(elw.rating.values,
                                  np.array([173.66, 170.33, 127.58, 103.5,
                                            89.128, 69.533, 67.829, 53.227,
                                            50.143, 39.633, 33.902, 33.012,
                                            32.083, 28.118, 27.125, 13.222,
                                            11.474, -1.2844, -5.3217, -11.126,
                                            -26.717, -28.142, -36.214, -53.35,
                                            -74.664, -83.319, -88.845, -109.28,
                                            -110.21, -130.1, -170.81,
                                            -194.12]),
                                  decimal=2)
        # Elo based on points
        elp = Elo(version=ratings.ELOPOINT,
                  K=32,
                  ks=1000,
                  HA=0,
                  starting_point=0).rate_from_file(
                      FP_FILENAME_NFL_2009_2010_FULLNAMES_GAMES_AND_PLAYOFFS,
                      sort=True)
        assert_array_almost_equal(elp.rating.values,
                                  np.array([58.825, 55.217, 49.495, 47.215,
                                            43.074, 40.357, 39.974, 39.26,
                                            37.86, 33.189, 18.447, 18.387,
                                            13.984, 9.1308, 6.1216, 5.2596,
                                            4.1006, -0.75014, -3.5097,
                                            -9.3122, -9.8351, -16.05, -23.287,
                                            -29.039, -34.647, -35.15, -37.05,
                                            -47.089, -54.373, -62.652,
                                            -72.8, -84.352]),
                                  decimal=2)

    @printdetails
    def test_offense_defense(self):
        """Test for Offense-Defense method"""
        mh = OffenseDefense(tol=0.0001).rate_from_file(FP_FILENAME_NCAA_2005_ATLANTIC,
                                                       sort=True)
        assert_array_almost_equal(mh.rating.values,
                                  np.array([279.8, 188.8, 84.8, 41.8, 20.1]),
                                  decimal=1)

    @printdetails
    def test_markov(self):
        """Test for Generalized Markov Method (GeM)"""
        # VotingWithLosses Test
        params_markov_sport_dict = {
            'TW': {'VOTE': 10, 'ITEM_I': 'FTHG', 'ITEM_J': 'FTAG',
                   'METHOD': 'VotingWithLosses'},
        }
        m = Markov(b=1,
                   stats_markov_dict=params_markov_sport_dict).rate_from_file(
            FP_FILENAME_NCAA_2005_ATLANTIC, sort=True)
        assert_array_almost_equal(m.rating.values, np.array(
            [0.438, 0.219, 0.146, 0.110, 0.087]), decimal=2)
        # LoserVotePointDiff Test
        params_markov_sport_dict = {
            'TG': {'VOTE': 10, 'ITEM_I': 'FTHG', 'ITEM_J': 'FTAG',
                   'METHOD': 'LosersVotePointDiff'},
        }
        m = Markov(b=1,
                   stats_markov_dict=params_markov_sport_dict).rate_from_file(
                       FP_FILENAME_NCAA_2005_ATLANTIC, sort=True)
        assert_array_almost_equal(m.rating.values, np.array(
            [0.442, 0.265, 0.110, 0.095, 0.088]), decimal=2)
        # WinnersAndLosersVotePoint Test
        params_markov_sport_dict = {
            'TG': {'VOTE': 10, 'ITEM_I': 'FTHG', 'ITEM_J': 'FTAG',
                   'METHOD': 'WinnersAndLosersVotePoint'},
        }
        m = Markov(b=1,
                   stats_markov_dict=params_markov_sport_dict).rate_from_file(
            FP_FILENAME_NCAA_2005_ATLANTIC, sort=True)
        assert_array_almost_equal(m.rating.values,
                                  np.array([
                                      0.296, 0.244, 0.216, 0.149, 0.095]),
                                  decimal=2)
        # LosersVotePointDiff b=0.85 Test
        params_markov_sport_dict = {
            'TW': {'VOTE': 10, 'ITEM_I': 'FTHG', 'ITEM_J': 'FTAG',
                   'METHOD': 'LosersVotePointDiff'},
        }
        m = Markov(b=0.85,
                   stats_markov_dict=params_markov_sport_dict).rate_from_file(
            FP_FILENAME_GOVAN_THESIS, sort=True)
        assert_array_almost_equal(m.rating.values, np.array(
            [0.3281, 0.2824, 0.2289, 0.0656, 0.056, 0.0389]), decimal=2)

    @printdetails
    def test_accurate(self):
        """Test for Accurate rating system"""
        # paper example todo [reference]
        ac = AccuRate().rate_from_file(FP_FILENAME_ACCURATE_PAPER_EXAMPLE)
        assert_array_almost_equal(ac.rating.values,
                                  np.array([1.68, -1.59]),
                                  decimal=2)

    @printdetails
    def test_aggregation(self):
        """Test rating aggregation"""
        massey = Massey().rate_from_file(FP_FILENAME_NCAA_2005_ATLANTIC)
        colley = Colley().rate_from_file(FP_FILENAME_NCAA_2005_ATLANTIC)
        od = OffenseDefense(tol=0.0001).rate_from_file(
            FP_FILENAME_NCAA_2005_ATLANTIC)
        items_df = massey[['Item']]
        # print(type(items_df))
        ratings_dict = {
            'Item': items_df['Item'].values,
            'massey': massey.rating.values,
            'colley': colley.rating.values,
            'od': od.rating.values
        }
        columns_dict = {'item': 'Item',
                        'ratings': list(ratings_dict.keys()-{'Item'})}
        data_df = pd.DataFrame.from_dict(ratings_dict)
        ra_borda = RankingAggregation(ratings.RANKINGBORDA).rate(data_df, items_df,
                                                                 columns_dict=columns_dict)
        self.assertListEqual(
            ra_borda['rating'].values.tolist(),
            [0.0, 11.0, 4.0, 5.0, 10.0])
        self.assertListEqual(
            ra_borda['ranking'].values.tolist(),
            [5, 1, 4, 3, 2])

        ra_avg = RankingAggregation(ratings.RANKINGAVG).rate(data_df, items_df,
                                                             columns_dict=columns_dict)
        print(ra_avg)
        assert_array_almost_equal(
            ra_avg['rating'].values.tolist(),
            np.array([5.0, 1.33, 3.66, 3.33, 1.66]),
            decimal=2)
        self.assertListEqual(
            ra_avg['ranking'].values.tolist(),
            [5, 1, 4, 3, 2])

        ra_markov = RatingAggregation(version=ratings.AGGREGATIONMARKOV, b=0.9).rate(
            data_df, items_df, columns_dict=columns_dict, sort=True)

        assert_array_almost_equal(
            ra_markov['rating'].values,
            -np.sort(-np.array([0.02, 0.466, 0.024, 0.024, 0.466])),
            decimal=3)

        ra_od = RatingAggregation(version=ratings.AGGREGATIONOD).rate(
            data_df, items_df, columns_dict=columns_dict, sort=True)
        assert_array_almost_equal(
            ra_od['rating'].values,
            -np.sort(-np.array([0.0, 17.8801, 0.3329, 0.3447, 25.3616])),
            decimal=3)

        ra_perron = RatingAggregation(version=ratings.AGGREGATIONPERRON).rate(
            data_df, items_df, columns_dict=columns_dict, sort=True)
        assert_array_almost_equal(
            ra_perron['rating'].values,
            -np.sort(-np.array([0., 0.463302, 0., 0., 0.536698])),
            decimal=3)

    @printdetails
    def test_rating_movies(self):

        user_movie_df = pd.read_csv(FP_FILENAME_BOOK_MOVIES, index_col='User')

        COLUMNS_MOVIE_DICT = {
            'item_i': 'MovieI',
            'item_j': 'MovieJ',
            'points_i': 'RatingI',
            'points_j': 'RatingJ'
        }

        movie_movie_df = create_pairs_data(
            user_movie_df, columns_dict=COLUMNS_MOVIE_DICT)
        # print(movie_movie_df)
        data_df, items_df = parse_pairs_data(
            movie_movie_df, columns_dict=COLUMNS_MOVIE_DICT)
        colley = Colley().rate(data_df,
                               items_df,
                               columns_dict=COLUMNS_MOVIE_DICT)
        massey = Massey().rate(data_df,
                               items_df,
                               columns_dict=COLUMNS_MOVIE_DICT)
        assert_array_almost_equal(
            colley['rating'].values,
            np.array([0.67, 0.63, 0.34, 0.35]),
            decimal=2)
        assert_array_almost_equal(
            massey['rating'].values,
            np.array([0.65, 1.01, -0.55, -1.11]),
            decimal=2)


class TestMetrics(unittest.TestCase):
    """
    A Class for testing functions of metrics module
    """

    @printdetails
    def test_kendalls_tau(self):
        """
        Test the results of Kendalls tau correlation coefficient.
        The data below is based on the example of
        chapter 16, p. 204 of Who's is First book
        """
        rating_list_1 = [4, 1, 2, 3]
        rating_list_2 = [4, 1, 3, 2]
        rating_list_3 = [1, 4, 2, 3]
        tau_12, _ = statsscipy.kendalltau(rating_list_1, rating_list_2)
        tau_13, _ = statsscipy.kendalltau(rating_list_1, rating_list_3)
        tau_23, _ = statsscipy.kendalltau(rating_list_2, rating_list_3)
        self.assertAlmostEqual(tau_12, 4 / 6, places=4)
        self.assertAlmostEqual(tau_13, -4 / 6, places=4)
        self.assertAlmostEqual(tau_23, -1, places=4)


class TestMethods(unittest.TestCase):

    @printdetails
    def test_calc_team_stats(self):
        """
        Test the calculation of the statistics of each team,
        Includes 4 statistics
        1. test the total number of teams
        2. test the DataFrame that contains team names under the column 'Item'
        """
        data, teams_df = parse_pairs_data(FP_FILENAME_EPL_2018_2019_20_GAMES,
                                          columns_dict=COLUMNS_DICT)
        teams_df = calc_items_stats(
            data, teams_df, normalization=False,
            stats_columns_dict=stats.STATS_COLUMNS_DICT,
            columns_dict=COLUMNS_DICT)
        assert_frame_equal(teams_df, pd.DataFrame(
            data={
                'Item': ['Arsenal', 'Bournemouth', 'Brighton', 'Burnley',
                         'Cardiff', 'Chelsea', 'Crystal Palace', 'Everton',
                         'Fulham', 'Huddersfield', 'Leicester', 'Liverpool',
                         'Man City', 'Man United', 'Newcastle', 'Southampton',
                         'Tottenham', 'Watford', 'West Ham', 'Wolves'],
                'TW': [0.0, 2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0,
                       1.0, 2.0, 2.0, 1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0],
                'TG': [2.0, 4.0, 3.0, 1.0, 0.0, 6.0, 2.0, 4.0, 1.0, 1.0,
                       3.0, 6.0, 8.0, 4.0, 1.0, 1.0, 5.0, 5.0, 1.0, 2.0],
                'TST': [9.0, 9.0, 3.0, 9.0, 2.0, 15.0, 11.0, 12.0, 9.0, 2.0,
                        6.0, 14.0, 22.0, 9.0, 8.0, 7.0, 16.0, 11.0, 7.0,
                        7.0],
                'TS': [24.0, 24.0, 12.0, 24.0, 22.0, 37.0, 18.0, 19.0,
                       25.0, 11.0, 19.0, 34.0, 49.0, 17.0, 27.0, 33.0,
                       40.0, 28.0, 16.0, 22.0],
                'NG': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                       2, 2, 2, 2]
            }))


if __name__ == '__main__':
    unittest.main()
