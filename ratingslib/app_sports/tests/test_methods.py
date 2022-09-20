"""Module for test methods module"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

import os
import unittest

from ratingslib.app_sports.methods import predict_hindsight
from ratingslib.application import SoccerOutcome
from ratingslib.datasets.filenames import FILENAME_EPL_2018_2019_20_GAMES
from ratingslib.datasets.parameters import COLUMNS_DICT
from ratingslib.datasets.parse import parse_pairs_data
from ratingslib.ratings.accurate import AccuRate
from ratingslib.tests.test_all import printdetails
from ratingslib.utils.methods import get_filename

current_dirname = os.path.dirname(__file__)
directory_path = r"../../datasets/"
FP_FILENAME_EPL_2018_2019_20_GAMES = get_filename(FILENAME_EPL_2018_2019_20_GAMES,
                                                  directory_path=directory_path,
                                                  current_dirname=current_dirname)


class TestMethods(unittest.TestCase):
    """
    A class to test functions from 
    :mod:`ratingslib.app_sports.methods` 
    module The results are based on the filename (FILENAME_EPL_2018_2019_20_GAMES)
    which contains the first two match-weeks of the English Premier League
    soccer championship during the season 2018-2019
    """
    @printdetails
    def test_calc_predictions(self):
        """
        Test the prediction procedure according to rating values.
        The logic of prediction is that a higher rating is preferred than
        the lower rating.
        For example in a match between teamA and teamB
        with ratingA and ratingB respectively,
        if ratingA > ratingB then prediction of the winner is teamA
        if ratingA < ratingB then prediction of the winner is teamB
        if ratingA = ratingB then prediction is the Draw.
        """
        columns_dict = COLUMNS_DICT
        outcome = SoccerOutcome()
        data, teams_df = parse_pairs_data(FP_FILENAME_EPL_2018_2019_20_GAMES,
                                          outcome=outcome,
                                          columns_dict=columns_dict)
        ac = AccuRate().rate(data, teams_df, columns_dict=columns_dict)
        # hindsight
        pred, _ = predict_hindsight(
            data, ac, outcome, columns_dict=columns_dict)
        self.assertListEqual(pred,
                             [2, 1, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 1, 2])


if __name__ == '__main__':
    unittest.main()
