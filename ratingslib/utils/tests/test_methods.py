"""
This module contains the unit tests for utils package
"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT


import os
import unittest

import pandas as pd
from ratingslib.datasets.filenames import FILENAME_EPL_2018_2019_20_GAMES
from ratingslib.datasets.soccer import COLUMNS_DICT
from ratingslib.datasets.parse import parse_pairs_data
from ratingslib.tests.test_all import printdetails
from ratingslib.utils.methods import create_items_dict, get_filename
from pandas.testing import assert_frame_equal

current_dirname = os.path.dirname(__file__)
directory_path = r"../../datasets/"
FP_FILENAME_EPL_2018_2019_20_GAMES = get_filename(FILENAME_EPL_2018_2019_20_GAMES,
                                                  directory_path=directory_path,
                                                  current_dirname=current_dirname)


class TestUtilsMethods(unittest.TestCase):
    """
    A class to test functions from utils.methods module
    The results are based on the filename (FILENAME_EPL_2018_2019_20_GAMES)
    which contains the first two match-weeks of the English Premier League
    soccer championship during the season 2018-2019
    """
    @printdetails
    def test_parse_pairs_data(self):
        """
        Test the parsing procedure of games data,
        Includes 2 tests:
        1. test the total number of teams
        2. test the DataFrame that contains team names under the column 'Item'
        """
        data, teams_df = parse_pairs_data(FP_FILENAME_EPL_2018_2019_20_GAMES,
                                          columns_dict=COLUMNS_DICT)
        self.assertEqual(len(data), 20)  # we have 2 matchweeks => 20 games
        self.assertEqual(len(teams_df), 20)  # 20 teams in PL
        assert_frame_equal(teams_df, pd.DataFrame(data={'Item': [
            'Arsenal', 'Bournemouth', 'Brighton', 'Burnley', 'Cardiff',
            'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Huddersfield',
            'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle',
            'Southampton', 'Tottenham', 'Watford', 'West Ham', 'Wolves']}))

    @printdetails
    def test_create_items_dict(self):
        """Test the correctness of teams dictionary"""
        _, teams_df = parse_pairs_data(FP_FILENAME_EPL_2018_2019_20_GAMES,
                                       columns_dict=COLUMNS_DICT)
        teams_dict = create_items_dict(teams_df)
        self.assertEqual(len(teams_dict), 20)  # 20 teams in PL
        self.assertDictEqual(teams_dict,
                             {'Arsenal': 0,
                              'Bournemouth': 1,
                              'Brighton': 2,
                              'Burnley': 3,
                              'Cardiff': 4,
                              'Chelsea': 5,
                              'Crystal Palace': 6,
                              'Everton': 7,
                              'Fulham': 8,
                              'Huddersfield': 9,
                              'Leicester': 10,
                              'Liverpool': 11,
                              'Man City': 12,
                              'Man United': 13,
                              'Newcastle': 14,
                              'Southampton': 15,
                              'Tottenham': 16,
                              'Watford': 17,
                              'West Ham': 18,
                              'Wolves': 19})


if __name__ == '__main__':
    unittest.main()
