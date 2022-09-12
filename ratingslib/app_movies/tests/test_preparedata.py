# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT


import os
import unittest

import pandas as pd

from ratingslib.datasets.filenames import FILENAME_MOVIES_EXAMPLE
from ratingslib.datasets.parse import create_pairs_data
from ratingslib.tests.test_all import printdetails
from ratingslib.utils.methods import get_filename

current_dirname = os.path.dirname(__file__)
directory_path = r"../../datasets/"
FP_FILENAME_MOVIES_EXAMPLE = get_filename(FILENAME_MOVIES_EXAMPLE,
                                          directory_path=directory_path,
                                          current_dirname=current_dirname)


class TestPrepareData(unittest.TestCase):
    """
    A class to test functions from :mod:`ratingslib.app_movies.preparedata`
    module. The results are based on the filename (FILENAME_MOVIES_EXAMPLE)
    which contains a small example where 5 users rate 3 movies
    (1-5 scale, 0 means that the user does not rate the movie).
    """
    @printdetails
    def test_movie_movie_matrix(self):
        """
        Test if user-movie matrix is converted to movie-movie matrix correct.
        We consider that a hypothetical mathup between two movies exists if
        at least one user has rated both movies.
        """
        user_movie_df = pd.read_csv(
            FP_FILENAME_MOVIES_EXAMPLE, index_col='User')

        COLUMNS_MOVIE_DICT = {
            'item_i': 'MovieI',
            'item_j': 'MovieJ',
            'points_i': 'RatingI',
            'points_j': 'RatingJ'
        }

        movie_movie_df = create_pairs_data(
            user_movie_df, columns_dict=COLUMNS_MOVIE_DICT)
        self.assertListEqual(movie_movie_df.values[:, -2:].tolist(), [
                             [1, 5], [4, 3], [2, 4], [4, 5], [1, 4], [3, 5], [2, 3]])


if __name__ == '__main__':
    unittest.main()
