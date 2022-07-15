"""Small example on investment selection based on rating and ranking aggregation methods"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

import pandas as pd
from apprate.datasets.filenames import FILENAME_INVESTMENTS, dataset_path
from apprate.ratings.aggregation import RankingAggregation, RatingAggregation
from apprate.utils.enums import ratings
from apprate.utils.methods import print_info

if __name__ == '__main__':
    pd.set_option('float_format', "{:.4f}".format)
    filename = dataset_path(FILENAME_INVESTMENTS)
    columns_dict = {'item': 'investment',
                    'ratings': ['ROI', 'PP']}
    # ==========================================================================
    # RANKING AGGREGATION
    # ==========================================================================

    versions = [ratings.RANKINGAVG, ratings.RANKINGBORDA]
    for version in versions:
        print_info(version)
        ra = RankingAggregation(version)
        ratings_df = ra.rate_from_file(
            filename, pairwise=False,
            reverse_attributes_cols=['PP'],
            columns_dict=columns_dict)
        print(ratings_df)

    # ==========================================================================
    # RATING AGGREGATION
    # ==========================================================================

    versions = [ratings.AGGREGATIONMARKOV,
                ratings.AGGREGATIONOD,
                ratings.AGGREGATIONPERRON]

    for version in versions:
        print_info(version)
        ra = RatingAggregation(version)
        ratings_df = ra.rate_from_file(
            filename, pairwise=False,
            reverse_attributes_cols=['PP'],
            columns_dict=columns_dict)
        print(ratings_df)
