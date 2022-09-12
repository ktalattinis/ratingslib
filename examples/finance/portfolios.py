"""Small example on portfolio rating and ranking based on aggregation methods"""

from ratingslib.datasets.filenames import FILENAME_PORTFOLIOS, dataset_path
from ratingslib.ratings.aggregation import RatingAggregation
from ratingslib.utils.enums import ratings
from ratingslib.utils.methods import print_info

if __name__ == '__main__':

    filename = dataset_path(FILENAME_PORTFOLIOS)
    columns_dict = {'item': 'portfolio',
                    'ratings': ['R2', 'AvgReturn', 'maxDD']}

    votes_or_weights = {
        'ENGINEER [RISK-SEEKING]': {
            'R2': 4.0,
            'AvgReturn': 10.0,
            'maxDD': 4.0},
        'STARTUP [NEUTRAL]': {
            'R2': 10.0,
            'AvgReturn': 10.0,
            'maxDD': 10.0},
        'FUND [RISK-AVERSE]': {
            'R2': 10.0,
            'AvgReturn': 4.0,
            'maxDD': 10.0}
    }

    versions = [ratings.AGGREGATIONMARKOV,
                ratings.AGGREGATIONOD,
                ratings.AGGREGATIONPERRON]

    for key, vw in votes_or_weights.items():
        print_info(key)
        for version in versions:
            print_info(version)
            ra = RatingAggregation(version, votes_or_weights=vw)
            ratings_df = ra.rate_from_file(
                filename, pairwise=False, columns_dict=columns_dict)
            print(ratings_df)
