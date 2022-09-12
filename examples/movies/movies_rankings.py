"""
This example illustrates the use of rating systems for movie rankings.
The dataset used [1]_ is obtained from: https://grouplens.org/datasets/movielens/ and it has the following details:
Small: 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users. Last updated 9/2018.
Link: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
Readme Link: https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html

.. [1] Harper, F. M., & Konstan, J. A. (2015, December). The MovieLens Datasets: History and Context. ACM Trans. Interact. Intell. Syst., 5. doi:10.1145/2827872

"""

import pandas as pd
from ratingslib.app_movies.preparedata import create_movie_users
from ratingslib.datasets.filenames import MOVIES_DATA_PATH, datasets_paths
from ratingslib.datasets.parse import create_pairs_data, parse_pairs_data
import ratingslib.ratings as rl
from ratingslib.utils.enums import ratings
from ratingslib.utils.methods import parse_columns, print_info, print_pandas

# set minimum votes from users
MIN_VOTES = 200
# get dataset paths
filename_ratings, filename_movies = datasets_paths(
    MOVIES_DATA_PATH+"ratings.csv", MOVIES_DATA_PATH+"movies.csv")

# load data
ratings_df = pd.read_csv(filename_ratings)
movies_df = pd.read_csv(filename_movies)

# prepare data
user_movie_df, mr_df, movies_dict, id_titles_dict = create_movie_users(
    ratings_df, movies_df, min_votes=MIN_VOTES)

COLUMNS_MOVIE_DICT = {
    'item_i': 'MovieI',
    'item_j': 'MovieJ',
    'points_i': 'RatingI',
    'points_j': 'RatingJ'
}
# create pairs. Create movie-movie dataframe which means that every pair is a
# hypothetical matchup. The columns of movie_movie dataframe are
# set in COLUMNS_MOVIE_DICT.
movie_movie_df = create_pairs_data(
    user_movie_df, columns_dict=COLUMNS_MOVIE_DICT)
# replace ids to titles
col_names = parse_columns(COLUMNS_MOVIE_DICT)
movie_movie_df.replace({col_names.item_i: id_titles_dict,
                        col_names.item_j: id_titles_dict}, inplace=True)

# parse movie-movie dataframe as pairs data.
data_df, items_df = parse_pairs_data(
    movie_movie_df, columns_dict=COLUMNS_MOVIE_DICT)

# RATE:

# Colley method
colley = rl.Colley().rate(data_df, items_df,
                          columns_dict=COLUMNS_MOVIE_DICT, sort=True)

# Massey method
massey = rl.Massey().rate(data_df, items_df,
                          columns_dict=COLUMNS_MOVIE_DICT, sort=True)
# Keener method
keener = rl.Keener(normalization=True).rate(data_df, items_df,
                                            columns_dict=COLUMNS_MOVIE_DICT,
                                            sort=True)
# Offense Defense method
od = rl.OffenseDefense(tol=0.0001).rate(data_df, items_df,
                                        columns_dict=COLUMNS_MOVIE_DICT, sort=True)

# print rating and ranking lists
print_pandas(colley)
print_pandas(massey)
print_pandas(keener)
print_pandas(od)

# We create a dictionary in order to compare ranking lists with Kendall Tau
ratings_dict = {
    'colley': colley.sort_values(by='Item'),
    'massey': massey.sort_values(by='Item'),
    'keener': keener.sort_values(by='Item'),
    'od': od.sort_values(by='Item')
}
kendall_tau_results = rl.metrics.kendall_tau_table(
    ratings_dict=ratings_dict, print_out=True)

# We aggregate rating values into one rating list by applying Perron method
print_info("RATING AGGREGATION: PERRON")
ra = rl.RatingAggregation(ratings.AGGREGATIONPERRON)

ratings_aggr_dict = {
    'Item': items_df['Item'].values,
    'colley': colley.sort_values(by='Item').rating.values,
    'massey': massey.sort_values(by='Item').rating.values,
    'keener': keener.sort_values(by='Item').rating.values,
    'od': od.sort_values(by='Item').rating.values
}

columns_dict = {'item': 'Item',
                'ratings': list(ratings_aggr_dict.keys()-{'Item'})}
data_df = pd.DataFrame.from_dict(ratings_aggr_dict)
movies_aggr_df = ra.rate(
    data_df, items_df, columns_dict=columns_dict, sort=True)
print_pandas(movies_aggr_df)
