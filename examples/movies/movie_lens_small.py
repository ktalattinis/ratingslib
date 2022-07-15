"""
This example illustrates the use of rating systems for movies ranking.
The dataset used [1]_ is obtained from: https://grouplens.org/datasets/movielens/ and it has the following details:
Small: 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users. Last updated 9/2018.
Link: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
Readme Link: https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html

.. [1] Harper, F. M., & Konstan, J. A. (2015, December). The MovieLens Datasets: History and Context. ACM Trans. Interact. Intell. Syst., 5. doi:10.1145/2827872

"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

import pandas as pd
from apprate.app_movies.preparedata import create_movie_users
from apprate.datasets.filenames import MOVIELENS_SMALL_PATH, datasets_paths
from apprate.datasets.parse import create_pairs_data, parse_pairs_data
import apprate.ratings as ar
from apprate.utils.enums import ratings
from apprate.utils.methods import parse_columns, print_info, print_pandas

# set minimum votes from users
MIN_VOTES = 200
# get dataset paths
filename_ratings, filename_movies = datasets_paths(
    MOVIELENS_SMALL_PATH+"ratings.csv", MOVIELENS_SMALL_PATH+"movies.csv")

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
# create pairs
movie_movie_df = create_pairs_data(
    user_movie_df, columns_dict=COLUMNS_MOVIE_DICT)
# replace ids to titles
col_names = parse_columns(COLUMNS_MOVIE_DICT)
movie_movie_df.replace({col_names.item_i: id_titles_dict,
                        col_names.item_j: id_titles_dict}, inplace=True)
# parse data as pairs
data_df, items_df = parse_pairs_data(
    movie_movie_df, columns_dict=COLUMNS_MOVIE_DICT)

# rate
winloss = ar.Winloss(normalization=True).rate(data_df, items_df,
                                              columns_dict=COLUMNS_MOVIE_DICT,
                                              sort=True)
colley = ar.Colley().rate(data_df, items_df,
                          columns_dict=COLUMNS_MOVIE_DICT, sort=True)
massey = ar.Massey().rate(data_df, items_df,
                          columns_dict=COLUMNS_MOVIE_DICT, sort=True)
keener = ar.Keener(normalization=True).rate(data_df, items_df,
                                            columns_dict=COLUMNS_MOVIE_DICT,
                                            sort=True)
od = ar.OffenseDefense(tol=0.0001).rate(data_df, items_df,
                                        columns_dict=COLUMNS_MOVIE_DICT, sort=True)
voting_losses_dict = {
    'ratings': {'VOTE': 10,
                'ITEM_I': col_names.points_i,
                'ITEM_J': col_names.points_i,
                'METHOD': 'VotingWithLosses'}
}
markov_vl = ar.Markov(stats_markov_dict=voting_losses_dict).rate(
    data_df, items_df,
    columns_dict=COLUMNS_MOVIE_DICT, sort=True)
voting_points_dict = {
    'ratings': {'VOTE': 10,
                'ITEM_I': col_names.points_i,
                'ITEM_J': col_names.points_i,
                'METHOD': 'WinnersAndLosersVotePoint'}
}
markov_vp = ar.Markov(stats_markov_dict=voting_points_dict).rate(
    data_df, items_df,
    columns_dict=COLUMNS_MOVIE_DICT, sort=True)

print_pandas(colley)
print_pandas(massey)
print_pandas(keener)
print_pandas(od)
print_pandas(markov_vl)
print_pandas(markov_vp)


ratings_dict = {
    'winloss': winloss.sort_values(by='Item'),
    'colley': colley.sort_values(by='Item'),
    'massey': massey.sort_values(by='Item'),
    'keener': keener.sort_values(by='Item'),
    'od': od.sort_values(by='Item'),
    'markov_vl': markov_vl.sort_values(by='Item'),
    'markov_vp': markov_vp.sort_values(by='Item')
}
kendall_tau_results = ar.metrics.kendall_tau_table(
    ratings_dict=ratings_dict, print_out=True)

print_info("RATING AGGREGATION: PERRON")
ra = ar.RatingAggregation(ratings.AGGREGATIONPERRON)

ratings_aggr_dict = {
    'Item': items_df['Item'].values,
    'winloss': winloss.sort_values(by='Item').rating.values,
    'colley': colley.sort_values(by='Item').rating.values,
    'massey': massey.sort_values(by='Item').rating.values,
    'keener': keener.sort_values(by='Item').rating.values,
    'od': od.sort_values(by='Item').rating.values,
    'markov_vl': markov_vl.sort_values(by='Item').rating.values,
    'markov_vp': markov_vp.sort_values(by='Item').rating.values
}

columns_dict = {'item': 'Item',
                'ratings': list(ratings_aggr_dict.keys()-{'Item'})}
data_df = pd.DataFrame.from_dict(ratings_aggr_dict)
movies_aggr_df = ra.rate(
    data_df, items_df, columns_dict=columns_dict, sort=True)
print_pandas(movies_aggr_df)
