"""Small example for rating/ranking movies"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

import apprate.ratings as ar
import pandas as pd
from apprate.datasets.filenames import FILENAME_MOVIES_EXAMPLE, dataset_path
from apprate.datasets.parse import create_pairs_data, parse_pairs_data
from apprate.utils.methods import parse_columns

filename = dataset_path(FILENAME_MOVIES_EXAMPLE)
user_movie_df = pd.read_csv(filename, index_col='User')

COLUMNS = {'item_i': 'MovieI',
           'item_j': 'MovieJ',
           'points_i': 'RatingI',
           'points_j': 'RatingJ'
           }
col_names = parse_columns(COLUMNS)
movie_movie_df = create_pairs_data(user_movie_df, columns_dict=COLUMNS)
# print(movie_movie_df)
data_df, items_df = parse_pairs_data(movie_movie_df, columns_dict=COLUMNS)
colley = ar.Colley().rate(data_df, items_df, columns_dict=COLUMNS)
massey = ar.Massey().rate(data_df, items_df, columns_dict=COLUMNS)
winloss = ar.Winloss(normalization=True).rate(data_df, items_df,
                                              columns_dict=COLUMNS,
                                              sort=True)
colley = ar.Colley().rate(data_df, items_df,
                          columns_dict=COLUMNS, sort=True)
massey = ar.Massey().rate(data_df, items_df,
                          columns_dict=COLUMNS, sort=True)
keener = ar.Keener(normalization=True).rate(data_df, items_df,
                                            columns_dict=COLUMNS,
                                            sort=True)
od = ar.OffenseDefense(tol=0.001).rate(data_df, items_df,
                                       columns_dict=COLUMNS, sort=True)
voting_losses_dict = {
    'ratings': {'VOTE': 10,
                'ITEM_I': col_names.points_i,
                'ITEM_J': col_names.points_i,
                'METHOD': 'VotingWithLosses'}
}
markov_vl = ar.Markov(stats_markov_dict=voting_losses_dict).rate(
    data_df, items_df,
    columns_dict=COLUMNS, sort=True)
voting_points_dict = {
    'ratings': {'VOTE': 10,
                'ITEM_I': col_names.points_i,
                'ITEM_J': col_names.points_i,
                'METHOD': 'WinnersAndLosersVotePoint'}
}
markov_vp = ar.Markov(stats_markov_dict=voting_points_dict).rate(
    data_df, items_df,
    columns_dict=COLUMNS, sort=True)

print("\ncolley\n", colley)
print("\nmassey\n", massey)
print("\nkeener\n", keener)
print("\nod\n", od)
print("\nmarkov-VotingWithLosses\n", markov_vl)
print("\nmarkov-WinnersAndLosersVotePoint\n", markov_vp)
