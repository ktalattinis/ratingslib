"""Small example for rating/ranking movies"""

import ratingslib.ratings as rl
import pandas as pd
from ratingslib.datasets.filenames import FILENAME_MOVIES_EXAMPLE, dataset_path
from ratingslib.datasets.parse import create_pairs_data, parse_pairs_data
from ratingslib.utils.methods import parse_columns

filename = dataset_path(FILENAME_MOVIES_EXAMPLE)
user_movie_df = pd.read_csv(filename, index_col='User')

COLUMNS = {'item_i': 'MovieI',
           'item_j': 'MovieJ',
           'points_i': 'RatingI',
           'points_j': 'RatingJ'
           }
col_names = parse_columns(COLUMNS)
movie_movie_df = create_pairs_data(user_movie_df, columns_dict=COLUMNS)

data_df, items_df = parse_pairs_data(movie_movie_df, columns_dict=COLUMNS)
# rate
colley = rl.Colley().rate(data_df, items_df,
                          columns_dict=COLUMNS, sort=True)
massey = rl.Massey().rate(data_df, items_df,
                          columns_dict=COLUMNS, sort=True)
keener = rl.Keener(normalization=True).rate(data_df, items_df,
                                            columns_dict=COLUMNS,
                                            sort=True)
od = rl.OffenseDefense(tol=0.001).rate(data_df, items_df,
                                       columns_dict=COLUMNS, sort=True)

print("\ncolley\n", colley)
print("\nmassey\n", massey)
print("\nkeener\n", keener)
print("\nod\n", od)
