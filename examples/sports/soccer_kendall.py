"""Ranking Lists comparisons"""

import ratingslib.ratings as rl
import pandas as pd
from ratingslib.datasets.filenames import (FILENAME_EPL_2018_2019_20_GAMES,
                                           dataset_path)
from ratingslib.ratings.methods import rating_systems_to_dict
from ratingslib.ratings.metrics import kendall_tau_table
from ratingslib.utils.enums import ratings
from ratingslib.utils.methods import print_info, print_pandas

# ==========================================================================
# Rating teams for the FILENAME_EPL_2018_2019_20_GAMES that contains the first
# two match weeks of EPL 2018-2019
# ==========================================================================

stats_markov_dict = {
    'TW': {'VOTE': 10, 'ITEM_I': 'FTHG', 'ITEM_J': 'FTAG',
           'METHOD': 'VotingWithLosses'},
    'TG': {'VOTE': 10, 'ITEM_I': 'FTHG', 'ITEM_J': 'FTAG',
           'METHOD': 'WinnersAndLosersVotePoint'},
    'TST': {'VOTE': 10, 'ITEM_I': 'HST', 'ITEM_J': 'AST',
            'METHOD': 'WinnersAndLosersVotePoint'},
    'TS': {'VOTE': 10, 'ITEM_I': 'HS', 'ITEM_J': 'AS',
           'METHOD': 'WinnersAndLosersVotePoint'},
}
attributes_votes = {'TW': 10.0, 'TG': 10.0, 'TST': 10.0, 'TS': 10.0}

ratings_list = [
    rl.Winloss(normalization=False),
    rl.Colley(),
    rl.Massey(),
    rl.Elo(version=ratings.ELOWIN, K=40, ks=400, HA=0,
           starting_point=0),
    rl.Elo(version=ratings.ELOPOINT, K=40, ks=400, HA=0,
           starting_point=0),
    rl.Keener(normalization=False),
    rl.OffenseDefense(tol=0.0001),
    rl.Markov(b=0.85, stats_markov_dict=stats_markov_dict),
    rl.AccuRate()
]
ratings_dict = rating_systems_to_dict(ratings_list, key_based_on='version')

filename = dataset_path(FILENAME_EPL_2018_2019_20_GAMES)
# Dictionary that maps column names of csv files
COLUMNS_DICT = {
    'item_i': 'HomeTeam',
    'item_j': 'AwayTeam',
    'points_i': 'FTHG',
    'points_j': 'FTAG',
    'ts_i': 'HS',
    'ts_j': 'AS',
    'tst_i': 'HST',
    'tst_j': 'AST',
}
rating_values_dict = {key: r.rate_from_file(
    filename, columns_dict=COLUMNS_DICT) for key, r in ratings_dict.items()}

# Print Ratings and Rankings
print_info("Rating and ranking results")
s = "-" * 100
pd.set_option('float_format', "{:.4f}".format)
for k, r in rating_values_dict.items():
    print(k)
    print_pandas(r)
    print(s)


# ==========================================================================
# Kendall Tau comparison of ranking lists.
# ==========================================================================
print_info("Kendall Tau comparison of ranking lists")
kendall_tau_results = kendall_tau_table(
    ratings_dict=rating_values_dict, print_out=True)
print("\n", s)
