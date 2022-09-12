"""Soccer predictions for EPL 2009-2018 seasons: ratings - Naive Bayes"""

import ratingslib.ratings as rl
from ratingslib.app_sports.methods import (Predictions, prepare_sports_seasons,
                                           rating_norm_features)
from ratingslib.application import SoccerOutcome
from ratingslib.datasets.filenames import get_seasons_dict_footballdata_online
from ratingslib.datasets.parameters import championships, stats
from ratingslib.utils.enums import ratings
from sklearn.naive_bayes import GaussianNB

outcome = SoccerOutcome()
filenames_dict = get_seasons_dict_footballdata_online(
    season_start=2009, season_end=2018,
    championship=championships.PREMIERLEAGUE)

ratings_list = [
    rl.Winloss(normalization=True),
    rl.Colley(),
    rl.Massey(data_limit=20),
    rl.Elo(version=ratings.ELOWIN, K=40, ks=400, HA=0,
           starting_point=0),
    rl.Elo(version=ratings.ELOPOINT, K=40, ks=400, HA=0,
           starting_point=0),
    rl.Keener(normalization=True),
    rl.OffenseDefense(tol=0.0001),
    rl.Markov(b=0.85, stats_markov_dict=stats.STATS_MARKOV_DICT),
    rl.AccuRate()
]

data = prepare_sports_seasons(filenames_dict,
                              outcome,
                              rating_systems=ratings_list,
                              start_week=2)
features_names_list = rating_norm_features(ratings_list)

results = Predictions(data, outcome, start_from_week=4).ml_pred_parallel(
    clf_list=[GaussianNB()], features_names_list=features_names_list)
