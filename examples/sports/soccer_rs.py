"""Predictions based only on rating values"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

import apprate.ratings as ar
import pandas as pd
from apprate.app_sports.methods import Predictions, prepare_sports_seasons
from apprate.application import SoccerOutcome
from apprate.datasets.filenames import get_seasons_dict_footballdata_online
from apprate.datasets.parameters import stats
from apprate.utils.enums import ratings

pd.set_option('float_format', "{:.4f}".format)

filenames_dict = get_seasons_dict_footballdata_online(
    season_start=2009, season_end=2018, championship='EPL')

ratings_list = [
    ar.Winloss(normalization=True),
    ar.Colley(),
    ar.Massey(data_limit=20),
    ar.Elo(version=ratings.ELOWIN, K=40, ks=400, HA=0,
           starting_point=0),
    ar.Elo(version=ratings.ELOPOINT, K=40, ks=400, HA=0,
           starting_point=0),
    ar.Keener(normalization=True),
    ar.OffenseDefense(tol=0.0001),
    ar.Markov(b=0.85, stats_markov_dict=stats.STATS_MARKOV_DICT),
    ar.AccuRate()
]
outcome = SoccerOutcome()
data = prepare_sports_seasons(filenames_dict,
                              outcome,
                              rating_systems=ratings_list,
                              start_week=2)


results = Predictions(data, outcome, start_from_week=4).rs_pred_parallel(
    rating_systems=ratings_list,
    pred_methods_list=['MLE', 'RANK'])
