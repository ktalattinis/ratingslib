"""Tuning number of neighbors KNN"""

from ratingslib.app_sports.methods import (Predictions, prepare_sports_seasons,
                                           rating_norm_features)
from ratingslib.application import SoccerOutcome
from ratingslib.datasets.filenames import get_seasons_dict_footballdata_online
from ratingslib.datasets.parameters import championships
from ratingslib.ratings.colley import Colley
from sklearn.neighbors import KNeighborsClassifier

outcome = SoccerOutcome()
filename = get_seasons_dict_footballdata_online(
    season_start=2009, season_end=2010, championship=championships.PREMIERLEAGUE)

colley = Colley()
data = prepare_sports_seasons(filename,
                              outcome,
                              rating_systems=colley,
                              start_week=2)

features_names = rating_norm_features(colley)
clf_list = [KNeighborsClassifier(n_neighbors=n) for n in range(7, 19, 2)]
best = Predictions(data, outcome, start_from_week=4, print_accuracy_report=False).ml_tuning_params(clf_list=clf_list,
                                                                                                   features_names=features_names,
                                                                                                   metric_name='f1',
                                                                                                   average='weighted')
