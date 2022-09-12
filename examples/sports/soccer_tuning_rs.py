"""Tuning ELO parameters"""

from ratingslib.app_sports.methods import Predictions, prepare_sports_seasons
from ratingslib.application import SoccerOutcome
from ratingslib.datasets.filenames import dataset_sports_path
from ratingslib.datasets.parameters import championships
from ratingslib.ratings.elo import Elo
from ratingslib.utils.enums import ratings
from sklearn.naive_bayes import GaussianNB

outcome = SoccerOutcome()
filename = dataset_sports_path(
    season=2009, championship=championships.PREMIERLEAGUE)

version_list = [ratings.ELOWIN, ratings.ELOPOINT]
ratings_dict = Elo.prepare_for_gridsearch_tuning(version_list=version_list,
                                                 k_range=[10, 20],
                                                 ks_range=[100, 200],
                                                 HA_range=[70, 80])

data = prepare_sports_seasons(filename,
                              outcome,
                              rating_systems=ratings_dict,
                              start_week=2)

prediction_methods = [GaussianNB(), 'MLE', 'RANK']
print()
for predict_with in prediction_methods:
    best = Predictions(data, outcome, start_from_week=4, print_accuracy_report=False).rs_tuning_params(
        ratings_dict=ratings_dict, predict_with=predict_with,
        metric_name='accuracy')
