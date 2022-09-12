"""Soccer outcome predictions: Colley, teams statistics, NaiveBayes"""


from ratingslib.app_sports.methods import (Predictions,
                                           prepare_sports_seasons)
from ratingslib.application import SoccerOutcome
from ratingslib.datasets.filenames import get_seasons_dict_footballdata_online
from ratingslib.datasets.parameters import championships
from ratingslib.ratings.colley import Colley
from sklearn.naive_bayes import GaussianNB

outcome = SoccerOutcome()
filenames_dict = get_seasons_dict_footballdata_online(
    season_start=2009, season_end=2010, championship=championships.PREMIERLEAGUE)

colley = Colley()
stats_attributes = {
    'TG': {'ITEM_I': 'FTHG', 'ITEM_J': 'FTAG', 'TYPE': 'POINTS'},
    'TST': {'ITEM_I': 'HST', 'ITEM_J': 'AST', 'TYPE': 'POINTS'},
}
data = prepare_sports_seasons(filenames_dict,
                              outcome,
                              rating_systems=colley,
                              start_week=2,
                              stats_attributes=stats_attributes)
# we use normalized ratings
# => for Home = H + ratingnorm + key = HratingnormColley
# => for Away = A + ratingnorm + key = AratingnormColley
features_names = ['HratingnormColley',
                  'AratingnormColley',
                  'HTG', 'ATG', 'HTST', 'ATST']
test_y, pred = Predictions(data, outcome, start_from_week=4).ml_pred(
    clf=GaussianNB(), features_names=features_names)
