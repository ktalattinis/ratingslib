"""Soccer outcome prediction: AccuRATE - Naive Bayes"""


from ratingslib.app_sports.methods import Predictions, prepare_sports_seasons
from ratingslib.application import SoccerOutcome
from ratingslib.datasets.filenames import get_seasons_dict_footballdata_online
from ratingslib.datasets.parameters import championships
from ratingslib.ratings.accurate import AccuRate
from sklearn.naive_bayes import GaussianNB

outcome = SoccerOutcome()
ratings_dict = {'AccuRATE': AccuRate()}
filenames_dict = get_seasons_dict_footballdata_online(
    season_start=2009, season_end=2010, championship=championships.PREMIERLEAGUE)
data_ml = prepare_sports_seasons(filenames_dict,
                                 outcome,
                                 rating_systems=ratings_dict,
                                 start_week=2)

features_names = ['HratingnormAccuRATE', 'AratingnormAccuRATE']
# or features_names = rating_norm_features(ratings_dict)
test_y, pred = Predictions(data_ml,
                           outcome,
                           start_from_week=4,
                           print_accuracy_report=True).ml_pred(
    clf=GaussianNB(), features_names=features_names)
