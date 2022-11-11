"""Soccer hindsight and foresight predictions"""

import ratingslib.ratings as rl
import pandas as pd
from ratingslib.app_sports.methods import (Predictions, predict_hindsight,
                                           prepare_sport_dataset,
                                           show_list_of_accuracy_results)
from ratingslib.application import SoccerOutcome
from ratingslib.datasets.filenames import (FILENAME_EPL_2018_2019_3RD_WEEK,
                                           FILENAME_EPL_2018_2019_20_GAMES,
                                           datasets_paths)
from ratingslib.datasets.parameters import (COLUMNS_DICT, DATE_COL, WEEK_PERIOD,
                                            stats)
from ratingslib.datasets.parse import parse_pairs_data
from ratingslib.ratings.methods import rating_systems_to_dict
from ratingslib.utils.enums import ratings
from ratingslib.utils.methods import print_info

# rating systems
ratings_list = [
    rl.Winloss(normalization=False),
    rl.Colley(),
    rl.Massey(data_limit=10),
    rl.Elo(version=ratings.ELOWIN, K=40, ks=400, HA=0,
           starting_point=0),
    rl.Elo(version=ratings.ELOPOINT, K=40, ks=400, HA=0,
           starting_point=0),
    rl.Keener(normalization=False),
    rl.OffenseDefense(tol=0.0001),
    rl.Markov(b=0.85, stats_markov_dict=stats.STATS_MARKOV_DICT),
    rl.AccuRate(),
]
# convert list to dictionary
ratings_dict = rating_systems_to_dict(ratings_list)
# get train and test filenames
filename, filename_test = datasets_paths(
    FILENAME_EPL_2018_2019_20_GAMES, FILENAME_EPL_2018_2019_3RD_WEEK)

# set columns dictionry and outcome
columns_dict = COLUMNS_DICT
outcome = SoccerOutcome()

# prediction methods
pred_methods_list = ['MLE', 'RANK']

# parse train and test data
data_train, teams_df = parse_pairs_data(filename,
                                        parse_dates=[DATE_COL],
                                        frequency=WEEK_PERIOD,
                                        outcome=outcome,
                                        columns_dict=columns_dict)

# ==========================================================================
# HINDSIGHT PREDICTIONS
# ==========================================================================

print_info("HINDSIGHT RESULTS")
for pm in pred_methods_list:
    pred_list = []
    test_Y_list = []
    for rs in ratings_list:
        pred, test_y = predict_hindsight(
            data_train, rs.rate_from_file(filename), outcome, pm)
        test_Y_list.append(test_y)
        pred_list.append(pred)
    print_info(pm)
    show_list_of_accuracy_results(
        list(ratings_dict.keys()), test_Y_list, pred_list, print_predictions=True)

# ==========================================================================
# FORESIGHT PREDICTIONS
# ==========================================================================

# the test file start from the 3rd week (start_period = 3)
data_test, teams_df = parse_pairs_data(filename_test,
                                       parse_dates=[DATE_COL],
                                       frequency=WEEK_PERIOD,
                                       outcome=outcome,
                                       start_period_from=3,
                                       columns_dict=columns_dict)


# concat 2 files
data_all = pd.concat([data_train, data_test]).reset_index(drop=True)
# prepare the dataset
data = prepare_sport_dataset(data_season=data_all,
                             teams_df=teams_df,
                             rating_systems=ratings_dict,
                             start_week=2,
                             columns_dict=columns_dict)

print_info("FORESIGHT RESULTS")
# make foresight predictions for the 3rd week (start_from_week=3)
results = Predictions(data, outcome, start_from_week=3,
                      print_accuracy_report=True,
                      print_predictions=True,
                      print_classification_report=False).rs_pred_parallel(
    rating_systems=ratings_dict,
    pred_methods_list=pred_methods_list)
