"""Predictions of sport outcome without backtester"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT


from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
import pandas as pd
import sklearn
from ratingslib.application import SportOutcome
from ratingslib.datasets.parse import parse_pairs_data
from ratingslib.datasets.preprocess import BasicPreprocess, Preprocess
from ratingslib.datasets.soccer import (DATE_COL, DAY_FIRST, WEEK_PERIOD)
from ratingslib.ratings.methods import (calc_items_stats, normalization_rating,
                                        rating_systems_to_dict)
from ratingslib.ratings.rating import RatingSystem

from ratingslib.utils.methods import (create_items_dict, get_indices,
                                      parse_columns, print_info, print_loading,
                                      print_pandas, str_info)
from ratingslib.utils.validation import (ValidationError, validate_from_set,
                                         validate_type, validate_type_of_elements)
from joblib import Parallel, delayed
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, get_scorer)
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None  # default='warn'

# ==============================================================================
# PREDICTION OF SPORT OUTCOME FUNCTIONS
# ==============================================================================


def predict_hindsight(data: pd.DataFrame,
                      teams_rating_df: pd.DataFrame,
                      outcome: SportOutcome,
                      pred_method: Literal['MLE', 'RANK'] = 'RANK',
                      columns_dict: Optional[Dict[str, Any]] = None
                      ) -> Tuple[list, list]:
    """Hindsight prediction refers to predicting past games using the ratings
    of entire games.

    Parameters
    ----------
    data : pd.DataFrame
        Data of games

    teams_rating_df : pd.DataFrame
        Rating values of teams. Note that 'rating' column must be in the
        DataFrame columns

    outcome : SportOutcome
        The `outcome` parameter is associated with application type
        e.g. for soccer the type of outcome is
        :class:`ratingslib.application.SoccerOutcome`.
        For more details see :mod:`ratingslib.application` module.

    pred_method : Literal['RANK', 'MLE'], default='RANK'
        Two available methods for predictions: 'RANK' or 'MLE'
        More details at :class:`ratingslib.application.SoccerOutcome`

    columns_dict : Optional[Dict[str, str]], default=None
        A dictionary mapping the column names of the dataset.
        See the module :mod:`ratingslib.datasets.parameters` for more details

    Returns
    -------
    Tuple[list, list]
        pred : List of predictions
        Y : Correct outcome values

    """

    col_names = parse_columns(columns_dict)
    home_col_ind, away_col_ind, outcome_col_index = get_indices(
        col_names.item_i, col_names.item_j, outcome.name, data=data)

    teams_dict = create_items_dict(teams_rating_df)

    data_np = data.to_numpy()
    Y = data[outcome.name].values
    X = []
    column_rating = 'rating'
    if (teams_rating_df[column_rating] == teams_rating_df[column_rating][0]).all():
        return [], []
    if pred_method == 'MLE':
        column_rating = 'norm' + column_rating
        teams_rating_df[column_rating] = normalization_rating(teams_rating_df,
                                                              'rating')
    for row in data_np:
        i = teams_dict[row[home_col_ind]]
        j = teams_dict[row[away_col_ind]]
        rating_home = teams_rating_df.iloc[i][column_rating]
        rating_away = teams_rating_df.iloc[j][column_rating]
        X.append([rating_home, rating_away])
    # print(X)
    if pred_method == 'RANK':
        pred, _ = outcome.fit_and_predict(
            None, None, X, method=pred_method)
    elif pred_method == 'MLE':
        pred, _ = outcome.fit_and_predict(
            X, Y, X, method=pred_method)
    return pred, Y


def accuracy_results(test_Y: list, predictions: list) -> Tuple[float, int]:
    """Returns the accuracy results in a percentage and as
    correctly classified samples.

    Parameters
    ----------
    test_Y : list
        Ground truth (correct) labels.

    predictions : list
        Predicted labels

    Returns
    -------
    accuracy : float
        Accuracy metric

    correct : int
        Correctly classified samples
    """
    accuracy = accuracy_score(test_Y, predictions)
    correct = accuracy_score(test_Y, predictions, normalize=False)
    return accuracy, correct


def show_list_of_accuracy_results(names_list: List[str], test_Y: list, predictions_list: list, print_predictions: bool):
    """Show accuracy results for a list of models

    Parameters
    ----------
    names_list : List[str]
        Model name list

    test_Y : list
        List of correct labels

    predictions_list : list
        List that contains lists of predicted labels

    print_predictions : bool
        If True then predictions are printed
    """
    if len(names_list) != len(predictions_list):
        ValidationError(
            'name list and prediction list should have the same size')
    results_list = []
    for i, name in enumerate(names_list):
        if len(predictions_list[i]) != 0:
            accuracy, correct = accuracy_results(
                test_Y[i], predictions_list[i])
            if print_predictions:
                pred = [str(int(item[0]))+"/"+str(int(item[1]))
                        for item in tuple(zip(predictions_list[i], test_Y[i]))]
            wrong_games = len(test_Y[i])-correct
        else:
            accuracy, correct, pred, wrong_games = 'NA', 'NA', 'NA', 'NA'
        result = [accuracy, correct, wrong_games, len(test_Y[i])]
        if print_predictions:
            result.append(pred)
        results_list.append(result)
    columns = ["Accuracy", "Correct Games", "Wrong Games", "Total Games"]
    if print_predictions:
        columns.append("Predictions")

    print_info("Accuracy results")
    print_pandas(pd.DataFrame(results_list,
                              columns=columns,
                              index=names_list))
    if print_predictions:
        notation = ("\n *** Predictions columns notation: (Predicted / Actual), " +
                    "1 = Home Win, 2 = Away Win, 3 = Draw")
        print(notation)


def classification_details(name: str, test_Y: list, pred: list) -> str:
    """Return classification details for a prediction model based on
    truth labels and predictions

    Parameters
    ----------
    name : str
        Name of prediction model

    test_Y : list
        List of correct labels

    pred : list
        List of predictions

    Returns
    -------
    str
        Classification details as string
    """
    result = f'No predictions made from: [{name}]'
    if len(pred) != 0:
        correct = accuracy_score(test_Y, pred, normalize=False)
        total = len(pred)
        wrong = total - correct
        result = ""
        result += str_info(name) + '\n'
        result += classification_report(test_Y, pred, digits=4) + '\n'
        result += 'confusion matrix:\n' + \
            np.array2string(confusion_matrix(test_Y, pred), prefix="\n") + '\n'
        result += 'Correct games: ' + str(correct) + '\n'
        result += 'Wrong games: ' + str(wrong) + '\n'
        result += 'Total predicted Games: ' + str(total) + '\n'
    return result


class Predictions():
    """Class for predict soccer match results

    Parameters
    ----------
    data : Union[Dict[int, pd.DataFrame], pd.DataFrame]
        Data of games in a dictionary or in a DataFrame. If dictionary passed
        then the key is the season and value is the data.

    outcome : SportOutcome
        The ``outcome`` parameter is related with
        application type. For sports application it must be an instance
        of subclass of SportOutcome class.
        e.g. for soccer the type of outcome is
        :class:`ratingslib.application.SoccerOutcome`.
        For more details see :mod:`ratingslib.application` module.

    pred_method : Union[Literal['MLE', 'RANK'], sklearn.base.BaseEstimator]
        Three available methods for predictions: 'RANK' or 'MLE'
        or a scikit classifier.
        More details for 'RANK' or 'MLE' at :class:`ratingslib.application.SoccerOutcome`

    features_names : List[str]
        List of feature names (each name refer to a column of the data)

    data_test : Optional[pd.DataFrame], default=None
        The test set. If data_test is passed then split and start_from_week
        parameters are ignored

    split: Optional[Union[float, int]], default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples.

    start_from_week : Optional[int], default=None
        The match week that the walk-forward procedure starts predictions

    walk_forward_window_size : int, default = -1
        Only valid if week is not ``None``.
        If ``-1`` then walk-forward procedure will not run.
        For example if walk_forward_window_size is ``1`` then
        the window size of walk-forward is one week.

    columns_dict : Optional[Dict[str, Any], default=None
        The column names of data file.
        See ``ratingslib.datasets.parameters.COLUMNS_DICT`` for more
        details.

    print_accuracy_report : bool, default=True
        If `True` accuracy report will be printed

    print_classification_report : bool, default=True
        If `True`, the classification report will be printed

    print_predictions : bool, default=False
        If `True`, the predictions will be printed
    """

    def __init__(self, data: Union[Dict[int, pd.DataFrame], pd.DataFrame],
                 outcome: SportOutcome,
                 data_test: Optional[pd.DataFrame] = None,
                 split: Optional[Union[float, int]] = None,
                 start_from_week: Optional[int] = None,
                 walk_forward_window_size: int = -1,
                 columns_dict: Optional[Dict[str, Any]] = None,
                 print_accuracy_report: bool = True,
                 print_classification_report: bool = False,
                 print_predictions: bool = False):
        validate_type(walk_forward_window_size, int,
                      'walk_forward_window_size')
        if split is not None:
            validate_type(split, float, 'split')
        if start_from_week is not None:
            validate_type(start_from_week, int, 'start_from_week')
        validate_type(outcome, SportOutcome, 'outcome')

        self.data = data
        self.outcome = outcome
        self.data_test = data_test
        self.split = split
        self.start_from_week = start_from_week
        self.walk_forward_window_size = walk_forward_window_size
        self.columns_dict = columns_dict
        self.print_accuracy_report = print_accuracy_report
        self.print_classification_report = print_classification_report
        self.print_predictions = print_predictions

    def _select_X_Y(self, data: pd.DataFrame,
                    features: List[str],
                    col_names: SimpleNamespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Selects from data the given features. In this function we remove the
        non-rated weeks.
        Non rated weeks is the case where all instances have the same value
        (e.g. massey case: sometimes massey rating system requires more data
        and as a result it starts from 4th week to rate teams. This means
        that the second and third week have rating 0 for all teams.
        First is not included if we have selected to remove it during
        preprocess

        Parameters
        ----------
        data : pd.DataFrame
            Games data

        features : List[str]
            List of feature names (each name refer to a column of the data)

        col_names : SimpleNamespace
            A simple object subclass that provides attribute access to its
            namespace. The attributes are the keys of :attr:`columns_dict`.

        Returns
        -------
        data_X : pandas.DataFrame
            Dataset that includes only the features after removing
            not rated weeks, if they have found.

        data_Y : pandas.DataFrame
            Dataset that contains only the outcomes after removing
            not rated weeks, if they have found.

        data : pandas.DataFrame
            Dataset after removing not rated weeks. If non rated weeks not
            found returns the dataset without any changes.
        """
        non_rated_flag = False
        data_X = data[features]
        data_Y = data[self.outcome.name]
        if col_names.period_number in data.columns:
            week_col = col_names.period_number
            min_week = data[col_names.period_number].min()
            max_week = data[col_names.period_number].max()
            for week in range(min_week, max_week+1):
                data_week_array = data_X[data[week_col] == week].to_numpy()
                data_week_flatten_array = np.ravel(data_week_array)
                if np.all(data_week_array == data_week_flatten_array[0]):
                    # only the first time we create a deep copy
                    if non_rated_flag is False:
                        data_X_mod = data_X.copy(deep=True)
                        data_Y_mod = data_Y.copy(deep=True)
                        data_mod = data.copy(deep=True)
                        non_rated_flag = True
                    data_X_mod = data_X_mod[data_mod[
                        col_names.period_number] != week]
                    data_Y_mod = data_Y_mod[data_mod[
                        col_names.period_number] != week]
                    data_mod = data_mod[data_mod[
                        col_names.period_number] != week]
                else:
                    # we check only for the first weeks, if a rated week is found
                    # then we stop searching for non_rated_flag
                    break
        # print(non_rated_flag)
        if not non_rated_flag:
            return data_X, data_Y, data
        else:
            data_X_mod.reset_index(inplace=True, drop=True)
            data_Y_mod.reset_index(inplace=True, drop=True)
            data_mod.reset_index(inplace=True, drop=True)
            return data_X_mod, data_Y_mod, data_mod

    def _predict(self, pred_method: Union[Literal['MLE', 'RANK'],
                                          sklearn.base.BaseEstimator],
                 train_X: pd.DataFrame,
                 train_Y: pd.Series,
                 test_X: pd.DataFrame) -> tuple:
        """Train first according to the given method and then predict

        Parameters
        ----------
        pred_method : Union[Literal['MLE', 'RANK'], sklearn.base.BaseEstimator]
            Three available methods for predictions: 'RANK' or 'MLE' or a scikit classifier
            More details for 'RANK' or 'MLE' at :class:`ratingslib.application.SoccerOutcome`

        train_X : pd.DataFrame
            The training set that includes only the features

        train_Y : pd.Series
            The outcome labels of training set

        test_X : pd.DataFrame
            The outcome labels of test set

        Returns
        -------
        tuple
            The predictions for the target outcome and the
            predictions distribution

        """
        predictions = []
        pred_prob = []
        if isinstance(pred_method, str):
            predictions, pred_prob = self.outcome.fit_and_predict(
                train_X.values, train_Y.values, test_X.values, pred_method)
        else:
            if len(train_X) != 0 and len(train_Y) != 0:
                pred_method = pred_method.fit(train_X, train_Y)
                predictions = pred_method.predict(test_X).tolist()
                pred_prob = pred_method.predict_proba(test_X).tolist()
        return predictions, pred_prob

    def _train_and_test(self, *,
                        pred_method: Union[Literal['MLE', 'RANK'],
                                           sklearn.base.BaseEstimator],
                        features_names: List[str]
                        ) -> tuple:
        """Training and testing based on the given method

        Parameters
        ----------
        pred_method : Union[Literal['MLE', 'RANK'], sklearn.base.BaseEstimator]
            Three available methods for predictions: 'RANK' or 'MLE' or a scikit classifier
            More details for 'RANK' or 'MLE' at :class:`ratingslib.application.SoccerOutcome`

        features_names : List[str]
            List of feature names (each name refer to a column of the data)

        Returns
        -------
        tuple
            test_Y (The outcome labels of test set) and predictions
        """
        col_names = parse_columns(self.columns_dict)
        data = self.data
        if isinstance(self.data, pd.DataFrame):
            data = {1: self.data}
        seasons = list(data.keys())
        season_start = min(seasons)
        season_end = max(seasons)
        pred_all = []
        test_Y_all = []
        for season in range(season_start, season_end+1):
            data_season = data[season]  # .copy(deep=True)
            data_X, data_Y, data_season = self._select_X_Y(
                data_season.copy(deep=True), features_names, col_names)
            if not data_season.empty:
                if self.data_test is not None:
                    train_X = data_X
                    train_Y = data_Y
                    test_X = self.data_test[features_names]
                    test_Y = self.data_test[self.outcome.name]
                    pred, pred_prob = self._predict(
                        pred_method, train_X,  train_Y, test_X)
                elif self.split is not None:
                    train_X, test_X, train_Y, test_Y = train_test_split(
                        data_X, data_Y, test_size=self.split, shuffle=False)
                    pred, pred_prob = self._predict(
                        pred_method, train_X,  train_Y, test_X)
                elif self.start_from_week is not None:
                    if self.walk_forward_window_size == -1:
                        self.walk_forward_window_size = 1
                    week_col = col_names.period_number
                    max_week = data_season[col_names.period_number].max()
                    pred = []
                    if self.walk_forward_window_size == 0:
                        train_X = data_X[data_season[week_col]
                                         < self.start_from_week]
                        train_Y = data_Y[data_season[week_col]
                                         < self.start_from_week]
                        test_X = data_X[data_season[week_col]
                                        >= self.start_from_week]
                        test_Y = data_Y[data_season[week_col]
                                        >= self.start_from_week]
                        pred, pred_prob = self._predict(
                            pred_method, train_X,  train_Y, test_X)
                    else:
                        # all test_Y for report
                        # test_Y = data_Y[data_season[week_col] >= start_from_week]
                        test_Y = pd.Series(dtype=int)
                        for w in range(self.start_from_week, max_week+1, self.walk_forward_window_size):
                            train_X = data_X[data_season[week_col] < w]
                            train_Y = data_Y[data_season[week_col] < w]
                            test_X = data_X[data_season[week_col] == w]
                            test_Y_current_week = data_Y[data_season[week_col] == w]
                            # if len(train_Y) != 0:
                            predictions, pred_prob = self._predict(
                                pred_method, train_X, train_Y, test_X)
                            if len(predictions) != 0:
                                pred.extend(predictions)
                                test_Y = pd.concat(
                                    [test_Y, test_Y_current_week], ignore_index=True)
                test_Y_all.extend(test_Y.to_list())
                pred_all.extend(pred)
        # test_Y = test_Y.to_list()
        pred_method_str = pred_method
        if len(pred_all) != 0:
            if isinstance(pred_method, sklearn.base.BaseEstimator):
                pred_method_str = self.classifier_features_repr(
                    pred_method, features_names)
            if self.print_classification_report:
                print(classification_details(
                    pred_method_str, test_Y_all, pred_all))
            if self.print_accuracy_report or self.print_predictions:
                print_info(pred_method_str)
                show_list_of_accuracy_results(
                    [pred_method_str], [test_Y_all], [pred_all], self.print_predictions)
        return test_Y_all, pred_all

    def classifier_features_repr(self, clf, feature_names):
        return str(clf) + "-[features: " + ' '.join(feature_names)+"]"

    def ml_pred(self, *,
                clf: sklearn.base.BaseEstimator,
                features_names: List[str],
                to_dict=False
                ) -> Union[Tuple[List, List], dict]:
        """Predict with ml classifiers

        Parameters
        ----------
        clf : sklearn.base.BaseEstimator
            A scikit classifier instance

        features_names : List[str]
            List of feature names (each name refer to a column of the data)

        to_dict : bool, default = False
            If `True` then results will be returned as a dictionary where
            the key is the name of clf

        Returns
        -------
        Union[Tuple[List, List], dict]
            Prediction results as tuple (test_Y, predictions) or
            dictionary {clf_repr: (test_Y, predictions)}
        """
        validate_type_of_elements(features_names, str)
        validate_type(clf, sklearn.base.BaseEstimator, 'clf')
        test_Y, pred = self._train_and_test(pred_method=clf,
                                            features_names=features_names)
        if not to_dict:
            return test_Y, pred
        else:
            clf_repr = self.classifier_features_repr(clf, features_names)
            return {clf_repr: (test_Y, pred)}

    def ml_pred_parallel(self, *,
                         clf_list: List[sklearn.base.BaseEstimator],
                         features_names_list: List[List[str]],
                         n_jobs: int = -1) -> dict:
        """Runs the ml predictions to test each one of the classifiers from the
        given list

        Parameters
        ----------
        clf_list : List[sklearn.base.BaseEstimator]
            List of scikit estimators

        features_names_list : List[List[str]]
            List that contains list of feature names
            (each name refer to a column of the data)

        n_jobs : int, default=-1
            Number of jobs to run in parallel.
            ``-1`` means using all processors.
            ``None`` means 1

        Returns
        -------
        dict
            Dictionary that maps classifier represenations to their
            test_Y (The outcome labels of test set) and predictions
        """
        classification_report_all = self.print_classification_report
        accuracy_report_all = self.print_accuracy_report
        print_predictions_all = self.print_predictions
        self.print_classification_report = False
        self.print_accuracy_report = False
        self.print_predictions = False
        validate_type_of_elements(clf_list, sklearn.base.BaseEstimator)
        parallel = Parallel(n_jobs=n_jobs, verbose=0)
        results = parallel(delayed(self.ml_pred)(clf=clf,
                                                 features_names=features_names,
                                                 to_dict=True)
                           for clf in clf_list
                           for features_names in features_names_list)
        results_dict = {
            list(r.keys())[0]: r[list(r.keys())[0]] for r in results}

        if classification_report_all:
            for key, (test_Y, pred) in results_dict.items():
                print(classification_details(key, test_Y, pred))

        if accuracy_report_all:
            show_list_of_accuracy_results(
                results_dict.keys(), [r[0] for r in results_dict.values()], [
                    r[1] for r in results_dict.values()],
                print_predictions=print_predictions_all)

        return results_dict

    def rs_pred(self, *,
                pred_method: Literal['MLE', 'RANK'],
                ratings: RatingSystem,
                to_dict: bool = False
                ) -> Union[Tuple[List, List], dict]:
        """Prediction with one of two available methods: MLE or RANK

        Parameters
        ----------
        pred_method : Literal['MLE', 'RANK']
            Two available methods for predictions: 'RANK' or 'MLE'
            More details at :class:`ratingslib.application.SoccerOutcome`

        ratings : RatingSystem
            Rating system instance

        to_dict : bool, default = False
            If `True` then results will be returned as a dictionary where
            the key is the name of pred_method

        Returns
        -------
        Union[Tuple[List, List], dict]
            prediction results as tuple (test_Y, predictions) or dictionary 
            {pred_name: (test_Y, predictions)}
        """
        if isinstance(pred_method, str):
            validate_from_set(pred_method, {'MLE', 'RANK'}, 'pred_method')
        else:
            raise ValidationError('pred_method should be string')
        ratingnorm = 'ratingnorm' if pred_method == 'MLE' else ''
        features_names = ['H'+ratingnorm+ratings.params_key,
                          'A'+ratingnorm+ratings.params_key]
        test_Y, pred = self._train_and_test(pred_method=pred_method,
                                            features_names=features_names)
        if not to_dict:
            return test_Y, pred
        else:
            return {pred_method: {ratings.params_key: (test_Y, pred)}}

    def rs_pred_parallel(self, *,
                         pred_methods_list: List[Literal['MLE', 'RANK']],
                         rating_systems: Union[Dict[str, RatingSystem],
                                               List[RatingSystem], RatingSystem],
                         n_jobs: int = -1,
                         ) -> dict:
        """Runs the rating prediction for each one from the methods in the
        given list

        Parameters
        ----------
        pred_methods_list : List[Literal['MLE', 'RANK']]
            List of prediction methods

        rating_systems : Dict[str, RatingSystem] or List[RatingSystem]]] or RatingSystem or None, default=None
            If type is dictionary then it is mapping names (or rating keys) to
            rating systems.
            If type is list of rating systems instances then it firstly converted
            to dictionary.
            If type is RatingSystem instance then it firstly converted
            to dictionary.
            If it is set to ``None`` then rating values are not included in data
            attributes for preparation.

        n_jobs : int, default=-1
            Number of jobs to run in parallel.
            ``-1`` means using all processors.
            ``None`` means 1

        Returns
        -------
        dict
            Dictionary that maps prediction name method to results per
            rating system

        """
        classification_report_all = self.print_classification_report
        accuracy_report_all = self.print_accuracy_report
        print_predictions_all = self.print_predictions
        self.print_classification_report = False
        self.print_accuracy_report = False
        self.print_predictions = False
        ratings_dict = rating_systems_to_dict(rating_systems)
        parallel = Parallel(n_jobs=n_jobs, verbose=0)
        results = parallel(delayed(self.rs_pred)(pred_method=pm,
                                                 ratings=ratings,
                                                 to_dict=True)
                           for pm in pred_methods_list
                           for ratings in ratings_dict.values())
        results_dict = {}
        for r in results:
            first_key = list(r.keys())[0]
            if first_key not in results_dict:
                results_dict[first_key] = r[first_key]
            else:
                first_of_first_key = list(r[first_key].keys())[0]
                results_dict[first_key][first_of_first_key] = r[first_key][first_of_first_key]

        if classification_report_all:
            for k, v in results_dict.items():
                for rkey, (test_Y, pred) in v.items():
                    print_info(f'{k} {rkey}')
                    if len(pred) != 0:
                        print(classification_report(test_Y, pred, digits=4))
                        print('confusion matrix:\n',
                              confusion_matrix(test_Y, pred))
                    else:
                        print('No predictions made')
        if accuracy_report_all:
            for k, v in results_dict.items():
                print_info(k)
                test_Y_list = [t[0] for t in v.values()]
                predictions_list = [pr[1] for pr in v.values()]
                show_list_of_accuracy_results(
                    v.keys(), test_Y_list, predictions_list,
                    print_predictions=print_predictions_all)
        return results_dict

    def rs_tuning_params(self, *,
                         ratings_dict: Dict[str, RatingSystem],
                         predict_with: Union[Literal['MLE', 'RANK'],
                                             sklearn.base.BaseEstimator],
                         use_norm_ratings: bool = True,
                         metric_name: str = 'accuracy',
                         maximize: bool = True,
                         print_out: bool = True,
                         **kwargs) -> dict:
        """Tuning of rating systems parameters for the given metric
        with grid-search.

        Parameters
        ----------
        ratings_dict : Dict[str, RatingSystem]
            Dictionary that maps names to ratings. Note that ratings are stored in
            a `pandas.DataFrame`.

        predict_with : Union[Literal['MLE', 'RANK'], sklearn.base.BaseEstimator]
            Three available methods for predictions: 'RANK' or 'MLE' or
            a scikit classifier.
            More details for 'RANK' or 'MLE' at :class:`ratingslib.application.SoccerOutcome`

        use_norm_ratings : bool, default=True
            if `True` then normalized rating values

        metric_name : str, default='accuracy'
            The optimization metric, available metrics name at
            https://scikit-learn.org/stable/modules/model_evaluation.html

        maximize : bool, default = True
            If True the maximize, else minimize

        print_out : bool, default=True
            Print results if True

        **kwargs : dict
            All keyword arguments are passed to _score_func of scikit

        Returns
        -------
        best : dict
            Dictionary that maps rating system versions with best values
        """
        version_list = set(rs.version for rs in ratings_dict.values())
        best = {v: {'params': None, metric_name: None} for v in version_list}
        scores = {v: {} for v in version_list}
        ratingsnorm = 'ratingnorm' if use_norm_ratings else ''

        for rs_key, rs in ratings_dict.items():
            if isinstance(predict_with, sklearn.base.BaseEstimator):
                test_y, pred = self.ml_pred(
                    clf=predict_with,
                    features_names=[
                        'H'+ratingsnorm+rs_key,
                        'A'+ratingsnorm+rs_key])
            else:
                test_y, pred = self.rs_pred(
                    pred_method=predict_with,
                    ratings=rs)
            scores[rs.version][rs_key] = get_scorer(
                metric_name)._score_func(test_y, pred, **kwargs)

        for v in version_list:
            if maximize:
                best_score = max(scores[v], key=scores[v].get)
            else:
                best_score = min(scores[v], key=scores[v].get)
            best[v]['params'] = best_score
            best[v][metric_name] = scores[v][best[v]['params']]
            if print_out:
                info = "Prediction method:"
                if isinstance(predict_with, sklearn.base.BaseEstimator):
                    print_info(info+" " + type(predict_with).__name__)
                else:
                    print_info(info+" " + predict_with)
                print(best[v]['params'], scores[v][best[v]['params']])
        return best

    def ml_tuning_params(self, *,
                         clf_list: List[sklearn.base.BaseEstimator],
                         features_names: List[str],
                         metric_name: str = 'accuracy',
                         maximize: bool = True,
                         print_out: bool = True,
                         n_jobs: int = -1,
                         **kwargs):
        """Tuning the classifiers hyper-parameters for
        the given metric with grid-search.

        Parameters
        ----------
        clf_list : List[sklearn.base.BaseEstimator]
            List of scikit estimators

        features_names_list : List[List[str]]
            List that contains list of feature names
            (each name refer to a column of the data)

        metric_name : str, default='accuracy'
            The optimization metric, available metrics name at
            https://scikit-learn.org/stable/modules/model_evaluation.html

        maximize : bool, default = True
            If True the maximize, else minimize

        print_out : bool, default=True
            Print results if True

        n_jobs : int, default=-1
            Number of jobs to run in parallel.
            ``-1`` means using all processors.
            ``None`` means 1

        **kwargs : dict
            All keyword arguments are passed to _score_func of scikit

        Returns
        -------
        best : dict
            Dictionary that maps classifier representations with best values
        """

        validate_type_of_elements(clf_list, sklearn.base.BaseEstimator)
        type_of_clf = type(clf_list[0])
        name_of_clf = type_of_clf.__name__
        validate_type_of_elements(clf_list, type(clf_list[0]))
        results = self.ml_pred_parallel(
            clf_list=clf_list,
            features_names_list=[features_names],
            n_jobs=n_jobs)
        scores = {name_of_clf: {}}
        best = {name_of_clf: {'params': None, metric_name: None}}
        for k, (test_Y, pred) in results.items():
            scores[name_of_clf][k] = get_scorer(
                metric_name)._score_func(test_Y, pred, **kwargs)
        # print(scores)
        if maximize:
            best_score = max(scores[name_of_clf], key=scores[name_of_clf].get)
        else:
            best_score = min(scores[name_of_clf], key=scores[name_of_clf].get)
        best[name_of_clf]['params'] = best_score
        best[name_of_clf][metric_name] = scores[name_of_clf][best[name_of_clf]['params']]
        if print_out:
            info = "Classifier:"
            print_info(info+" " + name_of_clf)
            print(best[name_of_clf]['params'],
                  scores[name_of_clf][best[name_of_clf]['params']])
        return best


def rating_norm_features(ratings) -> List[str]:
    """Function to use normalized ratings as ml features
    For example: For AccuRATE:
    => for Home = H + ratingnorm + key = HratingnormAccuRATE
    => for Away = A + ratingnorm + key = AratingnormAccuRATE

    Parameters
    ----------
    rating_systems : Dict[str, RatingSystem] or List[RatingSystem]]] or RatingSystem or None, default=None
        If type is dictionary then it is mapping names (or rating keys) to
        rating systems.
        If type is list of rating systems instances then it firstly converted
        to dictionary.
        If type is RatingSystem instance then it firstly converted
        to dictionary.
        If it is set to ``None`` then rating values are not included in data
        attributes for preparation.

    Returns
    -------
    features : List[str]
        List of normalized features (each name refer to a column of the data)
    """
    ratings_dict = rating_systems_to_dict(ratings)
    features = [['Hratingnorm'+rs_key,
                 'Aratingnorm'+rs_key] for rs_key in ratings_dict]
    if len(ratings_dict) == 1:
        return features[0]
    return features

# ==============================================================================
# SPORTS DATASET FUNCTIONS
# ==============================================================================


def enter_values(data: pd.DataFrame,
                 teams_df: pd.DataFrame,
                 teams_dict: Dict[Any, int],
                 rating_systems: Optional[Union[Dict[str, RatingSystem],
                                                List[RatingSystem], RatingSystem]] = None,
                 stats_attributes: Optional[Dict[str, Dict[Any, Any]]] = None,
                 columns_dict: Optional[Dict[str, Any]] = None
                 ) -> pd.DataFrame:
    """Enter the calculated values (from rating and statistic attributes)
    for each data-instance and return the data. Also, truncation is applied.

    Parameters
    ----------
    data : pd.DataFrame
        Games data with statistics and rating values for the teams

    teams_df : pd.DataFrame
        Set of teams.

    teams_dict : Dict[Any, int]
        Dictionary that maps teams' names to integer value.
        For instance ::

            teams_dict = {'Arsenal': 0,
                          'Aston Villa': 1,
                          'Birmingham': 2,
                          'Blackburn': 3
                          }

    rating_systems : Dict[str, RatingSystem] or List[RatingSystem]]] or RatingSystem or None, default=None
        If type is dictionary then it is mapping names (or rating keys) to
        rating systems.
        If type is list of rating systems instances then it firstly converted
        to dictionary.
        If type is RatingSystem instance then it firstly converted
        to dictionary.
        If it is set to ``None`` then rating values are not included in data
        attributes for preparation.

    stats_attributes : Optional[Dict[str, Dict[Any, Any]]], default=None
        The statistic attributes
        e.g. soccer sport: TW (Total Wins), TG (Total Goals),
        TS (Total Shots), TST (Total Shots on Target).

    columns_dict : Optional[Dict[str, str]], default=None
        A dictionary mapping the column names of the dataset.
        See the module :mod:`ratingslib.datasets.parameters` for more details.

    Returns
    -------
    data_truncate_df : pd.DataFrame
        Completed data-instances
    """
    col_names = parse_columns(columns_dict)
    data_to_iter = data.copy(deep=True)

    if rating_systems is None:
        rating_systems = {}
    if stats_attributes is None:
        stats_attributes = {}

    for index, row in data_to_iter.iterrows():
        item_i = teams_dict[row[col_names.item_i]]
        item_j = teams_dict[row[col_names.item_j]]
        for k in stats_attributes:
            data.at[index, 'H' + k] = teams_df.at[item_i, k]
            data.at[index, 'A' + k] = teams_df.at[item_j, k]
        # NG = number of games
        data.at[index, 'HNG'] = teams_df.at[item_i, 'NG']
        data.at[index, 'ANG'] = teams_df.at[item_j, 'NG']
        for k in rating_systems:
            # k=k#.value
            data.at[index, 'H' + k] = teams_df.at[item_i, k]
            data.at[index, 'A' + k] = teams_df.at[item_j, k]
            data.at[index, 'Hratingnorm' + k] = \
                teams_df.at[item_i, "ratingnorm" + k]
            data.at[index, 'Aratingnorm' + k] = \
                teams_df.at[item_j, "ratingnorm" + k]

    def trunc(x): return round(x, 6)  # math.trunc(1000000 * x) / 1000000
    data_truncate_df = data.copy(deep=True)
    for k in rating_systems:
        # k=k#.value
        data_truncate_df['H' + k] = data[['H' + k]].applymap(trunc)
        data_truncate_df['A' + k] = data[['A' + k]].applymap(trunc)
        data_truncate_df['Hratingnorm' + k] = \
            data[['Hratingnorm' + k]].applymap(trunc)
        data_truncate_df['Aratingnorm' + k] = \
            data[['Aratingnorm' + k]].applymap(trunc)
    for k in stats_attributes:
        data_truncate_df['H' + k] = data[['H' + k]].applymap(trunc)
        data_truncate_df['A' + k] = data[['A' + k]].applymap(trunc)
    return data_truncate_df


def _create_rating_data(rs_name: str,
                        rs: RatingSystem,
                        data_train: pd.DataFrame,
                        teams_df: pd.DataFrame):
    """Rate teams and also create column for normalized rating values

    Parameters
    ----------
    rs_name : str
        Name of rating system (from the key of dictionary)

    rs : RatingSystem
        RatingSystem instance

    data_train : pd.DataFrame
        Games data for training

    teams_df : pd.DataFrame
        Set of teams.

    Returns
    -------
    teams_df : pd.DataFrame
        Teams DataFrame with rating values, and normalized rating values.
    """
    teams_df = rs.rate(data_df=data_train, items_df=teams_df)
    # print_pandas(teams_df)
    teams_df[rs_name] = teams_df['rating']
    teams_df.drop(['rating'], axis=1, inplace=True)
    # print_pandas(teams_df)
    norm_colname = 'ratingnorm' + rs_name
    if (teams_df[rs_name] != 0).any():
        ratingnorm = normalization_rating(teams_df, rs_name)
        teams_df[norm_colname] = ratingnorm
    return teams_df


def prepare_sport_dataset(data_season: pd.DataFrame,
                          teams_df: pd.DataFrame,
                          rating_systems: Optional[
                              Union[Dict[str, RatingSystem],
                                    List[RatingSystem],
                                    RatingSystem]] = None,
                          stats_attributes=None,
                          start_week: int = 4,
                          preprocess: Optional[Preprocess] = BasicPreprocess(),
                          columns_dict: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Prepares the sport dataset in order to enter values of ratings and
    calculated games statistics to the teams every match-week.

    Parameters
    ----------
    data_season : pd.DataFrame
        Games data of season

    teams_df : pd.DataFrame
        Set of teams

    rating_systems : Dict[str, RatingSystem] or List[RatingSystem]]] or RatingSystem or None, default=None
        If type is dictionary then it is mapping names (or rating keys) to
        rating systems.
        If type is list of rating systems instances then it firstly converted
        to dictionary.
        If type is RatingSystem instance then it firstly converted
        to dictionary.
        If it is set to ``None`` then rating values are not included in data
        attributes for preparation.

    stats_attributes : Optional[Dict[str, Dict[Any, Any]]], default=None
        The statistic attributes
        e.g. soccer sport: TW (Total Wins), TG (Total Goals),
        TS (Total Shots), TST (Total Shots on Target).

    start_week : int, optional
        The match-week that rating procedure starts. For example if match-week
        is 4 then rating of teams will start from 4th week. Each week ratings
        are computed based on the previous weeks. e.g. 7th week -> 1,2,3,4,5,6

    preprocess : Preprocess
        The preprocess procedure for the dataset. It must be an instance of
        subclass of :class:`ratingslib.datasets.preprocess.Preprocess`

    columns_dict : Optional[Dict[str, str]], default=None
        A dictionary mapping the column names of the dataset.
        See the module :mod:`ratingslib.datasets.parameters` for more details

    Returns
    -------
    pd.DataFrame
        DataFrame of prepared data
    """

    col_names = parse_columns(columns_dict)
    sides = ['H', 'A']
    if rating_systems is None:
        ratings_dict = {}
    else:
        ratings_dict = rating_systems_to_dict(rating_systems)
    teams_dict = create_items_dict(teams_df)

    data_season[[side+norm+rs_name for rs_name in ratings_dict
                 for norm in ['', 'ratingnorm']
                 for side in sides]] = 0.0
    teams_df[[norm+rs_name for rs_name in ratings_dict
              for norm in ['', 'ratingnorm']]] = 0.0
    if stats_attributes is not None:
        data_season[[side+s for s in stats_attributes
                     for side in sides]] = 0.0
    max_week = max(data_season[col_names.period_number])
    for week in range(start_week, max_week+1):
        data_train = data_season.loc[
            data_season[col_names.period_number] < week]
        data_current_week = data_season.loc[
            data_season[col_names.period_number] == week]

        for rs_name, rs in ratings_dict.items():
            teams_df = _create_rating_data(rs_name, rs, data_train, teams_df)

        print_loading((week+1-start_week)/(max_week-start_week+1))
        teams_df = calc_items_stats(data_train, teams_df, teams_dict,
                                    normalization=True,
                                    stats_columns_dict=stats_attributes)

        data_season.loc[
            data_season[col_names.period_number] == week] = enter_values(
                data_current_week, teams_df, teams_dict,
                ratings_dict, stats_attributes)
    print()
    if preprocess is not None:
        data_season = preprocess.preprocessing(data_season, columns_dict)
    return data_season


def prepare_sports_seasons(filenames: Union[str, Dict[int, str]],
                           outcome: SportOutcome,
                           rating_systems: Optional[Union[Dict[str, RatingSystem],
                                                          List[RatingSystem],
                                                          RatingSystem]] = None,
                           stats_attributes=None,
                           start_week: int = 4,
                           preprocess: Optional[Preprocess] = BasicPreprocess(
),
    columns_dict: Optional[Dict[str, Any]] = None
) -> Dict[int, pd.DataFrame]:
    """Prepares datasets for multiple files that are passed as a dictionary.

    Parameters
    ----------
    filenames : Union[str, Dict[int, str]]
        Filename or dictionary that maps seasons to filename paths.
        e.g. {2009: 'sports/pl2009.csv'}

    outcome : SportOutcome
        The `outcome` parameter is associated with application type
        e.g. for soccer the type of outcome is
        :class:`ratingslib.application.SoccerOutcome`.
        For more details see :mod:`ratingslib.application` module.

    rating_systems : Dict[str, RatingSystem] or List[RatingSystem]]] or RatingSystem or None, default=None
        If type is dictionary then it is mapping names (or rating keys) to
        rating systems.
        If type is list of rating systems instances then it firstly converted
        to dictionary.
        If type is RatingSystem instance then it firstly converted
        to dictionary.
        If it is set to ``None`` then rating values are not included in data
        attributes for preparation.

    stats_attributes : Optional[Dict[str, Dict[Any, Any]]], default=None
        The statistic attributes
        e.g. soccer sport: TW (Total Wins), TG (Total Goals),
        TS (Total Shots), TST (Total Shots on Target).

    start_week : int, optional
        The match-week that rating procedure starts. For example if match-week
        is 4 then rating of teams will start from 4th week. Each week ratings
        are computed based on the previous weeks. e.g. 7th week -> 1,2,3,4,5,6

    preprocess : Preprocess
        The preprocess procedure for the dataset. It must be an instance of
        subclass of :class:`ratingslib.datasets.preprocess.Preprocess`

    columns_dict : Optional[Dict[str, str]], default=None
        A dictionary mapping the column names of the dataset.
        See the module :mod:`ratingslib.datasets.parameters` for more details

    Returns
    -------
    data_seasons_dict : Dict[int, pd.DataFrame]
        Dictionary that maps season to DataFrame prepared data. Note that if
        only one filename passed then the dictionary will be returned with
        the following structure {1: data}
    """
    show_load_msg = True
    if isinstance(filenames, str):
        filenames = {1: filenames}
        show_load_msg = False
    seasons = list(filenames.keys())
    season_start = min(seasons)
    season_end = max(seasons)
    data_seasons_dict = {}
    for season in range(season_start, season_end+1):
        if show_load_msg:
            print("Load season:", season, '-', season+1)
        # parse game data from file
        data_season, teams_df = parse_pairs_data(filenames[season],
                                                 parse_dates=[DATE_COL],
                                                 dayfirst=DAY_FIRST,
                                                 frequency=WEEK_PERIOD,
                                                 outcome=outcome,
                                                 columns_dict=columns_dict)
        data = prepare_sport_dataset(data_season=data_season,
                                     teams_df=teams_df,
                                     rating_systems=rating_systems,
                                     stats_attributes=stats_attributes,
                                     start_week=start_week,
                                     preprocess=preprocess,
                                     columns_dict=columns_dict)

        data_seasons_dict[season] = data
    return data_seasons_dict
