"""Module for preprocessing functions"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT


from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from ratingslib.utils.methods import parse_columns
from ratingslib.utils.validation import validate_type_of_elements


class Preprocess(ABC):
    """
    A base class for preprocessing data
    """

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def preprocessing(self,
                      data_df: pd.DataFrame,
                      col_names_or_dict: Union[SimpleNamespace,
                                               Optional[Dict[str, Any]]]) -> pd.DataFrame:
        """To be overridden in subclasses."""
        raise NotImplementedError(self.__class__.__name__ + "is a abstract")

    def __repr__(self):
        return self.__class__.__name__


class BasicPreprocess(Preprocess):
    """
    The basic preprocess class removes from sports-data the games
    of first match-week.
    """

    def __init__(self, weeks_to_remove: Optional[List[int]] = None):
        Preprocess.__init__(self)
        if weeks_to_remove is not None:
            validate_type_of_elements(weeks_to_remove, int)
            self.weeks_to_remove = weeks_to_remove
        else:
            self.weeks_to_remove = [1]

    def _remove_week(self, data_df, num_list, col_names) -> pd.DataFrame:
        """Remove match-weeks from a sport dataset according to the list
        of numbers passed.

        Parameters
        ----------
        data_df : pandas.DataFrame
            Dataset with games

        num_list : list
            match weeks to remove e.g. [1,2] the first two weeks will
            be removed from dataset

        col_names : SimpleNamespace
            column names based on dataset. For more details see at
            :meth:`ratingslib.utils.methods.parse_columns`

        Returns
        -------
        data_df : pandas.DataFrame
            The modified data
        """
        # print(self.col_names)
        for num in num_list:
            data_df = data_df[data_df[col_names.period_number] != num]
        return data_df

    def preprocessing(self,
                      data_df: pd.DataFrame,
                      col_names_or_dict: Union[SimpleNamespace,
                                               Optional[Dict[str, Any]]]) -> pd.DataFrame:
        """Removes the match-weeks and returns the modified
        dataset."""
        col_names = col_names_or_dict
        if col_names_or_dict is None or isinstance(col_names_or_dict, dict):
            col_names = parse_columns(col_names_or_dict)
        data_processed_df = data_df.copy(deep=True)
        data_processed_df = self._remove_week(data_processed_df,
                                              self.weeks_to_remove,
                                              col_names)
        data_processed_df.reset_index(inplace=True, drop=True)
        return data_processed_df
