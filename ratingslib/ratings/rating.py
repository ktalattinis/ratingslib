"""
This module provides the :class:`RatingMethod` abstract class for rating
systems.
"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

import collections
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from ratingslib.datasets.parse import parse_data, parse_pairs_data
from ratingslib.utils import logmsg
from ratingslib.utils.logmsg import log
from ratingslib.utils.validation import validate_not_none_and_type


class RatingSystem(object):
    """
    This class represents a Rating method / system.
    The logic behind class structure is that the rating system can be divided
    in two phases:

        * the preparation phase, :meth:`preparation_phase`
        * the computation phase, :meth:`computation_phase`

    Extend this class and override abstract methods to define
    a rating system.

    Warning: This class should not be used directly.
    Use derived classes instead.

    Parameters
    ----------
    version : str
        A string that shows the version of rating system. The available
        versions can be found in :class:`ratingslib.utils.enums.ratings` class.

    Attributes
    ----------
    score : float
        The score number that represents rating for a single item

    rating : numpy.ndarray
        The rating value of items are stored in a numpy array of shape (n,)

        | where n = the total number of teams

    ord_dict_attr : Optional[dict]
        An order dictionary that maps attributes

    params_key : str
        String representation of parameters. For rating systems without
        parameters their version is used.

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, version: str):
        self.version = version
        self.score: float = 0
        self.rating: Optional[np.ndarray] = None
        self.ord_dict_attr: Optional[dict] = None
        self.params_key: str = str(self.version)

    @abstractmethod
    def computation_phase(self):
        """To be overridden in subclasses."""
        raise NotImplementedError(self.__class__.__name__ + " is abstract")

    @abstractmethod
    def preparation_phase(self, data_df: pd.DataFrame, items_df: pd.DataFrame,
                          columns_dict: Optional[Dict[str, Any]] = None):
        """To be overridden in subclasses."""
        raise NotImplementedError(self.__class__.__name__ + " is abstract")

    @abstractmethod
    def rate(self, data_df: pd.DataFrame, items_df: pd.DataFrame,
             sort: bool = False,
             columns_dict: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        This method computes ratings for a pairwise data.
        (e.g. soccer teams games). To be overridden in subclasses.

        Parameters
        ----------
        data_df : pandas.DataFrame
            The pairwise data.

        items_df : pandas.DataFrame
            Set of items (e.g. teams) to be rated

        sort : bool, default=True.
            If true, the output is sorted by rating value

        columns_dict : Optional[Dict[str, str]]
            The column names of data file.
            See ``ratingslib.datasets.parameters.COLUMNS_DICT`` for more
            details.

        Returns
        -------
        items_df : pandas.DataFrame
            The set of items with their rating and ranking.

        """
        raise NotImplementedError(self.__class__.__name__ + " is abstract")

    def rate_from_file(self, filename: str, pairwise: bool = True,
                       reverse_attributes_cols: Optional[List[str]] = None,
                       sort: bool = False,
                       columns_dict: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        This method computes ratings for a pairwise data.
        (e.g. soccer teams games). Data are stored in a filename.

        Parameters
        ----------
        filename : str
            The name of file or file path+filename.

        sort : bool, default=True
            If ``True``, the output is sorted by rating value.

        pairwise : bool, default=True
            If ``True`` then data is in pairwise structure.

        reverse_attributes_cols : Optional[List[str]], default = None
            Name of columns (from csv file) where the numerical scoring scale runs
            in the opposite direction. Valid only if pairwise is ``False``.

        columns_dict : Optional[Dict[str, str]], default=None
            The column names of data file.
            See ``ratingslib.datasets.parameters.COLUMNS_DICT`` for more
            details.

        Returns
        -------
        items_df : pandas.DataFrame
            The set of items with their rating and ranking.

        """
        validate_not_none_and_type(filename, str, "filename")
        if pairwise:
            data_df, items_df = parse_pairs_data(filename,
                                                 columns_dict=columns_dict)
        else:
            data_df, items_df = parse_data(filename,
                                           reverse_attributes_cols,
                                           columns_dict=columns_dict)
        log(logmsg.EXAMPLE, self.params_key,
            "Rating method for filename: ", filename, sep='\n')
        items_df = self.rate(data_df, items_df, sort=sort,
                             columns_dict=columns_dict)
        # log_pandas(items_df)
        return items_df

    def set_rating(self, items_df: pd.DataFrame, rating_lower_best: bool = False,
                   sort: bool = False) -> pd.DataFrame:
        """Set the rating values and produce items rankings

        Parameters
        ----------
        items_df : pandas.DataFrame
            Set of items (e.g. teams) to be rated

        rating_lower_best : bool, default=True
            If ``True`` then lower rating is better, else ``False``

        sort : bool, default=True
            Sort the items by rating value if ``True``

        Returns
        -------
        items_df : pandas.DataFrame
            The set of items with their rating and ranking.
        """
        items_df['rating'] = self.rating
        items_df['rating'] = items_df['rating'].astype(float)
        items_df['ranking'] = items_df.rating.rank(method='dense',
                                                   ascending=rating_lower_best).astype(int)
        if sort:
            items_df = items_df.sort_values(by=['rating'], ascending=False)
        return items_df

    def create_ordered_dict(self, **kwargs):
        self.ord_dict_attr = collections.OrderedDict(sorted(kwargs.items()))
        if len(self.ord_dict_attr) > 0:
            self.params_key += '['+'_'.join(
                str(k) + '=' + str(v)
                for k, v in self.ord_dict_attr.items())+']'

    @property
    def params_key(self) -> str:
        return self._params_key

    @params_key.setter
    def params_key(self, params_key: str):
        self._params_key = params_key
