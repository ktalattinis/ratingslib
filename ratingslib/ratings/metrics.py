"""
This module implements evaluation metrics.
"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT


from typing import Dict, List

import pandas as pd
import scipy as sp
import scipy.stats as stats
from ratingslib.utils.validation import validate_type, validate_type_of_elements


def kendall_tau_table(ratings_dict: Dict[str, pd.DataFrame],
                      print_out: bool = True) -> List[List[float]]:
    """Kendall Tau comparison of ranking lists.

    Parameters
    ----------
    ratings_dict : Dict[str, pd.DataFrame]
        Dictionary that maps names to ratings. Note that ratings are stored in
        a `pandas.DataFrame`.

    print_out : bool
        If ``True`` then print results table.

    Returns
    -------
    kendall_results : List[List[float]]
        Table of Kendall tau results. The lower diagonal elements represent Kendall’s tau values
        of each pair, while the upper diagonal elements the p-values of each pair from the
        two-sided hypothesis test, whose null hypothesis is an absence of association
    """
    validate_type_of_elements(list(ratings_dict.keys()), str)
    validate_type_of_elements(list(ratings_dict.values()), pd.DataFrame)
    validate_type(print_out, bool, 'print_out')
    kendall_results = [[] for _ in range(len(ratings_dict))]
    for i, key_i in enumerate(ratings_dict):
        for j, key_j in enumerate(ratings_dict):
            tau, p_value = sp.stats.kendalltau(
                ratings_dict[key_i]['rating'], ratings_dict[key_j]['rating'])
            if i > j:
                kendall_results[i].insert(j, "{:.3}".format(tau))
            elif i < j:
                kendall_results[i].insert(j, "{:.3}".format(p_value))
            else:
                kendall_results[i].insert(j, "{:.3}".format(tau))
    if print_out:
        print(pd.DataFrame(kendall_results, columns=ratings_dict.keys(),
                           index=ratings_dict.keys()))
    return kendall_results
