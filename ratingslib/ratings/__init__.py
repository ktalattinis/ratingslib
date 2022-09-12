"""
Python package for Rating methods
This package includes the implementation of the following rating methods:

 #. WinLoss : :mod:`.winloss`
 #. Colley : :mod:`.colley`
 #. Massey : :mod:`.massey`
 #. Elo : :mod:`.elo`
 #. Keener : :mod:`.keener`
 #. OffenseDefense : :mod:`.od`
 #. AccuRATE : :mod:`.accurate`
 #. GeM : :mod:`.markov`

Several evaluation metrics that related with rating methods are included in
the module metrics.py.

All rating systems in this package have been implemented by exploiting
several functions of NumPy and SciPy libraries in python that
are intended for algebraic and scientific computations.
Particularly, NumPy was used:

 #.	matrices and vectors handling
 #.	linear systems solving
 #.	finding eigenvalues and eigenvectors
 #.	other problems of linear algebra required for the implementation of rating methods

As for the statistical tests, such as Kendalls’s Tau for the correlation of
ranking lists, SciPy was used.

"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

from .winloss import Winloss
from .accurate import AccuRate
from .colley import Colley
from .massey import Massey
from .elo import Elo
from .markov import Markov
from .keener import Keener
from .od import OffenseDefense
from .metrics import kendall_tau_table
from .aggregation import RatingAggregation, RankingAggregation

__all__ = ['Winloss', 'Keener', 'Massey', 'OffenseDefense',
           'Markov', 'AccuRate', 'Elo', 'Colley',
           'kendall_tau_table', 'RatingAggregation', 'RankingAggregation']
