"""This module connects constant variables with
variables of a data module according to the data_scheme."""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

DATA_SCHEME = "SOCCER_FOOTBALL-DATA.CO.UK"

# ==============================================================================
# Additional columns that are created during parsing.
# For more information check :func:`ratingslib.utils.methods.parse_pairs_data`
# 'period': column name for period name of each match-week
# 'period_number': column name for match-week number
# ==============================================================================

COLUMNS_EXTRA_DICT = {
    'period_number': 'Week_Number',
    'period': 'Period'
}


if DATA_SCHEME == 'SOCCER_FOOTBALL-DATA.CO.UK':
    from .soccer import (COLUMNS_DICT, COLUMNS_ODDS_DICT, DATE_COL,
                         TIME_FORMAT, WEEK_PERIOD, championships,
                         parser_date_func, stats)
    COLUMNS_DICT = {**COLUMNS_DICT, **COLUMNS_ODDS_DICT, **COLUMNS_EXTRA_DICT}
