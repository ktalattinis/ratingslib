"""
    This module includes all parameters and configuration for soccer.
    column names are based on football-data.co.uk data scheme.

    Notes
    ----------
    The dictionary COLUMNS_DICT is based on football-data.co.uk data files
    for other data-source change dictionary values specify column names:

        - key is the variable name in code
        - value is the column_name

        * 'ITEM_I': column name for home team
        * 'ITEM_J': column name for away team
        * 'points_i' : column name for home team goals
        * 'points_j' : column name for away team goals
        * 'HS' : column name for home team total shoots
        * 'AS' : column name for away team total shoots
        * 'HST' : column name of home team total shoots on target
        * 'AST' : column name of away team total shoots on target


    #. date column is written at the beginning of this module (`DATE_COL` var)

    #. `HS`, `AS`, `HST`, `AST` columns below are used in:
       :class:`ratingslib.ratings.markov.Markov`
       :class:`ratingslib.ratings.accurate.AccuRate`
"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

from datetime import datetime

DATE_COL = "Date"  # column of date
TIME_FORMAT = "%d/%m/%y"  # basic time format of Date in most of files
# there are two times format of Date in football-data.co.uk files
MULTIPLE_TIME_FORMAT = ("%d/%m/%y", "%d/%m/%Y")
WEEK_PERIOD = "W-THU"  # every match-week starts every Thursday
DAY_FIRST = True


def parser_date_func(x):
    for fmt in MULTIPLE_TIME_FORMAT:
        try:
            return datetime.strptime(x, fmt)
        except ValueError:
            pass

    raise ValueError('no valid date format found')


COLUMNS_DICT = {
    'item_i': 'HomeTeam',
    'item_j': 'AwayTeam',
    'points_i': 'FTHG',
    'points_j': 'FTAG',
    'ts_i': 'HS',
    'ts_j': 'AS',
    'tst_i': 'HST',
    'tst_j': 'AST',
}

COLUMNS_ODDS_DICT = {
    "B365H": "B365H",
    "BWH": "BWH",
    "IWH": "IWH",
    "B365A": "B365A",
    "BWA": "BWA",
    "IWA": "IWA",
    "B365D": "B365D",
    "BWD": "BWD",
    "IWD": "IWD"

}


# ==============================================================================


class championships:
    """Class for championship names"""
    PREMIERLEAGUE = 'EPL'  # ENGLISH PREMIER LEAGUE
    ENG_1 = 'E1'  # ENGLISH DIV 1
    ENG_2 = 'E2'  # ENGLISH DIV 2
    ENG_3 = 'E3'  # ENGLISH DIV 3
    ENG_CONF = 'ECONF'  # ENGLISH CONFERENCE
    SCOTLAND_PS = 'SPS'  # SCOTLAND PREMIERSHIP
    SCOTLAND_1 = 'SC1'  # SCOTLAND DIV 1
    SCOTLAND_2 = 'SC2'  # SCOTLAND DIV 2
    SCOTLAND_3 = 'SC3'  # SCOTLAND DIV 3
    BUNDESLIGAS_1 = 'GBL1'  # GERMANY BUNDESLIGAS 1
    BUNDESLIGAS_2 = 'GBL2'  # GERMANY BUNDESLIGAS 2
    ITALY_A = 'IA'  # ITALY SERIE A
    ITALY_B = 'IB'  # ITALY SERIE B
    SPAIN_PREMERA = 'SPP'  # SPAIN PREMERA
    SPAIN_SEGUNDA = 'SPS'  # SPAIN SEGUNDA
    FRANCE_CHAMP = 'FRCH'  # FRANCE LA CHAMPIONAT
    FRANCE_2 = 'FR2'  # FRANCE DIV 2
    NETHERLANDS = 'NKPN'  # KPN EREDIVISIE
    BELGIUM = 'BEL'  # JUPILER LEAGUE
    PORTUGAL = 'PLIGA'  # LIGA I
    TURKEY = 'TLIGI'  # LIGI 1
    GREECE_SL = 'GREK'  # ETHNIKI KATIGORIA

# ==============================================================================


class stats:
    """
    Class for soccer statistics,
    In this class all statistics can represented here.

    #. TW : TOTAL WINS/NUMBER OF GAMES

    #. TG : TOTAL GOALS/NUMBER OF GAMES

    #. TS : TOTAL SHOOTS/NUMBER OF GAMES (for soccer)

    #. TST : TOTAL SHOOTS TARGET/NUMBER OF GAMES (for soccer)

    * STATISTIC COLUMNS DICTIONARY

        `ITEM_I`: column name of statistic for home team e.g. in football-data.co.uk
        the column for goals is FTHG.

        `ITEM_J`: column name of statistic for away team e.g. in football-data.co.uk
        the column for goals is FTAG.

        TYPE: {WIN, POINTS}

            #. if type is WIN then compare if ITEM_I > ITEM_J or ITEM_J < ITEM_I

            #. if type is POINTS then count points

    for more information about the TYPE
    see the implementation of :func:`ratingslib.ratings.methods.calc_items_stats`

    * PARAMS DICTIONARY FOR MARKOV RATING SYSTEMS

        - key = statistic name e.g. `TW`
        - value = dictionary where

        `VOTE` : total votes (e.g. 10), then will be converted as weight.

        `ITEM_I`: H: column name of statistic for home team
        e.g. in football-data.co.uk the column for goals is `FTHG`.

        `ITEM_J`: column name of statistic for away team
        e.g. in football-data.co.uk the column for goals is `FTAG`.

    METHOD:{'VotingWithLosses', 'WinnersAndLosersVotePoint',
    'LosersVotePointDiff'}
    for more information about the methods see :mod:`ratingslib.ratings.markov`
    module.


    """
    TW, TG, TS, TST = ('TW', 'TG', 'TS', 'TST')

    STATS_COLUMNS_DICT = {
        TW: {'ITEM_I': 'FTHG', 'ITEM_J': 'FTAG', 'TYPE': 'WIN'},
        TG: {'ITEM_I': 'FTHG', 'ITEM_J': 'FTAG', 'TYPE': 'POINTS'},
        TST: {'ITEM_I': 'HST', 'ITEM_J': 'AST', 'TYPE': 'POINTS'},
        TS: {'ITEM_I': 'HS', 'ITEM_J': 'AS', 'TYPE': 'POINTS'}
    }

    STATS_MARKOV_EQUAL_DICT = {
        TW: {
            'VOTE': 10.0,
            'ITEM_I': 'FTHG',
            'ITEM_J': 'FTAG',
            'METHOD': 'VotingWithLosses'},
        TG: {
            'VOTE': 10.0,
            'ITEM_I': 'FTHG',
            'ITEM_J': 'FTAG',
            'METHOD': 'WinnersAndLosersVotePoint'},
        TST: {
            'VOTE': 10.0,
            'ITEM_I': 'HST',
            'ITEM_J': 'AST',
            'METHOD': 'WinnersAndLosersVotePoint'},
        TS: {
            'VOTE': 10.0,
            'ITEM_I': 'HS',
            'ITEM_J': 'AS',
            'METHOD': 'WinnersAndLosersVotePoint'},
    }
    STATS_MARKOV_DICT = {
        TW: {
            'VOTE': 10.0,
            'ITEM_I': 'FTHG',
            'ITEM_J': 'FTAG',
            'METHOD': 'VotingWithLosses'},
        TG: {
            'VOTE': 8.0,
            'ITEM_I': 'FTHG',
            'ITEM_J': 'FTAG',
            'METHOD': 'WinnersAndLosersVotePoint'},
        TST: {
            'VOTE': 6.0,
            'ITEM_I': 'HST',
            'ITEM_J': 'AST',
            'METHOD': 'WinnersAndLosersVotePoint'},
        TS: {
            'VOTE': 4.0,
            'ITEM_I': 'HS',
            'ITEM_J': 'AS',
            'METHOD': 'WinnersAndLosersVotePoint'},
    }
