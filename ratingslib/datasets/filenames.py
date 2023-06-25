"""This module contains filenames of examples and exposes various helper
functions for the diretory paths of datasets."""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

import os
from typing import Dict, Optional, Literal


import pandas as pd
from ratingslib.datasets.soccer import championships
from ratingslib.utils.logmsg import FILENAME, log

from ratingslib.utils.methods import get_filename, get_filenames, print_loading
from ratingslib.utils.validation import validate_type

# ==============================================================================
# Filenames path for datasets examples
# All paths have the parent directory ratingslib/datasets
# ==============================================================================
FINANCE_PATH = "finance/"
SPORTS_PATH = "sports/"
MOVIES_PATH = "movies/"
EXAMPLES_PATH = "examples/"
FOOTBALL_DATA_PATH = SPORTS_PATH + "soccer/footballdata-co-uk/"
NFL_DATA_PATH = SPORTS_PATH + "nfl/"
MOVIES_DATA_PATH = MOVIES_PATH + "data/"


FILENAME_NCAA_2005_ATLANTIC = EXAMPLES_PATH + "ncaa2005-atlantic-10games.csv"
FILENAME_CHARTIER_PAPER_MOVIES = EXAMPLES_PATH + "chartierPaperMovies.csv"
FILENAME_NFL_2009_2010_SHORTNAMES_NO_PLAYOFF = \
    NFL_DATA_PATH + "2009-2010NFLshortnames_no_playoffs.csv"
FILENAME_NFL_2009_2010_FULLNAMES_GAMES_AND_PLAYOFFS = \
    NFL_DATA_PATH + "2009-2010NFLfullnames_games_and_playoffs.csv"
FILENAME_GOVAN_THESIS = EXAMPLES_PATH + "govanThesisExample.csv"
FILENAME_ACCURATE_PAPER_EXAMPLE = EXAMPLES_PATH + "accuratePaperExample.csv"
FILENAME_EPL_2018_2019_20_GAMES = \
    EXAMPLES_PATH + "2018-2019EPL20games.csv"
FILENAME_EPL_2018_2019_3RD_WEEK = EXAMPLES_PATH + "2018-2019EPL3rd-weekgames.csv"
FILENAME_OOSA = EXAMPLES_PATH + "oosaExample.csv"
FILENAME_MOVIES_EXAMPLE = EXAMPLES_PATH + "moviesExample.csv"
FILENAME_DOMAIN_NAMES = EXAMPLES_PATH + "domainMarketExample.csv"
FILENAME_PORTFOLIOS = EXAMPLES_PATH + "portfoliosExample.csv"
FILENAME_INVESTMENTS = EXAMPLES_PATH + "investmentsExample.csv"

FOOTBALL_DATA_LINK = 'https://www.football-data.co.uk/mmz4281/'
FOOTBALL_DATA_LEAGUES: Dict[str, str] = {
    championships.PREMIERLEAGUE: 'E0',
    championships.ENG_1: 'E1',
    championships.ENG_2: 'E2',
    championships.ENG_3: 'EC',
    championships.SCOTLAND_PS: 'SC0',
    championships.SCOTLAND_1: 'SC1',
    championships.SCOTLAND_2: 'SC2',
    championships.SCOTLAND_3: 'SC3',
    championships.BUNDESLIGAS_1: 'D1',
    championships.BUNDESLIGAS_2: 'D2',
    championships.ITALY_A: 'I1',
    championships.ITALY_B: 'I2',
    championships.SPAIN_PREMERA: 'SP1',
    championships.SPAIN_SEGUNDA: 'SP2',
    championships.FRANCE_CHAMP: 'F1',
    championships.FRANCE_2: 'F2',
    championships.NETHERLANDS: 'N1',
    championships.BELGIUM: 'B1',
    championships.PORTUGAL: 'P1',
    championships.TURKEY: 'T1',
    championships.GREECE_SL: 'G1'

}


def dataset_path(filename):
    """Returns the absolute path of filename from dataset dicrectory"""
    current_dir = os.path.dirname(__file__)
    return get_filename(filename,
                        directory_path="",
                        current_dirname=current_dir,
                        check_if_not_exists=True)


def datasets_paths(*filenames):
    """Returns a tuple of filenames absolute path from dataset directory"""
    current_dir = os.path.dirname(__file__)
    return get_filenames(*filenames,
                         directory_path="",
                         current_dirname=current_dir)


def dataset_sports_path(
        season: int,
        path: Optional[str] = None, current_dir: Optional[str] = None,
        championship: str = 'EPL', file_suffix: str = 'footballdata',
        file_ext: str = '.csv') -> str:
    """Return filename path of a sports dataset for the given championship and
    season
    (e.g. filename of soccer games - English Premier League
    for the season 2007-2008)

    season : int
        The sport season-year, e.g. 2007 means 2007-2008.

    path : Optional[str], default=None
        The path of sports dataset.

    current_dir : Optional[str], default=None
        The path of current directory.

    championship : str, default='EPL'
        Championship initials. e.g. `EPL` for English Premier League.
        Championships initials are available at
        :mod:`data.parameters.championships`.

    file_suffix : str, default="footballdata",
        Filename suffix (before filename extension).

    file_ext : str, default=".csv"
        Filename extension.

    Returns
    -------
    filename : str
        The absolute path of filename (e.g. 2007-2008EPLfootballdata.csv)

    """
    if path is None:
        path = FOOTBALL_DATA_PATH
    if current_dir is None:
        current_dir = os.path.dirname(__file__)
    filename = get_filename(filename=str(season) + "-" + str(season + 1) +
                            championship + file_suffix + file_ext,
                            directory_path=path,
                            current_dirname=current_dir)
    log(FILENAME, "Dataset filename:", filename)
    return filename


def datasets_sports_seasons_path(
        season_start: int,
        season_end: int = -1,
        path: Optional[str] = None, current_dir: Optional[str] = None,
        championship: str = 'EPL', file_suffix: str = 'footballdata',
        file_ext: str = '.csv') -> Dict[int, str]:
    """Return filenames path of a sports dataset for the given championship
    and season range. The returned type is a dictionary where
    season number (int) is the key and filename path (str) is the value

    Parameters
    ----------
    season_start : int
        The starting sport season-year, e.g. 2007 means 2007-2008.

    season_end : Optional[int], default=-1
        The last sport season-year, e.g. 2017 means 2017-2018.
        The default value indicates only one season.

    path : Optional[str], default=None
        The path of sports dataset.

    current_dir : Optional[str], default=None
        The path of current directory.

    championship : str, default='EPL'
        Championship initials. e.g. `EPL` for English Premier League.
        Championships initials are available at
        :mod:`data.parameters.championships`.

    file_suffix : str, default="footballdata",
        Filename suffix (before filename extension).

    file_ext : str, default=".csv"
        Filename extension.

    Returns
    -------
    seasons_dict : Dict[int, str]
        Dictionary that maps seasons to csv file links

    Raises
    ------
    ValueError
        if ``season_start`` or ``season_end`` are not positive integers.

    """
    validate_type(season_start, int, 'season_start')
    validate_type(season_end, int, 'season_end')
    if season_end == -1:
        season_end = season_start+1
    if season_start >= season_end:
        raise ValueError('season_start must be smaller than season_end')
    return {season: dataset_sports_path(
        season=season, path=path, current_dir=current_dir,
        championship=championship,
        file_suffix=file_suffix, file_ext=file_ext)
        for season in range(season_start, season_end)}


def get_season_footballdata_online(
        season: int,
        championship: str = championships.PREMIERLEAGUE):
    """Get a link of football-data for a csv file for the given championship
    and sport season. Link example for 2021-2022 English Premier League:
    https://www.football-data.co.uk/mmz4281/2122/E0.csv

    Parameters
    ----------
    season : int
        The sport season-year, e.g. 2007 means 2007-2008.

    championship : Optional[str], default=championships.PREMIERLEAGUE
        Championship initials. e.g. `EPL` for English Premier League.
        Championships initials are available at
        :mod:`data.soccer.championships`.

    Returns
    -------
    link : str
        The csv file link

    Raises
    ------
    ValueError
        If ``season`` is not positive int
    ValueError
        If ``championship`` is not listed in ``FOOTBALL_DATA_LEAGUES`` dictionary
    """
    validate_type(season, int, 'season')

    if season < 0:
        raise ValueError(
            'season parameter should not be negative. For example 2009 is the 2009-2010 sport season.')
    if championship not in FOOTBALL_DATA_LEAGUES.keys():
        raise ValueError('The championship ' + championship +
                         ' not included in football-data.co.uk')
    s1 = str(season)
    s2 = str(season+1)
    if len(s1) == 4:
        s1 = s1[-2:]
    if len(s2) == 4:
        s2 = s2[-2:]
    link = FOOTBALL_DATA_LINK + s1+s2+"/" + \
        FOOTBALL_DATA_LEAGUES[championship] + ".csv"
    return link


def get_seasons_dict_footballdata_online(
        season_start: int,
        season_end: int = -1,
        championship: str = championships.PREMIERLEAGUE) -> Dict[int, str]:
    """Get data links of csv files https://www.football-data.co.uk for
    the given championship and season range.

    Parameters
    ----------
    season_start : int
        The starting sport season-year, e.g. 2007 means 2007-2008.

    season_end : Optional[int], default=-1
        The last sport season-year, e.g. 2017 means 2017-2018.
        The default value indicates only one season.

    championship : Optional[str], default=championships.PREMIERLEAGUE
        Championship initials. e.g. `EPL` for English Premier League.
        Championships initials are available at
        :mod:`data.soccer.championships`.

    Returns
    -------
    seasons_dict : Dict[int, str]
        Dictionary that maps seasons to csv file links

    Raises
    ------
    ValueError
        if ``season_start`` or ``season_end`` are not positive integers.
    """
    validate_type(season_start, int, 'season_start')
    validate_type(season_end, int, 'season_end')
    if season_end == -1:
        season_end = season_start+1
    if season_start >= season_end:
        raise ValueError('season_start must be smaller than season_end')

    return {season: get_season_footballdata_online(
            season=season, championship=championship)
            for season in range(season_start, season_end)}


def download_and_store_footballdata(season_start: int,
                                    season_end: int = -1,
                                    championship: str = championships.PREMIERLEAGUE,
                                    save_path: Optional[str] = None):
    """Download and store data from www.football-data.co.uk for the given
    championship and season range.

    Parameters
    ----------
    season_start : int
        The starting sport season-year, e.g. 2007 means 2007-2008.

    season_end : Optional[int], default=-1
        The last sport season-year, e.g. 2018 means 2017-2018.
        The default value indicates only one season.

    championship : Optional[str], default=championships.PREMIERLEAGUE
        Championship initials. e.g. `EPL` for English Premier League.
        Championships initials are available at
        :mod:`data.soccer.championships`.

    save_path : Optional[str], default=None
        The path to save csv files

    Examples
    --------
    Download and store csv files from 2005/06 to 2017/18

    >>> from ratingslib.datasets.filenames import download_and_store_footballdata
    >>> from ratingslib.datasets.soccer import championships

    >>> epl = championships.PREMIERLEAGUE  # ENGLISH PREMIER LEAGUE
    >>> download_and_store_footballdata(2005, 2018, championship=epl)
    """
    if save_path is None:
        save_path = FOOTBALL_DATA_PATH
    if season_end == -1:
        season_end = season_start+1
    season_dict = get_seasons_dict_footballdata_online(
        season_start, season_end, championship=championship)
    current_dirname = os.path.dirname(__file__)
    for season, filename in season_dict.items():
        i = season-season_start+1
        filename_save = str(season)+"-"+str(season+1) + \
            championship+'footballdata.csv'
        outfilename = get_filename(
            filename_save,
            directory_path=save_path,
            current_dirname=current_dirname)

        file_df = pd.read_csv(filename)
        file_df.dropna(axis=0, how='all', inplace=True)
        file_df.to_csv(outfilename, index=False)
        print_loading((i/(season_end-season_start)))
