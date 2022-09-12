"""NFL teams ratings with Elo-win and Elo-point"""

from ratingslib.datasets.filenames import (
    FILENAME_NFL_2009_2010_FULLNAMES_GAMES_AND_PLAYOFFS, dataset_path)
from ratingslib.ratings.elo import Elo
from ratingslib.utils.enums import ratings
from ratingslib.utils.methods import print_pandas

filename = dataset_path(FILENAME_NFL_2009_2010_FULLNAMES_GAMES_AND_PLAYOFFS)
elowin = Elo(version=ratings.ELOWIN, K=32, ks=1000, HA=0,
             starting_point=0).rate_from_file(filename, sort=True)
print_pandas(elowin)


filename = dataset_path(FILENAME_NFL_2009_2010_FULLNAMES_GAMES_AND_PLAYOFFS)
elopoint = Elo(version=ratings.ELOPOINT, K=32, ks=1000, HA=0,
               starting_point=0).rate_from_file(filename, sort=True)
print_pandas(elopoint)
