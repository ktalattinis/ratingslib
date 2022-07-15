"""NFL teams ratings with Keener"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

from apprate.datasets.filenames import (
    FILENAME_NFL_2009_2010_SHORTNAMES_NO_PLAYOFF, dataset_path)
from apprate.ratings.keener import Keener
from apprate.utils.methods import print_pandas

filename = dataset_path(FILENAME_NFL_2009_2010_SHORTNAMES_NO_PLAYOFF)
keener = Keener(normalization=False).rate_from_file(filename)
print_pandas(keener)
