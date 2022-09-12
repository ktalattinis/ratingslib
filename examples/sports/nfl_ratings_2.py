"""NFL teams ratings with Keener"""


from ratingslib.datasets.filenames import (
    FILENAME_NFL_2009_2010_SHORTNAMES_NO_PLAYOFF, dataset_path)
from ratingslib.ratings.keener import Keener
from ratingslib.utils.methods import print_pandas

filename = dataset_path(FILENAME_NFL_2009_2010_SHORTNAMES_NO_PLAYOFF)
keener = Keener(normalization=False).rate_from_file(filename)
print_pandas(keener)
