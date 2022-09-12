"""Small example on domain names ranking"""

import ratingslib.ratings as rl
import pandas as pd
from ratingslib.datasets.filenames import FILENAME_DOMAIN_NAMES, dataset_path
from ratingslib.utils import logmsg
from ratingslib.utils.logmsg import set_logger
from ratingslib.utils.methods import print_pandas

if __name__ == '__main__':

    pd.set_option('float_format', "{:.4f}".format)
    set_logger(logmsg.EXAMPLE)

    filename_dn = dataset_path(FILENAME_DOMAIN_NAMES)

    COLUMNS_DOMAIN_DICT = {
        'item_i': 'DomainNameI',
        'item_j': 'DomainNameJ',
        'points_i': 'TrendsI',
        'points_j': 'TrendsJ',
    }

    s = "-" * 100
    winloss = rl.Winloss(
        normalization=False).rate_from_file(
        filename_dn,
        columns_dict=COLUMNS_DOMAIN_DICT)
    print_pandas(winloss)
    print(s)

    colley = rl.Colley().rate_from_file(filename_dn,
                                        columns_dict=COLUMNS_DOMAIN_DICT)
    print_pandas(colley)
    print(s)

    massey = rl.Massey().rate_from_file(filename_dn,
                                        columns_dict=COLUMNS_DOMAIN_DICT)
    print_pandas(massey)
    print(s)

    od = rl.OffenseDefense().rate_from_file(filename_dn,
                                            columns_dict=COLUMNS_DOMAIN_DICT)
    print_pandas(od)
    print(s)
