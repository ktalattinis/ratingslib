"""Small example on domain names ranking"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

import apprate.ratings as ar
import pandas as pd
from apprate.datasets.filenames import FILENAME_DOMAIN_NAMES, dataset_path
from apprate.utils import logmsg
from apprate.utils.logmsg import set_logger
from apprate.utils.methods import print_pandas

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
    winloss = ar.Winloss(
        normalization=False).rate_from_file(
        filename_dn,
        columns_dict=COLUMNS_DOMAIN_DICT)
    print_pandas(winloss)
    print(s)

    colley = ar.Colley().rate_from_file(filename_dn,
                                        columns_dict=COLUMNS_DOMAIN_DICT)
    print_pandas(colley)
    print(s)

    massey = ar.Massey().rate_from_file(filename_dn,
                                        columns_dict=COLUMNS_DOMAIN_DICT)
    print_pandas(massey)
    print(s)

    od = ar.OffenseDefense().rate_from_file(filename_dn,
                                            columns_dict=COLUMNS_DOMAIN_DICT)
    print_pandas(od)
    print(s)
