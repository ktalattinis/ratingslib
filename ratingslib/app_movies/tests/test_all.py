# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

import unittest
from ratingslib.tests.test_all import all_test_suite
import os


pack_prefix = 'ratingslib.app_movies.tests.test_'
modules_names = [
    'preparedata'
]
TEST_MODULES = [pack_prefix + t for t in modules_names]


TEST_FILES_PATH = os.path.dirname(__file__)+"/files/"


def suite(run_type="MINIMAL"):
    return all_test_suite(unittest.TestSuite(), TEST_MODULES,
                          run_type=run_type)


if __name__ == "__main__":
    test_all_suite = suite()
    unittest.TextTestRunner().run(test_all_suite)
