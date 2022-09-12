"""
Run all test of rating package from this module

"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

import unittest
from ratingslib.tests.test_all import all_test_suite

pack_prefix = 'ratingslib.utils.tests.test_'
modules_names = [
    'methods',
]
TEST_MODULES = [pack_prefix + t for t in modules_names]


def suite():
    return all_test_suite(unittest.TestSuite(), TEST_MODULES)


if __name__ == "__main__":
    test_all_suite = suite()
    unittest.TextTestRunner().run(test_all_suite)
