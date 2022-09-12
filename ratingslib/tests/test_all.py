"""
Module for testing all unit-test of the ratingslib project
"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT


from ratingslib.utils.logmsg import FILENAME, set_logger
from ratingslib.utils.methods import print_info, clean_args


import unittest

set_logger(FILENAME)
TEST_MODULES = [
    "ratingslib.utils.tests.test_all",
    "ratingslib.ratings.tests.test_all",
    "ratingslib.app_sports.tests.test_all",
    "ratingslib.app_movies.tests.test_all"
]


def printdetails(fn):
    """Print the docstring of test"""
    from functools import wraps

    @wraps(fn)
    def wrapper(*args, **kwds):
        print_info(fn.__name__+" TEST")
        print(fn.__doc__)
        print_info("RUN")
        print("")
        return fn(*args, **kwds)
    return wrapper


def all_test_suite(suite, test_modules, func_name_str='suite', **kwargs):
    """Run all tests"""
    for t in test_modules:
        try:
            # If the module defines a suite() function,
            # call it to get the suite.
            mod = __import__(t, globals(), locals(), [func_name_str])
            suitefn = getattr(mod, func_name_str)
            fparameters = clean_args(kwargs, suitefn)
            suite.addTest(suitefn(**fparameters))
        except (ImportError, AttributeError):
            # else, just load all the test cases from the module.
            suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))

    return suite


if __name__ == '__main__':
    test_all_suite = all_test_suite(unittest.TestSuite(), TEST_MODULES,
                                    run_type="MINIMAL")
    unittest.TextTestRunner().run(test_all_suite)
