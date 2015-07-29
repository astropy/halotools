# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.

from astropy.tests.pytest_plugins import *

## Uncomment the following line to treat all DeprecationWarnings as
## exceptions
# enable_deprecations_as_exceptions()


def pytest_addoption_decorator(func):

    def wrapper(parser):
        baseline_behavior = func(parser)

        parser.addoption('--slow', action='store_true', default=False,
            help='Also run slow tests')

    return wrapper
pytest_addoption = pytest_addoption_decorator(pytest_addoption)

# import pytest

# def pytest_addoption(parser):
#     parser.addoption('--slow', action='store_true', default=False,
#                       help='Also run slow tests')

# def pytest_runtest_setup(item):
#     """Skip tests if they are marked as slow and --slow is not given"""
#     if getattr(item.obj, 'slow', None) and not item.config.getvalue('slow'):
#         pytest.skip('slow tests not requested')
