# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.

from astropy.tests.pytest_plugins import *
import os
from . import version

# Uncomment the following line to treat all DeprecationWarnings as
# exceptions
enable_deprecations_as_exceptions()

# Uncomment and customize the following lines to add/remove entries from
# the list of packages for which version numbers are displayed when running
# the tests. Making it pass for KeyError is essential in some cases when
# the package uses other astropy affiliated packages.
try:
    PYTEST_HEADER_MODULES['Astropy'] = 'astropy'
    del PYTEST_HEADER_MODULES['Pandas']
except (NameError, KeyError):  # NameError is needed to support Astropy < 1.0
    pass


# This is to figure out the affiliated package version, rather than
# using Astropy's

packagename = os.path.basename(os.path.dirname(__file__))
TESTED_VERSIONS[packagename] = version.version
