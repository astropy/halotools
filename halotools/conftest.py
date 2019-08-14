# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.
from distutils.version import LooseVersion

from astropy.version import version as astropy_version

if LooseVersion(astropy_version) < LooseVersion('2.0.3'):
    # Astropy is not compatible with the standalone plugins prior this while
    # astroquery requires them, so we need this workaround. This will mess
    # up the test header, but everything else will work.
    from astropy.tests.pytest_plugins import (PYTEST_HEADER_MODULES,
                                              enable_deprecations_as_exceptions,
                                              TESTED_VERSIONS)
elif LooseVersion(astropy_version) < LooseVersion('3.0'):
    # With older versions of Astropy, we actually need to import the pytest
    # plugins themselves in order to make them discoverable by pytest.
    from astropy.tests.pytest_plugins import *
else:
    # As of Astropy 3.0, the pytest plugins provided by Astropy are
    # automatically made available when Astropy is installed. This means it's
    # not necessary to import them here, but we still need to import global
    # variables that are used for configuration.
    from astropy.tests.plugins.display import (PYTEST_HEADER_MODULES,
                                               TESTED_VERSIONS)

from astropy.tests.helper import enable_deprecations_as_exceptions

from .version import version, astropy_helpers_version

if LooseVersion(astropy_version) > LooseVersion('1'):
    # The warnings_to_ignore_by_pyver parameter was added in astropy 2.0
    enable_deprecations_as_exceptions(modules_to_ignore_on_import=['requests'])

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

TESTED_VERSIONS['halotools'] = version
TESTED_VERSIONS['astropy_helpers'] = astropy_helpers_version
