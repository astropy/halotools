# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Halotools is a specialized python package
for building and testing models of the galaxy-halo connection,
and analyzing catalogs of dark matter halos.
"""

# ----------------------------------------------------------------------------
# keep this content at the top.
from ._astropy_init import *

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    pass
    # ----------------------------------------------------------------------------

from . import custom_exceptions
