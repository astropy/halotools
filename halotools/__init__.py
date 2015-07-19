# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Halotools is a python package designed to analyze N-body simulations 
and constrain models of cosmology and galaxy evolution. 
"""

# ----------------------------------------------------------------------------
# keep this content at the top.
from ._astropy_init import *

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
	pass
	# ----------------------------------------------------------------------------

from halotools_exceptions import *