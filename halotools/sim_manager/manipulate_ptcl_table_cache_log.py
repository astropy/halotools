""" This module contains functions used to read, interpret and 
update the ascii data file that keeps track of N-body simulation data. 
"""

__all__ = ('get_formatted_halo_table_cache_log_line', 
    'get_halo_table_cache_log_header', 
    'overwrite_halo_table_cache_log', 'read_halo_table_cache_log')

import numpy as np
import os, tempfile, fnmatch
from copy import copy, deepcopy 

from astropy.config.paths import get_cache_dir as get_astropy_cache_dir
from astropy.config.paths import _find_home
from astropy.table import Table
from astropy.table import vstack as table_vstack

from astropy.io import ascii as astropy_ascii

from warnings import warn 
import datetime

try:
    import h5py
except ImportError:
    warn("\nMost of the functionality of the sim_manager sub-package"
    " requires h5py to be installed,\n"
        "which can be accomplished either with pip or conda")

from . import sim_defaults

from ..custom_exceptions import HalotoolsError
