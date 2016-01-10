#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
import pytest 
import warnings, os, shutil

import numpy as np 
from copy import copy, deepcopy 

from astropy.config.paths import _find_home 
from astropy.table import Table
from astropy.table import vstack as table_vstack

from . import helper_functions

from ..halo_table_cache_log_entry import HaloTableCacheLogEntry, get_redshift_string
from ..halo_table_cache import HaloTableCache

from ..ptcl_table_cache_log_entry import PtclTableCacheLogEntry
from ..ptcl_table_cache import PtclTableCache

from ...custom_exceptions import InvalidCacheLogEntry, HalotoolsError

### Determine whether the machine is mine
# This will be used to select tests whose 
# returned values depend on the configuration 
# of my personal cache directory files
aph_home = u'/Users/aphearin'
detected_home = _find_home()
if aph_home == detected_home:
    APH_MACHINE = True
else:
    APH_MACHINE = False

__all__ = ('TestPtclTableCache',  )

class TestPtclTableCache(TestCase):
    """
    """

    def setUp(self):
        import h5py
        self.h5py = h5py

        self.dummy_cache_baseloc = helper_functions.dummy_cache_baseloc
        try:
            shutil.rmtree(self.dummy_cache_baseloc)
        except:
            pass
        os.makedirs(self.dummy_cache_baseloc)



    def tearDown(self):
        try:
            shutil.rmtree(self.dummy_cache_baseloc)
        except:
            pass



























