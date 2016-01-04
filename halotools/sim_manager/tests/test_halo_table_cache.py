#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
import pytest 
import warnings, os

import numpy as np 
from copy import copy, deepcopy 

from astropy.config.paths import _find_home 

from ..log_entry import HaloTableCacheLogEntry
from ..halo_table_cache import HaloTableCache

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

__all__ = ('TestHaloTableCache' )

class TestHaloTableCache(TestCase):
	"""
	"""

	def setUp(self):
		self.empty_cache = HaloTableCache(read_log_from_standard_loc = False)


	def tearDown(self):
		pass

