#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
import pytest 
import warnings, os

import numpy as np 
from copy import copy, deepcopy 

from astropy.table import Table
from astropy.config.paths import _find_home 

from .. import manipulate_cache_log

from ...custom_exceptions import HalotoolsError

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

__all__ = ('TestCacheManipulation', )

class TestCacheManipulation(TestCase):
    """ 
    """

    def setup_class(self):
        """ Pre-load various arrays into memory for use by all tests. 
        """

        self.temp_dirname = os.path.join(detected_home, 
            'Desktop', 'temp_dummy_cache_dirname')

        if os.path.isdir(self.temp_dirname) is False:
            os.mkdir(self.temp_dirname)
        else:
            os.system('rm -rf ' + self.temp_dirname)
            os.mkdir(self.temp_dirname)


    @pytest.mark.skipif('not APH_MACHINE')
    def test_read_write(self):
        dummy_fname = os.path.join(self.temp_dirname, 'dummy_cache_log.txt')

        simname = ['bolshoi', 'bolshoi']
        redshift = [0, 1]
        halo_finder = ['rockstar', 'rockstar']
        version_name = ['beta_v0', 'beta_v0']
        fname = ['whatever_fname1', 'whatever_fname2']

        table = Table({'simname': simname, 'redshift': redshift, 
            'halo_finder': halo_finder, 'version_name': version_name, 
            'fname': fname})













