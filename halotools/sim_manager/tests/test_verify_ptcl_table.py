#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
import pytest 
import warnings, os

import numpy as np 
from copy import copy, deepcopy 

from astropy.table import Table
from astropy.table import vstack as table_vstack

from astropy.config.paths import _find_home 

from . import helper_functions 

from .. import manipulate_ptcl_table_cache_log

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

__all__ = ('TestLoadCachedPtclTable' )


class TestLoadCachedPtclTable(TestCase):
    """ 
    """

    def setUp(self):
        """ Pre-load various arrays into memory for use by all tests. 
        """
        self.dummy_cache_baseloc = helper_functions.dummy_cache_baseloc

        try:
            os.system('rm -rf ' + self.dummy_cache_baseloc)
        except OSError:
            pass


    def test_cache_existence_check(self):
        """ Verify that the appropriate HalotoolsError is raised 
        if trying to load a non-existent cache log.
        """
        scenario = 0
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)
        try:
            os.makedirs(cache_dirname)
        except OSError:
            pass

        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'rockstar', 0.00004, 'halotools.alpha.version0')
        helper_functions.create_ptcl_table_hdf5(updated_log[-1])

        fname = updated_log['fname'][0]
        with pytest.raises(HalotoolsError):
            _ = manipulate_ptcl_table_cache_log.return_ptcl_table_fname_after_verification(
                fname = fname, cache_fname = cache_fname)

        assert not os.path.isfile(cache_fname)
        manipulate_ptcl_table_cache_log.overwrite_ptcl_table_cache_log(
            updated_log, cache_fname = cache_fname)
        assert os.path.isfile(cache_fname)

        _ = manipulate_ptcl_table_cache_log.return_ptcl_table_fname_after_verification(
            fname = fname, cache_fname = cache_fname)









