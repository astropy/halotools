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

__all__ = ('TestLoadCachedHaloTableFromSimname', )

class TestLoadCachedHaloTableFromSimname(TestCase):
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

    def test_scenario6(self):
        """ There are harmless duplicate entries in the log
        """
        #################### SETUP ####################
        scenario = 6
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)

        # Create a new log entry and accompanying halo table 
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.004, 'halotools.alpha.version0')
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        # Create a new log entry and accompanying halo table 
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.004, 'halotools.alpha.version0', 
            existing_table = updated_log)
        # Do not create another table as it already exists 
        
        # Now write the log file to disk using a dummy location so that the real cache is left alone
        manipulate_cache_log.overwrite_halo_table_cache_log(
            updated_log, cache_fname = cache_fname)

        #################################################################
        ##### First we perform tests passing in absolute fnames #####

        log = manipulate_cache_log.read_halo_table_cache_log(cache_fname = cache_fname)
        assert len(log) == 2
        fname = updated_log['fname'][1]
        _ = manipulate_cache_log.load_cached_halo_table_from_fname(fname = fname, 
            cache_fname = cache_fname)
        log = manipulate_cache_log.read_halo_table_cache_log(cache_fname = cache_fname)
        assert len(log) == 1

        #################################################################
        ##### Now we perform various tests using the simname shorthands #####

        # re-add the duplicate log entry
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.004, 'halotools.alpha.version0', 
            existing_table = log)
        # Do not create another table as it already exists 
        log2 = manipulate_cache_log.read_halo_table_cache_log(cache_fname = cache_fname)
        assert len(log2) == 1
        manipulate_cache_log.overwrite_halo_table_cache_log(
            updated_log, cache_fname = cache_fname)
        log2 = manipulate_cache_log.read_halo_table_cache_log(cache_fname = cache_fname)
        assert len(log2) == 2

        # Load the first halo table with the correct simname arguments
        _ = manipulate_cache_log.load_cached_halo_table_from_simname(
            cache_fname = cache_fname, 
            simname = log2['simname'][0], 
            halo_finder = log2['halo_finder'][0], 
            redshift = log2['redshift'][0], 
            version_name = log2['version_name'][0])




        
