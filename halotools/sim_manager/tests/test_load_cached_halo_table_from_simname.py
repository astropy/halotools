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


    def test_scenario9(self):
        """ There are duplicate entries in the log that have mutually inconsistent data. 
        """
        #################### SETUP ####################
        scenario = 9
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)

        # Create a custom location that differs from where the log is stored
        alt_halo_table_loc = os.path.join(
            helper_functions.dummy_cache_baseloc, 'alt_halo_table_loc')
        try:
            os.makedirs(alt_halo_table_loc)
        except OSError:
            pass

        # Create a new log entry and accompanying halo table 
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.004, 'halotools.alpha.version0', 
            fname = os.path.join(alt_halo_table_loc, 'dummy_halo_table.hdf5'))
        helper_functions.create_halo_table_hdf5(updated_log[0])

        # Create a new log entry and accompanying halo table 
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.004, 'beta.version0', 
            fname = os.path.join(alt_halo_table_loc, 'dummy_halo_table.hdf5'), 
            existing_table = updated_log)
        # Note that we have not created a new halo table, and that the version_name 
        # specified in the above line of code is incorrect but we have added it to the log

        # Now write the log file to disk using a dummy location so that the real cache is left alone
        manipulate_cache_log.overwrite_halo_table_cache_log(
            updated_log, cache_fname = cache_fname)

        # Verify that we catch the error when passing in an fname
        with pytest.raises(HalotoolsError) as err:
            _ = manipulate_cache_log.load_cached_halo_table_from_fname(
                fname = updated_log['fname'][0], cache_fname = cache_fname)
        assert 'appears multiple times in the halo table cache log,' in err.value.message

        with pytest.raises(HalotoolsError) as err:
            _ = manipulate_cache_log.load_cached_halo_table_from_simname(
                cache_fname = cache_fname, 
                simname = updated_log['simname'][0], 
                halo_finder = updated_log['halo_finder'][0], 
                redshift = updated_log['redshift'][0], 
                version_name = updated_log['version_name'][0]
                )

        _ = manipulate_cache_log.load_cached_halo_table_from_simname(
            cache_fname = cache_fname, 
            simname = updated_log['simname'][1], 
            halo_finder = updated_log['halo_finder'][1], 
            redshift = updated_log['redshift'][1], 
            version_name = updated_log['version_name'][1]
            )

        # Now correct the log and ensure the error goes away
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.004, 'halotools.alpha.version0', 
            fname = os.path.join(alt_halo_table_loc, 'dummy_halo_table.hdf5'))
        manipulate_cache_log.overwrite_halo_table_cache_log(
            updated_log, cache_fname = cache_fname)
        _ = manipulate_cache_log.load_cached_halo_table_from_fname(
            fname = updated_log['fname'][0], cache_fname = cache_fname)









        
