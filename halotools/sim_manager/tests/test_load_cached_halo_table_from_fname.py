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

__all__ = ('TestLoadCachedHaloTableFromFname' )


class TestLoadCachedHaloTableFromFname(TestCase):
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
            'bolshoi', 'rockstar', 0.00004, 'alpha.version0')
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        fname = updated_log['fname'][0]
        with pytest.raises(HalotoolsError):
            _ = manipulate_cache_log.load_cached_halo_table_from_fname(
                fname = fname, cache_fname = cache_fname)

        assert not os.path.isfile(cache_fname)
        manipulate_cache_log.overwrite_halo_table_cache_log(
            updated_log, cache_fname = cache_fname)
        assert os.path.isfile(cache_fname)

        _ = manipulate_cache_log.load_cached_halo_table_from_fname(
            fname = fname, cache_fname = cache_fname)

    def test_scenario1(self):
        """ There is a one-to-one match between log entries and halo tables. 
        Only one version exists. 
        All entries have exactly matching metadata. 
        """
        scenario = 1
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)

        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'rockstar', 0.00004, 'alpha.version0')
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'rockstar', 1.23456, 'alpha.version0', 
            existing_table = updated_log)
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.010101, 'alpha.version0', 
            existing_table = updated_log)
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        manipulate_cache_log.overwrite_halo_table_cache_log(
            updated_log, cache_fname = cache_fname)

        for entry in updated_log:
            fname = entry['fname']
            _ = manipulate_cache_log.load_cached_halo_table_from_fname(fname = fname, 
                cache_fname = cache_fname)


    def test_scenario2(self):
        """ There is a one-to-one match between log entries and halo tables. 
        Only one version exists. 
        One entry has mis-matched simname metadata. 
        """
        scenario = 2
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)

        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'rockstar', 0.00004, 'alpha.version0')
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'rockstar', 1.23456, 'alpha.version0', 
            existing_table = updated_log)
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.010101, 'alpha.version0', 
            existing_table = updated_log)
        helper_functions.create_halo_table_hdf5(updated_log[-1], simname='marf')

        manipulate_cache_log.overwrite_halo_table_cache_log(
            updated_log, cache_fname = cache_fname)

        fname = updated_log['fname'][0]
        _ = manipulate_cache_log.load_cached_halo_table_from_fname(fname = fname, 
            cache_fname = cache_fname)
        fname = updated_log['fname'][1]
        _ = manipulate_cache_log.load_cached_halo_table_from_fname(fname = fname, 
            cache_fname = cache_fname)
        with pytest.raises(HalotoolsError) as err:
            fname = updated_log['fname'][-1]
            _ = manipulate_cache_log.load_cached_halo_table_from_fname(fname = fname, 
                cache_fname = cache_fname)
        assert 'If you are using your own halo catalog' in err.value.message


    def test_scenario3(self):
        """ There are two identical entries that differ only by a version name
        """
        scenario = 3
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)

        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'rockstar', 0.004, 'alpha.version0')
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'rockstar', 0.004, 'alpha.version1', 
            existing_table = updated_log)
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        manipulate_cache_log.overwrite_halo_table_cache_log(
            updated_log, cache_fname = cache_fname)

        fname = updated_log['fname'][0]
        _ = manipulate_cache_log.load_cached_halo_table_from_fname(fname = fname, 
            cache_fname = cache_fname)

        fname = updated_log['fname'][1]
        _ = manipulate_cache_log.load_cached_halo_table_from_fname(fname = fname, 
            cache_fname = cache_fname)

    def test_scenario4(self):
        """ There are two identical entries that differ only by a halo-finder
        """
        scenario = 4
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)

        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.004, 'alpha.version0')
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'rockstar', 0.004, 'alpha.version0', 
            existing_table = updated_log)
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        manipulate_cache_log.overwrite_halo_table_cache_log(
            updated_log, cache_fname = cache_fname)

        fname = updated_log['fname'][0]
        _ = manipulate_cache_log.load_cached_halo_table_from_fname(fname = fname, 
            cache_fname = cache_fname)

        fname = updated_log['fname'][1]
        _ = manipulate_cache_log.load_cached_halo_table_from_fname(fname = fname, 
            cache_fname = cache_fname)

    def test_scenario4(self):
        """ A non-existent file is requested
        """
        scenario = 4
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)

        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.004, 'alpha.version0')
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'rockstar', 0.004, 'alpha.version0', 
            existing_table = updated_log)
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        manipulate_cache_log.overwrite_halo_table_cache_log(
            updated_log, cache_fname = cache_fname)

        with pytest.raises(HalotoolsError) as err:
            fname = 'marf'
            _ = manipulate_cache_log.load_cached_halo_table_from_fname(fname = fname, 
                cache_fname = cache_fname)
        assert 'does not exist' in err.value.message

    def test_scenario5(self):
        """ There are harmless duplicate entries in the log
        """
        scenario = 5
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)

        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.004, 'alpha.version0')
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.004, 'alpha.version0', 
            existing_table = updated_log)
        # Do not create another table as it already exists 
        
        manipulate_cache_log.overwrite_halo_table_cache_log(
            updated_log, cache_fname = cache_fname)

        fname = updated_log['fname'][1]
        _ = manipulate_cache_log.load_cached_halo_table_from_fname(fname = fname, 
            cache_fname = cache_fname)

    def test_scenario6(self):
        """ There are entries in the log that point to nonexistent files. 
        """
        scenario = 6
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)

        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.004, 'alpha.version0')
        helper_functions.create_halo_table_hdf5(updated_log[0])

        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.004, 'alpha.version0', 
            existing_table = updated_log)

        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolplanck', 'bdm', 0.104, 'beta.version0', 
            existing_table = updated_log)

        manipulate_cache_log.overwrite_halo_table_cache_log(
            updated_log, cache_fname = cache_fname)

        _ = manipulate_cache_log.load_cached_halo_table_from_fname(
            fname = updated_log['fname'][0], cache_fname = cache_fname)

        with pytest.raises(HalotoolsError) as err:
            _ = manipulate_cache_log.load_cached_halo_table_from_fname(
                fname = updated_log['fname'][2], cache_fname = cache_fname)
        assert 'does not exist' in err.value.message

    def test_scenario7(self):
        """ There is a halo table stored in a custom location.  
        """
        scenario = 7
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)

        alt_halo_table_loc = os.path.join(
            helper_functions.dummy_cache_baseloc, 'alt_halo_table_loc')
        try:
            os.makedirs(alt_halo_table_loc)
        except OSError:
            pass

        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.004, 'alpha.version0', 
            fname = os.path.join(alt_halo_table_loc, 'dummy_halo_table.hdf5'))
        helper_functions.create_halo_table_hdf5(updated_log[0])

        manipulate_cache_log.overwrite_halo_table_cache_log(
            updated_log, cache_fname = cache_fname)

        _ = manipulate_cache_log.load_cached_halo_table_from_fname(
            fname = updated_log['fname'][0], cache_fname = cache_fname)

    def test_scenario8(self):
        """ There are duplicate entries in the log that have mutually inconsistent data. 
        """
        scenario = 8
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)

        alt_halo_table_loc = os.path.join(
            helper_functions.dummy_cache_baseloc, 'alt_halo_table_loc')
        try:
            os.makedirs(alt_halo_table_loc)
        except OSError:
            pass

        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.004, 'alpha.version0', 
            fname = os.path.join(alt_halo_table_loc, 'dummy_halo_table.hdf5'))
        helper_functions.create_halo_table_hdf5(updated_log[0])

        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.004, 'beta.version0', 
            fname = os.path.join(alt_halo_table_loc, 'dummy_halo_table.hdf5'), 
            existing_table = updated_log)

        manipulate_cache_log.overwrite_halo_table_cache_log(
            updated_log, cache_fname = cache_fname)

        with pytest.raises(HalotoolsError) as err:
            _ = manipulate_cache_log.load_cached_halo_table_from_fname(
                fname = updated_log['fname'][0], cache_fname = cache_fname)
        assert 'appears multiple times in the halo table cache log,' in err.value.message

        # Now correct the log and ensure the error goes away
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.004, 'alpha.version0', 
            fname = os.path.join(alt_halo_table_loc, 'dummy_halo_table.hdf5'))
        manipulate_cache_log.overwrite_halo_table_cache_log(
            updated_log, cache_fname = cache_fname)
        _ = manipulate_cache_log.load_cached_halo_table_from_fname(
            fname = updated_log['fname'][0], cache_fname = cache_fname)

    def tearDown(self):
        try:
            os.system('rm -rf ' + self.dummy_cache_baseloc)
        except OSError:
            pass


























        
