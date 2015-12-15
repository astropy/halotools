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
            'bolshoi', 'rockstar', 0.00004, 'halotools.alpha.version0')
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

    @pytest.mark.skipif('not APH_MACHINE')
    def test_scenario1(self):
        """ There is a one-to-one match between log entries and halo tables. 
        Only one version exists. 
        All entries have exactly matching metadata. 
        """

        #################### SETUP ####################
        scenario = 1
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)

        # Create a new log entry and accompanying halo table 
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'rockstar', 0.00004, 'halotools.alpha.version0')
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        # Create a new log entry and accompanying halo table 
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'rockstar', 1.23456, 'halotools.alpha.version0', 
            existing_table = updated_log)
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        # Create a new log entry and accompanying halo table 
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.010101, 'halotools.alpha.version0', 
            existing_table = updated_log)
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        # Now write the log file to disk using a dummy location so that the real cache is left alone
        manipulate_cache_log.overwrite_halo_table_cache_log(
            updated_log, cache_fname = cache_fname)

        #################################################################
        ##### First we perform tests passing in absolute fnames #####

        # For each entry in the log, load the halo table into memory 
        # with an input fname taken directly from the log entry
        for ii, entry in enumerate(updated_log):
            fname = entry['fname']
            _ = manipulate_cache_log.load_cached_halo_table_from_fname(fname = fname, 
                cache_fname = cache_fname)

        #################################################################
        ##### Now we perform various tests using the simname shorthands #####

        # Pass in a complete, correct set of metadata
        _ = manipulate_cache_log.load_cached_halo_table_from_simname(
            cache_fname = cache_fname, 
            simname = 'bolshoi', halo_finder = 'bdm', redshift = 0.01, 
            version_name = 'halotools.alpha.version0')

        # Pass in an incomplete-but-sufficient-and-correct set of metadata
        _ = manipulate_cache_log.load_cached_halo_table_from_simname(
            cache_fname = cache_fname, 
            halo_finder = 'bdm', redshift = 0.01, 
            version_name = 'halotools.alpha.version0')

        # Pass in a complete set of metadata with a slightly incorrect redshift
        _ = manipulate_cache_log.load_cached_halo_table_from_simname(
            cache_fname = cache_fname, 
            simname = 'bolshoi', halo_finder = 'bdm', redshift = 0.02, 
            version_name = 'halotools.alpha.version0')

        # Pass in a complete set of metadata with a badly incorrect redshift
        with pytest.raises(HalotoolsError) as err:
            _ = manipulate_cache_log.load_cached_halo_table_from_simname(
                cache_fname = cache_fname, 
                simname = 'bolshoi', halo_finder = 'rockstar', redshift = 2., 
                version_name = 'halotools.alpha.version0')
        assert 'closest available redshift is' in err.value.message
        assert '0.0' not in err.value.message
        assert '2.0' in err.value.message
        assert '1.235' in err.value.message

        # Pass in a complete set of metadata with a slightly incorrect, negative redshift
        _ = manipulate_cache_log.load_cached_halo_table_from_simname(
            cache_fname = cache_fname, 
            simname = 'bolshoi', halo_finder = 'bdm', redshift = -0.001, 
            version_name = 'halotools.alpha.version0')

        # Pass in a complete set of metadata with a matching default halo_finder
        _ = manipulate_cache_log.load_cached_halo_table_from_simname(
            cache_fname = cache_fname, 
            simname = 'bolshoi', redshift = 1.25, 
            version_name = 'halotools.alpha.version0')

        # Pass in the wrong version name 
        with pytest.raises(HalotoolsError):
            _ = manipulate_cache_log.load_cached_halo_table_from_simname(
                cache_fname = cache_fname, 
                simname = 'bolshoi', redshift = 1.25, 
                version_name = 'halotools.beta.version0')


    def test_scenario2(self):
        """ There is a one-to-one match between log entries and halo tables. 
        Only one version exists. 
        One entry has mis-matched simname metadata. 
        """
        #################### SETUP ####################
        scenario = 2
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)

        # Create a new log entry and accompanying halo table 
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'rockstar', 0.00004, 'halotools.alpha.version0')
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        # Create a new log entry and accompanying halo table 
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'rockstar', 1.23456, 'halotools.alpha.version0', 
            existing_table = updated_log)
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        # Create a new log entry and accompanying halo table 
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.010101, 'halotools.alpha.version0', 
            existing_table = updated_log)
        helper_functions.create_halo_table_hdf5(updated_log[-1], simname='Jose Canseco')

        # Now write the log file to disk using a dummy location so that the real cache is left alone
        manipulate_cache_log.overwrite_halo_table_cache_log(
            updated_log, cache_fname = cache_fname)

        #################################################################
        ##### First we perform tests passing in absolute fnames #####

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

        #################################################################
        ##### Now we perform various tests using the simname shorthands #####

        # Pass in a complete set of metadata that disagrees with the 
        # metadata stored in the hdf5 file
        with pytest.raises(HalotoolsError) as err:
            _ = manipulate_cache_log.load_cached_halo_table_from_simname(
                cache_fname = cache_fname, 
                simname = 'bolshoi', halo_finder = 'bdm', redshift = 0.01, 
                version_name = 'halotools.alpha.version0')
        assert 'You can make the correction as follows:' in err.value.message

        ## Now verify that the proposed solution works
        import h5py
        fname = updated_log['fname'][-1]
        f = h5py.File(fname)
        f.attrs.create('simname', 'bolshoi')
        f.close()

        _ = manipulate_cache_log.load_cached_halo_table_from_simname(
            cache_fname = cache_fname, 
            simname = 'bolshoi', halo_finder = 'bdm', redshift = 0.01, 
            version_name = 'halotools.alpha.version0')

    def test_scenario3(self):
        """ There are two entries that differ by both a redshift and a version name
        """
        #################### SETUP ####################
        scenario = 3
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)

        # Create a new log entry and accompanying halo table 
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'rockstar', 0.004, 'halotools.alpha.version0')
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        # Create a new log entry and accompanying halo table 
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'rockstar', 0.4, 'alpha.version1', 
            existing_table = updated_log)
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        # Now write the log file to disk using a dummy location so that the real cache is left alone
        manipulate_cache_log.overwrite_halo_table_cache_log(
            updated_log, cache_fname = cache_fname)

        #################################################################
        ##### First we perform tests passing in absolute fnames #####

        # Load the first halo table with the correct arguments
        fname = updated_log['fname'][0]
        _ = manipulate_cache_log.load_cached_halo_table_from_fname(fname = fname, 
            cache_fname = cache_fname)

        # Load the second halo table with the correct arguments
        fname = updated_log['fname'][1]
        _ = manipulate_cache_log.load_cached_halo_table_from_fname(fname = fname, 
            cache_fname = cache_fname)

        #################################################################
        ##### Now we perform various tests using the simname shorthands #####

        # Load the first halo table with the correct arguments
        _ = manipulate_cache_log.load_cached_halo_table_from_simname(
            cache_fname = cache_fname, 
            simname = 'bolshoi', halo_finder = 'rockstar', redshift = 0.004, 
            version_name = 'halotools.alpha.version0')

        # Load the second halo table with the correct arguments
        _ = manipulate_cache_log.load_cached_halo_table_from_simname(
            cache_fname = cache_fname, 
            simname = 'bolshoi', halo_finder = 'rockstar', redshift = 0.4, 
            version_name = 'alpha.version1')

        # Load the first halo table with incorrect redshift that gives a suggested alternative
        with pytest.raises(HalotoolsError) as err:
            _ = manipulate_cache_log.load_cached_halo_table_from_simname(
                cache_fname = cache_fname, 
                simname = 'bolshoi', halo_finder = 'rockstar', redshift = 0.4, 
                version_name = 'halotools.alpha.version0')
        assert ('Alternatively, you do have an alternate version of this catalog' in 
            err.value.message)
        assert 'the closest available redshift is ' in err.value.message

        # Load the first halo table with incorrect redshift but a correct version name
        # this triggers the 'len(matches_no_redshift_mask) > 0' branch
        with pytest.raises(HalotoolsError) as err:
            _ = manipulate_cache_log.load_cached_halo_table_from_simname(
                cache_fname = cache_fname, 
                simname = 'bolshoi', halo_finder = 'rockstar', redshift = 4, 
                version_name = 'halotools.alpha.version0')
        assert ('Alternatively, you do have an alternate version of this catalog' not in 
            err.value.message)
        assert 'the closest available redshift is ' in err.value.message

        # Load the first halo table with the correct redshift but a mis-spelled version name
        # this triggers the 'if len(matches_no_redshift_mask) == 0' branch
        with pytest.raises(HalotoolsError) as err:
            _ = manipulate_cache_log.load_cached_halo_table_from_simname(
                cache_fname = cache_fname, 
                simname = 'bolshoi', halo_finder = 'rockstar', redshift = 0.4, 
                version_name = 'halotools.alpersion0')
        assert ('Alternatively, you do have an alternate version of this catalog' in 
            err.value.message)
        assert 'the closest available redshift is ' not in err.value.message


    def test_scenario4(self):
        """ There are two identical entries that differ only by a halo-finder
        """
        #################### SETUP ####################
        scenario = 4
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)

        # Create a new log entry and accompanying halo table 
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.004, 'halotools.alpha.version0')
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        # Create a new log entry and accompanying halo table 
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'rockstar', 0.004, 'halotools.alpha.version0', 
            existing_table = updated_log)
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        # Now write the log file to disk using a dummy location so that the real cache is left alone
        manipulate_cache_log.overwrite_halo_table_cache_log(
            updated_log, cache_fname = cache_fname)


        #################################################################
        ##### First we perform tests passing in absolute fnames #####

        # Load the first halo table with the correct fname arguments
        _ = manipulate_cache_log.load_cached_halo_table_from_fname(
            fname = updated_log['fname'][0], 
            cache_fname = cache_fname)

        #################################################################
        ##### Now we perform various tests using the simname shorthands #####

        # Load the first halo table with the correct simname arguments
        _ = manipulate_cache_log.load_cached_halo_table_from_simname(
            cache_fname = cache_fname, 
            simname = updated_log['simname'][0], 
            halo_finder = updated_log['halo_finder'][0], 
            redshift = updated_log['redshift'][0], 
            version_name = updated_log['version_name'][0])

        # Change the hdf5 metadata of entry 0 by swapping the version name from version 1
        import h5py
        f = h5py.File(updated_log['fname'][0])
        f.attrs.create('halo_finder', updated_log['halo_finder'][1])
        f.close()

        # Verify that an error is raised when loading from input fname
        with pytest.raises(HalotoolsError) as err:
            _ = manipulate_cache_log.load_cached_halo_table_from_fname(
                fname = updated_log['fname'][0], 
                cache_fname = cache_fname)
        assert 'inconsistent with the ``bdm`` value that you requested' in err.value.message

        # Verify that an error is raised when loading from input simname
        with pytest.raises(HalotoolsError) as err:
            _ = manipulate_cache_log.load_cached_halo_table_from_simname(
                cache_fname = cache_fname, 
                simname = updated_log['simname'][0], 
                halo_finder = updated_log['halo_finder'][0], 
                redshift = updated_log['redshift'][0], 
                version_name = updated_log['version_name'][0])
        assert 'inconsistent with the ``bdm`` value that you requested' in err.value.message


        ## Now verify that the proposed solution works
        import h5py
        fname = updated_log['fname'][0]
        f = h5py.File(fname)
        f.attrs.create('halo_finder', 'bdm')
        f.close()

        _ = manipulate_cache_log.load_cached_halo_table_from_simname(
            cache_fname = cache_fname, 
            simname = updated_log['simname'][0], 
            halo_finder = updated_log['halo_finder'][0], 
            redshift = updated_log['redshift'][0], 
            version_name = updated_log['version_name'][0])

    def test_scenario5(self):
        """ A non-existent file is requested
        """
        #################### SETUP ####################
        scenario = 5
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)

        # Create a new log entry and accompanying halo table 
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.004, 'halotools.alpha.version0')
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        # Create a new log entry and accompanying halo table 
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'rockstar', 0.004, 'halotools.alpha.version0', 
            existing_table = updated_log)
        helper_functions.create_halo_table_hdf5(updated_log[-1])

        # Now write the log file to disk using a dummy location so that the real cache is left alone
        manipulate_cache_log.overwrite_halo_table_cache_log(
            updated_log, cache_fname = cache_fname)

        # Verify that the appropriate exception is raised when passing in a nonsense fname
        with pytest.raises(HalotoolsError) as err:
            fname = 'Jose Canseco'
            _ = manipulate_cache_log.load_cached_halo_table_from_fname(fname = fname, 
                cache_fname = cache_fname)
        assert 'does not exist' in err.value.message

        # Verify that the file can be loaded from the correct simname 
        _ = manipulate_cache_log.load_cached_halo_table_from_simname(
            cache_fname = cache_fname, 
            simname = updated_log['simname'][0], 
            halo_finder = updated_log['halo_finder'][0], 
            redshift = updated_log['redshift'][0], 
            version_name = updated_log['version_name'][0])

        # Manually delete the file
        os.system('rm ' + updated_log['fname'][0])

        # Verify that the appropriate exception is raised now that the file is gone 
        with pytest.raises(HalotoolsError) as err:
            _ = manipulate_cache_log.load_cached_halo_table_from_simname(
                cache_fname = cache_fname, 
                simname = updated_log['simname'][0], 
                halo_finder = updated_log['halo_finder'][0], 
                redshift = updated_log['redshift'][0], 
                version_name = updated_log['version_name'][0])
        assert 'This file does not exist' in err.value.message

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
        final_log = manipulate_cache_log.read_halo_table_cache_log(cache_fname = cache_fname)
        assert len(final_log) == 1



    def test_scenario7(self):
        """ There are entries in the log that point to nonexistent files. 
        """
        #################### SETUP ####################
        scenario = 7
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)

        # Create a new log entry and accompanying halo table 
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.004, 'halotools.alpha.version0')
        helper_functions.create_halo_table_hdf5(updated_log[0])

        # Create a new log entry and accompanying halo table 
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.004, 'halotools.alpha.version0', 
            existing_table = updated_log)
        # Do not create a table as we need it to be non-existent 

        # Create a new log entry and accompanying halo table 
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolplanck', 'bdm', 0.104, 'beta.version0', 
            existing_table = updated_log)
        # Do not create a table as we need it to be non-existent 

        # Now write the log file to disk using a dummy location so that the real cache is left alone
        manipulate_cache_log.overwrite_halo_table_cache_log(
            updated_log, cache_fname = cache_fname)

        # The first file does exist and can be loaded from an explicit fname
        _ = manipulate_cache_log.load_cached_halo_table_from_fname(
            fname = updated_log['fname'][0], cache_fname = cache_fname)

        # The first file does exist and can be loaded from metadata 
        _ = manipulate_cache_log.load_cached_halo_table_from_simname(
            cache_fname = cache_fname, 
            simname = updated_log['simname'][0], 
            halo_finder = updated_log['halo_finder'][0], 
            redshift = updated_log['redshift'][0], 
            version_name = updated_log['version_name'][0], 
            )

        # The third file in the log does not exist 
        # and an exception is raised when passing an explicit fname
        with pytest.raises(HalotoolsError) as err:
            _ = manipulate_cache_log.load_cached_halo_table_from_fname(
                fname = updated_log['fname'][2], cache_fname = cache_fname)
        assert 'located on an external disk that is' in err.value.message
        assert 'You tried to load a halo catalog by' in err.value.message 

        # The third file in the log does not exist 
        # and an exception is raised when passing in metadata
        with pytest.raises(HalotoolsError) as err:
            _ = manipulate_cache_log.load_cached_halo_table_from_simname(
                cache_fname = cache_fname, 
                simname = updated_log['simname'][2], 
                halo_finder = updated_log['halo_finder'][2], 
                redshift = updated_log['redshift'][2], 
                version_name = updated_log['version_name'][2], 
                )
        assert 'You requested to load a halo catalog' in err.value.message
    def test_scenario8(self):
        """ There is a halo table stored in a custom location.  
        """
        #################### SETUP ####################
        scenario = 8
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

        # Now write the log file to disk using a dummy location so that the real cache is left alone
        # The fact that alt_halo_table_loc is not a sub-directory of 
        # os.path.dirname(cache_fname) is what makes this scenario different
        manipulate_cache_log.overwrite_halo_table_cache_log(
            updated_log, cache_fname = cache_fname)

        # Verify that we can load the file when passing in an explicit fname
        _ = manipulate_cache_log.load_cached_halo_table_from_fname(
            fname = updated_log['fname'][0], cache_fname = cache_fname)

        # Verify that we can load the file when passing in metadata
        _ = manipulate_cache_log.load_cached_halo_table_from_simname(
            cache_fname = cache_fname, 
            simname = updated_log['simname'][0], 
            halo_finder = updated_log['halo_finder'][0],
            redshift = updated_log['redshift'][0], 
            version_name = updated_log['version_name'][0]        
            )

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
        assert 'appears multiple times in the halo table cache log,' in err.value.message

        # Now correct the log and ensure the error goes away
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 'bdm', 0.004, 'halotools.alpha.version0', 
            fname = os.path.join(alt_halo_table_loc, 'dummy_halo_table.hdf5'))
        manipulate_cache_log.overwrite_halo_table_cache_log(
            updated_log, cache_fname = cache_fname)

        # Passing in an fname should work now
        _ = manipulate_cache_log.load_cached_halo_table_from_fname(
            fname = updated_log['fname'][0], cache_fname = cache_fname)

        # Passing in metadata should also work now
        _ = manipulate_cache_log.load_cached_halo_table_from_simname(
            cache_fname = cache_fname, 
            simname = updated_log['simname'][0], 
            halo_finder = updated_log['halo_finder'][0], 
            redshift = updated_log['redshift'][0], 
            version_name = updated_log['version_name'][0]
            )
        
    def tearDown(self):
        try:
            os.system('rm -rf ' + self.dummy_cache_baseloc)
        except OSError:
            pass


























        
