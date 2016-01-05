#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
import pytest 
import warnings, os, shutil

import numpy as np 
from copy import copy, deepcopy 

from astropy.table import Table
from astropy.table import vstack as table_vstack

from astropy.config.paths import _find_home 

from . import ptcl_helper_functions as helper_functions

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
            shutil.rmtree(self.dummy_cache_baseloc)
        except:
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
            'bolshoi', 0.00004, 'halotools.alpha.version0')
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

    @pytest.mark.xfail
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

        # Create a new log entry and accompanying particle table 
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 0.00004, 'halotools_alpha_version1')
        helper_functions.create_ptcl_table_hdf5(updated_log[-1])

        # Create a new log entry and accompanying particle table
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 1.23456, 'halotools_alpha_version1', 
            existing_table = updated_log)
        helper_functions.create_ptcl_table_hdf5(updated_log[-1])

        # Create a new log entry and accompanying particle table
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 0.110101, 'halotools_alpha_version1', 
            existing_table = updated_log)
        helper_functions.create_ptcl_table_hdf5(updated_log[-1])

        # Now write the log file to disk using a dummy location so that the real cache is left alone
        manipulate_ptcl_table_cache_log.overwrite_ptcl_table_cache_log(
            updated_log, cache_fname = cache_fname)

        #################################################################
        ##### First we perform tests passing in absolute fnames #####

        # Verify each entry in the log for an input  
        # fname taken directly from the log entry
        for ii, entry in enumerate(updated_log):
            fname = entry['fname']
            _ = manipulate_ptcl_table_cache_log.return_ptcl_table_fname_after_verification(fname = fname, 
                cache_fname = cache_fname)

        #################################################################
        ##### Now we perform various tests using the simname shorthands #####

        # Pass in a complete, correct set of metadata
        _ = manipulate_ptcl_table_cache_log.return_ptcl_table_fname_from_simname_inputs(
            cache_fname = cache_fname, 
            simname = 'bolshoi', redshift = .11, 
            version_name = 'halotools_alpha_version1')

        # Pass in an incomplete-but-sufficient-and-correct set of metadata
        _ = manipulate_ptcl_table_cache_log.return_ptcl_table_fname_from_simname_inputs(
            cache_fname = cache_fname, 
            redshift = .11, 
            version_name = 'halotools_alpha_version1')

        # Pass in a complete set of metadata with a slightly incorrect redshift
        _ = manipulate_ptcl_table_cache_log.return_ptcl_table_fname_from_simname_inputs(
            cache_fname = cache_fname, 
            simname = 'bolshoi', redshift = 0.02, 
            version_name = 'halotools_alpha_version1')

        # Pass in a complete set of metadata with a badly incorrect redshift
        with pytest.raises(HalotoolsError) as err:
            _ = manipulate_ptcl_table_cache_log.return_ptcl_table_fname_from_simname_inputs(
                cache_fname = cache_fname, 
                simname = 'bolshoi',  redshift = 2., 
                version_name = 'halotools_alpha_version1')
        assert 'corresponding entry of `halo_table_cache_log.txt' in err.value.message
        assert '0.0' not in err.value.message
        assert '2.0' in err.value.message

        assert 'closest available redshift is' in err.value.message
        # The line above is the error message raised by the corresponding halo_table test
        # For some reason, I instead to raise the line above it to get the right message
        # I don't understand what the cause of the different logical branching is, 
        # and this would be good to test since the ptcl_table versions of the code were made rather hastily

        # Pass in a complete set of metadata with a slightly incorrect, negative redshift
        _ = manipulate_ptcl_table_cache_log.return_ptcl_table_fname_from_simname_inputs(
            cache_fname = cache_fname, 
            simname = 'bolshoi', redshift = -0.001, 
            version_name = 'halotools_alpha_version1')

        # Pass in a complete set of matching metadata 
        _ = manipulate_ptcl_table_cache_log.return_ptcl_table_fname_from_simname_inputs(
            cache_fname = cache_fname, 
            simname = 'bolshoi', redshift = 1.25, 
            version_name = 'halotools_alpha_version1')

        # Pass in the wrong version name 
        with pytest.raises(HalotoolsError):
            _ = manipulate_ptcl_table_cache_log.return_ptcl_table_fname_from_simname_inputs(
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
            'bolshoi', 0.00004, 'halotools_alpha_version1')
        helper_functions.create_ptcl_table_hdf5(updated_log[-1])

        # Create a new log entry and accompanying halo table 
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi', 1.23456, 'halotools_alpha_version1', 
            existing_table = updated_log)
        helper_functions.create_ptcl_table_hdf5(updated_log[-1])

        # Create a new log entry and accompanying halo table 
        updated_log = helper_functions.add_new_row_to_cache_log(scenario, 
            'bolshoi',  0.100101, 'halotools_alpha_version1', 
            existing_table = updated_log)
        helper_functions.create_ptcl_table_hdf5(updated_log[-1], simname='Jose Canseco')

        # Now write the log file to disk using a dummy location so that the real cache is left alone
        manipulate_ptcl_table_cache_log.overwrite_ptcl_table_cache_log(
            updated_log, cache_fname = cache_fname)

        #################################################################
        ##### First we perform tests passing in absolute fnames #####

        fname = updated_log['fname'][0]
        _ = manipulate_ptcl_table_cache_log.return_ptcl_table_fname_after_verification(fname = fname, 
            cache_fname = cache_fname)
        fname = updated_log['fname'][1]
        _ = manipulate_ptcl_table_cache_log.return_ptcl_table_fname_after_verification(fname = fname, 
            cache_fname = cache_fname)
        with pytest.raises(HalotoolsError) as err:
            fname = updated_log['fname'][-1]
            _ = manipulate_ptcl_table_cache_log.return_ptcl_table_fname_after_verification(fname = fname, 
                cache_fname = cache_fname)
        assert 'If you are using your own particle catalog' in err.value.message

        #################################################################
        ##### Now we perform various tests using the simname shorthands #####

        # Pass in a complete set of metadata that disagrees with the 
        # metadata stored in the hdf5 file
        with pytest.raises(HalotoolsError) as err:
            _ = manipulate_ptcl_table_cache_log.return_ptcl_table_fname_from_simname_inputs(
                cache_fname = cache_fname, 
                simname = 'bolshoi',  redshift = 0.10, 
                version_name = 'halotools_alpha_version1')

        assert 'attribute that is inconsistent with the stored value.' in err.value.message

        ## Now verify that the proposed solution works
        import h5py
        fname = updated_log['fname'][-1]
        f = h5py.File(fname)
        f.attrs.create('simname', 'bolshoi')
        f.close()

        _ = manipulate_ptcl_table_cache_log.return_ptcl_table_fname_from_simname_inputs(
            cache_fname = cache_fname, 
            simname = 'bolshoi',  redshift = 0.10, 
            version_name = 'halotools_alpha_version1')


    def tearDown(self):
        try:
            shutil.rmtree(self.dummy_cache_baseloc)
        except:
            pass






