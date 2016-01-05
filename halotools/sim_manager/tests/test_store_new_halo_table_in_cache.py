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

from . import helper_functions 

from .. import manipulate_cache_log
from ..halo_catalog import OverhauledHaloCatalog
from ..user_defined_halo_catalog import UserDefinedHaloCatalog

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

__all__ = ('TestStoreNewHaloTable' )


class TestStoreNewHaloTable(TestCase):
    """ Class to test manipulate_cache_log.store_new_halo_table_in_cache
    """

    def setUp(self):
        """ Pre-load various arrays into memory for use by all tests. 
        """
        self.dummy_cache_baseloc = helper_functions.dummy_cache_baseloc
        try:
            shutil.rmtree(self.dummy_cache_baseloc)
        except:
            pass

        # Create a fake halo catalog 
        self.Nhalos = 1e2
        self.Lbox = 100
        self.halo_x = np.linspace(0, self.Lbox, self.Nhalos)
        self.halo_y = np.linspace(0, self.Lbox, self.Nhalos)
        self.halo_z = np.linspace(0, self.Lbox, self.Nhalos)
        self.halo_mass = np.logspace(10, 15, self.Nhalos)
        self.halo_id = np.arange(0, self.Nhalos)
        self.good_halocat_args = (
            {'halo_x': self.halo_x, 'halo_y': self.halo_y, 
            'halo_z': self.halo_z, 'halo_id': self.halo_id, 'halo_mass': self.halo_mass}
            )
        self.halocat_obj = UserDefinedHaloCatalog(Lbox = 200, particle_mass = 100, redshift = 0., 
            **self.good_halocat_args)

    @pytest.mark.xfail
    def test_scenario0(self):
        """ The cache has never been used before, 
        and the first time it is used is to store a perfectly kosher 
        user-defined halo catalog. 
        """

        #################### SETUP ####################
        scenario = 0
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)
        try:
            os.makedirs(cache_dirname)
        except OSError:
            pass

        # Store the halo table 
        temp_fname = os.path.join(self.dummy_cache_baseloc, 'temp_halocat.hdf5')
        manipulate_cache_log.store_new_halo_table_in_cache(self.halocat_obj.halo_table, 
            cache_fname = cache_fname, 
            simname = 'fakesim', halo_finder = 'fake_halo_finder', 
            redshift = 0.0, version_name = 'phony_version', 
            Lbox = self.halocat_obj.Lbox, particle_mass = self.halocat_obj.particle_mass, 
            fname = temp_fname
            )

        # Load the newly created table using a simname
        loaded_halocat = OverhauledHaloCatalog(
            simname = 'fakesim', halo_finder = 'fake_halo_finder',
            redshift = 0.0, version_name = 'phony_version', 
            cache_fname = cache_fname)
        assert loaded_halocat.redshift == 0.0
        assert hasattr(loaded_halocat, 'halo_table')

        # Load the newly created table using an explicit fname
        loaded_halocat2 = OverhauledHaloCatalog(
            fname = temp_fname, cache_fname = cache_fname)
        assert loaded_halocat2.redshift == 0.0
        assert hasattr(loaded_halocat2, 'halo_table')

    def test_scenario1(self):
        """ The cache has never been used before, 
        and the first time it is used is to store a user-defined halo catalog 
        that is flawed in some way or another. 
        """

        #################### SETUP ####################
        scenario = 1
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)
        try:
            os.makedirs(cache_dirname)
        except OSError:
            pass

        temp_fname = os.path.join(self.dummy_cache_baseloc, 'temp_halocat.hdf5')

        # Some points in the x-positions are not bounded by [0, Lbox]
        with pytest.raises(HalotoolsError) as err:
            bad_table = deepcopy(self.halocat_obj.halo_table)
            bad_table['halo_x'] = -1
            # Store the halo table 
            manipulate_cache_log.store_new_halo_table_in_cache(bad_table, 
                cache_fname = cache_fname, 
                simname = 'fakesim', halo_finder = 'fake_halo_finder', 
                redshift = 0.0, version_name = 'phony_version', 
                Lbox = self.halocat_obj.Lbox, particle_mass = self.halocat_obj.particle_mass, 
                fname = temp_fname
                )
            del bad_table
        substr = 'There are points in the input halo table that lie outside'
        assert substr in err.value.message


        # User forgot to pass in one of the required pieces of metadata
        with pytest.raises(HalotoolsError) as err:
            # Store the halo table 
            manipulate_cache_log.store_new_halo_table_in_cache(
                self.halocat_obj.halo_table, 
                cache_fname = cache_fname, 
                simname = 'fakesim', halo_finder = 'fake_halo_finder', 
                redshift = 0.0, version_name = 'phony_version', 
                particle_mass = self.halocat_obj.particle_mass, 
                fname = temp_fname
                )
        substr = 'All calls to the `store_new_halo_table_in_cache` function'
        assert substr in err.value.message


    @pytest.mark.xfail
    def test_scenario2(self):
        """ There is an existing halo table stored in cache. 
        We will store an identical one differing by a substantially distinct redshift. 
        Both catalogs successfully store and reload. 
        """


        #################### SETUP ####################
        scenario = 2
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)
        try:
            os.makedirs(cache_dirname)
        except OSError:
            pass

        # Store the first halo table 
        temp_fname = os.path.join(self.dummy_cache_baseloc, 'temp_halocat.hdf5')
        manipulate_cache_log.store_new_halo_table_in_cache(self.halocat_obj.halo_table, 
            cache_fname = cache_fname, 
            simname = 'fakesim', halo_finder = 'fake_halo_finder', 
            redshift = 0.0, version_name = 'phony_version', 
            Lbox = self.halocat_obj.Lbox, particle_mass = self.halocat_obj.particle_mass, 
            fname = temp_fname
            )

        # Store the second halo table 
        temp_fname2 = os.path.join(self.dummy_cache_baseloc, 'temp_halocat2.hdf5')
        manipulate_cache_log.store_new_halo_table_in_cache(self.halocat_obj.halo_table, 
            cache_fname = cache_fname, 
            simname = 'fakesim', halo_finder = 'fake_halo_finder', 
            redshift = 1.0, version_name = 'phony_version', 
            Lbox = self.halocat_obj.Lbox, particle_mass = self.halocat_obj.particle_mass, 
            fname = temp_fname2
            )

        # Load the two halo tables 
        halocat1 = OverhauledHaloCatalog(
            simname = 'fakesim', halo_finder = 'fake_halo_finder',
            redshift = 0.0, version_name = 'phony_version', 
            cache_fname = cache_fname)
        halocat2 = OverhauledHaloCatalog(
            simname = 'fakesim', halo_finder = 'fake_halo_finder',
            redshift = 1.0, version_name = 'phony_version', 
            cache_fname = cache_fname)

        assert halocat2.redshift == 1.0
        assert halocat1.redshift == 0.0

    def test_scenario3(self):
        """ There is an existing halo table stored in cache. 
        We will attempt to store a identical halo table 
        with the same metadata but a different fname, raising an error. 
        """
        #################### SETUP ####################
        scenario = 3
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)
        try:
            os.makedirs(cache_dirname)
        except OSError:
            pass

        # Store the first halo table 
        temp_fname = os.path.join(self.dummy_cache_baseloc, 'temp_halocat.hdf5')
        manipulate_cache_log.store_new_halo_table_in_cache(self.halocat_obj.halo_table, 
            cache_fname = cache_fname, 
            simname = 'fakesim', halo_finder = 'fake_halo_finder', 
            redshift = 0.0, version_name = 'phony_version', 
            Lbox = self.halocat_obj.Lbox, particle_mass = self.halocat_obj.particle_mass, 
            fname = temp_fname
            )

        # Store the second halo table 
        with pytest.raises(HalotoolsError) as err:
            temp_fname2 = os.path.join(self.dummy_cache_baseloc, 'temp_halocat2.hdf5')
            manipulate_cache_log.store_new_halo_table_in_cache(self.halocat_obj.halo_table, 
                cache_fname = cache_fname, 
                simname = 'fakesim', halo_finder = 'fake_halo_finder', 
                redshift = 0.0, version_name = 'phony_version', 
                Lbox = self.halocat_obj.Lbox, particle_mass = self.halocat_obj.particle_mass, 
                fname = temp_fname2
                )
        substr = 'If this matching halo catalog is one you want to continue keeping track of'
        assert substr in err.value.message

        # Now verify that the solution proposed by the error message does indeed resolve the problem
        manipulate_cache_log.store_new_halo_table_in_cache(self.halocat_obj.halo_table, 
            cache_fname = cache_fname, 
            simname = 'fakesim', halo_finder = 'fake_halo_finder', 
            redshift = 0.0, version_name = 'phony_version2', 
            Lbox = self.halocat_obj.Lbox, particle_mass = self.halocat_obj.particle_mass, 
            fname = temp_fname2
            )

    def test_scenario4(self):
        """ There is an existing halo table stored in cache. 
        We will attempt to store a identical halo table 
        with the different metadata but the same fname, raising an error. 
        """
        #################### SETUP ####################
        scenario = 4
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)
        try:
            os.makedirs(cache_dirname)
        except OSError:
            pass

        # Store the first halo table 
        temp_fname = os.path.join(self.dummy_cache_baseloc, 'temp_halocat.hdf5')
        manipulate_cache_log.store_new_halo_table_in_cache(self.halocat_obj.halo_table, 
            cache_fname = cache_fname, 
            simname = 'fakesim', halo_finder = 'fake_halo_finder', 
            redshift = 0.0, version_name = 'phony_version', 
            Lbox = self.halocat_obj.Lbox, particle_mass = self.halocat_obj.particle_mass, 
            fname = temp_fname
            )

        # Store the second halo table 
        temp_fname2 = os.path.join(self.dummy_cache_baseloc, 'temp_halocat.hdf5')
        with pytest.raises(HalotoolsError) as err:
            manipulate_cache_log.store_new_halo_table_in_cache(self.halocat_obj.halo_table, 
                cache_fname = cache_fname, 
                simname = 'fakesim', halo_finder = 'fake_halo_finder', 
                redshift = 0.0, version_name = 'phony_version', 
                Lbox = self.halocat_obj.Lbox, particle_mass = self.halocat_obj.particle_mass, 
                fname = temp_fname2
                )
        substr = 'A file at this location already exists.'
        assert substr in err.value.message

        # Now verify that the solution proposed by the error message does indeed resolve the problem
        temp_fname2 = os.path.join(self.dummy_cache_baseloc, 'temp_halocat2.hdf5')
        manipulate_cache_log.store_new_halo_table_in_cache(self.halocat_obj.halo_table, 
            cache_fname = cache_fname, 
            simname = 'fakesim', halo_finder = 'fake_halo_finder', 
            redshift = 0.0, version_name = 'phony_version2', 
            Lbox = self.halocat_obj.Lbox, particle_mass = self.halocat_obj.particle_mass, 
            fname = temp_fname2
            )


    @pytest.mark.xfail
    def test_scenario5(self):
        """ There is an existing halo table stored in cache. 
        We will store an identical one differing by an extremely similar redshift. 
        Both catalogs successfully store and reload, provided that the 
        ``ignore_nearby_redshifts`` flag is thrown. 
        """


        #################### SETUP ####################
        scenario = 5
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)
        try:
            os.makedirs(cache_dirname)
        except OSError:
            pass

        # Store the first halo table 
        temp_fname = os.path.join(self.dummy_cache_baseloc, 'temp_halocat.hdf5')
        manipulate_cache_log.store_new_halo_table_in_cache(self.halocat_obj.halo_table, 
            cache_fname = cache_fname, 
            simname = 'fakesim', halo_finder = 'fake_halo_finder', 
            redshift = 0.0, version_name = 'phony_version', 
            Lbox = self.halocat_obj.Lbox, particle_mass = self.halocat_obj.particle_mass, 
            fname = temp_fname
            )

        # Try to store the second halo table without the necessary flag
        temp_fname2 = os.path.join(self.dummy_cache_baseloc, 'temp_halocat2.hdf5')

        with pytest.raises(HalotoolsError) as err:
            manipulate_cache_log.store_new_halo_table_in_cache(self.halocat_obj.halo_table, 
                cache_fname = cache_fname, 
                simname = 'fakesim', halo_finder = 'fake_halo_finder', 
                redshift = 0.001, version_name = 'phony_version', 
                Lbox = self.halocat_obj.Lbox, particle_mass = self.halocat_obj.particle_mass, 
                fname = temp_fname2
                )
        substr = 'and a very similar redshift.'
        assert substr in err.value.message

        # Now verify that the solution proposed by the error message does indeed resolve the problem
        manipulate_cache_log.store_new_halo_table_in_cache(self.halocat_obj.halo_table, 
            cache_fname = cache_fname, 
            simname = 'fakesim', halo_finder = 'fake_halo_finder', 
            redshift = 0.001, version_name = 'phony_version', 
            Lbox = self.halocat_obj.Lbox, particle_mass = self.halocat_obj.particle_mass, 
            fname = temp_fname2, ignore_nearby_redshifts = True
            )

        # Load the two halo tables 
        with pytest.raises(HalotoolsError) as err:
            halocat1 = OverhauledHaloCatalog(
                simname = 'fakesim', halo_finder = 'fake_halo_finder',
                redshift = 0.0, version_name = 'phony_version', 
                cache_fname = cache_fname)
        substr = 'Try decreasing the value of the ``dz_tol`` parameter.'
        assert substr in err.value.message

        halocat1 = OverhauledHaloCatalog(
            simname = 'fakesim', halo_finder = 'fake_halo_finder',
            redshift = 0.0, version_name = 'phony_version', 
            cache_fname = cache_fname, dz_tol = 0.0001)

        halocat2 = OverhauledHaloCatalog(
            simname = 'fakesim', halo_finder = 'fake_halo_finder',
            redshift = 0.001, version_name = 'phony_version', 
            cache_fname = cache_fname, dz_tol = 0.0001)

        assert halocat1.redshift == 0.0
        assert halocat2.redshift == 0.001



    def tearDown(self):
        try:
            shutil.rmtree(self.dummy_cache_baseloc)
        except:
            pass













        
