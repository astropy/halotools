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

from .. import manipulate_cache_log, halo_catalog
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
            os.system('rm -rf ' + self.dummy_cache_baseloc)
        except OSError:
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
        self.halocat_obj = UserDefinedHaloCatalog(Lbox = 200, ptcl_mass = 100, 
            **self.good_halocat_args)

    def test_manipulate_cache_log_storage_function(self):
        """
        """

        scenario = 0
        cache_dirname = helper_functions.get_scenario_cache_fname(scenario)
        cache_fname = os.path.join(cache_dirname, helper_functions.cache_basename)
        try:
            os.makedirs(cache_dirname)
        except OSError:
            pass

        temp_fname = os.path.join(self.dummy_cache_baseloc, 'temp_halocat.hdf5')
        manipulate_cache_log.store_new_halo_table_in_cache(self.halocat_obj.halo_table, 
            cache_fname = cache_fname, 
            simname = 'fakesim', halo_finder = 'fake_halo_finder', 
            redshift = 0.0, version_name = 'phony_version', 
            Lbox = self.halocat_obj.Lbox, ptcl_mass = self.halocat_obj.ptcl_mass, 
            fname = temp_fname
            )


    def tearDown(self):
        try:
            os.system('rm -rf ' + self.dummy_cache_baseloc)
        except OSError:
            pass













        
