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
from ..log_entry import HaloTableCacheLogEntry

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

__all__ = ('TestHaloTableCacheLogEntry' )



class TestHaloTableCacheLogEntry(TestCase):
    """ 
    """
    import h5py

    def setUp(self):
        """ Pre-load various arrays into memory for use by all tests. 
        """
        self.dummy_cache_baseloc = helper_functions.dummy_cache_baseloc

        try:
            os.system('rm -rf ' + self.dummy_cache_baseloc)
        except OSError:
            pass

        os.makedirs(self.dummy_cache_baseloc)

        self.simnames = ('bolshoi', 'consuelo')
        self.halo_finders = ('rockstar', 'bdm')
        self.version_names = ('testing_version1', 'testing_version2')
        self.redshifts = (1.2, -0.1)

        self.basenames = ('non_existent.hdf5', 'existent.file')
        self.fnames = tuple(os.path.join(self.dummy_cache_baseloc, name) 
            for name in self.basenames)
            

    def get_scenario_kwargs(self, num_scenario):
        return ({'simname': self.simnames[num_scenario], 'halo_finder': self.halo_finders[num_scenario], 
            'version_name': self.version_names[num_scenario], 'redshift': self.redshifts[num_scenario], 
            'fname': self.fnames[num_scenario]})

    def test_instantiation(self):
        """ We can instantiate the log entry with a complete set of metadata
        """
        hard_coded_attrs = ['simname', 'halo_finder', 'version_name', 'redshift', 'fname']
        for i in range(len(self.simnames)):
            constructor_kwargs = self.get_scenario_kwargs(i)
            log_entry = HaloTableCacheLogEntry(**constructor_kwargs)
            assert set(log_entry.log_attributes) == set(hard_coded_attrs)

    def test_scenario0(self):
        log_entry = HaloTableCacheLogEntry(**self.get_scenario_kwargs(0))
        assert log_entry.safe_for_cache == False
        assert "The input filename does not exist." in log_entry._cache_safety_message

    def test_scenario1(self):

        with open(self.fnames[1], 'w') as f:
            f.write('abc')
        log_entry = HaloTableCacheLogEntry(**self.get_scenario_kwargs(1))
        assert log_entry.safe_for_cache == False
        assert "The input filename does not exist." not in log_entry._cache_safety_message
        assert "The input file must have '.hdf5' extension" in log_entry._cache_safety_message



    def tearDown(self):
        try:
            os.system('rm -rf ' + self.dummy_cache_baseloc)
        except OSError:
            pass













