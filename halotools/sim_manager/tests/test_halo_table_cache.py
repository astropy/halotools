#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
import pytest 
import warnings, os

import numpy as np 
from copy import copy, deepcopy 

from astropy.config.paths import _find_home 
from astropy.table import Table
from astropy.table import vstack as table_vstack

from . import helper_functions

from ..log_entry import HaloTableCacheLogEntry, get_redshift_string
from ..halo_table_cache import HaloTableCache

from ...custom_exceptions import InvalidCacheLogEntry

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

__all__ = ('TestHaloTableCache' )

class TestHaloTableCache(TestCase):
    """
    """

    def setUp(self):
        import h5py
        self.h5py = h5py

        self.dummy_cache_baseloc = helper_functions.dummy_cache_baseloc
        try:
            os.system('rm -rf ' + self.dummy_cache_baseloc)
        except OSError:
            pass
        os.makedirs(self.dummy_cache_baseloc)


        # Create a good halo catalog and log entry
        self.good_table = Table(
            {'halo_id': [1, 2, 3], 
            'halo_x': [1, 2, 3], 
            'halo_y': [1, 2, 3], 
            'halo_z': [1, 2, 3], 
            'halo_mass': [1, 2, 3], 
            })
        self.good_table_fname = os.path.join(self.dummy_cache_baseloc, 
            'good_table.hdf5')
        self.good_table.write(self.good_table_fname, path='data')

        self.good_log_entry = HaloTableCacheLogEntry('good_simname1', 
            'good_halo_finder', 'good_version_name', 
            get_redshift_string(0.0), self.good_table_fname)

        f = self.h5py.File(self.good_table_fname)
        for attr in self.good_log_entry.log_attributes:
            f.attrs.create(str(attr), str(getattr(self.good_log_entry, attr)))
        f.attrs.create('Lbox', 100.)
        f.attrs.create('particle_mass', 1e8)
        f.close()

        # Create a second good halo catalog and log entry

        self.good_table2 = deepcopy(self.good_table)
        self.good_table2_fname = os.path.join(self.dummy_cache_baseloc, 
            'good_table2.hdf5')
        self.good_table2.write(self.good_table2_fname, path='data')

        self.good_log_entry2 = HaloTableCacheLogEntry('good_simname2', 
            'good_halo_finder2', 'good_version_name', 
            get_redshift_string(1.0), self.good_table2_fname)

        f = self.h5py.File(self.good_table2_fname)
        for attr in self.good_log_entry2.log_attributes:
            f.attrs.create(str(attr), str(getattr(self.good_log_entry2, attr)))
        f.attrs.create('Lbox', 100.)
        f.attrs.create('particle_mass', 1e8)
        f.close()

        # Create a bad halo catalog and log entry

        self.bad_table = Table(
            {'halo_id': [1, 2, 3], 
            'halo_y': [1, 2, 3], 
            'halo_z': [1, 2, 3], 
            'halo_mass': [1, 2, 3], 
            })
        bad_table_fname = os.path.join(self.dummy_cache_baseloc, 
            'bad_table.hdf5')
        self.bad_table.write(bad_table_fname, path='data')

        self.bad_log_entry = HaloTableCacheLogEntry('1', '2', '3', '4', '5')


    def test_clean_log_of_repeated_entries(self):
        pass

    def test_add_entry_to_cache_log(self):
        cache = HaloTableCache(read_log_from_standard_loc = False)
        assert len(cache.log) == 0

        with pytest.raises(TypeError) as err:
            cache.add_entry_to_cache_log('abc', update_ascii = False)
        substr = "You can only add instances of HaloTableCacheLogEntry to the cache log"
        assert substr in err.value.message

        cache.add_entry_to_cache_log(self.good_log_entry, update_ascii = False)
        assert len(cache.log) == 1

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cache.add_entry_to_cache_log(self.good_log_entry, update_ascii = False)
            substr = "cache log already contains the entry"
            assert substr in str(w[-1].message)
        assert len(cache.log) == 1

        cache.add_entry_to_cache_log(self.good_log_entry2, update_ascii = False)
        assert len(cache.log) == 2

        with pytest.raises(InvalidCacheLogEntry) as err:
            cache.add_entry_to_cache_log(self.bad_log_entry, update_ascii = False)
        substr = "The input filename does not exist."
        assert substr in err.value.message




    def tearDown(self):
        try:
            os.system('rm -rf ' + self.dummy_cache_baseloc)
        except OSError:
            pass

