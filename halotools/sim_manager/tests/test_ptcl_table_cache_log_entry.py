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
from ..log_entry import PtclTableCacheLogEntry

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

__all__ = ('TestPtclTableCacheLogEntry' )



class TestPtclTableCacheLogEntry(TestCase):
    """ Class providing unit testing for `~halotools.sim_manager.PtclTableCacheLogEntry`. 
    """
    import h5py
    hard_coded_log_attrs = ['simname', 'version_name', 'redshift', 'fname']

    def setUp(self):
        """ Pre-load various arrays into memory for use by all tests. 
        """
        self.dummy_cache_baseloc = helper_functions.dummy_cache_baseloc

        try:
            os.system('rm -rf ' + self.dummy_cache_baseloc)
        except OSError:
            pass

        os.makedirs(self.dummy_cache_baseloc)

        self.simnames = ('bolshoi', 'consuelo', 
            'bolshoi', 'bolshoi', 'multidark')
        self.version_names = ('v0', 'v1', 'v2', 'v3', 'v4')
        self.redshifts = (1.2, -0.1, 1.339, 1.3, 100.)

        self.basenames = ('non_existent.hdf5', 'existent.file', 
            'existent.hdf5', 'existent.hdf5', 'good.hdf5')
        self.fnames = tuple(os.path.join(self.dummy_cache_baseloc, name) 
            for name in self.basenames)

        self.table1 = Table({'x': [1, 2, 3]})
        
        self.good_table = Table(
            {'ptcl_id': [1, 2, 3], 
            'x': [1, 2, 3], 
            'y': [1, 2, 3], 
            'z': [1, 2, 3], 
            'vx': [1, 2, 3], 
            'vy': [1, 2, 3], 
            'vz': [1, 2, 3], 
            })

    def get_scenario_kwargs(self, num_scenario):
        return ({'simname': self.simnames[num_scenario],  
            'version_name': self.version_names[num_scenario], 
            'redshift': self.redshifts[num_scenario], 
            'fname': self.fnames[num_scenario]})

    def test_instantiation(self):
        """ We can instantiate the log entry with a complete set of metadata
        """
        
        for i in range(len(self.simnames)):
            constructor_kwargs = self.get_scenario_kwargs(i)
            log_entry = PtclTableCacheLogEntry(**constructor_kwargs)
            assert set(log_entry.log_attributes) == set(self.hard_coded_log_attrs)







