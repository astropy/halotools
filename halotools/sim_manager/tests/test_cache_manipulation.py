#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
import pytest 
import warnings, os

import numpy as np 
from copy import copy, deepcopy 

from astropy.table import Table
from astropy.config.paths import _find_home 

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

__all__ = ('TestCacheManipulation', )

class TestCacheManipulation(TestCase):
    """ 
    """

    def setUp(self):
        """ Pre-load various arrays into memory for use by all tests. 
        """

        self.cache_log_tables_to_test = []
        simname_list = ['bolshoi', 'bolshoi']
        redshift_list = [0, 1]
        halo_finder_list = ['rockstar', 'rockstar']
        version_name_list = ['beta_v0', 'beta_v0']
        fname_list = ['whatever_fname1', 'whatever_fname2']
        self.cache_log_tables_to_test.append(
            self.create_dummy_cache_log('temp_dummy_cache_dirname', 
            simname_list, redshift_list,
            halo_finder_list, version_name_list, fname_list)
            )

        for name, table in zip(self.dummy_cache_log_fnames, self.cache_log_tables_to_test):
            self.create_dummy_cache_directories(name, table)

    def create_dummy_cache_log(self, tmp_name, simname_list, redshift_list, 
        halo_finder_list, version_name_list, fname_list):

        new_cache_name = os.path.join(detected_home, 'Desktop', tmp_name)
        setattr(self, tmp_name, new_cache_name)

        try:
            self.temp_cache_dirnames.append(new_cache_name)
        except AttributeError:
            self.temp_cache_dirnames = [new_cache_name]

        if os.path.isdir(getattr(self, tmp_name)) is False:
            os.mkdir(new_cache_name)
        else:
            os.system('rm -rf ' + new_cache_name)
            os.mkdir(new_cache_name)

        new_cache_log_fname = os.path.join(new_cache_name, 'dummy_cache_log.txt')
        try:
            self.dummy_cache_log_fnames.append(new_cache_log_fname)
        except AttributeError:
            self.dummy_cache_log_fnames = [new_cache_log_fname]

        return Table({'simname': simname_list, 'redshift': redshift_list, 
            'halo_finder': halo_finder_list, 'version_name': version_name_list, 
            'fname': fname_list}
            )

    def create_dummy_cache_directories(self, basename, table):
        """
        """
        basedir = os.path.dirname(basename)
        for entry in table:
            simname = entry['simname']
            new_simname_path = os.path.join(basedir, simname)
            try:
                os.mkdir(new_simname_path)
            except OSError:
                pass
            halo_finder = entry['halo_finder']
            new_halo_finder_path = os.path.join(new_simname_path, halo_finder)
            try:
                os.mkdir(new_halo_finder_path)
            except OSError:
                pass


    @pytest.mark.xfail
    def test_rebuild_halo_table_cache_log(self):
        raise HalotoolsError("The rebuild_halo_table_cache_log function is not implemented yet.")


    @pytest.mark.xfail
    def test_overwrite_existing_halo_table_in_cache(self):
        raise HalotoolsError("The overwrite_existing_halo_table_in_cache function "
            "has been implemented but remains untested.")
        

    def tearDown(self):
        for name in self.temp_cache_dirnames:
            os.system('rm -rf ' + name)













