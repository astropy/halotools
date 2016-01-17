#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
import warnings, os, shutil

from astropy.config.paths import _find_home 
from astropy.tests.helper import remote_data, pytest

try:
    import h5py 
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

import numpy as np 
from copy import copy, deepcopy 

from . import helper_functions

from astropy.table import Table

from ..user_supplied_ptcl_catalog import UserSuppliedPtclCatalog
from ..ptcl_table_cache import PtclTableCache

from ...custom_exceptions import HalotoolsError, InvalidCacheLogEntry

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

__all__ = ('TestUserSuppliedPtclCatalog', )

class TestUserSuppliedPtclCatalog(TestCase):
    """ Class providing tests of the `~halotools.sim_manager.UserSuppliedPtclCatalog`. 
    """

    def setUp(self):
        """ Pre-load various arrays into memory for use by all tests. 
        """
        self.Nptcls = 1e4
        self.Lbox = 100
        self.redshift = 0.0
        self.x = np.linspace(0, self.Lbox, self.Nptcls)
        self.y = np.linspace(0, self.Lbox, self.Nptcls)
        self.z = np.linspace(0, self.Lbox, self.Nptcls)

        self.good_ptclcat_args = (
            {'x': self.x, 'y': self.y, 
            'z': self.z}
            )

        self.good_ptcl_table = Table(self.good_ptclcat_args)

        self.dummy_cache_baseloc = helper_functions.dummy_cache_baseloc

        try:
            shutil.rmtree(self.dummy_cache_baseloc)
        except:
            pass
        os.makedirs(self.dummy_cache_baseloc)

    def test_particle_mass_requirement(self):

        with pytest.raises(HalotoolsError):
            halocat = UserSuppliedPtclCatalog(Lbox = 200, 
                **self.good_ptclcat_args)

    def test_lbox_requirement(self):

        with pytest.raises(HalotoolsError):
            halocat = UserSuppliedPtclCatalog(particle_mass = 200, 
                **self.good_ptclcat_args)

    def test_ptcls_contained_inside_lbox(self):

        with pytest.raises(HalotoolsError):
            halocat = UserSuppliedPtclCatalog(Lbox = 20, particle_mass = 100, 
                **self.good_ptclcat_args)

    def test_redshift_is_float(self):

        with pytest.raises(HalotoolsError) as err:
            halocat = UserSuppliedPtclCatalog(
                Lbox = 200, particle_mass = 100, redshift = '1.0', 
                **self.good_ptclcat_args)
        substr = "The ``redshift`` metadata must be a float."
        assert substr in err.value.message

    def tearDown(self):
        try:
            shutil.rmtree(self.dummy_cache_baseloc)
        except:
            pass














