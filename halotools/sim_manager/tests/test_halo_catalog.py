#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
import warnings, os, pytest
from astropy.config.paths import _find_home 


import numpy as np 

from ..halo_catalog import OverhauledHaloCatalog
from ..halo_table_cache import HaloTableCache
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

__all__ = ('TestOverhauledHaloCatalog' )


class TestOverhauledHaloCatalog(TestCase):
    """ 
    """

    @pytest.mark.skipif('not APH_MACHINE')
    def test_load_all_catalogs(self):
        """ Verify that the default halo catalog loads. 
        """
        cache = HaloTableCache()
        for entry in cache.log:
            constructor_kwargs = (
                {attr: getattr(entry, attr) 
                for attr in entry.log_attributes})
            halocat = OverhauledHaloCatalog(**constructor_kwargs)
            assert hasattr(halocat, 'redshift')
            assert hasattr(halocat, 'Lbox')

    @pytest.mark.skipif('not APH_MACHINE')
    def test_default_catalog(self):
        """ Verify that the default halo catalog loads. 
        """
        halocat = OverhauledHaloCatalog()
        assert hasattr(halocat, 'redshift')
        assert hasattr(halocat, 'Lbox')

    @pytest.mark.skipif('not APH_MACHINE')
    def test_load_bad_catalog1(self):
        """ Verify that the appropriate errors are raised when 
        attempting to load catalogs without matches in cache.  
        """
        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = OverhauledHaloCatalog(simname = 'bolshoi', 
                halo_finder = 'bdm', version_name = 'halotools_alpha_version1', 
                redshift = 5, dz_tol = 1)
        assert 'The following entries in the cache log' in err.value.message

    @pytest.mark.skipif('not APH_MACHINE')
    def test_load_bad_catalog2(self):
        """ Verify that the appropriate errors are raised when 
        attempting to load catalogs without matches in cache.  
        """
        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = OverhauledHaloCatalog(simname = 'bolshoi', 
                halo_finder = 'bdm', version_name = 'halotools_alpha_version1', 
                redshift = 5, dz_tol = 1)
        assert 'The following entries in the cache log' in err.value.message

    @pytest.mark.skipif('not APH_MACHINE')
    def test_load_bad_catalog3(self):
        """ Verify that the appropriate errors are raised when 
        attempting to load catalogs without matches in cache.  
        """
        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = OverhauledHaloCatalog(simname = 'bolshoi', 
                halo_finder = 'bdm', version_name = 'Jose Canseco')
        assert 'The following entries in the cache log' in err.value.message
        assert '(set by sim_defaults.default_redshift)' in err.value.message

    @pytest.mark.skipif('not APH_MACHINE')
    def test_load_bad_catalog4(self):
        """ Verify that the appropriate errors are raised when 
        attempting to load catalogs without matches in cache.  
        """
        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = OverhauledHaloCatalog(simname = 'bolshoi', 
                halo_finder = 'Jose Canseco')
        assert 'The following entries in the cache log' in err.value.message
        assert '(set by sim_defaults.default_version_name)' in err.value.message

    @pytest.mark.skipif('not APH_MACHINE')
    def test_load_bad_catalog5(self):
        """ Verify that the appropriate errors are raised when 
        attempting to load catalogs without matches in cache.  
        """
        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = OverhauledHaloCatalog(simname = 'Jose Canseco')
        assert 'There are no simulations matching your input simname' in err.value.message
        assert '(set by sim_defaults.default_halo_finder)' in err.value.message

    @pytest.mark.skipif('not APH_MACHINE')
    def test_load_bad_catalog6(self):
        """ Verify that the appropriate errors are raised when 
        attempting to load catalogs without matches in cache.  
        """
        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = OverhauledHaloCatalog(simname = 'Jose Canseco', 
                halo_finder = 'bdm', version_name = 'halotools_alpha_version1', 
                redshift = 5, dz_tol = 1)
        assert 'There are no simulations matching your input simname' in err.value.message

    @pytest.mark.skipif('not APH_MACHINE')
    def test_load_bad_catalog7(self):
        """ Verify that the appropriate errors are raised when 
        attempting to load catalogs without matches in cache.  
        """
        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = OverhauledHaloCatalog(dz_tol = 100)
        assert 'There are multiple entries in the cache log' in err.value.message
        assert '(set by sim_defaults.default_simname)' in err.value.message

    @pytest.mark.slow
    @pytest.mark.skipif('not APH_MACHINE')
    def test_load_ptcl_table(self):
        """ Verify that the default particle catalog loads. 
        """
        halocat = OverhauledHaloCatalog()
        ptcls = halocat.ptcl_table






