#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
import warnings, os, pytest, shutil
from astropy.config.paths import _find_home 


import numpy as np 

from . import helper_functions
from ..cached_halo_catalog import CachedHaloCatalog
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

__all__ = ('TestCachedHaloCatalog' )


class TestCachedHaloCatalog(TestCase):
    """ 
    """

    def setUp(self):
        """ Pre-load various arrays into memory for use by all tests. 
        """
        import h5py
        self.h5py = h5py

        self.dummy_cache_baseloc = helper_functions.dummy_cache_baseloc
        try:
            shutil.rmtree(self.dummy_cache_baseloc)
        except:
            pass
        os.makedirs(self.dummy_cache_baseloc)

    @pytest.mark.skipif('not APH_MACHINE')
    def test_load_all_catalogs(self):
        """ Verify that the default halo catalog loads. 
        """
        cache = HaloTableCache()
        for entry in cache.log:
            constructor_kwargs = (
                {attr: getattr(entry, attr) 
                for attr in entry.log_attributes})
            del constructor_kwargs['fname']
            halocat = CachedHaloCatalog(**constructor_kwargs)
            assert hasattr(halocat, 'redshift')
            assert hasattr(halocat, 'Lbox')

    @pytest.mark.skipif('not APH_MACHINE')
    def test_default_catalog(self):
        """ Verify that the default halo catalog loads. 
        """
        halocat = CachedHaloCatalog()
        assert hasattr(halocat, 'redshift')
        assert hasattr(halocat, 'Lbox')

    @pytest.mark.skipif('not APH_MACHINE')
    def test_load_bad_catalog1(self):
        """ Verify that the appropriate errors are raised when 
        attempting to load catalogs without matches in cache.  
        """
        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = CachedHaloCatalog(simname = 'bolshoi', 
                halo_finder = 'bdm', version_name = 'halotools_alpha_version1', 
                redshift = 5, dz_tol = 1)
        assert 'The following entries in the cache log' in err.value.message

    @pytest.mark.skipif('not APH_MACHINE')
    def test_load_bad_catalog2(self):
        """ Verify that the appropriate errors are raised when 
        attempting to load catalogs without matches in cache.  
        """
        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = CachedHaloCatalog(simname = 'bolshoi', 
                halo_finder = 'bdm', version_name = 'halotools_alpha_version1', 
                redshift = 5, dz_tol = 1)
        assert 'The following entries in the cache log' in err.value.message

    @pytest.mark.skipif('not APH_MACHINE')
    def test_load_bad_catalog3(self):
        """ Verify that the appropriate errors are raised when 
        attempting to load catalogs without matches in cache.  
        """
        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = CachedHaloCatalog(simname = 'bolshoi', 
                halo_finder = 'bdm', version_name = 'Jose Canseco')
        assert 'The following entries in the cache log' in err.value.message
        assert '(set by sim_defaults.default_redshift)' in err.value.message

    @pytest.mark.skipif('not APH_MACHINE')
    def test_load_bad_catalog4(self):
        """ Verify that the appropriate errors are raised when 
        attempting to load catalogs without matches in cache.  
        """
        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = CachedHaloCatalog(simname = 'bolshoi', 
                halo_finder = 'Jose Canseco')
        assert 'The following entries in the cache log' in err.value.message
        assert '(set by sim_defaults.default_version_name)' in err.value.message

    @pytest.mark.skipif('not APH_MACHINE')
    def test_load_bad_catalog5(self):
        """ Verify that the appropriate errors are raised when 
        attempting to load catalogs without matches in cache.  
        """
        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = CachedHaloCatalog(simname = 'Jose Canseco')
        assert 'There are no simulations matching your input simname' in err.value.message
        assert '(set by sim_defaults.default_halo_finder)' in err.value.message

    @pytest.mark.skipif('not APH_MACHINE')
    def test_load_bad_catalog6(self):
        """ Verify that the appropriate errors are raised when 
        attempting to load catalogs without matches in cache.  
        """
        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = CachedHaloCatalog(simname = 'Jose Canseco', 
                halo_finder = 'bdm', version_name = 'halotools_alpha_version1', 
                redshift = 5, dz_tol = 1)
        assert 'There are no simulations matching your input simname' in err.value.message

    @pytest.mark.skipif('not APH_MACHINE')
    def test_load_bad_catalog7(self):
        """ Verify that the appropriate errors are raised when 
        attempting to load catalogs without matches in cache.  
        """
        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = CachedHaloCatalog(dz_tol = 100)
        assert 'There are multiple entries in the cache log' in err.value.message
        assert '(set by sim_defaults.default_simname)' in err.value.message

    @pytest.mark.slow
    @pytest.mark.skipif('not APH_MACHINE')
    def test_load_ptcl_table(self):
        """ Verify that the default particle catalog loads. 
        """
        halocat = CachedHaloCatalog()
        ptcls = halocat.ptcl_table

    @pytest.mark.skipif('not APH_MACHINE')
    def test_fname_optional_load(self):
        fname = '/Users/aphearin/.astropy/cache/halotools/halo_catalogs/bolplanck/rockstar/hlist_0.33406.list.halotools_alpha_version1.hdf5'
        halocat = CachedHaloCatalog(fname = fname)
        assert halocat.simname == 'bolplanck'

    @pytest.mark.slow
    @pytest.mark.skipif('not APH_MACHINE')
    def test_all_fname_loads(self):
        cache = HaloTableCache()
        for entry in cache.log:
            fname = entry.fname
            halocat = CachedHaloCatalog(fname = fname)
            for attr in entry.log_attributes:
                if attr == 'redshift':
                    assert float(getattr(entry, attr)) == float(getattr(halocat, attr))
                else:
                    assert str(getattr(entry, attr)) == str(getattr(halocat, attr))

    @pytest.mark.skipif('not APH_MACHINE')
    def test_acceptable_arguments1(self):
        fname = os.path.join(self.dummy_cache_baseloc, 'abc.hdf5')
        os.system('touch '+ fname)

        with pytest.raises(HalotoolsError) as err:
            halocat = CachedHaloCatalog(fname = fname, simname = 'bolshoi')
        substr = "If you specify an input ``fname``"
        assert substr in err.value.message
        substr = "do not also specify ``simname``"
        assert substr in err.value.message

    @pytest.mark.skipif('not APH_MACHINE')
    def test_acceptable_arguments2(self):
        fname = os.path.join(self.dummy_cache_baseloc, 'abc.hdf5')
        os.system('touch '+ fname)

        with pytest.raises(HalotoolsError) as err:
            halocat = CachedHaloCatalog(fname = fname, version_name = 'dummy')
        substr = "If you specify an input ``fname``"
        assert substr in err.value.message
        substr = "do not also specify ``version_name``"
        assert substr in err.value.message

    @pytest.mark.skipif('not APH_MACHINE')
    def test_acceptable_arguments3(self):
        fname = os.path.join(self.dummy_cache_baseloc, 'abc.hdf5')
        os.system('touch '+ fname)

        with pytest.raises(HalotoolsError) as err:
            halocat = CachedHaloCatalog(fname = fname, halo_finder = 'dummy')
        substr = "If you specify an input ``fname``"
        assert substr in err.value.message
        substr = "do not also specify ``halo_finder``"
        assert substr in err.value.message

    @pytest.mark.skipif('not APH_MACHINE')
    def test_acceptable_arguments4(self):
        fname = os.path.join(self.dummy_cache_baseloc, 'abc.hdf5')
        os.system('touch '+ fname)

        with pytest.raises(HalotoolsError) as err:
            halocat = CachedHaloCatalog(fname = fname, redshift = 0)
        substr = "If you specify an input ``fname``"
        assert substr in err.value.message
        substr = "do not also specify ``redshift``"
        assert substr in err.value.message

    @pytest.mark.skipif('not APH_MACHINE')
    def test_acceptable_arguments5(self):
        fname = 'abc'

        with pytest.raises(HalotoolsError) as err:
            halocat = CachedHaloCatalog(fname = fname, redshift = 0)
        substr = "non-existent path"
        assert substr in err.value.message

    @pytest.mark.skipif('not APH_MACHINE')
    def test_acceptable_arguments6(self):
        cache = HaloTableCache()
        fname = cache.log[0].fname
        halocat = CachedHaloCatalog(fname = fname)

    @pytest.mark.skipif('not APH_MACHINE')
    def test_acceptable_arguments7(self):
        cache = HaloTableCache()
        correct_fname = cache.log[10].fname
        # print("printing correct fname")
        # print(correct_fname)
        temporary_bad_fname = 'abc.hdf5'

        f = self.h5py.File(correct_fname)
        f.attrs['fname'] = temporary_bad_fname
        f.close()

        with pytest.raises(HalotoolsError) as err:
            halocat = CachedHaloCatalog(fname = correct_fname)
        print(err.value.message)

        f = self.h5py.File(correct_fname)
        f.attrs['fname'] = correct_fname
        f.close()
       


    def tearDown(self):
        try:
            shutil.rmtree(self.dummy_cache_baseloc)
        except:
            pass




