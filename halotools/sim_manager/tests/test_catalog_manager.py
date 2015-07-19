#!/usr/bin/env python

import pytest
slow = pytest.mark.slow

import os, fnmatch
import numpy as np

from ..catalog_manager import CatalogManager
from astropy.config.paths import _find_home 

from unittest import TestCase

class TestCatalogManager(TestCase):


    def setup_class(self):

        homedir = _find_home()

        def defensively_create_empty_dir(dirname):

            if os.path.isdir(dirname) is False:
                os.mkdir(dirname)
            else:
                os.system('rm -rf ' + dirname)
                os.mkdir(dirname)

        # First create an empty directory where we will 
        # temporarily store a collection of empty files
        dummydir = os.path.join(homedir, 'temp_directory_for_halotools_testing')
        defensively_create_empty_dir(dummydir)
        self.dummyloc = os.path.join(dummydir, 'halotools')
        defensively_create_empty_dir(self.dummyloc)

        self.halocat_dir = os.path.join(self.dummyloc, 'halo_catalogs')
        defensively_create_empty_dir(self.halocat_dir)

        self.ptclcat_dir = os.path.join(self.dummyloc, 'particle_catalogs')
        defensively_create_empty_dir(self.ptclcat_dir)

        self.raw_halocat_dir = os.path.join(self.dummyloc, 'raw_halo_catalogs')
        defensively_create_empty_dir(self.raw_halocat_dir)

        self.simnames = ['bolshoi', 'bolshoiplanck', 'multidark', 'consuelo']
        self.halo_finders = ['rockstar', 'bdm']
        self.dummy_version_names = ['halotools.alpha']
        self.extension = '.hdf5'

        self.bolshoi_fnames = ['hlist_0.33035', 'hlist_0.54435', 'hlist_0.67035', 'hlist_1.00035']
        self.bolshoi_bdm_fnames = ['hlist_0.33030', 'hlist_0.49830', 'hlist_0.66430', 'hlist_1.00035']
        self.bolshoiplanck_fnames = ['hlist_0.33035', 'hlist_0.54435', 'hlist_0.67035', 'hlist_1.00035']
        self.consuelo_fnames = ['hlist_0.33324', 'hlist_0.50648', 'hlist_0.67540', 'hlist_1.00000']
        self.multidark_fnames = ['hlist_0.31765', 'hlist_0.49990', 'hlist_0.68215', 'hlist_1.00109']

        # make all relevant subdirectories and dummy files
        for simname in self.simnames:
            simdir = os.path.join(self.halocat_dir, simname)
            defensively_create_empty_dir(simdir)
            rockstardir = os.path.join(simdir, 'rockstar')
            defensively_create_empty_dir(rockstardir)

            if simname == 'bolshoi':
                fnames = self.bolshoi_fnames
            elif simname == 'bolshoiplanck':
                fnames = self.bolshoiplanck_fnames
            elif simname == 'consuelo':
                fnames = self.consuelo_fnames
            elif simname == 'multidark':
                fnames = self.multidark_fnames

            for name in fnames:
                for version in self.dummy_version_names:
                    full_fname = name + '.' + version + self.extension
                    abs_fname = os.path.join(rockstardir, full_fname)
                    os.system('touch ' + abs_fname)

            if simname == 'bolshoi':
                simdir = os.path.join(self.halocat_dir, simname)
                bdmdir = os.path.join(simdir, 'bdm')
                defensively_create_empty_dir(bdmdir)
                fnames = self.bolshoi_bdm_fnames
                for name in fnames:
                    for version in self.dummy_version_names:
                        full_fname = name + '.' + version + self.extension
                        abs_fname = os.path.join(bdmdir, full_fname)
                        os.system('touch ' + abs_fname)

        p = os.path.join(self.halocat_dir, 'bolshoi', 'bdm')
        assert os.path.isdir(p)
        f = 'hlist_0.33030.halotools.alpha.hdf5'
        full_fname = os.path.join(p, f)
        assert os.path.isfile(full_fname)

    def test_processed_halocats_in_cache(self):
        catman = CatalogManager()

        for simname in self.simnames:
            attrname = simname + '_fnames'
            basenames_from_self = getattr(self, attrname)

            for version in self.dummy_version_names:
                basenames_from_setup = [f + '.' + version + self.extension for f in basenames_from_self]

                result = catman.processed_halocats_in_cache(external_cache_loc=self.halocat_dir, 
                    simname = simname, halo_finder = 'rockstar', version_name = version)
                basenames_from_catman = [os.path.basename(f) for f in result]

                assert set(basenames_from_catman) == set(basenames_from_setup)

        simname = 'bolshoi'
        version = 'halotools.alpha'
        result_allargs = catman.processed_halocats_in_cache(external_cache_loc=self.halocat_dir, 
            simname = simname, halo_finder = 'rockstar', version_name = version)
        result_nosim = catman.processed_halocats_in_cache(external_cache_loc=self.halocat_dir, 
            halo_finder = 'rockstar', version_name = version)
        result_noversion = catman.processed_halocats_in_cache(external_cache_loc=self.halocat_dir, 
            simname = simname, halo_finder = 'rockstar')
        result_nohf = catman.processed_halocats_in_cache(external_cache_loc=self.halocat_dir, 
            simname = simname, version_name = version)

        assert result_allargs != []
        assert result_nosim != []
        assert result_noversion != []
        assert result_nohf != []

        assert set(result_allargs).issubset(set(result_nosim))
        assert set(result_allargs).issubset(set(result_noversion))
        assert set(result_allargs).issubset(set(result_nohf))

        assert set(result_allargs) != set(result_nohf)
        assert set(result_allargs) != set(result_nosim)
        assert set(result_allargs) == set(result_noversion)

        assert set(result_nohf) != (set(result_nosim))
        assert set(result_nohf) != (set(result_noversion))
        assert set(result_nosim) != (set(result_noversion))







    def teardown_class(self):
        os.system('rm -rf ' + self.dummyloc)
















