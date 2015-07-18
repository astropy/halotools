#!/usr/bin/env python

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
        dummydir = 'temp_directory_for_halotools_testing'
        self.dummyloc = os.path.join(homedir, dummydir)
        defensively_create_empty_dir(self.dummyloc)

        self.simnames = ['bolshoi', 'bolshoiplanck', 'multidark', 'consuelo']
        self.halo_finders = ['rockstar', 'bdm']
        self.dummy_version_names = ['halotools.alpha', 'some_other_cuts']
        self.extension = ['.hdf5']

        self.dummy_fnames = ['hlist_1.00030', 'hlist_0.547', 'hlist_0.3397']

#        for simname in self.dummy_simnames:
#            os.mkdir(os.path.join(self.dummyloc, simname))







"""
    def test_processed_halocats_in_cache(self):


        catman = CatalogManager()


        self.defensively_create_empty_dir(dummyloc)
        self.defensively_create_empty_dir(os.path.join(dummyloc, 'dummy_sim1'))
        self.defensively_create_empty_dir(os.path.join(dummyloc, 'dummy_sim1', 'dummy_hfinder1'))
        self.defensively_create_empty_dir(os.path.join(dummyloc, 'dummy_sim1', 'dummy_hfinder2'))
        self.defensively_create_empty_dir(os.path.join(dummyloc, 'dummy_sim2'))
        self.defensively_create_empty_dir(os.path.join(dummyloc, 'dummy_sim2', 'dummy_hfinder1'))
        self.defensively_create_empty_dir(os.path.join(dummyloc, 'dummy_sim2', 'dummy_hfinder2'))

        temp_fnames = []
        temp_fnames.append(dummyloc + '/dummy_sim1/dummy_hfinder1/dummy_fname2.dummyversion1.hdf5')
        temp_fnames.append(dummyloc + '/dummy_sim1/dummy_hfinder1/dummy_fname3.dummyversion1.hdf5')
        temp_fnames.append(dummyloc + '/dummy_sim1/dummy_hfinder1/dummy_fname1.dummyversion1.hdf5')
        temp_fnames.append(dummyloc + '/dummy_sim2/dummy_hfinder1/dummy_fname1.dummyversion1.hdf5')
        temp_fnames.append(dummyloc + '/dummy_sim2/dummy_hfinder1/dummy_fname1.dummyversion2.hdf5')
        temp_fnames.append(dummyloc + '/dummy_sim2/dummy_hfinder1/dummy_fname2.dummyversion1.hdf5')
        temp_fnames.append(dummyloc + '/dummy_sim2/dummy_hfinder2/dummy_fname1.dummyversion1.hdf5')
        temp_fnames = list(set(temp_fnames))

        for f in temp_fnames:
            os.system('touch ' + f)
                
        result = catman.processed_halocats_in_cache(external_cache_loc=dummyloc)
        assert set(result) == set(temp_fnames)
"""

#   result = catman.processed_halocats_in_cache(external_cache_loc=dummyloc)
#   assert set(result) == set(temp_fnames)











