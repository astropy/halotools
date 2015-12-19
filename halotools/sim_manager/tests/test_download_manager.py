#!/usr/bin/env python


import os, fnmatch
import numpy as np
from astropy.config.paths import _find_home 
from astropy.tests.helper import remote_data, pytest
from unittest import TestCase

from ..download_manager import DownloadManager

from ...custom_exceptions import UnsupportedSimError

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


class TestDownloadManager(TestCase):


    def setup_class(self):

        homedir = _find_home()

        self.downman = DownloadManager()

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

        self.raw_halo_table_dir = os.path.join(self.dummyloc, 'raw_halo_catalogs')
        defensively_create_empty_dir(self.raw_halo_table_dir)

        self.simnames = ['bolshoi', 'bolplanck', 'multidark', 'consuelo']
        self.halo_finders = ['rockstar', 'bdm']
        self.dummy_version_names = ['halotools.alpha']
        self.extension = '.hdf5'

        self.bolshoi_fnames = ['hlist_0.33035', 'hlist_0.54435', 'hlist_0.67035', 'hlist_1.00035']
        self.bolshoi_bdm_fnames = ['hlist_0.33030', 'hlist_0.49830', 'hlist_0.66430', 'hlist_1.00035']
        self.bolplanck_fnames = ['hlist_0.33035', 'hlist_0.54435', 'hlist_0.67035', 'hlist_1.00035']
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
            elif simname == 'bolplanck':
                fnames = self.bolplanck_fnames
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


    @remote_data
    def test_ptcl_tables_available_for_download(self):

        file_list = self.downman.ptcl_tables_available_for_download(simname='bolshoi')
        assert len(file_list) == 1
        assert 'hlist_1.00035.particles.hdf5' == os.path.basename(file_list[0])

        file_list = self.downman.ptcl_tables_available_for_download(simname='multidark')
        assert len(file_list) == 1
        assert 'hlist_1.00109.particles.hdf5' == os.path.basename(file_list[0])

        consuelo_set = set(
            ['hlist_0.33324.particles.hdf5', 
            'hlist_0.50648.particles.hdf5',
            'hlist_0.67540.particles.hdf5', 
            'hlist_1.00000.particles.hdf5']
            )
        file_list = self.downman.ptcl_tables_available_for_download(simname='consuelo')
        assert len(file_list) == 4
        file_set = set([os.path.basename(f) for f in file_list])
        assert file_set == consuelo_set

        bolplanck_set = set(
            ['hlist_0.33406.particles.hdf5', 
            'hlist_0.50112.particles.hdf5',
            'hlist_0.66818.particles.hdf5', 
            'hlist_1.00231.particles.hdf5']
            )
        file_list = self.downman.ptcl_tables_available_for_download(simname='bolplanck')
        assert len(file_list) == 4
        file_set = set([os.path.basename(f) for f in file_list])
        assert file_set == bolplanck_set

    @remote_data
    def test_processed_halo_tables_available_for_download(self):

        file_list = self.downman.processed_halo_tables_available_for_download(
            simname='bolshoi', halo_finder='rockstar')
        assert file_list != []


    @pytest.mark.skipif('not APH_MACHINE')
    @remote_data
    def test_unsupported_sim_download_attempt(self): 
        simname = 'consuelo'
        redshift = 2
        halo_finder = 'bdm'
        with pytest.raises(UnsupportedSimError) as exc:
            self.downman.download_processed_halo_table(simname = simname, 
                halo_finder = halo_finder, desired_redshift = redshift, 
                overwrite = False)

    @pytest.mark.skipif('not APH_MACHINE')
    @remote_data
    def test_raw_halo_tables_available_for_download(self):
        l = self.downman.raw_halo_tables_available_for_download(
            simname='bolshoi', halo_finder = 'bdm')
        assert len(l) > 0

        l = self.downman.raw_halo_tables_available_for_download(
            simname='bolshoi', halo_finder = 'rockstar')
        assert len(l) > 0

        l = self.downman.raw_halo_tables_available_for_download(
            simname='bolplanck', halo_finder = 'rockstar')
        assert len(l) > 0

        l = self.downman.raw_halo_tables_available_for_download(
            simname='multidark', halo_finder = 'rockstar')
        assert len(l) > 0

    def teardown_class(self):
        os.system('rm -rf ' + self.dummyloc)
















