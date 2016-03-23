#!/usr/bin/env python

import os, shutil
import numpy as np
from unittest import TestCase
import warnings
from astropy.tests.helper import pytest 
from astropy.table import Table 

from astropy.config.paths import _find_home 

from ..rockstar_hlist_reader import RockstarHlistReader, _infer_redshift_from_input_fname
from ..halo_table_cache import HaloTableCache 

from ...custom_exceptions import HalotoolsError

try:
    import h5py 
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


### Determine whether the machine is mine
# This will be used to select tests whose 
# returned values depend on the configuration 
# of my personal cache directory files
aph_home = '/Users/aphearin'
detected_home = _find_home()
if aph_home == detected_home:
    APH_MACHINE = True
else:
    APH_MACHINE = False

__all__ = ('TestRockstarHlistReader', )


class TestRockstarHlistReader(TestCase):

    def setUp(self):

        self.tmpdir = os.path.join(_find_home(), 'Desktop', 'tmp_testingdir')
        try:
            os.makedirs(self.tmpdir)
        except OSError:
            pass

        basename = 'abc.txt'
        self.dummy_fname = os.path.join(self.tmpdir, basename)
        _t = Table({'x': [0]})
        _t.write(self.dummy_fname, format='ascii')


        self.good_columns_to_keep_dict = ({
            'halo_x': (1, 'f4'), 
            'halo_y': (2, 'f4'), 
            'halo_z': (3, 'f4'), 
            'halo_id': (4, 'i8'), 
            'halo_mvir': (5, 'f4')
            })

        self.good_output_fname = os.path.join(self.tmpdir, 'def.hdf5')

        self.bad_columns_to_keep_dict1 = ({
            'halo_x': (1, 'f4'), 
            'halo_z': (3, 'f4'), 
            'halo_id': (4, 'i8'), 
            'halo_mvir': (5, 'f4')
            })

        self.bad_columns_to_keep_dict2 = ({
            'halo_x': (1, 'f4'), 
            'halo_y': (2, 'f4'), 
            'halo_z': (3, 'f4'), 
            'halo_mvir': (5, 'f4')
            })

        self.bad_columns_to_keep_dict3 = ({
            'halo_x': (1, 'f4'), 
            'halo_y': (2, 'f4'), 
            'halo_z': (3, 'f4'), 
            'halo_id': (4, 'i8'), 
            })

        self.bad_columns_to_keep_dict4 = ({
            'halo_x': (1, 'f4'), 
            'halo_y': (1, 'f4'), 
            'halo_z': (3, 'f4'), 
            'halo_id': (4, 'i8'), 
            'halo_mvir': (5, 'f4')
            })

    @pytest.mark.skipif('not HAS_H5PY')
    def test_good_args(self):

        reader = RockstarHlistReader(
            input_fname = self.dummy_fname, 
            columns_to_keep_dict = self.good_columns_to_keep_dict, 
            output_fname = self.good_output_fname, 
            simname = 'Jean Claude van Damme', halo_finder = 'ok usa',
            redshift = 4, version_name = 'dummy', Lbox = 100, particle_mass = 1e8 
            )

    @pytest.mark.skipif('not HAS_H5PY')
    def test_bad_columns_to_keep_dict1(self):

        with pytest.raises(HalotoolsError) as err:
            reader = RockstarHlistReader(
                input_fname = self.dummy_fname, 
                columns_to_keep_dict = self.bad_columns_to_keep_dict1, 
                output_fname = self.good_output_fname, 
                simname = 'Jean Claude van Damme', halo_finder = 'ok usa',
                redshift = 4, version_name = 'dummy', Lbox = 100, particle_mass = 1e8 
                )
        substr = "at least have the following columns"
        assert substr in err.value.message

    @pytest.mark.skipif('not HAS_H5PY')
    def test_bad_columns_to_keep_dict2(self):

        with pytest.raises(HalotoolsError) as err:
            reader = RockstarHlistReader(
                input_fname = self.dummy_fname, 
                columns_to_keep_dict = self.bad_columns_to_keep_dict2, 
                output_fname = self.good_output_fname, 
                simname = 'Jean Claude van Damme', halo_finder = 'ok usa',
                redshift = 4, version_name = 'dummy', Lbox = 100, particle_mass = 1e8 
                )
        substr = "at least have the following columns"
        assert substr in err.value.message

    @pytest.mark.skipif('not HAS_H5PY')
    def test_bad_columns_to_keep_dict3(self):

        with pytest.raises(HalotoolsError) as err:
            reader = RockstarHlistReader(
                input_fname = self.dummy_fname, 
                columns_to_keep_dict = self.bad_columns_to_keep_dict3, 
                output_fname = self.good_output_fname, 
                simname = 'Jean Claude van Damme', halo_finder = 'ok usa',
                redshift = 4, version_name = 'dummy', Lbox = 100, particle_mass = 1e8 
                )
        substr = "at least have the following columns"
        assert substr in err.value.message

    @pytest.mark.skipif('not HAS_H5PY')
    def test_bad_columns_to_keep_dict4(self):

        with pytest.raises(ValueError) as err:
            reader = RockstarHlistReader(
                input_fname = self.dummy_fname, 
                columns_to_keep_dict = self.bad_columns_to_keep_dict4, 
                output_fname = self.good_output_fname, 
                simname = 'Jean Claude van Damme', halo_finder = 'ok usa',
                redshift = 4, version_name = 'dummy', Lbox = 100, particle_mass = 1e8 
                )
        substr = "appears more than once in your ``columns_to_keep_dict``"
        assert substr in err.value.message


    @pytest.mark.slow
    @pytest.mark.skipif('not HAS_H5PY')
    @pytest.mark.skipif('not APH_MACHINE')
    def test_read_dummy_halo_catalog1(self):
        """
        """
        fname = "/Users/aphearin/.astropy/cache/halotools/raw_halo_catalogs/bolplanck/rockstar/hlist_0.07812.list"

        columns_to_keep_dict = ({
            'halo_spin_bullock': (43, 'f4'), 'halo_id': (1, 'i8'), 'halo_upid': (6, 'i8'), 
            'halo_x': (17, 'f4'), 'halo_y': (18, 'f4'), 'halo_z': (19, 'f4')
            })

        reader = RockstarHlistReader(
            input_fname = fname, 
            columns_to_keep_dict = columns_to_keep_dict, 
            output_fname = self.good_output_fname, 
            simname = 'bolplanck', halo_finder = 'rockstar', redshift = 11.8008, 
            version_name = 'dummy', Lbox = 250., particle_mass = 1.35e8, 
            row_cut_min_dict = {'halo_spin_bullock': 0.1}, 
            row_cut_max_dict = {'halo_spin_bullock': 0.9}, 
            row_cut_eq_dict = {'halo_upid': -1}, 
            row_cut_neq_dict = {'halo_id': -1}
            )

        reader.read_halocat(columns_to_convert_from_kpc_to_mpc=[], 
            write_to_disk = True)


    @pytest.mark.slow
    @pytest.mark.skipif('not HAS_H5PY')
    @pytest.mark.skipif('not APH_MACHINE')
    def test_read_dummy_halo_catalog2(self):
        """
        """
        fname = "/Users/aphearin/.astropy/cache/halotools/raw_halo_catalogs/bolplanck/rockstar/hlist_0.07812.list"

        columns_to_keep_dict = ({
            'halo_spin_bullock': (43, 'f4'), 'halo_id': (1, 'i8'), 'halo_upid': (6, 'i8'), 
            'halo_x': (17, 'f4'), 'halo_y': (18, 'f4'), 'halo_z': (19, 'f4')
            })


        cache = HaloTableCache()
        entry = cache.log[0]

        with pytest.raises(HalotoolsError) as err:
            reader = RockstarHlistReader(
                input_fname = fname, 
                columns_to_keep_dict = columns_to_keep_dict, 
                output_fname = entry.fname, 
                simname = entry.simname, halo_finder = entry.halo_finder, redshift = entry.redshift, 
                version_name = entry.version_name, Lbox = 250., particle_mass = 1.35e8)
        substr = "There is already an existing entry in the Halotools cache log"
        assert substr in err.value.message

    @pytest.mark.slow
    @pytest.mark.skipif('not HAS_H5PY')
    @pytest.mark.skipif('not APH_MACHINE')
    def test_read_dummy_halo_catalog3(self):
        """
        """
        fname = "/Users/aphearin/.astropy/cache/halotools/raw_halo_catalogs/bolplanck/rockstar/hlist_0.07812.list"

        columns_to_keep_dict = ({
            'spin_bullock': (43, 'f4'), 'halo_id': (1, 'i8'), 'halo_upid': (6, 'i8'), 
            'halo_x': (17, 'f4'), 'halo_y': (18, 'f4'), 'halo_z': (19, 'f4')
            })

        with pytest.raises(HalotoolsError) as err:
            reader = RockstarHlistReader(
                input_fname = fname, 
                columns_to_keep_dict = columns_to_keep_dict, 
                output_fname = self.good_output_fname, 
                simname = 'bolplanck', halo_finder = 'rockstar', redshift = 11.8008, 
                version_name = 'dummy', Lbox = 250., particle_mass = 1.35e8, 
                )
        substr = "must begin with the substring ``halo_``"
        assert substr in err.value.message

    @pytest.mark.slow
    @pytest.mark.skipif('not HAS_H5PY')
    @pytest.mark.skipif('not APH_MACHINE')
    def test_read_dummy_halo_catalog4(self):
        """
        """
        fname = "/Users/aphearin/.astropy/cache/halotools/raw_halo_catalogs/bolplanck/rockstar/hlist_0.07812.list"

        columns_to_keep_dict = ({
            'halo_spin_bullock': (43, 'f4'), 'halo_id': (1, 'i8'), 'halo_upid': (6, 'i8'), 
            'halo_x': (17, 'f4'), 'halo_y': (18, 'f4'), 'halo_z': (19, 'f4')
            })

        reader = RockstarHlistReader(
            input_fname = fname, 
            columns_to_keep_dict = columns_to_keep_dict, 
            output_fname = 'std_cache_loc', 
            simname = 'bolplanck', halo_finder = 'rockstar', redshift = 11.8008, 
            version_name = 'dummy', Lbox = 250., particle_mass = 1.35e8, 
            )
 
    def test_infer_redshift_from_fname(self):
        fname = 'hlist_0.07812.list'
        result = _infer_redshift_from_input_fname(fname)
        assert result == 11.8008

    def tearDown(self):
        try:
            shutil.rmtree(self.tmpdir)
        except:
            pass


