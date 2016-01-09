#!/usr/bin/env python

import os, shutil
import numpy as np
from unittest import TestCase
import warnings, pytest

from astropy.config.paths import _find_home 

from ..rockstar_hlist_reader import RockstarHlistReader

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
        os.system('touch '+self.dummy_fname)


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


    def test_good_args(self):

        reader = RockstarHlistReader(
            input_fname = self.dummy_fname, 
            columns_to_keep_dict = self.good_columns_to_keep_dict, 
            output_fname = self.good_output_fname, 
            simname = 'Jean Claude van Damme', halo_finder = 'ok usa',
            redshift = 4, version_name = 'dummy', Lbox = 100, particle_mass = 1e8 
            )

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

    def tearDown(self):
        try:
            shutil.rmtree(self.tmpdir)
        except:
            pass


