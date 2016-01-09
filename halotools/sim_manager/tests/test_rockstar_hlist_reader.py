#!/usr/bin/env python

import os, shutil
import numpy as np
from unittest import TestCase
import pytest 

from astropy.config.paths import _find_home 

from ..rockstar_hlist_reader import RockstarHlistReader


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

    def test_get_fname(self):

        reader = RockstarHlistReader(
            input_fname = self.dummy_fname, 
            columns_to_keep_dict = self.good_columns_to_keep_dict, 
            output_fname = self.good_output_fname, 
            simname = 'Jean Claude van Damme', halo_finder = 'ok usa',
            redshift = 4, version_name = 'dummy', Lbox = 100, particle_mass = 1e8 
            )


    def tearDown(self):
        try:
            shutil.rmtree(self.tmpdir)
        except:
            pass


