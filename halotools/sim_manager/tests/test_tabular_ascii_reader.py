#!/usr/bin/env python

import os
import numpy as np
from unittest import TestCase
import pytest 

from astropy.config.paths import _find_home 

from ..tabular_ascii_reader import TabularAsciiReader


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

class TestTabularAsciiReader(TestCase):

    def setUp(self):

        self.tmpdir = os.path.join(_find_home(), 'Desktop', 'tmp_testingdir')
        try:
            os.makedirs(self.tmpdir)
        except OSError:
            pass

        basename = 'abc.txt'
        self.dummy_fname = os.path.join(self.tmpdir, basename)
        os.system('touch '+self.dummy_fname)

    def test_get_fname(self):
        reader = TabularAsciiReader(
            self.dummy_fname, columns_to_keep_dict = {'mass': (3, 'f4')})

        with pytest.raises(IOError) as err:
            reader = TabularAsciiReader(
                os.path.basename(self.dummy_fname), 
                columns_to_keep_dict = {'mass': (3, 'f4')})
        substr = 'is not a file'
        assert substr in err.value.message

    def test_get_header_char(self):
        reader = TabularAsciiReader(
            self.dummy_fname, columns_to_keep_dict = {'mass': (3, 'f4')}, 
            header_char = '*')

        with pytest.raises(TypeError) as err:
            reader = TabularAsciiReader(self.dummy_fname, 
                columns_to_keep_dict = {'mass': (3, 'f4')}, 
                header_char = '###')
        substr = 'must be a single string character'
        assert substr in err.value.message

    def test_process_columns_to_keep(self):

        with pytest.raises(TypeError) as err:
            reader = TabularAsciiReader(
                self.dummy_fname, columns_to_keep_dict = {'mass': (3, 'f4', 'c')}, 
                header_char = '*')
        substr = 'must be a two-element tuple.'
        assert substr in err.value.message

        with pytest.raises(TypeError) as err:
            reader = TabularAsciiReader(
                self.dummy_fname, columns_to_keep_dict = {'mass': (3.5, 'f4')}, 
                header_char = '*')
        substr = 'The first element of the two-element tuple'
        assert substr in err.value.message

        with pytest.raises(TypeError) as err:
            reader = TabularAsciiReader(
                self.dummy_fname, columns_to_keep_dict = {'mass': (3, 'Jose Canseco')}, 
                header_char = '*')
        substr = 'The second element of the two-element tuple'
        assert substr in err.value.message

    def test_verify_input_row_cuts(self):

        reader = TabularAsciiReader(
            self.dummy_fname, columns_to_keep_dict = {'mass': (3, 'f4')}, 
            row_cut_min_dict = {'mass': 8})

        reader = TabularAsciiReader(
            self.dummy_fname, columns_to_keep_dict = {'mass': (3, 'f4')}, 
            row_cut_max_dict = {'mass': 8})

        reader = TabularAsciiReader(
            self.dummy_fname, columns_to_keep_dict = {'mass': (3, 'f4')}, 
            row_cut_eq_dict = {'mass': 8})

        reader = TabularAsciiReader(
            self.dummy_fname, columns_to_keep_dict = {'mass': (3, 'f4')}, 
            row_cut_neq_dict = {'mass': 8})

    def test_verify_min_max_consistency(self):

        reader = TabularAsciiReader(
            self.dummy_fname, columns_to_keep_dict = {'mass': (3, 'f4')}, 
            row_cut_min_dict = {'mass': 8}, row_cut_max_dict = {'mass': 9})
      
        with pytest.raises(ValueError) as err:
            reader = TabularAsciiReader(
                self.dummy_fname, columns_to_keep_dict = {'mass': (3, 'f4')}, 
                row_cut_min_dict = {'mass': 9}, row_cut_max_dict = {'mass': 8})
        substr = 'This will result in zero selected rows '
        assert substr in err.value.message

        with pytest.raises(KeyError) as err:
            reader = TabularAsciiReader(
                self.dummy_fname, columns_to_keep_dict = {'mass': (3, 'f4')}, 
                row_cut_min_dict = {'mass': 9}, row_cut_max_dict = {'vmax': 8})
        substr = 'The ``vmax`` key does not appear in the input'
        assert substr in err.value.message

        with pytest.raises(KeyError) as err:
            reader = TabularAsciiReader(
                self.dummy_fname, columns_to_keep_dict = {'vmax': (3, 'f4')}, 
                row_cut_min_dict = {'mass': 9}, row_cut_max_dict = {'vmax': 8})
        substr = 'The ``mass`` key does not appear in the input'
        assert substr in err.value.message

    def test_verify_eq_neq_consistency(self):

        with pytest.raises(ValueError) as err:
            reader = TabularAsciiReader(
                self.dummy_fname, columns_to_keep_dict = {'mass': (3, 'f4')}, 
                row_cut_eq_dict = {'mass': 8}, row_cut_neq_dict = {'mass': 8})
        substr = 'This will result in zero selected rows '
        assert substr in err.value.message


    def tearDown(self):
        os.system("rm -rf " + self.tmpdir)







