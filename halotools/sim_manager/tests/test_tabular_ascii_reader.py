#!/usr/bin/env python

import os
import numpy as np
from unittest import TestCase
import pytest 

from astropy.config.paths import _find_home 

from ..tabular_ascii_reader import TabularAsciiReader
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

class TestTabularAsciiReader(TestCase):

    def setup_class(self):

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

        with pytest.raises(HalotoolsError) as err:
            reader = TabularAsciiReader(
                os.path.basename(self.dummy_fname), 
                columns_to_keep_dict = {'mass': (3, 'f4')})
        substr = 'is not a file'
        assert substr in err.value.message











