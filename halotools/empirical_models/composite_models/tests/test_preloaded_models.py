#!/usr/bin/env python

from unittest import TestCase
import pytest
import numpy as np
from astropy.table import Table

from ...composite_models import *
from ...factories import HodModelFactory, SubhaloModelFactory
from ...factories import PrebuiltHodModelFactory, PrebuiltSubhaloModelFactory

from ....utils.table_utils import compute_conditional_percentiles
from ....sim_manager import CachedHaloCatalog, FakeSim
from ....custom_exceptions import *

### Determine whether the machine is mine
# This will be used to select tests whose 
# returned values depend on the configuration 
# of my personal cache directory files
from astropy.config.paths import _find_home 
aph_home = '/Users/aphearin'
detected_home = _find_home()
if aph_home == detected_home:
    APH_MACHINE = True
else:
    APH_MACHINE = False


__all__ = ('TestHearin15', )

class TestHearin15(TestCase):

    def setup_class(self):
        pass

    def test_Hearin15(self):

        model = PrebuiltHodModelFactory('hearin15')
        halocat = FakeSim()
        model.populate_mock(halocat)

    def test_Leauthaud11(self):

        model = PrebuiltHodModelFactory('leauthaud11')
        halocat = FakeSim()
        model.populate_mock(halocat)

    def test_Leauthaud11b(self):

        model = PrebuiltHodModelFactory('leauthaud11') 
        halocat = FakeSim(redshift = 2.)
        # Test that an attempt to repopulate with a different halocat raises an exception
        with pytest.raises(HalotoolsError) as exc:
            model.populate_mock(halocat) #default redshift != 2

    def test_Leauthaud11c(self):

        model_highz = PrebuiltHodModelFactory('leauthaud11', redshift = 2.)
        halocat = FakeSim(redshift = 2.)
        model_highz.populate_mock(halocat)

    @pytest.mark.skipif('not APH_MACHINE')
    @pytest.mark.slow
    def test_hearin15_fullpop(self):
        halocat = CachedHaloCatalog(simname = 'bolshoi', redshift = 0)
        model = PrebuiltHodModelFactory('hearin15', threshold = 11)
        model.populate_mock(halocat)
        del model






