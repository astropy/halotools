#!/usr/bin/env python

from unittest import TestCase
from astropy.tests.helper import pytest

from ....factories import PrebuiltHodModelFactory

from .....sim_manager import FakeSim

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

__all__ = ('TestTinker13', )


class TestTinker13(TestCase):

    def test_tinker13_default(self):
        model = PrebuiltHodModelFactory('tinker13')

    def test_tinker13_abscissa(self):
        model = PrebuiltHodModelFactory('tinker13',
                quiescent_fraction_abscissa=[1e12, 1e13, 1e14, 1e15],
                quiescent_fraction_ordinates=[0.25, 0.5, 0.75, 0.9])

    @pytest.mark.slow
    def test_tinker13_populate1(self):
        model = PrebuiltHodModelFactory('tinker13')
        fake_sim = FakeSim()
        model.populate_mock(fake_sim)
