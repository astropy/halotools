#!/usr/bin/env python
import numpy as np

from astropy.tests.helper import pytest
from unittest import TestCase

from ..binary_galprop_models import BinaryGalpropInterpolModel

from ....custom_exceptions import HalotoolsError

__all__ = ['TestBinaryGalpropInterpolModel']


class TestBinaryGalpropInterpolModel(TestCase):
    """ Class providing testing for the
    `~halotools.empirical_models.BinaryGalpropInterpolModel`.
    """

    def setUp(self):

        self.abscissa, self.ordinates = [12, 15], [1/3., 0.9]
        self.model = BinaryGalpropInterpolModel(galprop_name='late_type',
            galprop_abscissa=self.abscissa, galprop_ordinates=self.ordinates,
            prim_haloprop_key='vpeak_host', gal_type='sats')

        Npts = int(5e3)
        self.testmass12 = np.ones(Npts)*1e12
        self.testmass135 = np.ones(Npts)*10.**13.5
        self.testmass15 = np.ones(Npts)*1e15

    def test_mean_galprop_fraction(self):
        m = self.model

        frac12 = m.mean_late_type_fraction(prim_haloprop=self.testmass12)
        frac135 = m.mean_late_type_fraction(prim_haloprop=self.testmass135)
        frac15 = m.mean_late_type_fraction(prim_haloprop=self.testmass15)

        midval = 0.5*np.sum(self.ordinates)
        assert np.all(frac12 == self.ordinates[0])
        assert np.all(frac135 == midval)
        assert np.all(frac15 == self.ordinates[1])

    def test_mc_galprop_fraction(self):
        m = self.model

        frac12 = m.mean_late_type_fraction(prim_haloprop=self.testmass12)
        frac135 = m.mean_late_type_fraction(prim_haloprop=self.testmass135)
        frac15 = m.mean_late_type_fraction(prim_haloprop=self.testmass15)

        mean_mcfrac12 = np.mean(m.mc_late_type(prim_haloprop=self.testmass12, seed=43))
        mean_mcfrac135 = np.mean(m.mc_late_type(prim_haloprop=self.testmass135, seed=43))
        mean_mcfrac15 = np.mean(m.mc_late_type(prim_haloprop=self.testmass15, seed=43))

        np.testing.assert_allclose(mean_mcfrac12, frac12, rtol=1e-2, atol=1.e-2)
        np.testing.assert_allclose(mean_mcfrac135, frac135, rtol=1e-2, atol=1.e-2)
        np.testing.assert_allclose(mean_mcfrac15, frac15, rtol=1e-2, atol=1.e-2)

    def test_abscissa_check(self):
        with pytest.raises(HalotoolsError) as err:
            model = BinaryGalpropInterpolModel(galprop_name='late_type',
                galprop_abscissa=[12, 12], galprop_ordinates=[0.5, 0.9],
                prim_haloprop_key='vpeak_host', gal_type='sats')
        substr = "Your input ``galprop_abscissa`` cannot have any repeated values"
        assert substr in err.value.args[0]

    def test_ordinates_check(self):
        with pytest.raises(HalotoolsError) as err:
            model = BinaryGalpropInterpolModel(galprop_name='late_type',
                galprop_abscissa=[12, 13], galprop_ordinates=[0.5, 1.9],
                prim_haloprop_key='vpeak_host', gal_type='sats')
        substr = "All values of the input ``galprop_ordinates`` must be between 0 and 1, inclusive."
        assert substr in err.value.args[0]

    def test_galprop_ordinates_consistency(self):
        with pytest.raises(HalotoolsError) as err:
            model = BinaryGalpropInterpolModel(galprop_name='late_type',
                galprop_abscissa=[12, 13], galprop_ordinates=[0.5, 0.7, 0.9],
                prim_haloprop_key='vpeak_host', gal_type='sats')
        substr = "Input ``galprop_abscissa`` and ``galprop_ordinates`` must have the same length"
        assert substr in err.value.args[0]
