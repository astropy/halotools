"""
"""
import numpy as np

import pytest

from ..binary_galprop_models import BinaryGalpropInterpolModel

from ....custom_exceptions import HalotoolsError

__all__ = ('test_mean_galprop_fraction', 'test_mc_galprop_fraction')



abscissa, ordinates = [12, 15], [1/3., 0.9]
Npts = int(5e3)
testmass12 = np.ones(Npts)*1e12
testmass135 = np.ones(Npts)*10.**13.5
testmass15 = np.ones(Npts)*1e15


def test_mean_galprop_fraction():
    model = BinaryGalpropInterpolModel(galprop_name='late_type',
        galprop_abscissa=abscissa, galprop_ordinates=ordinates,
        prim_haloprop_key='vpeak_host', gal_type='sats')

    frac12 = model.mean_late_type_fraction(prim_haloprop=testmass12)
    frac135 = model.mean_late_type_fraction(prim_haloprop=testmass135)
    frac15 = model.mean_late_type_fraction(prim_haloprop=testmass15)

    midval = 0.5*np.sum(ordinates)
    assert np.all(frac12 == ordinates[0])
    assert np.all(frac135 == midval)
    assert np.all(frac15 == ordinates[1])


def test_mc_galprop_fraction():
    model = BinaryGalpropInterpolModel(galprop_name='late_type',
        galprop_abscissa=abscissa, galprop_ordinates=ordinates,
        prim_haloprop_key='vpeak_host', gal_type='sats')

    frac12 = model.mean_late_type_fraction(prim_haloprop=testmass12)
    frac135 = model.mean_late_type_fraction(prim_haloprop=testmass135)
    frac15 = model.mean_late_type_fraction(prim_haloprop=testmass15)

    mean_mcfrac12 = np.mean(model.mc_late_type(prim_haloprop=testmass12, seed=43))
    mean_mcfrac135 = np.mean(model.mc_late_type(prim_haloprop=testmass135, seed=43))
    mean_mcfrac15 = np.mean(model.mc_late_type(prim_haloprop=testmass15, seed=43))

    np.testing.assert_allclose(mean_mcfrac12, frac12, rtol=1e-2, atol=1.e-2)
    np.testing.assert_allclose(mean_mcfrac135, frac135, rtol=1e-2, atol=1.e-2)
    np.testing.assert_allclose(mean_mcfrac15, frac15, rtol=1e-2, atol=1.e-2)


def test_abscissa_check():
    with pytest.raises(HalotoolsError) as err:
        model = BinaryGalpropInterpolModel(galprop_name='late_type',
            galprop_abscissa=[12, 12], galprop_ordinates=[0.5, 0.9],
            prim_haloprop_key='vpeak_host', gal_type='sats')
    substr = "Your input ``galprop_abscissa`` cannot have any repeated values"
    assert substr in err.value.args[0]


def test_ordinates_check():
    with pytest.raises(HalotoolsError) as err:
        model = BinaryGalpropInterpolModel(galprop_name='late_type',
            galprop_abscissa=[12, 13], galprop_ordinates=[0.5, 1.9],
            prim_haloprop_key='vpeak_host', gal_type='sats')
    substr = "All values of the input ``galprop_ordinates`` must be between 0 and 1, inclusive."
    assert substr in err.value.args[0]


def test_galprop_ordinates_consistency():
    with pytest.raises(HalotoolsError) as err:
        model = BinaryGalpropInterpolModel(galprop_name='late_type',
            galprop_abscissa=[12, 13], galprop_ordinates=[0.5, 0.7, 0.9],
            prim_haloprop_key='vpeak_host', gal_type='sats')
    substr = "Input ``galprop_abscissa`` and ``galprop_ordinates`` must have the same length"
    assert substr in err.value.args[0]
