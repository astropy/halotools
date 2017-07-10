"""
"""
import numpy as np
import pytest

from ..zu_mandelbaum16 import ZuMandelbaum16QuenchingCens, ZuMandelbaum16QuenchingSats


def test_respects_bounds():
    for model in (ZuMandelbaum16QuenchingCens(), ZuMandelbaum16QuenchingSats()):
        mass = np.logspace(10, 15, 10)
        red_frac = model.mean_quiescent_fraction(prim_haloprop=mass)
        assert np.all(red_frac >= 0)
        assert np.all(red_frac <= 1)
        assert np.all(np.diff(red_frac) >= 0)


def test_raises_key_error():
    for model in (ZuMandelbaum16QuenchingCens(), ZuMandelbaum16QuenchingSats()):
        mass = np.logspace(10, 15, 10)
        with pytest.raises(KeyError) as err:
            __ = model.mean_quiescent_fraction(halo_mass=mass)
        substr = "Must pass one of the following keyword arguments"
        assert substr in err.value.args[0]


def test_expected_param_behavior1():
    """ Verify that we get the correct red fraction when evaluating right at
    the characteristic mass
    """
    for model in (ZuMandelbaum16QuenchingCens(), ZuMandelbaum16QuenchingSats()):
        key = 'quenching_mass_' + model.gal_type
        testmass = np.copy(model.param_dict[key])
        red_frac = model.mean_quiescent_fraction(prim_haloprop=testmass)
        assert np.allclose(1 - 1./np.e, red_frac)


def test_expected_param_behavior2():
    """ Verify that we get the correct behavior when increasing the characteristic mass
    """
    for model in (ZuMandelbaum16QuenchingCens(), ZuMandelbaum16QuenchingSats()):
        key = 'quenching_mass_' + model.gal_type
        testmass = np.copy(model.param_dict[key])
        red_frac1 = model.mean_quiescent_fraction(prim_haloprop=testmass)
        model.param_dict[key] *= 2
        red_frac2 = model.mean_quiescent_fraction(prim_haloprop=testmass)
        assert red_frac2 < red_frac1


def test_expected_param_behavior3():
    """ Verify that we get the correct behavior when increasing the exponential power
    """
    for model in (ZuMandelbaum16QuenchingCens(), ZuMandelbaum16QuenchingSats()):
        key = 'quenching_mass_' + model.gal_type
        testmass = np.copy(model.param_dict[key])*5
        red_frac1 = model.mean_quiescent_fraction(prim_haloprop=testmass)

        key = 'quenching_exp_power_' + model.gal_type
        model.param_dict[key] *= 2
        red_frac2 = model.mean_quiescent_fraction(prim_haloprop=testmass)
        assert red_frac2 > red_frac1

