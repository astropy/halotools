""" Module providing unit-testing for the component models in
`halotools.empirical_models.occupation_components.cacciato09_components` module"
"""
import numpy as np
from scipy.stats import kstest
from astropy.tests.helper import pytest

from .. import Cacciato09Cens, Cacciato09Sats
from ....custom_exceptions import HalotoolsError


__all__ = ('test_Cacciato09Cens1', 'test_Cacciato09Sats1')


def test_Cacciato09Cens1():
    """
    Verify that the mean and Monte Carlo occupations are both reasonable and
    in agreement.
    """
    model = Cacciato09Cens(threshold=9.5)
    ncen_exp = model.mean_occupation(prim_haloprop=5e11)
    ncen_mc = model.mc_occupation(prim_haloprop=np.ones(int(1e5))*5e11, seed=1)
    assert np.isclose(np.average(ncen_mc), ncen_exp, rtol=1e-2, atol=1.e-2)


def test_Cacciato09Cens2():
    """Check that the model behavior is altered in the expected way by changing
    param_dict values."""
    model = Cacciato09Cens(threshold=9.5)
    ncen_exp = model.mean_occupation(prim_haloprop=5e11)

    # Increasing log L_0 does increase occupation.
    model.param_dict['log_L_0'] += 0.1
    ncen_exp_new = model.mean_occupation(prim_haloprop=5e11)
    assert ncen_exp_new > ncen_exp

    # Decreasing log M_1 has the same effect.
    model.param_dict['log_M_1'] -= 0.1
    ncen_exp_new = model.mean_occupation(prim_haloprop=5e11)
    assert ncen_exp_new > ncen_exp


def test_Cacciato09Cens3():
    """Check that increasing stellar mass thresholds
    decreases the mean occupation.
    """
    model_1 = Cacciato09Cens(threshold=9.5)
    model_2 = Cacciato09Cens(threshold=10.0)
    model_3 = Cacciato09Cens(threshold=10.5)
    ncen_exp_1 = model_1.mean_occupation(prim_haloprop=5e11)
    ncen_exp_2 = model_2.mean_occupation(prim_haloprop=5e11)
    ncen_exp_3 = model_3.mean_occupation(prim_haloprop=5e11)
    assert ncen_exp_1 > ncen_exp_2 > ncen_exp_3


def test_Cacciato09Cens4():
    """Check that increasing the halo mass increases the mean occupation.
    """
    model = Cacciato09Cens(threshold=9.5)
    ncen_exp = model.mean_occupation(prim_haloprop=np.logspace(9, 12, 100))
    assert np.all(np.diff(ncen_exp) >= 0)


def test_Cacciato09Cens5():
    """Check that the CLF behaves as expected.
    """
    model = Cacciato09Cens(threshold=9.5)
    clf = model.clf(model.median_prim_galprop(prim_haloprop=1e12),
                    prim_haloprop=1e12)
    assert np.isclose(np.sqrt(2 * np.pi * model.param_dict['sigma']**2)**(-1),
                      clf, rtol=1e-6, atol=1.e-6)
    model.param_dict['sigma'] = 0.24
    clf = model.clf(model.median_prim_galprop(prim_haloprop=1e13),
                    prim_haloprop=1e13)
    assert np.isclose(
        np.sqrt(2 * np.pi * model.param_dict['sigma'] ** 2) ** (-1),
        clf, rtol=1e-6, atol=1.e-6)


def test_Cacciato09Cens6():
    """Check that the median primary galaxy property behave accordingly.
    """
    model = Cacciato09Cens(threshold=9.5)
    prim_galprop_1 = model.median_prim_galprop(prim_haloprop=1e14)
    model.param_dict['log_M_1'] += 0.1
    prim_galprop_2 = model.median_prim_galprop(prim_haloprop=1e14*10**0.1)
    assert np.isclose(prim_galprop_1, prim_galprop_2, rtol=1e-6, atol=1.e-2)

    model.param_dict['log_L_0'] += 0.1
    prim_galprop_3 = model.median_prim_galprop(prim_haloprop=1e14*10**0.1)
    assert np.isclose(prim_galprop_2 * 10**0.1, prim_galprop_3, rtol=1e-6,
                      atol=1.e-2)

    model.param_dict['gamma_1'] += 0.1
    prim_galprop_4 = model.median_prim_galprop(prim_haloprop=1e14*10**0.1)
    assert prim_galprop_3 != prim_galprop_4

    model.param_dict['gamma_2'] += 0.1
    prim_galprop_5 = model.median_prim_galprop(prim_haloprop=1e14 * 10 ** 0.1)
    assert prim_galprop_4 != prim_galprop_5

    model.param_dict['sigma'] += 0.1
    prim_galprop_6 = model.median_prim_galprop(prim_haloprop=1e14 * 10 ** 0.1)
    assert np.isclose(prim_galprop_5, prim_galprop_6, rtol=1e-6, atol=1.e-2)


def test_Cacciato09Cens7():
    """heck that no luminosity is below the threshold.
    """
    model = Cacciato09Cens(threshold=9.5)
    lum_mc = model.mc_prim_galprop(prim_haloprop=np.ones(int(1e5))*5e11,
                                   seed=1)
    assert np.all(lum_mc >= 10**model.threshold)

    # Check that luminosities follow the expected distribution.
    def cdf(lum):
        return np.array([(model.mean_occupation(prim_haloprop=5e11) -
                          model.mean_occupation(prim_haloprop=5e11,
                                                prim_galprop_min=l)) /
                         model.mean_occupation(prim_haloprop=5e11) for l in
                         lum])

    p_value = kstest(lum_mc, cdf)[1]

    assert p_value > 0.001


def test_Cacciato09Cens_median_prim_galprop_raises_exception():
    model = Cacciato09Cens(threshold=10.5)
    with pytest.raises(HalotoolsError) as err:
        __ = model.median_prim_galprop(x=7)
    substr = "You must pass either a ``table`` or ``prim_haloprop``"
    assert substr in err.value.args[0]


def test_Cacciato09Cens_clf_raises_exception():
    model = Cacciato09Cens(threshold=10.5)
    with pytest.raises(AssertionError) as err:
        __ = model.clf(prim_galprop=np.zeros(5)+1e10, prim_haloprop=np.zeros(4)+1e12)
    substr = "with multiple elements, they must have the same length."
    assert substr in err.value.args[0]


def test_Cacciato09Cens_mc_prim_galprop_raises_exception1():
    model = Cacciato09Cens(threshold=10.5)
    with pytest.raises(HalotoolsError) as err:
        __ = model.mc_prim_galprop(mass=4)
    substr = "You must pass either a ``table`` or ``prim_haloprop``"
    assert substr in err.value.args[0]


def test_Cacciato09Cens_mc_prim_galprop_raises_exception2():
    model = Cacciato09Cens(threshold=10.5)
    with pytest.raises(HalotoolsError) as err:
        __ = model.mc_prim_galprop(prim_haloprop=4)
    substr = "has (virtually) no expected"
    assert substr in err.value.args[0]


def test_Cacciato09Sats1():
    """
    Verify that the mean and Monte Carlo occupations are both reasonable and
    in agreement.
    """
    model = Cacciato09Sats(threshold=9.5)
    nsat_exp = model.mean_occupation(prim_haloprop=5e13)
    nsat_mc = model.mc_occupation(prim_haloprop=np.ones(int(1e5))*5e13, seed=1)
    np.testing.assert_allclose(np.average(nsat_mc), nsat_exp, rtol=1e-2,
                               atol=1.e-2)


def test_Cacciato09Sats2():
    """
    Check that the model behavior is altered in the expected way by changing
    param_dict values.
    """
    model = Cacciato09Sats(threshold=9.5)
    nsat_exp = model.mean_occupation(prim_haloprop=5e13)
    # Increasing b_0 by x should increase the occupation by exactly 10**x.
    model.param_dict['b_0'] += 0.1
    nsat_exp_new = model.mean_occupation(prim_haloprop=5e13)
    assert np.isclose(nsat_exp_new, nsat_exp * 10**0.1, rtol=1e-2, atol=1.e-2)

    # Increasing b_1 by x should increase the occupation by exactly
    # 10**(x * (log prim_haloprop - 12.0)).
    model.param_dict['b_0'] -= 0.1
    model.param_dict['b_1'] += 0.1
    nsat_exp_new = model.mean_occupation(prim_haloprop=5e13)
    assert np.isclose(nsat_exp_new, nsat_exp * 10**(
        0.1 * (np.log10(5e13) - 12.0)), rtol=1e-2, atol=1.e-2)

    # Increasing b_2 by x should increase the occupation by exactly
    # 10**(x * (log prim_haloprop - 12.0)**2).
    model.param_dict['b_1'] -= 0.1
    model.param_dict['b_2'] += 0.1
    nsat_exp_new = model.mean_occupation(prim_haloprop=5e13)
    assert np.isclose(nsat_exp_new, nsat_exp * 10 ** (
        0.1 * (np.log10(5e13) - 12.0)**2), rtol=1e-2, atol=1.e-2)


def test_Cacciato09Sats3():
    """
    Check that increasing stellar mass thresholds decreases the mean
    occupation.
    """
    model_1 = Cacciato09Sats(threshold=9.5)
    model_2 = Cacciato09Sats(threshold=10.0)
    model_3 = Cacciato09Sats(threshold=10.5)
    nsat_exp_1 = model_1.mean_occupation(prim_haloprop=5e11)
    nsat_exp_2 = model_2.mean_occupation(prim_haloprop=5e11)
    nsat_exp_3 = model_3.mean_occupation(prim_haloprop=5e11)
    assert nsat_exp_1 > nsat_exp_2
    assert nsat_exp_2 > nsat_exp_3


def test_Cacciato09Sats4():
    """
    Check that all dictionary parameters have an effect on the CLF.
    Particularly, this checks that all central occupation parameters are
    successfully passed to the internal Cacciato09Cens model.
    """
    model = Cacciato09Sats(threshold=9.0)

    clf_orig = model.clf(prim_haloprop=1e14, prim_galprop=2e11)
    for param_key in model.param_dict:
        model_new = Cacciato09Sats(threshold=9.0)
        model_new.param_dict[param_key] += 0.1
        clf_new = model_new.clf(prim_haloprop=1e14, prim_galprop=2e11)
        assert clf_new != clf_orig


def test_Cacciato09Sats5():
    """
    Check that no luminosity is below the threshold.
    """
    model = Cacciato09Sats(threshold=9.0)
    lum_mc = model.mc_prim_galprop(prim_haloprop=np.ones(int(1e5))*5e13,
                                   seed=1)
    assert np.all(lum_mc >= 10**model.threshold)

    # Check that luminosities follow the expected distribution.
    def cdf(lum):
        return np.array([(model.mean_occupation(prim_haloprop=5e13) -
                          model.mean_occupation(prim_haloprop=5e13,
                                                prim_galprop_min=l)) /
                         model.mean_occupation(prim_haloprop=5e13) for l in
                         lum])

    p_value = kstest(lum_mc, cdf)[1]
    assert p_value > 0.001


def test_Cacciato09Sats_phi_sat_raises_exception():
    model = Cacciato09Sats(threshold=11.0)
    with pytest.raises(HalotoolsError) as err:
        __ = model.phi_sat(x=4)
    substr = "You must pass either a ``table`` or ``prim_haloprop``"
    assert substr in err.value.args[0]


def test_Cacciato09Sats_alpha_sat_raises_exception():
    model = Cacciato09Sats(threshold=11.0)
    with pytest.raises(HalotoolsError) as err:
        __ = model.alpha_sat(x=4)
    substr = "You must pass either a ``table`` or ``prim_haloprop``"
    assert substr in err.value.args[0]


def test_Cacciato09Sats_prim_galprop_cut_raises_exception():
    model = Cacciato09Sats(threshold=11.0)
    with pytest.raises(HalotoolsError) as err:
        __ = model.prim_galprop_cut(x=4)
    substr = "You must pass either a ``table`` or ``prim_haloprop``"
    assert substr in err.value.args[0]


def test_Cacciato09Sats_mean_occupation_raises_exception1():
    model = Cacciato09Sats(threshold=11.0)
    with pytest.raises(HalotoolsError) as err:
        __ = model.mean_occupation(x=4)
    substr = "You must pass either a ``table`` or ``prim_haloprop``"
    assert substr in err.value.args[0]


def test_Cacciato09Sats_mean_occupation_raises_exception2():
    model = Cacciato09Sats(threshold=11.0)
    with pytest.raises(HalotoolsError) as err:
        __ = model.mean_occupation(table={'halo_m200b': np.array((1e11, 1e12))},
                prim_galprop_min=5, prim_galprop_max=4)
    substr = "keyword must be bigger than 10^threshold"
    assert substr in err.value.args[0]


def test_Cacciato09Sats_clf_raises_exception():
    model = Cacciato09Sats(threshold=10.5)
    with pytest.raises(HalotoolsError) as err:
        __ = model.clf(prim_galprop=np.zeros(5)+1e10, prim_haloprop=np.zeros(4)+1e12)
    substr = "with multiple elements, they must have the same length"
    assert substr in err.value.args[0]


def test_Cacciato09Sats_mc_prim_galprop_raises_exception1():
    model = Cacciato09Sats(threshold=10.5)
    with pytest.raises(HalotoolsError) as err:
        __ = model.mc_prim_galprop(mass=8)
    substr = "You must pass either a ``table`` or ``prim_haloprop``"
    assert substr in err.value.args[0]
