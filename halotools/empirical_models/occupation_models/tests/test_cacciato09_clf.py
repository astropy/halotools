""" Module providing unit-testing for the component models in
`halotools.empirical_models.occupation_components.cacciato09_components` module"
"""
import numpy as np
from scipy.stats import kstest
from scipy.interpolate import interp1d
import pytest
from scipy.integrate import cumtrapz

from .. import Cacciato09Cens, Cacciato09Sats
from ....custom_exceptions import HalotoolsError


__all__ = ("test_Cacciato09Cens1", "test_Cacciato09Sats1")


@pytest.mark.installation_test
def test_Cacciato09Cens1():
    """
    Verify that the mean and Monte Carlo occupations are both reasonable and
    in agreement.
    """
    model = Cacciato09Cens(threshold=9.5)
    ncen_exp = model.mean_occupation(prim_haloprop=5e11)
    ncen_mc = model.mc_occupation(prim_haloprop=np.ones(int(1e5)) * 5e11, seed=1)
    assert np.isclose(np.average(ncen_mc), ncen_exp, rtol=1e-2, atol=1.0e-2)


def test_Cacciato09Cens2():
    """Check that the model behavior is altered in the expected way by changing
    param_dict values."""
    model = Cacciato09Cens(threshold=9.5)
    ncen_exp = model.mean_occupation(prim_haloprop=5e11)

    # Increasing log L_0 does increase occupation.
    model.param_dict["log_L_0"] += 0.1
    ncen_exp_new = model.mean_occupation(prim_haloprop=5e11)
    assert ncen_exp_new > ncen_exp

    # Decreasing log M_1 has the same effect.
    model.param_dict["log_M_1"] -= 0.1
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
    clf = model.clf(model.median_prim_galprop(prim_haloprop=1e12), prim_haloprop=1e12)
    assert np.isclose(
        np.sqrt(2 * np.pi * model.param_dict["sigma"] ** 2) ** (-1),
        clf,
        rtol=1e-6,
        atol=1.0e-6,
    )
    model.param_dict["sigma"] = 0.24
    clf = model.clf(model.median_prim_galprop(prim_haloprop=1e13), prim_haloprop=1e13)
    assert np.isclose(
        np.sqrt(2 * np.pi * model.param_dict["sigma"] ** 2) ** (-1),
        clf,
        rtol=1e-6,
        atol=1.0e-6,
    )


def test_Cacciato09Cens6():
    """Check that the median primary galaxy property behave accordingly.
    """
    model = Cacciato09Cens(threshold=9.5)
    prim_galprop_1 = model.median_prim_galprop(prim_haloprop=1e14)
    model.param_dict["log_M_1"] += 0.1
    prim_galprop_2 = model.median_prim_galprop(prim_haloprop=1e14 * 10 ** 0.1)
    assert np.isclose(prim_galprop_1, prim_galprop_2, rtol=1e-6, atol=1.0e-2)

    model.param_dict["log_L_0"] += 0.1
    prim_galprop_3 = model.median_prim_galprop(prim_haloprop=1e14 * 10 ** 0.1)
    assert np.isclose(
        prim_galprop_2 * 10 ** 0.1, prim_galprop_3, rtol=1e-6, atol=1.0e-2
    )

    model.param_dict["gamma_1"] += 0.1
    prim_galprop_4 = model.median_prim_galprop(prim_haloprop=1e14 * 10 ** 0.1)
    assert prim_galprop_3 != prim_galprop_4

    model.param_dict["gamma_2"] += 0.1
    prim_galprop_5 = model.median_prim_galprop(prim_haloprop=1e14 * 10 ** 0.1)
    assert prim_galprop_4 != prim_galprop_5

    model.param_dict["sigma"] += 0.1
    prim_galprop_6 = model.median_prim_galprop(prim_haloprop=1e14 * 10 ** 0.1)
    assert np.isclose(prim_galprop_5, prim_galprop_6, rtol=1e-6, atol=1.0e-2)


def test_Cacciato09Cens7():
    """heck that no luminosity is below the threshold.
    """
    model = Cacciato09Cens(threshold=9.5)
    lum_mc = model.mc_prim_galprop(prim_haloprop=np.ones(int(1e5)) * 5e11, seed=1)
    assert np.all(lum_mc >= 10 ** model.threshold)

    # Check that luminosities follow the expected distribution.
    def cdf(lum):
        return np.array(
            [
                (
                    model.mean_occupation(prim_haloprop=5e11)
                    - model.mean_occupation(prim_haloprop=5e11, prim_galprop_min=l)
                )
                / model.mean_occupation(prim_haloprop=5e11)
                for l in lum
            ]
        )

    p_value = kstest(lum_mc, cdf)[1]

    assert p_value > 0.001


def test_Cacciato09Cens8():
    """Verify that the prim_galprop_max keyword argument behaves as expected.
    """
    model = Cacciato09Cens(threshold=10)
    marr = np.logspace(10, 13, 100)
    ncen1 = model.mean_occupation(prim_haloprop=marr)
    ncen2 = model.mean_occupation(prim_haloprop=marr, prim_galprop_max=10 ** 15)
    assert np.allclose(ncen1, ncen2, atol=0.01)

    ncen3 = model.mean_occupation(prim_haloprop=marr, prim_galprop_max=10 ** 10.3)
    assert not np.allclose(ncen1, ncen3, atol=0.01)


def test_Cacciato09Cens_median_prim_galprop_raises_exception():
    model = Cacciato09Cens(threshold=10.5)
    with pytest.raises(HalotoolsError) as err:
        __ = model.median_prim_galprop(x=7)
    substr = "You must pass either a ``table`` or ``prim_haloprop``"
    assert substr in err.value.args[0]


def test_Cacciato09Cens_clf_raises_exception():
    model = Cacciato09Cens(threshold=10.5)
    with pytest.raises(AssertionError) as err:
        __ = model.clf(
            prim_galprop=np.zeros(5) + 1e10, prim_haloprop=np.zeros(4) + 1e12
        )
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


@pytest.mark.installation_test
def test_Cacciato09Sats1():
    """
    Verify that the mean and Monte Carlo occupations are both reasonable and
    in agreement.
    """
    model = Cacciato09Sats(threshold=9.5)
    nsat_exp = model.mean_occupation(prim_haloprop=5e13)
    nsat_mc = model.mc_occupation(prim_haloprop=np.ones(int(1e5)) * 5e13, seed=1)
    np.testing.assert_allclose(np.average(nsat_mc), nsat_exp, rtol=1e-2, atol=1.0e-2)


def test_Cacciato09Sats2():
    """
    Check that the model behavior is altered in the expected way by changing
    param_dict values.
    """
    model = Cacciato09Sats(threshold=9.5)
    nsat_exp = model.mean_occupation(prim_haloprop=5e13)
    # Increasing b_0 by x should increase the occupation by exactly 10**x.
    model.param_dict["b_0"] += 0.1
    nsat_exp_new = model.mean_occupation(prim_haloprop=5e13)
    assert np.isclose(nsat_exp_new, nsat_exp * 10 ** 0.1, rtol=1e-2, atol=1.0e-2)

    # Increasing b_1 by x should increase the occupation by exactly
    # 10**(x * (log prim_haloprop - 12.0)).
    model.param_dict["b_0"] -= 0.1
    model.param_dict["b_1"] += 0.1
    nsat_exp_new = model.mean_occupation(prim_haloprop=5e13)
    assert np.isclose(
        nsat_exp_new,
        nsat_exp * 10 ** (0.1 * (np.log10(5e13) - 12.0)),
        rtol=1e-2,
        atol=1.0e-2,
    )

    # Increasing b_2 by x should increase the occupation by exactly
    # 10**(x * (log prim_haloprop - 12.0)**2).
    model.param_dict["b_1"] -= 0.1
    model.param_dict["b_2"] += 0.1
    nsat_exp_new = model.mean_occupation(prim_haloprop=5e13)
    assert np.isclose(
        nsat_exp_new,
        nsat_exp * 10 ** (0.1 * (np.log10(5e13) - 12.0) ** 2),
        rtol=1e-2,
        atol=1.0e-2,
    )


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
    lum_mc = model.mc_prim_galprop(prim_haloprop=np.ones(int(1e5)) * 5e13, seed=1)
    assert np.all(lum_mc >= 10 ** model.threshold)

    # Check that luminosities follow the expected distribution.
    def cdf(lum):
        return np.array(
            [
                (
                    model.mean_occupation(prim_haloprop=5e13)
                    - model.mean_occupation(prim_haloprop=5e13, prim_galprop_min=l)
                )
                / model.mean_occupation(prim_haloprop=5e13)
                for l in lum
            ]
        )

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
        __ = model.mean_occupation(
            table={"halo_m180b": np.array((1e11, 1e12))},
            prim_galprop_min=5,
            prim_galprop_max=4,
        )
    substr = "keyword must be bigger than 10^threshold"
    assert substr in err.value.args[0]


def test_Cacciato09Sats_clf_raises_exception():
    model = Cacciato09Sats(threshold=10.5)
    with pytest.raises(HalotoolsError) as err:
        __ = model.clf(
            prim_galprop=np.zeros(5) + 1e10, prim_haloprop=np.zeros(4) + 1e12
        )
    substr = "with multiple elements, they must have the same length"
    assert substr in err.value.args[0]


def test_Cacciato09Sats_mc_prim_galprop_raises_exception1():
    model = Cacciato09Sats(threshold=10.5)
    with pytest.raises(HalotoolsError) as err:
        __ = model.mc_prim_galprop(mass=8)
    substr = "You must pass either a ``table`` or ``prim_haloprop``"
    assert substr in err.value.args[0]


def test_Cacciato09_gap():
    cens = Cacciato09Cens(threshold=(0.4 * (19 + 4.76)))
    sats = Cacciato09Sats(threshold=(0.4 * (19 + 4.76)))

    for model in [cens, sats]:
        model.param_dict["log_L_0"] = 9.95
        model.param_dict["log_M_1"] = 11.27
        model.param_dict["sigma"] = 0.156
        model.param_dict["gamma_1"] = 2.94
        model.param_dict["gamma_2"] = 0.244
        model.param_dict["a_1"] = 2.0 - 1.17
        model.param_dict["a_2"] = 0
        model.param_dict["b_0"] = -1.42
        model.param_dict["b_1"] = 1.82
        model.param_dict["b_2"] = -0.30
        model.param_dict["log_M_2"] = 14.28

    lum_cen = cens.mc_prim_galprop(prim_haloprop=np.repeat(10 ** 14.5, 30000), seed=1)
    lum_sat = sats.mc_prim_galprop(
        prim_haloprop=np.repeat(10 ** 14.5, len(lum_cen) * 30), seed=1
    )

    gap = np.zeros(len(lum_cen))
    for i in range(len(gap)):
        lum_cen_i = lum_cen[i]
        lum_sat_i = lum_sat[i * 30 : (i + 1) * (30)]
        lum_sat_i = lum_sat_i[lum_sat_i < lum_cen_i]  # remove bright satellites
        gap[i] = 2.5 * np.log10(lum_cen_i / np.amax(lum_sat_i[:20]))

    gap_more = np.linspace(0, 1.98, 100)
    pdf_more = np.array(
        [
            0.749018,
            0.761080,
            0.772940,
            0.784557,
            0.795891,
            0.806902,
            0.817547,
            0.827784,
            0.837571,
            0.846866,
            0.855625,
            0.863806,
            0.871370,
            0.878274,
            0.884480,
            0.889951,
            0.894650,
            0.898544,
            0.901600,
            0.903791,
            0.905089,
            0.905471,
            0.904918,
            0.903413,
            0.900942,
            0.897497,
            0.893073,
            0.887667,
            0.881284,
            0.873930,
            0.865615,
            0.856356,
            0.846172,
            0.835086,
            0.823125,
            0.810321,
            0.796708,
            0.782324,
            0.767210,
            0.751410,
            0.734971,
            0.717942,
            0.700375,
            0.682321,
            0.663836,
            0.644974,
            0.625791,
            0.606345,
            0.586690,
            0.566884,
            0.546981,
            0.527037,
            0.507105,
            0.487235,
            0.467480,
            0.447885,
            0.428498,
            0.409362,
            0.390517,
            0.372001,
            0.353850,
            0.336096,
            0.318767,
            0.301891,
            0.285491,
            0.269585,
            0.254192,
            0.239325,
            0.224996,
            0.211213,
            0.197980,
            0.185302,
            0.173179,
            0.161609,
            0.150587,
            0.140109,
            0.130165,
            0.120747,
            0.111843,
            0.103441,
            0.095527,
            0.088087,
            0.081104,
            0.074563,
            0.068447,
            0.062738,
            0.057419,
            0.052471,
            0.047878,
            0.043622,
            0.039683,
            0.036046,
            0.032693,
            0.029607,
            0.026772,
            0.024172,
            0.021791,
            0.019615,
            0.017630,
            0.015821,
        ]
    )
    cdf_more = np.concatenate([[0], cumtrapz(pdf_more, x=gap_more)])
    cdf_more = cdf_more / cdf_more[-1]
    cdf_more = interp1d(gap_more, cdf_more)

    gap = gap[gap < gap_more[-1]]

    p_value = kstest(gap, cdf_more)[1]

    assert p_value > 0.001
