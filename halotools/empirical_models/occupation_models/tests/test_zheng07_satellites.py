"""
"""
import numpy as np
from copy import deepcopy

import pytest
import warnings

from ....empirical_models import OccupationComponent, Zheng07Sats, Zheng07Cens

from ....custom_exceptions import HalotoolsError

__all__ = ("test_alternate_threshold_models",)


supported_thresholds = np.arange(-22, -17.5, 0.5)


def test_default_model():

    # First test the model with all default settings
    default_model = Zheng07Sats()

    enforce_required_attributes(default_model)
    enforce_mean_occupation_behavior(default_model)
    enforce_mc_occupation_behavior(default_model)


def test_alternate_threshold_models():

    for threshold in supported_thresholds:
        thresh_model = Zheng07Sats(threshold=threshold)
        enforce_required_attributes(thresh_model)
        enforce_mean_occupation_behavior(thresh_model)


def enforce_required_attributes(model):
    assert isinstance(model, OccupationComponent)
    assert model.gal_type == "satellites"

    assert hasattr(model, "prim_haloprop_key")

    assert model._upper_occupation_bound == float("inf")


def enforce_mean_occupation_behavior(model):

    assert hasattr(model, "mean_occupation")

    mvir_array = np.logspace(10, 16, 10)
    mean_occ = model.mean_occupation(prim_haloprop=mvir_array)

    # Check non-negative
    assert np.all(mean_occ >= 0)
    # The mean occupation should be monotonically increasing
    assert np.all(np.diff(mean_occ) >= 0)


def enforce_mc_occupation_behavior(model):

    # Check the Monte Carlo realization method
    assert hasattr(model, "mc_occupation")

    model.param_dict["alpha"] = 1
    model.param_dict["logM0"] = 11.25
    model.param_dict["logM1"] = model.param_dict["logM0"] + np.log10(20.0)

    Npts = int(1e3)
    masses = np.ones(Npts) * 10.0 ** model.param_dict["logM1"]
    mc_occ = model.mc_occupation(prim_haloprop=masses, seed=43)
    # We chose a specific seed that has been pre-tested,
    # so we should always get the same result
    expected_result = 1.0
    np.testing.assert_allclose(mc_occ.mean(), expected_result, rtol=1e-2, atol=1.0e-2)


def test_ncen_inheritance_behavior1():
    satmodel_nocens = Zheng07Sats()
    cenmodel = Zheng07Cens()
    satmodel_cens = Zheng07Sats(modulate_with_cenocc=True)

    Npts = 100
    masses = np.logspace(10, 15, Npts)
    mean_occ_satmodel_nocens = satmodel_nocens.mean_occupation(prim_haloprop=masses)
    mean_occ_satmodel_cens = satmodel_cens.mean_occupation(prim_haloprop=masses)
    assert np.all(mean_occ_satmodel_cens <= mean_occ_satmodel_nocens)

    diff = mean_occ_satmodel_cens - mean_occ_satmodel_nocens
    assert diff.sum() < 0

    mean_occ_cens = satmodel_cens.central_occupation_model.mean_occupation(
        prim_haloprop=masses
    )
    assert np.all(mean_occ_satmodel_cens == mean_occ_satmodel_nocens * mean_occ_cens)


def test_ncen_inheritance_behavior2():
    """Verify that the ``modulate_with_cenocc`` and ``cenocc_model``
    keyword arguments behave as expected, including propagation of
    param_dict values.
    """
    from .. import OccupationComponent

    class MyCenModel(OccupationComponent):
        def __init__(self, threshold):
            OccupationComponent.__init__(
                self,
                gal_type="centrals",
                threshold=threshold,
                upper_occupation_bound=1.0,
            )
            self.param_dict["new_cen_param"] = 0.5

        def mean_occupation(self, **kwargs):
            try:
                halo_table = kwargs["table"]
                num_halos = len(halo_table)
            except KeyError:
                mass_array = kwargs["prim_haloprop"]
                num_halos = len(mass_array)

            result = np.zeros(num_halos) + self.param_dict["new_cen_param"]
            return result

    with warnings.catch_warnings(record=True) as w:
        my_cen_model = MyCenModel(threshold=-20)
        my_sat_model1 = Zheng07Sats(
            threshold=-20, modulate_with_cenocc=True, cenocc_model=my_cen_model
        )
        my_sat_model2 = Zheng07Sats(threshold=-20, modulate_with_cenocc=True)
        my_sat_model3 = Zheng07Sats(
            threshold=-20, modulate_with_cenocc=False, cenocc_model=my_cen_model
        )
        assert "modulate_with_cenocc" in str(w[-1].message)

    mass_array = np.logspace(11, 15, 10)
    result1a = my_sat_model1.mean_occupation(prim_haloprop=mass_array)
    result2 = my_sat_model2.mean_occupation(prim_haloprop=mass_array)
    result3 = my_sat_model3.mean_occupation(prim_haloprop=mass_array)

    assert not np.allclose(result1a, result2, rtol=0.001)
    assert not np.allclose(result1a, result3, rtol=0.001)
    assert not np.allclose(result2, result3, rtol=0.001)

    my_sat_model1.param_dict["new_cen_param"] = 1.0
    result1b = my_sat_model1.mean_occupation(prim_haloprop=mass_array)
    assert not np.allclose(result1a, result1b, rtol=0.001)
    assert np.allclose(result1b, result3, rtol=0.001)


def test_alpha_scaling1_mean_occupation():
    default_model = Zheng07Sats()
    model2 = Zheng07Sats()
    model2.param_dict["alpha"] *= 1.25

    logmass = model2.param_dict["logM1"] + np.log10(5)
    mass = 10.0**logmass
    assert model2.mean_occupation(prim_haloprop=mass) > default_model.mean_occupation(
        prim_haloprop=mass
    )


def test_alpha_scaling2_mc_occupation():
    default_model = Zheng07Sats()
    model2 = Zheng07Sats()
    model2.param_dict["alpha"] *= 1.25

    logmass = model2.param_dict["logM1"] + np.log10(5)
    mass = 10.0**logmass
    Npts = 1000
    masses = np.ones(Npts) * mass

    assert (
        model2.mc_occupation(prim_haloprop=masses, seed=43).mean()
        > default_model.mc_occupation(prim_haloprop=masses, seed=43).mean()
    )


def test_alpha_propagation():
    default_model = Zheng07Sats()
    model2 = Zheng07Sats()
    model2.param_dict["alpha"] *= 1.25

    logmass = model2.param_dict["logM1"] + np.log10(5)
    mass = 10.0**logmass
    Npts = 1000
    masses = np.ones(Npts) * mass

    alt_default_model = deepcopy(default_model)

    alt_default_model.param_dict["alpha"] = model2.param_dict["alpha"]

    assert (
        model2.mc_occupation(prim_haloprop=masses, seed=43).mean()
        == alt_default_model.mc_occupation(prim_haloprop=masses, seed=43).mean()
    )


def test_logM0_scaling1_mean_occupation():
    default_model = Zheng07Sats()
    model3 = Zheng07Sats()
    model3.param_dict["logM0"] += np.log10(2)

    # At very low mass, both models should have zero satellites
    lowmass = 1e10
    assert model3.mean_occupation(
        prim_haloprop=lowmass
    ) == default_model.mean_occupation(prim_haloprop=lowmass)


def test_logM0_scaling2_mean_occupation():
    default_model = Zheng07Sats()
    model3 = Zheng07Sats()
    model3.param_dict["logM0"] += np.log10(2)

    # At intermediate masses, there should be fewer satellites for larger M0
    midmass = 1e12
    assert model3.mean_occupation(
        prim_haloprop=midmass
    ) < default_model.mean_occupation(prim_haloprop=midmass)


def test_logM0_scaling3_mean_occupation():
    default_model = Zheng07Sats()
    model3 = Zheng07Sats()
    model3.param_dict["logM0"] += np.log10(2)

    # At high masses, the difference should be negligible
    highmass = 1e15
    np.testing.assert_allclose(
        model3.mean_occupation(prim_haloprop=highmass),
        default_model.mean_occupation(prim_haloprop=highmass),
        rtol=1e-3,
        atol=1.0e-3,
    )


def test_logM1_scaling1_mean_occupation():
    default_model = Zheng07Sats()
    model2 = Zheng07Sats()
    model2.param_dict["alpha"] *= 1.25
    model3 = Zheng07Sats()
    model3.param_dict["logM0"] += np.log10(2)
    model4 = Zheng07Sats()
    model4.param_dict["logM1"] += np.log10(2)

    # At very low mass, both models should have zero satellites
    lowmass = 1e10
    assert model4.mean_occupation(
        prim_haloprop=lowmass
    ) == default_model.mean_occupation(prim_haloprop=lowmass)


def test_logM1_scaling2_mean_occupation():
    default_model = Zheng07Sats()
    model2 = Zheng07Sats()
    model2.param_dict["alpha"] *= 1.25
    model3 = Zheng07Sats()
    model3.param_dict["logM0"] += np.log10(2)
    model4 = Zheng07Sats()
    model4.param_dict["logM1"] += np.log10(2)

    # At intermediate masses, there should be fewer satellites for larger M1
    midmass = 1e12
    fracdiff_midmass = (
        model4.mean_occupation(prim_haloprop=midmass)
        - default_model.mean_occupation(prim_haloprop=midmass)
    ) / default_model.mean_occupation(prim_haloprop=midmass)
    assert fracdiff_midmass < 0

    highmass = 1e14
    fracdiff_highmass = (
        model4.mean_occupation(prim_haloprop=highmass)
        - default_model.mean_occupation(prim_haloprop=highmass)
    ) / default_model.mean_occupation(prim_haloprop=highmass)
    assert fracdiff_highmass < 0

    # The fractional change due to alterations of logM1 should be identical at all mass
    assert fracdiff_highmass == fracdiff_midmass


def test_raises_correct_exception():
    default_model = Zheng07Sats()
    with pytest.raises(HalotoolsError) as err:
        _ = default_model.mean_occupation(x=4)
    substr = "You must pass either a ``table`` or ``prim_haloprop`` argument"
    assert substr in err.value.args[0]


def test_occupation_component_requirement():
    """Verify that the optional ``cenocc_model`` input is correctly
    enforced to be an instance of the OccupationComponent class.
    """
    with pytest.raises(HalotoolsError) as err:
        __ = Zheng07Sats(modulate_with_cenocc=True, cenocc_model=7)
    substr = "``OccupationComponent`` or one of its sub-classes"
    assert substr in err.value.args[0]


def test_get_published_parameters1():
    default_model = Zheng07Sats()
    d1 = default_model.get_published_parameters(default_model.threshold)


def test_get_published_parameters2():
    default_model = Zheng07Sats()
    with pytest.raises(KeyError) as err:
        d2 = default_model.get_published_parameters(
            default_model.threshold, publication="Parejko13"
        )
    substr = "For Zheng07Sats, only supported best-fit models are currently Zheng et al. 2007"
    assert substr == err.value.args[0]


def test_get_published_parameters3():
    default_model = Zheng07Sats()
    with warnings.catch_warnings(record=True) as w:

        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        d1 = default_model.get_published_parameters(-11.3)

        assert "does not match any of the Table 1 values" in str(w[-1].message)

        d2 = default_model.get_published_parameters(default_model.threshold)

        assert d1 == d2


def test_weighted_nearest_integer():
    model1 = Zheng07Sats()
    model2 = Zheng07Sats(second_moment="weighted_nearest_integer")

    mtest = np.zeros(int(1e5)) + 1e13
    nsat1 = model1.mc_occupation(prim_haloprop=mtest)
    nsat2 = model2.mc_occupation(prim_haloprop=mtest)

    assert np.allclose(nsat1.mean(), nsat2.mean(), rtol=0.05)
    assert np.std(nsat1) > np.std(nsat2) + 0.1
