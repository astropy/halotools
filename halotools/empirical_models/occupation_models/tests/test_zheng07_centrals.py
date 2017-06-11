"""
"""
import numpy as np
from astropy.table import Table

import pytest
import warnings

from ....empirical_models import OccupationComponent, Zheng07Cens
from ....custom_exceptions import HalotoolsError

__all__ = ('test_default_model', )


supported_thresholds = np.arange(-22, -17.5, 0.5)

_default_model = Zheng07Cens()
lowmass = (10.**_default_model.param_dict['logMmin'])/1.1
highmass = (10.**_default_model.param_dict['logMmin'])*1.1


def enforce_required_attributes(model):
    assert isinstance(model, OccupationComponent)
    assert model.gal_type == 'centrals'

    assert hasattr(model, 'prim_haloprop_key')

    assert model._upper_occupation_bound == 1


def enforce_mean_occupation_behavior(model):

    assert hasattr(model, 'mean_occupation')

    mvir_array = np.logspace(10, 16, 10)
    mean_occ = model.mean_occupation(prim_haloprop=mvir_array)

    # Check that the range is in [0,1]
    assert np.all(mean_occ <= 1)
    assert np.all(mean_occ >= 0)

    # The mean occupation should be monotonically increasing
    assert np.all(np.diff(mean_occ) >= 0)


def enforce_mc_occupation_behavior(model):

    # Check the Monte Carlo realization method
    assert hasattr(model, 'mc_occupation')

    # First check that the mean occuation is ~0.5 when model is evaulated at Mmin
    mvir_midpoint = 10.**model.param_dict['logMmin']
    Npts = int(1e3)
    masses = np.ones(Npts)*mvir_midpoint
    mc_occ = model.mc_occupation(prim_haloprop=masses, seed=43)
    assert set(mc_occ).issubset([0, 1])
    expected_result = 0.48599999
    np.testing.assert_allclose(mc_occ.mean(), expected_result, rtol=1e-5, atol=1.e-5)

    # Now check that the model is ~ 1.0 when evaluated for a cluster
    masses = np.ones(Npts)*5.e15
    mc_occ = model.mc_occupation(prim_haloprop=masses, seed=43)
    assert set(mc_occ).issubset([0, 1])
    expected_result = 1.0
    np.testing.assert_allclose(mc_occ.mean(), expected_result, rtol=1e-2, atol=1.e-2)

    # Now check that the model is ~ 0.0 when evaluated for a tiny halo
    masses = np.ones(Npts)*1.e10
    mc_occ = model.mc_occupation(prim_haloprop=masses, seed=43)
    assert set(mc_occ).issubset([0, 1])
    expected_result = 0.0
    np.testing.assert_allclose(mc_occ.mean(), expected_result, rtol=1e-2, atol=1.e-2)


def enforce_correct_argument_inference_from_halo_catalog(model):
    mvir_array = np.logspace(10, 16, 10)
    key = model.prim_haloprop_key
    mvir_dict = {key: mvir_array}
    halo_catalog = Table(mvir_dict)
    # First test mean occupations
    meanocc_from_array = model.mean_occupation(prim_haloprop=mvir_array)
    meanocc_from_halos = model.mean_occupation(table=halo_catalog)
    assert np.all(meanocc_from_array == meanocc_from_halos)
    # Now test Monte Carlo occupations
    mcocc_from_array = model.mc_occupation(prim_haloprop=mvir_array, seed=43)
    mcocc_from_halos = model.mc_occupation(table=halo_catalog, seed=43)
    assert np.all(mcocc_from_array == mcocc_from_halos)


def test_default_model():
    default_model = Zheng07Cens()

    enforce_required_attributes(default_model)
    enforce_mean_occupation_behavior(default_model)
    enforce_mc_occupation_behavior(default_model)
    enforce_correct_argument_inference_from_halo_catalog(default_model)


def test_alternate_threshold_models():

    for threshold in supported_thresholds:
        thresh_model = Zheng07Cens(threshold=threshold)
        enforce_required_attributes(thresh_model)
        enforce_mean_occupation_behavior(thresh_model)
        enforce_mc_occupation_behavior(thresh_model)


def test_mean_ncen_scaling1():
    """Verify the value of <Ncen> scales reasonably with the parameters
    """
    default_model = Zheng07Cens()
    model2 = Zheng07Cens()
    model2.param_dict['logMmin'] += np.log10(2.)
    model3 = Zheng07Cens()
    model3.param_dict['sigma_logM'] *= 2.0

    defocc_lowmass = default_model.mean_occupation(prim_haloprop=lowmass)
    occ2_lowmass = model2.mean_occupation(prim_haloprop=lowmass)
    occ3_lowmass = model3.mean_occupation(prim_haloprop=lowmass)
    assert occ3_lowmass > defocc_lowmass
    assert defocc_lowmass > occ2_lowmass


def test_mean_ncen_scaling2():
    default_model = Zheng07Cens()
    model2 = Zheng07Cens()
    model2.param_dict['logMmin'] += np.log10(2.)
    model3 = Zheng07Cens()
    model3.param_dict['sigma_logM'] *= 2.0

    defocc_highmass = default_model.mean_occupation(prim_haloprop=highmass)
    occ2_highmass = model2.mean_occupation(prim_haloprop=highmass)
    occ3_highmass = model3.mean_occupation(prim_haloprop=highmass)
    assert defocc_highmass > occ3_highmass
    assert occ3_highmass > occ2_highmass


def test_param_dict_propagation1():
    """
    Verify that directly changing model parameters
    without a new instantiation also behaves properly
    """
    default_model = Zheng07Cens()

    defocc_lowmass = default_model.mean_occupation(prim_haloprop=lowmass)

    alt_model = Zheng07Cens()
    alt_model.param_dict['sigma_logM'] *= 2.
    updated_defocc_lowmass = alt_model.mean_occupation(prim_haloprop=lowmass)

    assert updated_defocc_lowmass > defocc_lowmass


def test_param_dict_propagation2():
    """
    Verify that directly changing model parameters
    without a new instantiation also behaves properly
    """
    default_model = Zheng07Cens()

    defocc_highmass = default_model.mean_occupation(prim_haloprop=highmass)

    alt_model = Zheng07Cens()
    alt_model.param_dict['sigma_logM'] *= 2.
    updated_defocc_highmass = alt_model.mean_occupation(prim_haloprop=highmass)

    assert updated_defocc_highmass < defocc_highmass


def test_param_dict_propagation3():
    """
    Verify that directly changing model parameters
    without a new instantiation also behaves properly
    """
    default_model = Zheng07Cens()

    defocc_lowmass = default_model.mean_occupation(prim_haloprop=lowmass)

    alt_model = Zheng07Cens()
    alt_model.param_dict['sigma_logM'] *= 2.
    updated_defocc_lowmass = alt_model.mean_occupation(prim_haloprop=lowmass)

    assert updated_defocc_lowmass > defocc_lowmass


def test_param_dict_propagation4():
    """
    Verify that directly changing model parameters
    without a new instantiation also behaves properly
    """
    default_model = Zheng07Cens()

    defocc_highmass = default_model.mean_occupation(prim_haloprop=highmass)

    alt_model = Zheng07Cens()
    alt_model.param_dict['sigma_logM'] *= 2.
    updated_defocc_highmass = alt_model.mean_occupation(prim_haloprop=highmass)

    assert updated_defocc_highmass < defocc_highmass


def test_raises_correct_exception():
    default_model = Zheng07Cens()
    with pytest.raises(HalotoolsError) as err:
        _ = default_model.mean_occupation(x=4)
    substr = "You must pass either a ``table`` or ``prim_haloprop`` argument"
    assert substr in err.value.args[0]


def test_get_published_parameters1():
    default_model = Zheng07Cens()
    d1 = default_model.get_published_parameters(default_model.threshold)


def test_get_published_parameters2():
    default_model = Zheng07Cens()
    with pytest.raises(KeyError) as err:
        d2 = default_model.get_published_parameters(default_model.threshold,
            publication='Parejko13')
    substr = "For Zheng07Cens, only supported best-fit models are currently Zheng et al. 2007"
    assert substr == err.value.args[0]


def test_get_published_parameters3():
    default_model = Zheng07Cens()
    with warnings.catch_warnings(record=True) as w:

        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        d1 = default_model.get_published_parameters(-11.3)

        assert "does not match any of the Table 1 values" in str(w[-1].message)

        d2 = default_model.get_published_parameters(default_model.threshold)

        assert d1 == d2
