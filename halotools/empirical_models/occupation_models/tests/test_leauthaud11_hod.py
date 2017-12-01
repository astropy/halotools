""" Module providing unit-testing for the component models in
`halotools.empirical_models.occupation_components.leauthaud11_components` module"
"""
import numpy as np
import pytest

from .. import Leauthaud11Cens, Leauthaud11Sats

__all__ = ('test_Leauthaud11Cens', 'test_Leauthaud11Sats')


@pytest.mark.installation_test
def test_Leauthaud11Cens():
    """ Function to test
    `~halotools.empirical_models.Leauthaud11Cens`.
    """

    # Verify that the mean and Monte Carlo occupations are both reasonable and in agreement
    # for halos of mass 1e12
    model = Leauthaud11Cens()
    ncen1 = model.mean_occupation(prim_haloprop=5e11)
    mcocc = model.mc_occupation(prim_haloprop=np.ones(int(1e4))*5e11, seed=43)

    # Check that the model behavior is altered in the expected way by changing param_dict values

    # Increasing scatter picks up more galaxies in low-mass halos
    model.param_dict['scatter_model_param1'] *= 1.5
    ncen2 = model.mean_occupation(prim_haloprop=5e11)
    assert ncen2 > ncen1
#
    model.param_dict['smhm_m1_0'] *= 1.1
    ncen3 = model.mean_occupation(prim_haloprop=5e11)
    assert ncen3 < ncen2
#
    model.param_dict['smhm_m1_a'] *= 1.1
    ncen4 = model.mean_occupation(prim_haloprop=5e11)
    assert ncen4 == ncen3

    # Check that increasing stellar mass thresholds decreases the mean occupation
    model2 = Leauthaud11Cens(threshold=10.75)
    ncen5 = model2.mean_occupation(prim_haloprop=5e11)
    model3 = Leauthaud11Cens(threshold=11.25)
    ncen6 = model3.mean_occupation(prim_haloprop=5e11)
    assert ncen6 < ncen5 < ncen1

    __ = model2.get_published_parameters()


@pytest.mark.installation_test
def test_Leauthaud11Sats():
    """ Function to test
    `~halotools.empirical_models.Leauthaud11Cens`.
    """

    # Verify that the mean and Monte Carlo occupations are both reasonable and in agreement
    # for halos of mass 1e12
    model = Leauthaud11Sats()
    nsat1 = model.mean_occupation(prim_haloprop=5.e12)
    mcocc = model.mc_occupation(prim_haloprop=np.ones(int(1e4))*5e12, seed=43)

    # Check that the model behavior is altered in the expected way by changing param_dict values
    model.param_dict['alphasat'] *= 1.1
    nsat2 = model.mean_occupation(prim_haloprop=5.e12)
    assert nsat2 < nsat1

    # Msat := Bsat(mhalo_thresh/1e12)**betasat
    # since mhalo_thresh(logM*=10.5) > 1e12, increasing betasat should increase Msat
    msat1 = np.copy(model._msat)
    model.param_dict['betasat'] *= 1.1
    model._update_satellite_params()
    msat2 = np.copy(model._msat)
    assert msat2 > msat1

    # Msat := Bsat(mhalo_thresh/1e12)**betasat
    # Increasing Bsat should increase Msat regardless of mass
    model._update_satellite_params()
    msat1 = np.copy(model._msat)
    model.param_dict['bsat'] *= 1.1
    model._update_satellite_params()
    msat2 = np.copy(model._msat)
    assert msat2 > msat1
#
    # Mcut := Bcut(mhalo_thresh/1e12)**betacut
    # Default value of betacut is -0.13 for default threshold of 10.5
    # since mhalo_thresh(logM*=10.5) > 1e12, increasing betacut should increase Mcut
    model.param_dict['betacut'] = -0.13
    model._update_satellite_params()
    mcut1 = np.copy(model._mcut)
    model.param_dict['betacut'] = -0.1
    model._update_satellite_params()
    mcut2 = np.copy(model._mcut)
    assert mcut2 > mcut1
# #
    # Mcut := Bcut(mhalo_thresh/1e12)**betacut
    # Increasing Bcut should increase Mcut regardless of mass
    model._update_satellite_params()
    mcut1 = np.copy(model._mcut)
    model.param_dict['bcut'] *= 1.1
    model._update_satellite_params()
    mcut2 = np.copy(model._mcut)
    assert mcut2 > mcut1
# #
#
    # Check that modulate_with_cenocc strictly decreases the mean occupations
    model2a = Leauthaud11Sats(modulate_with_cenocc=False)
    model2b = Leauthaud11Sats(modulate_with_cenocc=True)
    nsat2a = model2a.mean_occupation(prim_haloprop=5.e12)
    nsat2b = model2b.mean_occupation(prim_haloprop=5.e12)
    assert model2b.central_occupation_model.mean_occupation(prim_haloprop=5.e12) < 1
    assert nsat2b < nsat2a

    # Check that increasing stellar mass thresholds decreases the mean occupation
    model10 = Leauthaud11Sats(threshold=10)
    model105 = Leauthaud11Sats(threshold=10.5)
    model11 = Leauthaud11Sats(threshold=11)
    nsat10 = model10.mean_occupation(prim_haloprop=1e13)
    nsat105 = model105.mean_occupation(prim_haloprop=1e13)
    nsat11 = model11.mean_occupation(prim_haloprop=1e13)
    assert nsat10 > nsat105 > nsat11

    # Check that increasing stellar mass thresholds decreases the central occupations
    ncen10 = model10.central_occupation_model.mean_occupation(prim_haloprop=5e12)
    ncen105 = model105.central_occupation_model.mean_occupation(prim_haloprop=5e12)
    ncen11 = model11.central_occupation_model.mean_occupation(prim_haloprop=5e12)
    assert ncen10 > ncen105 > ncen11
