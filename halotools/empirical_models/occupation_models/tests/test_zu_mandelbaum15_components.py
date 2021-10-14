""" Module providing unit-testing for the component models in
`halotools.empirical_models.occupation_components.ZuMandelbaum15_components` module"
"""
import numpy as np

from .. import ZuMandelbaum15Cens, ZuMandelbaum15Sats

__all__ = ("test_ZuMandelbaum15Cens1", "test_ZuMandelbaum15Cens2")


def test_ZuMandelbaum15Cens1():
    """Verify that the mean and Monte Carlo occupations
    are both reasonable and in agreement for a few halo masses
    """
    model = ZuMandelbaum15Cens()

    testmass = 1e12
    npts = int(1e4)
    mass_array = np.zeros(npts) + testmass
    ncen = model.mean_occupation(prim_haloprop=testmass)
    mcocc = model.mc_occupation(prim_haloprop=mass_array, seed=42)
    assert np.allclose(mcocc.mean(), ncen, rtol=0.05)

    testmass = 5e12
    mass_array = np.zeros(npts) + testmass
    ncen = model.mean_occupation(prim_haloprop=testmass)
    mcocc = model.mc_occupation(prim_haloprop=mass_array, seed=42)
    assert np.allclose(mcocc.mean(), ncen, rtol=0.05)

    testmass = 1e13
    mass_array = np.zeros(npts) + testmass
    ncen = model.mean_occupation(prim_haloprop=testmass)
    mcocc = model.mc_occupation(prim_haloprop=mass_array, seed=42)
    assert np.allclose(mcocc.mean(), ncen, rtol=0.05)


def test_ZuMandelbaum15Cens2():
    """Check that the model behavior is altered in the expected way
    by changing smhm_sigma
    """
    model = ZuMandelbaum15Cens(threshold=11)

    testmass = 1e12
    ncen1 = model.mean_occupation(prim_haloprop=testmass)

    # Increasing scatter picks up more galaxies in low-mass halos
    model.param_dict["smhm_sigma"] *= 1.5
    ncen2 = model.mean_occupation(prim_haloprop=testmass)
    assert ncen2 > ncen1


def test_ZuMandelbaum15Cens3():
    """Check that the model behavior is altered in the expected way
    by changing smhm_m1
    """
    model = ZuMandelbaum15Cens(threshold=11)

    testmass = 1e12
    ncen1 = model.mean_occupation(prim_haloprop=testmass)

    model.param_dict["smhm_m1"] *= 1.1
    ncen3 = model.mean_occupation(prim_haloprop=testmass)
    assert ncen3 < ncen1


def test_ZuMandelbaum15Cens4():
    """Check that increasing stellar mass thresholds decreases the mean occupation"""
    testmass = 1e12

    model1 = ZuMandelbaum15Cens(threshold=11)
    ncen1 = model1.mean_occupation(prim_haloprop=testmass)

    model2 = ZuMandelbaum15Cens(threshold=10.75)
    ncen2 = model2.mean_occupation(prim_haloprop=testmass)
    assert ncen2 > ncen1

    model3 = ZuMandelbaum15Cens(threshold=11.25)
    ncen3 = model3.mean_occupation(prim_haloprop=testmass)
    assert ncen3 < ncen1


def test_ZuMandelbaum15Sats1():
    """Verify that the mean and Monte Carlo occupations
    are both reasonable and in agreement for a few halo masses
    """
    model = ZuMandelbaum15Sats()

    npts = int(1e5)
    testmass = 5e12
    mass_array = np.zeros(npts) + testmass
    nsat = model.mean_occupation(prim_haloprop=testmass)
    mcocc = model.mc_occupation(prim_haloprop=mass_array, seed=42)
    assert np.allclose(mcocc.mean(), nsat, rtol=0.05)

    testmass = 1e13
    mass_array = np.zeros(npts) + testmass
    nsat = model.mean_occupation(prim_haloprop=testmass)
    mcocc = model.mc_occupation(prim_haloprop=mass_array, seed=42)
    assert np.allclose(mcocc.mean(), nsat, rtol=0.05)

    testmass = 5e13
    mass_array = np.zeros(npts) + testmass
    nsat = model.mean_occupation(prim_haloprop=testmass)
    mcocc = model.mc_occupation(prim_haloprop=mass_array, seed=42)
    assert np.allclose(mcocc.mean(), nsat, rtol=0.05)

    testmass = 1e14
    mass_array = np.zeros(npts) + testmass
    nsat = model.mean_occupation(prim_haloprop=testmass)
    mcocc = model.mc_occupation(prim_haloprop=mass_array, seed=42)
    assert np.allclose(mcocc.mean(), nsat, rtol=0.05)


def test_ZuMandelbaum15Sats2():
    """Check that the model behavior is altered in the expected way
    by changing alphasat values
    """
    model = ZuMandelbaum15Sats()

    testmass = 5e13
    nsat1 = model.mean_occupation(prim_haloprop=testmass)
    model.param_dict["alphasat"] *= 1.1
    nsat2 = model.mean_occupation(prim_haloprop=testmass)
    assert nsat2 > nsat1


def test_ZuMandelbaum15Sats3():
    """Check that the model behavior is altered in the expected way
    by changing betasat values
    """
    model = ZuMandelbaum15Sats()

    # Msat := Bsat(mhalo_thresh/1e12)**betasat
    # since mhalo_thresh(logM*=10.5) > 1e12, increasing betasat should increase Msat
    msat1 = np.copy(model._msat)
    model.param_dict["betasat"] *= 1.1
    model._update_satellite_params()
    msat2 = np.copy(model._msat)
    assert msat2 > msat1


def test_ZuMandelbaum15Sats4():
    """Check that the model behavior is altered in the expected way
    by changing bsat values
    """
    model = ZuMandelbaum15Sats()

    # Msat := Bsat(mhalo_thresh/1e12)**betasat
    # Increasing Bsat should increase Msat
    model._update_satellite_params()
    msat1 = np.copy(model._msat)
    model.param_dict["bsat"] *= 1.1
    model._update_satellite_params()
    msat2 = np.copy(model._msat)
    assert msat2 > msat1


def test_ZuMandelbaum15Sats5():
    """Check that the model behavior is altered in the expected way
    by changing betacut values
    """
    model = ZuMandelbaum15Sats(threshold=11)

    # Mcut := Bcut(mhalo_thresh/1e12)**betacut
    # since mean_halo_mass(10**11) > knee_mass=1e12,
    # increasing betacut should increase Mcut
    model.param_dict["betacut"] = 0.41
    model._update_satellite_params()
    mcut1 = np.copy(model._mcut)
    model.param_dict["betacut"] = 0.5
    model._update_satellite_params()
    mcut2 = np.copy(model._mcut)
    assert mcut2 > mcut1


def test_ZuMandelbaum15Sats6():
    """Check that the model behavior is altered in the expected way
    by changing bcut values
    """
    model = ZuMandelbaum15Sats(threshold=10)

    # Mcut := Bcut(mhalo_thresh/1e12)**betacut
    # Increasing Bcut should increase Mcut
    model._update_satellite_params()
    mcut1 = np.copy(model._mcut)
    model.param_dict["bcut"] *= 1.1
    model._update_satellite_params()
    mcut2 = np.copy(model._mcut)
    assert mcut2 > mcut1


def test_ZuMandelbaum15Sats7():
    """Check that increasing stellar mass thresholds decreases the mean occupation"""
    model10 = ZuMandelbaum15Sats(threshold=10)
    model105 = ZuMandelbaum15Sats(threshold=10.5)
    model11 = ZuMandelbaum15Sats(threshold=11)

    nsat10 = model10.mean_occupation(prim_haloprop=1e13)
    nsat105 = model105.mean_occupation(prim_haloprop=1e13)
    nsat11 = model11.mean_occupation(prim_haloprop=1e13)
    assert nsat10 > nsat105
    assert nsat105 > nsat11

    ncen10 = model10.central_occupation_model.mean_occupation(prim_haloprop=5e12)
    ncen105 = model105.central_occupation_model.mean_occupation(prim_haloprop=5e12)
    ncen11 = model11.central_occupation_model.mean_occupation(prim_haloprop=5e12)
    assert ncen10 > ncen105
    assert ncen105 > ncen11


def test_ZuMandelbaum15Sats8():
    """Check that changes to centrals parameters propagate through to satellites"""
    model = ZuMandelbaum15Sats(threshold=10)
    model.param_dict["smhm_sigma"] = 0.5

    # Verify that we have identified a mass range with mean_ncen <~ 1
    # In such a mass range, decreasing scatter increases <Ncen>
    testmass = 1.5e12
    ncen1 = model.central_occupation_model.mean_occupation(prim_haloprop=testmass)
    assert ncen1 < 0.99
    assert ncen1 > 0.95

    # Verify that we have non-zero satellites so that this test is non-trivial
    nsat1 = model.mean_occupation(prim_haloprop=testmass)
    assert nsat1 > 0.01

    # Modifying scatter should only impact <Nsat> via <Ncen>
    model.param_dict["smhm_sigma"] = 0.25
    nsat2 = model.mean_occupation(prim_haloprop=testmass)

    assert nsat2 > nsat1
