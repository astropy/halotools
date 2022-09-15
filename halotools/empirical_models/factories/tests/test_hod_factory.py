"""
"""
import warnings
from copy import copy

from ...factories import PrebuiltHodModelFactory
from ... import factories
from ... import occupation_models as occm

__all__ = ["test_Zheng07_composite"]


def test_Zheng07_composite():
    """Method to test the basic behavior of
    `~halotools.empirical_models.Zheng07`,
    a specific pre-loaded model of
    `~halotools.empirical_models.PrebuiltHodModelFactory`.

    The suite includes the following tests:

        * Changes to ``self.param_dict`` properly propagate through to occupation component models.

        * Default behavior is recovered after calling the `~halotools.empirical_models.HodModelFactory.restore_init_param_dict` method.
    """
    model = PrebuiltHodModelFactory("zheng07", threshold=-18)

    # Verify that changes param_dict properly propagate
    testmass1 = 5.0e11
    cenocc_orig = model.mean_occupation_centrals(prim_haloprop=testmass1)
    orig_logMmin_centrals = model.param_dict["logMmin"]
    model.param_dict["logMmin"] = 11.5
    cenocc_new = model.mean_occupation_centrals(prim_haloprop=testmass1)
    assert cenocc_new < cenocc_orig

    testmass2 = 5.0e12
    satocc_orig = model.mean_occupation_satellites(prim_haloprop=testmass2)
    model.param_dict["logM0"] = 11.4
    satocc_new = model.mean_occupation_satellites(prim_haloprop=testmass2)
    assert satocc_new < satocc_orig

    # Test that we can recover our initial behavior
    model.restore_init_param_dict()
    assert model.param_dict["logMmin"] == orig_logMmin_centrals
    cenocc_restored = model.mean_occupation_centrals(prim_haloprop=testmass1)
    assert cenocc_restored == cenocc_orig
    satocc_restored = model.mean_occupation_satellites(prim_haloprop=testmass2)
    assert satocc_restored == satocc_orig


def test_alt_Zheng07_composites():

    # First build two models that are identical except for the satellite occupations
    default_model = PrebuiltHodModelFactory("zheng07")
    default_model_dictionary = default_model._input_model_dictionary
    default_satocc_component = default_model_dictionary["satellites_occupation"]
    default_cenocc_component = default_model_dictionary["centrals_occupation"]

    cenmod_satocc_component = occm.Zheng07Sats(
        threshold=default_satocc_component.threshold,
        modulate_with_cenocc=True,
        gal_type_centrals="centrals",
    )

    cenmod_model_dictionary = copy(default_model_dictionary)
    cenmod_model_dictionary["satellites_occupation"] = cenmod_satocc_component
    cenmod_model_dictionary["centrals_occupation"] = default_cenocc_component
    with warnings.catch_warnings(record=True) as w:
        cenmod_model = factories.HodModelFactory(**cenmod_model_dictionary)

    # Now we test whether changes to the param_dict keys of the composite model
    # that pertain to the centrals properly propagate through to the behavior
    # of the satellites, only for cases where satellite occupations are modulated
    # by central occupations
    assert set(cenmod_model.param_dict) == set(default_model.param_dict)

    nsat1 = default_model.mean_occupation_satellites(prim_haloprop=2.0e12)
    nsat2 = cenmod_model.mean_occupation_satellites(prim_haloprop=2.0e12)
    assert nsat2 < nsat1

    cenmod_model.param_dict["logMmin"] *= 1.1
    nsat3 = cenmod_model.mean_occupation_satellites(prim_haloprop=2.0e12)
    assert nsat3 < nsat2

    nsat3 = default_model.mean_occupation_satellites(prim_haloprop=2.0e12)
    default_model.param_dict["logMmin"] *= 1.1
    nsat4 = default_model.mean_occupation_satellites(prim_haloprop=2.0e12)
    assert nsat3 == nsat4


def test_Leauthaud11_composite():
    """ """
    model = PrebuiltHodModelFactory("leauthaud11", threshold=10.5)

    # Verify that changes param_dict properly propagate
    testmass1 = 5.0e11
    ncen1 = model.mean_occupation_centrals(prim_haloprop=testmass1)
    nsat1 = model.mean_occupation_satellites(prim_haloprop=testmass1)
    model.param_dict["smhm_m1_0"] /= 1.02
    ncen2 = model.mean_occupation_centrals(prim_haloprop=testmass1)
    nsat2 = model.mean_occupation_satellites(prim_haloprop=testmass1)
    assert ncen2 > ncen1
    assert nsat2 > nsat1

    model.param_dict["smhm_m1_a"] *= 1.1
    ncen3 = model.mean_occupation_centrals(prim_haloprop=testmass1)
    nsat3 = model.mean_occupation_satellites(prim_haloprop=testmass1)
    assert ncen3 == ncen2
    assert nsat3 == nsat2
