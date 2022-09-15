"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from ...biased_nfw_phase_space import BiasedNFWPhaseSpace
from ...sfr_biased_nfw_phase_space import SFRBiasedNFWPhaseSpace

from .......factories import PrebuiltHodModelFactory, HodModelFactory
from .......sfr_models import HaloMassInterpolQuenching

from ........sim_manager import FakeSim


__all__ = ("test_conc_gal_bias1",)


conc_bins = np.linspace(2, 30, 3)
gal_bias_bins = np.linspace(0.1, 20, 2)
gal_bias_bins = np.insert(gal_bias_bins, np.searchsorted(gal_bias_bins, 1), 1)


def test_conc_gal_bias1():
    zheng07_model = PrebuiltHodModelFactory("zheng07", threshold=-18)
    model_dict = zheng07_model.model_dictionary

    log_lowmass_value, log_highmass_value = 14, 16
    conc_gal_bias_logM_abscissa = [log_lowmass_value, log_highmass_value]
    biased_nfw = BiasedNFWPhaseSpace(
        concentration_bins=conc_bins,
        conc_gal_bias_bins=gal_bias_bins,
        conc_gal_bias_logM_abscissa=conc_gal_bias_logM_abscissa,
        conc_mass_model="dutton_maccio14",
    )

    model_dict["satellites_profile"] = biased_nfw
    model = HodModelFactory(**model_dict)

    model.param_dict["conc_gal_bias_param0"] = gal_bias_bins.min()
    model.param_dict["conc_gal_bias_param1"] = gal_bias_bins.max()

    a = model.calculate_conc_gal_bias_satellites(prim_haloprop=10**log_lowmass_value)
    assert np.allclose(a, model.param_dict["conc_gal_bias_param0"], atol=1e-3)
    b = model.calculate_conc_gal_bias_satellites(prim_haloprop=10**log_highmass_value)
    assert np.allclose(b, model.param_dict["conc_gal_bias_param1"], atol=1e-3)


def test_sfr_biased_nfw_phase_space_conc_gal_bias():
    zheng07_model = PrebuiltHodModelFactory("zheng07", threshold=-18)
    model_dict = zheng07_model.model_dictionary

    log_lowmass_value, log_highmass_value = 14, 16

    conc_gal_bias_logM_abscissa = [log_lowmass_value, log_highmass_value]
    biased_nfw = SFRBiasedNFWPhaseSpace(
        concentration_bins=conc_bins,
        conc_gal_bias_bins=gal_bias_bins,
        conc_gal_bias_logM_abscissa=conc_gal_bias_logM_abscissa,
        conc_mass_model="dutton_maccio14",
    )

    model_dict["satellites_profile"] = biased_nfw
    model = HodModelFactory(**model_dict)
    assert "quiescent_conc_gal_bias_param0" in model.param_dict
    assert "conc_gal_bias_logM_abscissa_param0" in model.param_dict
    assert model.param_dict["conc_gal_bias_logM_abscissa_param0"] == 14, "whoops"

    model.param_dict["quiescent_conc_gal_bias_param0"] = gal_bias_bins.min()
    model.param_dict["quiescent_conc_gal_bias_param1"] = gal_bias_bins.max()
    model.param_dict["star_forming_conc_gal_bias_param0"] = gal_bias_bins.max()
    model.param_dict["star_forming_conc_gal_bias_param1"] = gal_bias_bins.max()

    aq = model.calculate_conc_gal_bias_satellites(
        prim_haloprop=10**log_lowmass_value, quiescent=True
    )
    assert np.allclose(
        aq, model.param_dict["quiescent_conc_gal_bias_param0"], atol=1e-3
    )
    bq = model.calculate_conc_gal_bias_satellites(
        prim_haloprop=10**log_highmass_value, quiescent=True
    )
    assert np.allclose(
        bq, model.param_dict["quiescent_conc_gal_bias_param1"], atol=1e-3
    )

    asf = model.calculate_conc_gal_bias_satellites(
        prim_haloprop=10**log_lowmass_value, quiescent=False
    )
    assert np.allclose(
        asf, model.param_dict["star_forming_conc_gal_bias_param0"], atol=1e-3
    )
    bsf = model.calculate_conc_gal_bias_satellites(
        prim_haloprop=10**log_highmass_value, quiescent=False
    )
    assert np.allclose(
        bsf, model.param_dict["star_forming_conc_gal_bias_param1"], atol=1e-3
    )

    masses = np.zeros(100) + 10**log_lowmass_value
    quiescent = np.random.randint(0, 1, 100).astype(bool)
    c = model.calculate_conc_gal_bias_satellites(
        prim_haloprop=masses, quiescent=quiescent
    )
    mask = quiescent == True
    assert np.allclose(
        c[mask], model.param_dict["quiescent_conc_gal_bias_param0"], atol=1e-3
    )
    assert np.allclose(
        c[~mask], model.param_dict["star_forming_conc_gal_bias_param0"], atol=1e-3
    )

    masses = np.zeros(100) + 10**log_highmass_value
    quiescent = np.random.randint(0, 1, 100).astype(bool)
    c = model.calculate_conc_gal_bias_satellites(
        prim_haloprop=masses, quiescent=quiescent
    )
    mask = quiescent == True
    assert np.allclose(
        c[mask], model.param_dict["quiescent_conc_gal_bias_param1"], atol=1e-3
    )
    assert np.allclose(
        c[~mask], model.param_dict["star_forming_conc_gal_bias_param1"], atol=1e-3
    )


def test_sfr_biased_nfw_phase_space_mockpop():
    zheng07_model = PrebuiltHodModelFactory("zheng07", threshold=-20)
    model_dict = zheng07_model.model_dictionary

    log_lowmass_value, log_highmass_value = 14, 16

    conc_gal_bias_logM_abscissa = [log_lowmass_value, log_highmass_value]
    biased_nfw = SFRBiasedNFWPhaseSpace(
        concentration_bins=conc_bins,
        conc_gal_bias_bins=gal_bias_bins,
        conc_gal_bias_logM_abscissa=conc_gal_bias_logM_abscissa,
        conc_mass_model="dutton_maccio14",
    )
    model_dict["satellites_profile"] = biased_nfw

    sfr_centrals = HaloMassInterpolQuenching(
        "halo_mvir", [1e12, 1e15], [0.2, 0.9], gal_type="centrals"
    )
    sfr_satellites = HaloMassInterpolQuenching(
        "halo_mvir", [1e12, 1e15], [0.2, 0.9], gal_type="satellites"
    )

    model_dict["centrals_sfr"] = sfr_centrals
    model_dict["satellites_sfr"] = sfr_satellites

    model = HodModelFactory(**model_dict)

    halocat = FakeSim(seed=43)
    model.populate_mock(halocat)
