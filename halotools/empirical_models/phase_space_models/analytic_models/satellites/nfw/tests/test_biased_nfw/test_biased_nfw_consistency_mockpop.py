"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.config.paths import _find_home
import pytest

from ...biased_nfw_phase_space import BiasedNFWPhaseSpace
from .......factories import PrebuiltHodModelFactory, HodModelFactory
from ........sim_manager import FakeSim

__all__ = ('test_consistency1', 'test_consistency2', 'test_consistency3' )

# Determine whether the machine is mine
# This will be used to select tests whose
# returned values depend on the configuration
# of my personal cache directory files
aph_home = '/Users/aphearin'
detected_home = _find_home()
if aph_home == detected_home:
    APH_MACHINE = True
else:
    APH_MACHINE = False


conc_bins = np.linspace(2, 30, 3)
gal_bias_bins = np.linspace(0.1, 20, 2)
gal_bias_bins = np.insert(gal_bias_bins, np.searchsorted(gal_bias_bins, 1), 1)


@pytest.mark.skipif('not APH_MACHINE')
def test_consistency1():

    halocat = FakeSim(seed=43, num_halos_per_massbin=25)

    unbiased_model = PrebuiltHodModelFactory('zheng07', threshold=-20.5,
            conc_mass_model='dutton_maccio14')
    unbiased_model.populate_mock(halocat, seed=43)

    model_dict = unbiased_model.model_dictionary
    assert model_dict['satellites_profile'].conc_mass_model == 'dutton_maccio14'

    biased_nfw = BiasedNFWPhaseSpace(concentration_bins=conc_bins,
        conc_gal_bias_bins=gal_bias_bins, conc_mass_model='dutton_maccio14')
    model_dict['satellites_profile'] = biased_nfw
    model = HodModelFactory(**model_dict)
    model.populate_mock(halocat, seed=43)
    assert model.threshold == unbiased_model.threshold
    assert biased_nfw.conc_mass_model == 'dutton_maccio14'

    model.param_dict['conc_gal_bias'] = 1
    model.mock.populate(seed=43)
    satmask = model.mock.galaxy_table['gal_type'] == 'satellites'
    sats = model.mock.galaxy_table[satmask]
    lowmass_value = sats['halo_mvir'][sats['halo_mvir'] >= 10**14].min()
    lowmass_sats_mask = sats['halo_mvir'] == lowmass_value
    lowmass_sats = sats[lowmass_sats_mask]
    mean_r_by_R_lowmass_biased = np.mean(
        lowmass_sats['host_centric_distance']/lowmass_sats['halo_rvir'])

    satmask = unbiased_model.mock.galaxy_table['gal_type'] == 'satellites'
    sats = unbiased_model.mock.galaxy_table[satmask]
    lowmass_value = sats['halo_mvir'][sats['halo_mvir'] >= 10**14].min()
    lowmass_sats_mask = sats['halo_mvir'] == lowmass_value
    lowmass_sats = sats[lowmass_sats_mask]
    mean_r_by_R_lowmass_unbiased = np.mean(
        lowmass_sats['host_centric_distance']/lowmass_sats['halo_rvir'])

    msg = ("Mocks made with unbiased vs. biased NFW profiles should have \n"
        "comparable values of <r / Rvir> when conc_gal_bias=1")
    assert np.allclose(mean_r_by_R_lowmass_unbiased, mean_r_by_R_lowmass_biased, atol=0.02), msg


@pytest.mark.skipif('not APH_MACHINE')
def test_consistency2():

    halocat = FakeSim(seed=43, num_halos_per_massbin=25)

    unbiased_model = PrebuiltHodModelFactory('zheng07', threshold=-20.5,
            conc_mass_model='dutton_maccio14')
    unbiased_model.populate_mock(halocat, seed=43)

    model_dict = unbiased_model.model_dictionary
    assert model_dict['satellites_profile'].conc_mass_model == 'dutton_maccio14'

    biased_nfw = BiasedNFWPhaseSpace(concentration_bins=conc_bins,
        conc_gal_bias_bins=gal_bias_bins, conc_mass_model='dutton_maccio14')
    model_dict['satellites_profile'] = biased_nfw
    model = HodModelFactory(**model_dict)
    model.populate_mock(halocat, seed=43)
    assert model.threshold == unbiased_model.threshold
    assert biased_nfw.conc_mass_model == 'dutton_maccio14'

    model.param_dict['conc_gal_bias'] = gal_bias_bins.max()
    model.mock.populate(seed=43)
    satmask = model.mock.galaxy_table['gal_type'] == 'satellites'
    sats = model.mock.galaxy_table[satmask]
    lowmass_value = sats['halo_mvir'][sats['halo_mvir'] >= 10**14].min()
    lowmass_sats_mask = sats['halo_mvir'] == lowmass_value
    lowmass_sats = sats[lowmass_sats_mask]
    mean_r_by_R_lowmass_biased = np.mean(
        lowmass_sats['host_centric_distance']/lowmass_sats['halo_rvir'])

    satmask = unbiased_model.mock.galaxy_table['gal_type'] == 'satellites'
    sats = unbiased_model.mock.galaxy_table[satmask]
    lowmass_value = sats['halo_mvir'][sats['halo_mvir'] >= 10**14].min()
    lowmass_sats_mask = sats['halo_mvir'] == lowmass_value
    lowmass_sats = sats[lowmass_sats_mask]
    mean_r_by_R_lowmass_unbiased = np.mean(
        lowmass_sats['host_centric_distance']/lowmass_sats['halo_rvir'])

    msg = ("Mocks made with unbiased profiles should have \n"
    "larger values of <r / Rvir> relative to a biased model with conc_gal_bias=10")
    assert mean_r_by_R_lowmass_unbiased - mean_r_by_R_lowmass_biased > 0.01, msg


@pytest.mark.skipif('not APH_MACHINE')
def test_consistency3():

    halocat = FakeSim(seed=43, num_halos_per_massbin=25)

    unbiased_model = PrebuiltHodModelFactory('zheng07', threshold=-20.5,
            conc_mass_model='dutton_maccio14')
    unbiased_model.populate_mock(halocat, seed=43)

    model_dict = unbiased_model.model_dictionary
    assert model_dict['satellites_profile'].conc_mass_model == 'dutton_maccio14'

    biased_nfw = BiasedNFWPhaseSpace(concentration_bins=conc_bins,
        conc_gal_bias_bins=gal_bias_bins, conc_mass_model='dutton_maccio14')
    model_dict['satellites_profile'] = biased_nfw
    model = HodModelFactory(**model_dict)
    model.populate_mock(halocat, seed=43)
    assert model.threshold == unbiased_model.threshold
    assert biased_nfw.conc_mass_model == 'dutton_maccio14'

    model.param_dict['conc_gal_bias'] = gal_bias_bins.min()
    model.mock.populate(seed=43)
    satmask = model.mock.galaxy_table['gal_type'] == 'satellites'
    sats = model.mock.galaxy_table[satmask]
    lowmass_value = sats['halo_mvir'][sats['halo_mvir'] >= 10**14].min()
    lowmass_sats_mask = sats['halo_mvir'] == lowmass_value
    lowmass_sats = sats[lowmass_sats_mask]
    mean_r_by_R_lowmass_biased = np.mean(
        lowmass_sats['host_centric_distance']/lowmass_sats['halo_rvir'])

    satmask = unbiased_model.mock.galaxy_table['gal_type'] == 'satellites'
    sats = unbiased_model.mock.galaxy_table[satmask]
    lowmass_value = sats['halo_mvir'][sats['halo_mvir'] >= 10**14].min()
    lowmass_sats_mask = sats['halo_mvir'] == lowmass_value
    lowmass_sats = sats[lowmass_sats_mask]
    mean_r_by_R_lowmass_unbiased = np.mean(
        lowmass_sats['host_centric_distance']/lowmass_sats['halo_rvir'])

    msg = ("Mocks made with unbiased profiles should have \n"
    "smaller values of <r / Rvir> relative to a biased model with conc_gal_bias=0.1")
    assert mean_r_by_R_lowmass_unbiased + 0.05 < mean_r_by_R_lowmass_biased, msg
