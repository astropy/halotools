"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from unittest import TestCase
import numpy as np

from ...biased_nfw_phase_space import BiasedNFWPhaseSpace
from .......factories import PrebuiltHodModelFactory, HodModelFactory
from ........sim_manager import FakeSim

__all__ = ('TestBiasedNFWConsistency', )


conc_bins = np.linspace(2, 30, 3)
gal_bias_bins = np.linspace(0.1, 20, 2)
gal_bias_bins = np.insert(gal_bias_bins, np.searchsorted(gal_bias_bins, 1), 1)


class TestBiasedNFWConsistency(TestCase):

    def setUp(self):
        halocat = FakeSim(seed=43, num_halos_per_massbin=25)

        self.unbiased_model = PrebuiltHodModelFactory('zheng07', threshold=-20.5,
                conc_mass_model='dutton_maccio14')
        self.unbiased_model.populate_mock(halocat, seed=43)

        model_dict = self.unbiased_model.model_dictionary
        assert model_dict['satellites_profile'].conc_mass_model == 'dutton_maccio14'

        biased_nfw = BiasedNFWPhaseSpace(concentration_bins=conc_bins,
            conc_gal_bias_bins=gal_bias_bins, conc_mass_model='dutton_maccio14')
        model_dict['satellites_profile'] = biased_nfw
        self.model = HodModelFactory(**model_dict)
        self.model.populate_mock(halocat, seed=43)
        assert self.model.threshold == self.unbiased_model.threshold
        assert biased_nfw.conc_mass_model == 'dutton_maccio14'

    def test_consistency1(self):

        self.model.param_dict['conc_gal_bias'] = 1
        self.model.mock.populate(seed=43)
        satmask = self.model.mock.galaxy_table['gal_type'] == 'satellites'
        sats = self.model.mock.galaxy_table[satmask]
        lowmass_value = sats['halo_mvir'][sats['halo_mvir'] >= 10**14].min()
        lowmass_sats_mask = sats['halo_mvir'] == lowmass_value
        lowmass_sats = sats[lowmass_sats_mask]
        mean_r_by_R_lowmass_biased = np.mean(
            lowmass_sats['host_centric_distance']/lowmass_sats['halo_rvir'])

        satmask = self.unbiased_model.mock.galaxy_table['gal_type'] == 'satellites'
        sats = self.unbiased_model.mock.galaxy_table[satmask]
        lowmass_value = sats['halo_mvir'][sats['halo_mvir'] >= 10**14].min()
        lowmass_sats_mask = sats['halo_mvir'] == lowmass_value
        lowmass_sats = sats[lowmass_sats_mask]
        mean_r_by_R_lowmass_unbiased = np.mean(
            lowmass_sats['host_centric_distance']/lowmass_sats['halo_rvir'])

        msg = ("Mocks made with unbiased vs. biased NFW profiles should have \n"
            "comparable values of <r / Rvir> when conc_gal_bias=1")
        assert np.allclose(mean_r_by_R_lowmass_unbiased, mean_r_by_R_lowmass_biased, atol=0.02), msg

    def test_consistency2(self):

        self.model.param_dict['conc_gal_bias'] = gal_bias_bins.max()
        self.model.mock.populate(seed=43)
        satmask = self.model.mock.galaxy_table['gal_type'] == 'satellites'
        sats = self.model.mock.galaxy_table[satmask]
        lowmass_value = sats['halo_mvir'][sats['halo_mvir'] >= 10**14].min()
        lowmass_sats_mask = sats['halo_mvir'] == lowmass_value
        lowmass_sats = sats[lowmass_sats_mask]
        mean_r_by_R_lowmass_biased = np.mean(
            lowmass_sats['host_centric_distance']/lowmass_sats['halo_rvir'])

        satmask = self.unbiased_model.mock.galaxy_table['gal_type'] == 'satellites'
        sats = self.unbiased_model.mock.galaxy_table[satmask]
        lowmass_value = sats['halo_mvir'][sats['halo_mvir'] >= 10**14].min()
        lowmass_sats_mask = sats['halo_mvir'] == lowmass_value
        lowmass_sats = sats[lowmass_sats_mask]
        mean_r_by_R_lowmass_unbiased = np.mean(
            lowmass_sats['host_centric_distance']/lowmass_sats['halo_rvir'])

        msg = ("Mocks made with unbiased profiles should have \n"
        "larger values of <r / Rvir> relative to a biased model with conc_gal_bias=10")
        assert mean_r_by_R_lowmass_unbiased - 0.05 > mean_r_by_R_lowmass_biased, msg

    def test_consistency3(self):

        self.model.param_dict['conc_gal_bias'] = gal_bias_bins.min()
        self.model.mock.populate(seed=43)
        satmask = self.model.mock.galaxy_table['gal_type'] == 'satellites'
        sats = self.model.mock.galaxy_table[satmask]
        lowmass_value = sats['halo_mvir'][sats['halo_mvir'] >= 10**14].min()
        lowmass_sats_mask = sats['halo_mvir'] == lowmass_value
        lowmass_sats = sats[lowmass_sats_mask]
        mean_r_by_R_lowmass_biased = np.mean(
            lowmass_sats['host_centric_distance']/lowmass_sats['halo_rvir'])

        satmask = self.unbiased_model.mock.galaxy_table['gal_type'] == 'satellites'
        sats = self.unbiased_model.mock.galaxy_table[satmask]
        lowmass_value = sats['halo_mvir'][sats['halo_mvir'] >= 10**14].min()
        lowmass_sats_mask = sats['halo_mvir'] == lowmass_value
        lowmass_sats = sats[lowmass_sats_mask]
        mean_r_by_R_lowmass_unbiased = np.mean(
            lowmass_sats['host_centric_distance']/lowmass_sats['halo_rvir'])

        msg = ("Mocks made with unbiased profiles should have \n"
        "smaller values of <r / Rvir> relative to a biased model with conc_gal_bias=0.1")
        assert mean_r_by_R_lowmass_unbiased + 0.05 < mean_r_by_R_lowmass_biased, msg

    def tearDown(self):
        del self.model
