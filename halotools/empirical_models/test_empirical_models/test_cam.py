#!/usr/bin/env python

import numpy as np 
from astropy.table import Table 
from scipy.stats import spearmanr

from .. abunmatch import ConditionalAbunMatch
from .. import model_defaults
from ...sim_manager import FakeMock

def test_cam():
	galprop_key = 'gr_color'
	prim_galprop_key = 'stellar_mass'
	sec_haloprop_key = 'halo_zhalf'

	fake_data = FakeMock()
	sm_min = fake_data.galaxy_table['stellar_mass'].min()
	sm_max = fake_data.galaxy_table['stellar_mass'].max()
	sm_bins = np.logspace(np.log10(sm_min)-0.01, np.log10(sm_max)+0.01, 50)

	cam_noscatter = ConditionalAbunMatch(
		galprop_key=galprop_key, 
		prim_galprop_key = prim_galprop_key, 
		sec_haloprop_key = sec_haloprop_key, 
		input_galaxy_table = fake_data.galaxy_table, 
		prim_galprop_bins = sm_bins
		)

	fake_mock = FakeMock(approximate_ngals = 1e5)
	fake_mock.galaxy_table['gr_color'] = (
		cam_noscatter.mc_gr_color(galaxy_table = fake_mock.galaxy_table))


	def check_conditional_one_point(sm_low, sm_high):
		idx_sm_range_mock = np.where(
			(fake_mock.galaxy_table['stellar_mass'] > sm_low) & 
			(fake_mock.galaxy_table['stellar_mass'] < sm_high))[0]
		mock_sm_range = fake_mock.galaxy_table[idx_sm_range_mock]

		idx_sm_range_data = np.where(
			(fake_data.galaxy_table['stellar_mass'] > sm_low) & 
			(fake_data.galaxy_table['stellar_mass'] < sm_high))[0]
		data_sm_range = fake_data.galaxy_table[idx_sm_range_data]

		np.testing.assert_almost_equal(
			data_sm_range['gr_color'].mean(), 
			mock_sm_range['gr_color'].mean(), decimal=1)

	def check_spearmanr(sm_low, sm_high, desired_correlation):
		idx_sm_range_mock = np.where(
			(fake_mock.galaxy_table['stellar_mass'] > sm_low) & 
			(fake_mock.galaxy_table['stellar_mass'] < sm_high))[0]
		mock_sm_range = fake_mock.galaxy_table[idx_sm_range_mock]

		idx_sm_range_data = np.where(
			(fake_data.galaxy_table['stellar_mass'] > sm_low) & 
			(fake_data.galaxy_table['stellar_mass'] < sm_high))[0]
		data_sm_range = fake_data.galaxy_table[idx_sm_range_data]

		corr = spearmanr(mock_sm_range['gr_color'], mock_sm_range['halo_zhalf'])[0]
		np.testing.assert_almost_equal(corr, desired_correlation, decimal=1)


	sm_low, sm_high = 1.e10, 5.e10
	check_conditional_one_point(sm_low, sm_high)
	check_spearmanr(sm_low, sm_high, 0.99)
	sm_low, sm_high = 5.e10, 1.e11
	check_conditional_one_point(sm_low, sm_high)
	check_spearmanr(sm_low, sm_high, 0.99)
















