#!/usr/bin/env python

import numpy as np 
from astropy.table import Table 
from scipy.stats import spearmanr

from .. abunmatch import ConditionalAbunMatch
from .. import model_defaults
from ...sim_manager import FakeMock

def test_cam_gr_color():
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
	fake_mock_noscatter = FakeMock(approximate_ngals = 1e5)
	fake_mock_noscatter.galaxy_table['gr_color'] = (
		cam_noscatter.mc_gr_color(galaxy_table = fake_mock_noscatter.galaxy_table))

	cam_scatter_50 = ConditionalAbunMatch(
		galprop_key=galprop_key, 
		prim_galprop_key = prim_galprop_key, 
		sec_haloprop_key = sec_haloprop_key, 
		input_galaxy_table = fake_data.galaxy_table, 
		prim_galprop_bins = sm_bins, 
		correlation_strength = 0.5
		)
	fake_mock_scatter_50 = FakeMock(approximate_ngals = 1e5)
	fake_mock_scatter_50.galaxy_table['gr_color'] = (
		cam_scatter_50.mc_gr_color(galaxy_table = fake_mock_scatter_50.galaxy_table))

	cam_variable_scatter = ConditionalAbunMatch(
		galprop_key=galprop_key, 
		prim_galprop_key = prim_galprop_key, 
		sec_haloprop_key = sec_haloprop_key, 
		input_galaxy_table = fake_data.galaxy_table, 
		prim_galprop_bins = sm_bins, 
		correlation_strength = [0.25, 0.75], 
		correlation_strength_abcissa = [2.e10, 7.e10]
		)
	fake_mock_variable_scatter = FakeMock(approximate_ngals = 1e5)
	fake_mock_variable_scatter.galaxy_table['gr_color'] = (
		cam_variable_scatter.mc_gr_color(galaxy_table = fake_mock_variable_scatter.galaxy_table))

	def check_conditional_one_point(mock, data, sm_low, sm_high):
		idx_sm_range_mock = np.where(
			(mock.galaxy_table['stellar_mass'] > sm_low) & 
			(mock.galaxy_table['stellar_mass'] < sm_high))[0]
		mock_sm_range = mock.galaxy_table[idx_sm_range_mock]

		idx_sm_range_data = np.where(
			(data.galaxy_table['stellar_mass'] > sm_low) & 
			(data.galaxy_table['stellar_mass'] < sm_high))[0]
		data_sm_range = data.galaxy_table[idx_sm_range_data]

		np.testing.assert_almost_equal(
			data_sm_range['gr_color'].mean(), 
			mock_sm_range['gr_color'].mean(), decimal=1)

	def check_spearmanr(mock, data, sm_low, sm_high, desired_correlation):
		idx_sm_range_mock = np.where(
			(mock.galaxy_table['stellar_mass'] > sm_low) & 
			(mock.galaxy_table['stellar_mass'] < sm_high))[0]
		mock_sm_range = mock.galaxy_table[idx_sm_range_mock]

		idx_sm_range_data = np.where(
			(data.galaxy_table['stellar_mass'] > sm_low) & 
			(data.galaxy_table['stellar_mass'] < sm_high))[0]
		data_sm_range = data.galaxy_table[idx_sm_range_data]

		corr = spearmanr(mock_sm_range['gr_color'], mock_sm_range['halo_zhalf'])[0]
		np.testing.assert_almost_equal(corr, desired_correlation, decimal=1)


	# Check no-scatter mock
	sm_low, sm_high = 1.e10, 5.e10
	check_conditional_one_point(fake_mock_noscatter, fake_data, sm_low, sm_high)
	check_spearmanr(fake_mock_noscatter, fake_data, sm_low, sm_high, 0.99)
	sm_low, sm_high = 5.e10, 1.e11
	check_conditional_one_point(fake_mock_noscatter, fake_data, sm_low, sm_high)
	check_spearmanr(fake_mock_noscatter, fake_data, sm_low, sm_high, 0.99)

	# Check mock with 50% correlation strength
	sm_low, sm_high = 1.e10, 5.e10
	check_conditional_one_point(fake_mock_scatter_50, fake_data, sm_low, sm_high)
	check_spearmanr(fake_mock_scatter_50, fake_data, sm_low, sm_high, 0.5)
	sm_low, sm_high = 5.e10, 1.e11
	check_conditional_one_point(fake_mock_scatter_50, fake_data, sm_low, sm_high)
	check_spearmanr(fake_mock_scatter_50, fake_data, sm_low, sm_high, 0.5)

	# Check mock with variable correlation strength
	sm_low, sm_high = 1.e10, 5.e10
	check_conditional_one_point(fake_mock_variable_scatter, fake_data, sm_low, sm_high)
	check_spearmanr(fake_mock_variable_scatter, fake_data, sm_low, sm_high, 0.34)
	sm_low, sm_high = 5.e10, 1.e11
	check_conditional_one_point(fake_mock_variable_scatter, fake_data, sm_low, sm_high)
	check_spearmanr(fake_mock_variable_scatter, fake_data, sm_low, sm_high, 0.835)

def test_cam_ssfr():
	galprop_key = 'ssfr'
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
	fake_mock_noscatter = FakeMock(approximate_ngals = 1e5)
	fake_mock_noscatter.galaxy_table['ssfr'] = (
		cam_noscatter.mc_ssfr(galaxy_table = fake_mock_noscatter.galaxy_table))

	cam_scatter_50 = ConditionalAbunMatch(
		galprop_key=galprop_key, 
		prim_galprop_key = prim_galprop_key, 
		sec_haloprop_key = sec_haloprop_key, 
		input_galaxy_table = fake_data.galaxy_table, 
		prim_galprop_bins = sm_bins, 
		correlation_strength = 0.5
		)
	fake_mock_scatter_50 = FakeMock(approximate_ngals = 1e5)
	fake_mock_scatter_50.galaxy_table['ssfr'] = (
		cam_scatter_50.mc_ssfr(galaxy_table = fake_mock_scatter_50.galaxy_table))

	cam_variable_scatter = ConditionalAbunMatch(
		galprop_key=galprop_key, 
		prim_galprop_key = prim_galprop_key, 
		sec_haloprop_key = sec_haloprop_key, 
		input_galaxy_table = fake_data.galaxy_table, 
		prim_galprop_bins = sm_bins, 
		correlation_strength = [0.25, 0.75], 
		correlation_strength_abcissa = [2.e10, 7.e10]
		)
	fake_mock_variable_scatter = FakeMock(approximate_ngals = 1e5)
	fake_mock_variable_scatter.galaxy_table['ssfr'] = (
		cam_variable_scatter.mc_ssfr(galaxy_table = fake_mock_variable_scatter.galaxy_table))

	def check_conditional_one_point(mock, data, sm_low, sm_high):
		idx_sm_range_mock = np.where(
			(mock.galaxy_table['stellar_mass'] > sm_low) & 
			(mock.galaxy_table['stellar_mass'] < sm_high))[0]
		mock_sm_range = mock.galaxy_table[idx_sm_range_mock]

		idx_sm_range_data = np.where(
			(data.galaxy_table['stellar_mass'] > sm_low) & 
			(data.galaxy_table['stellar_mass'] < sm_high))[0]
		data_sm_range = data.galaxy_table[idx_sm_range_data]

		np.testing.assert_almost_equal(
			data_sm_range['ssfr'].mean(), 
			mock_sm_range['ssfr'].mean(), decimal=1)

	def check_spearmanr(mock, data, sm_low, sm_high, desired_correlation):
		idx_sm_range_mock = np.where(
			(mock.galaxy_table['stellar_mass'] > sm_low) & 
			(mock.galaxy_table['stellar_mass'] < sm_high))[0]
		mock_sm_range = mock.galaxy_table[idx_sm_range_mock]

		idx_sm_range_data = np.where(
			(data.galaxy_table['stellar_mass'] > sm_low) & 
			(data.galaxy_table['stellar_mass'] < sm_high))[0]
		data_sm_range = data.galaxy_table[idx_sm_range_data]

		corr = spearmanr(mock_sm_range['ssfr'], mock_sm_range['halo_zhalf'])[0]
		np.testing.assert_almost_equal(corr, desired_correlation, decimal=1)

	# Check no-scatter mock
	sm_low, sm_high = 1.e10, 5.e10
	check_conditional_one_point(fake_mock_noscatter, fake_data, sm_low, sm_high)
	check_spearmanr(fake_mock_noscatter, fake_data, sm_low, sm_high, 0.99)
	sm_low, sm_high = 5.e10, 1.e11
	check_conditional_one_point(fake_mock_noscatter, fake_data, sm_low, sm_high)
	check_spearmanr(fake_mock_noscatter, fake_data, sm_low, sm_high, 0.99)

	# Check mock with 50% correlation strength
	sm_low, sm_high = 1.e10, 5.e10
	check_conditional_one_point(fake_mock_scatter_50, fake_data, sm_low, sm_high)
	check_spearmanr(fake_mock_scatter_50, fake_data, sm_low, sm_high, 0.5)
	sm_low, sm_high = 5.e10, 1.e11
	check_conditional_one_point(fake_mock_scatter_50, fake_data, sm_low, sm_high)
	check_spearmanr(fake_mock_scatter_50, fake_data, sm_low, sm_high, 0.5)

	# Check mock with variable correlation strength
	sm_low, sm_high = 1.e10, 5.e10
	check_conditional_one_point(fake_mock_variable_scatter, fake_data, sm_low, sm_high)
	check_spearmanr(fake_mock_variable_scatter, fake_data, sm_low, sm_high, 0.34)
	sm_low, sm_high = 5.e10, 1.e11
	check_conditional_one_point(fake_mock_variable_scatter, fake_data, sm_low, sm_high)
	check_spearmanr(fake_mock_variable_scatter, fake_data, sm_low, sm_high, 0.835)












