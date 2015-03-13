#!/usr/bin/env python
import numpy as np

from .. import preloaded_hod_blueprints
from .. import model_defaults
from .. import hod_components
from .. import gal_prof_factory

from astropy.table import Table
from copy import copy


def test_Kravtsov04Cens():

	def test_attributes(model, gal_type='centrals'):
		assert isinstance(model, hod_components.OccupationComponent)
		assert model.gal_type == gal_type

		correct_haloprops = {'halo_boundary', 'prim_haloprop_key'}
		assert set(model.haloprop_key_dict.keys()) == correct_haloprops

		assert model.num_haloprops == 1
		assert model.occupation_bound == 1
		assert model.prim_func_dict.keys() == [None]

	def test_mean_occupation(model):

		assert hasattr(model, 'mean_occupation')

		mvir_array = np.logspace(10, 16, 10)
		mean_occ = model.mean_occupation(mvir_array) 

		# Check that the range is in [0,1]
		assert np.all(mean_occ<= 1)
		assert np.all(mean_occ >= 0)

		# The mean occupation should be monotonically increasing
		assert np.all(np.diff(mean_occ) >= 0)

	def test_mc_occupation(model):

		### Check the Monte Carlo realization method
		assert hasattr(model, 'mc_occupation')

		# First check that the mean occuation is ~0.5 when model is evaulated at Mmin
		mvir_midpoint = 10.**model.param_dict[model.logMmin_key]
		Npts = 1e3
		masses = np.ones(Npts)*mvir_midpoint
		mc_occ = model.mc_occupation(masses, seed=43)
		assert set(mc_occ).issubset([0,1])
		expected_result = 0.48599999
		np.testing.assert_allclose(mc_occ.mean(), expected_result, rtol=1e-5, atol=1.e-5)

		# Now check that the model is ~ 1.0 when evaluated for a cluster
		masses = np.ones(Npts)*5.e15
		mc_occ = model.mc_occupation(masses, seed=43)
		assert set(mc_occ).issubset([0,1])
		expected_result = 1.0
		np.testing.assert_allclose(mc_occ.mean(), expected_result, rtol=1e-2, atol=1.e-2)

		# Now check that the model is ~ 0.0 when evaluated for a tiny halo
		masses = np.ones(Npts)*1.e10
		mc_occ = model.mc_occupation(masses, seed=43)
		assert set(mc_occ).issubset([0,1])
		expected_result = 0.0
		np.testing.assert_allclose(mc_occ.mean(), expected_result, rtol=1e-2, atol=1.e-2)

	def test_correct_argument_inference_from_halo_catalog(model):
		mvir_array = np.logspace(10, 16, 10)
		key = model.haloprop_key_dict['prim_haloprop_key']
		mvir_dict = {key:mvir_array}
		halo_catalog = Table(mvir_dict)
		# First test mean occupations
		meanocc_from_array = model.mean_occupation(mvir_array)
		meanocc_from_halos = model.mean_occupation(halos=halo_catalog)
		assert np.all(meanocc_from_array == meanocc_from_halos)
		# Now test Monte Carlo occupations
		mcocc_from_array = model.mc_occupation(mvir_array,seed=43)
		mcocc_from_halos = model.mc_occupation(halos=halo_catalog,seed=43)
		assert np.all(mcocc_from_array == mcocc_from_halos)

	### First test the model with all default settings
	default_model = hod_components.Kravtsov04Cens()
	test_attributes(default_model)
	test_mean_occupation(default_model)
	test_mc_occupation(default_model)
	test_correct_argument_inference_from_halo_catalog(default_model)

	### Now test the various threshold settings
	for threshold in np.arange(-22, -17.5, 0.5):
		thresh_model = hod_components.Kravtsov04Cens(threshold = threshold,gal_type='cens')
		test_attributes(thresh_model,gal_type='cens')
		test_mean_occupation(thresh_model)
		test_mc_occupation(thresh_model)

	### Test models with manually perturbed param_dict values
	default_dict = default_model.param_dict
	# Increase Mmin by a factor of 2: 
	# decreases <Ncen> at fixed mass, and so there should be fewer total centrals in <Ncen> < 1 regime
	model2_dict = copy(default_dict)
	model2_dict[default_model.logMmin_key] += np.log10(2.)
	model2 = hod_components.Kravtsov04Cens(input_param_dict = model2_dict)
	#
	# Increase sigma_logM by a factor of 2: 
	# broadens <Ncen> ==> more centrals in halos with Mvir < Mmin, no change whatsoever to mid-mass abundance 
	model3_dict = copy(default_dict)
	model3_dict[default_model.sigma_logM_key] *= 2.0
	model3 = hod_components.Kravtsov04Cens(input_param_dict = model3_dict)
	### First test to make sure models run ok
	test_attributes(model2)
	test_mean_occupation(model2)
	test_mc_occupation(model2)
	#
	test_attributes(model3)
	test_mean_occupation(model3)
	test_mc_occupation(model3)
	# Check that the dictionaries were correctly implemented
	assert model2.param_dict[model2.logMmin_key] > default_model.param_dict[default_model.logMmin_key]
	assert model3.param_dict[model3.sigma_logM_key] > default_model.param_dict[default_model.sigma_logM_key]

	### Now make sure the value of <Ncen> scales reasonably with the parameters
	lowmass = (10.**default_model.param_dict[default_model.logMmin_key])/1.1
	defocc_lowmass = default_model.mean_occupation(lowmass)
	occ2_lowmass = model2.mean_occupation(lowmass)
	occ3_lowmass = model3.mean_occupation(lowmass)
	assert occ3_lowmass > defocc_lowmass
	assert defocc_lowmass > occ2_lowmass
	#
	highmass = (10.**default_model.param_dict[default_model.logMmin_key])*1.1
	defocc_highmass = default_model.mean_occupation(highmass)
	occ2_highmass = model2.mean_occupation(highmass)
	occ3_highmass = model3.mean_occupation(highmass)
	assert defocc_highmass > occ3_highmass 
	assert occ3_highmass > occ2_highmass
	### Verify that directly changing model parameters 
	# without a new instantiation also behaves properly
	default_model.param_dict[default_model.sigma_logM_key] *= 2.
	updated_defocc_lowmass = default_model.mean_occupation(lowmass)
	updated_defocc_highmass = default_model.mean_occupation(highmass)
	assert updated_defocc_lowmass > defocc_lowmass
	assert updated_defocc_highmass < defocc_highmass
	# Check that updating parameters produces identical behavior 
	# to a new instantiatino with the same parameters
	assert updated_defocc_lowmass == occ3_lowmass
	assert updated_defocc_highmass == occ3_highmass


	










