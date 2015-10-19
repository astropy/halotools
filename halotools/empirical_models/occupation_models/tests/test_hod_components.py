#!/usr/bin/env python
import numpy as np
from astropy.table import Table
from copy import copy

from .. import hod_components, zheng07_components

from ... import model_defaults

__all__ = ['test_Zheng07Cens','test_Zheng07Sats']


def test_Zheng07Cens():
	""" Function to test 
	`~halotools.empirical_models.Zheng07Cens`. 
	Here's a brief summary of the tests performed: 

		* The basic metadata of the model is correct, e.g., ``self._upper_occupation_bound = 1`` 

		* The `mean_occupation` function is bounded by zero and unity for the full range of reasonable input masses, :math:`0 <= \\langle N_{\mathrm{cen}}(M) \\rangle <=1` for :math:`\\log_{10}M/M_{\odot} \\in [10, 16]`

		* The `mean_occupation` function increases monotonically for the full range of reasonable input masses, :math:`\\langle N_{\mathrm{cen}}(M_{2}) \\rangle > \\langle N_{\mathrm{cen}}(M_{1}) \\rangle` for :math:`M_{2}>M_{1}`

		* The model correctly navigates having either array or halo catalog arguments, and returns the identical result regardless of how the inputs are bundled

		* The `mean_occupation` function scales properly as a function of variations in :math:`\\sigma_{\\mathrm{log}M}`, and also variations in :math:`\\log M_{\mathrm{min}}`, for both low and high halo masses. 


	"""

	def test_attributes(model):
		assert isinstance(model, hod_components.OccupationComponent)
		assert model.gal_type == 'centrals'

		assert hasattr(model, 'prim_haloprop_key')

		assert model._upper_occupation_bound == 1

	def test_mean_occupation(model):

		assert hasattr(model, 'mean_occupation')

		mvir_array = np.logspace(10, 16, 10)
		mean_occ = model.mean_occupation(prim_haloprop=mvir_array) 

		# Check that the range is in [0,1]
		assert np.all(mean_occ<= 1)
		assert np.all(mean_occ >= 0)

		# The mean occupation should be monotonically increasing
		assert np.all(np.diff(mean_occ) >= 0)

	def test_mc_occupation(model):

		### Check the Monte Carlo realization method
		assert hasattr(model, 'mc_occupation')

		# First check that the mean occuation is ~0.5 when model is evaulated at Mmin
		mvir_midpoint = 10.**model.param_dict['logMmin']
		Npts = 1e3
		masses = np.ones(Npts)*mvir_midpoint
		mc_occ = model.mc_occupation(prim_haloprop=masses, seed=43)
		assert set(mc_occ).issubset([0,1])
		expected_result = 0.48599999
		np.testing.assert_allclose(mc_occ.mean(), expected_result, rtol=1e-5, atol=1.e-5)

		# Now check that the model is ~ 1.0 when evaluated for a cluster
		masses = np.ones(Npts)*5.e15
		mc_occ = model.mc_occupation(prim_haloprop=masses, seed=43)
		assert set(mc_occ).issubset([0,1])
		expected_result = 1.0
		np.testing.assert_allclose(mc_occ.mean(), expected_result, rtol=1e-2, atol=1.e-2)

		# Now check that the model is ~ 0.0 when evaluated for a tiny halo
		masses = np.ones(Npts)*1.e10
		mc_occ = model.mc_occupation(prim_haloprop=masses, seed=43)
		assert set(mc_occ).issubset([0,1])
		expected_result = 0.0
		np.testing.assert_allclose(mc_occ.mean(), expected_result, rtol=1e-2, atol=1.e-2)

	def test_correct_argument_inference_from_halo_catalog(model):
		mvir_array = np.logspace(10, 16, 10)
		key = model.prim_haloprop_key
		mvir_dict = {key:mvir_array}
		halo_catalog = Table(mvir_dict)
		# First test mean occupations
		meanocc_from_array = model.mean_occupation(prim_haloprop=mvir_array)
		meanocc_from_halos = model.mean_occupation(halo_table=halo_catalog)
		assert np.all(meanocc_from_array == meanocc_from_halos)
		# Now test Monte Carlo occupations
		mcocc_from_array = model.mc_occupation(prim_haloprop=mvir_array,seed=43)
		mcocc_from_halos = model.mc_occupation(halo_table=halo_catalog,seed=43)
		assert np.all(mcocc_from_array == mcocc_from_halos)

	### First test the model with all default settings
	default_model = zheng07_components.Zheng07Cens()
	test_attributes(default_model)
	test_mean_occupation(default_model)
	test_mc_occupation(default_model)
	test_correct_argument_inference_from_halo_catalog(default_model)

	### Now test the various threshold settings
	for threshold in np.arange(-22, -17.5, 0.5):
		thresh_model = zheng07_components.Zheng07Cens(threshold = threshold)
		test_attributes(thresh_model)
		test_mean_occupation(thresh_model)
		test_mc_occupation(thresh_model)

	### Test models with manually perturbed param_dict values
	default_dict = default_model.param_dict
	# Increase Mmin by a factor of 2: 
	# decreases <Ncen> at fixed mass, and so there should be fewer total centrals in <Ncen> < 1 regime
	model2_dict = copy(default_dict)
	model2_dict['logMmin'] += np.log10(2.)
	model2 = zheng07_components.Zheng07Cens()
	model2.param_dict = model2_dict
	#
	# Increase sigma_logM by a factor of 2: 
	# broadens <Ncen> ==> more centrals in halos with Mvir < Mmin, no change whatsoever to mid-mass abundance 
	model3_dict = copy(default_dict)
	model3_dict['sigma_logM'] *= 2.0
	model3 = zheng07_components.Zheng07Cens()
	model3.param_dict = model3_dict
	### First test to make sure models run ok
	test_attributes(model2)
	test_mean_occupation(model2)
	test_mc_occupation(model2)
	#
	test_attributes(model3)
	test_mean_occupation(model3)
	test_mc_occupation(model3)
	# Check that the dictionaries were correctly implemented
	assert model2.param_dict['logMmin'] > default_model.param_dict['logMmin']
	assert model3.param_dict['sigma_logM'] > default_model.param_dict['sigma_logM']

	### Now make sure the value of <Ncen> scales reasonably with the parameters
	lowmass = (10.**default_model.param_dict['logMmin'])/1.1
	defocc_lowmass = default_model.mean_occupation(prim_haloprop=lowmass)
	occ2_lowmass = model2.mean_occupation(prim_haloprop=lowmass)
	occ3_lowmass = model3.mean_occupation(prim_haloprop=lowmass)
	assert occ3_lowmass > defocc_lowmass
	assert defocc_lowmass > occ2_lowmass
	#
	highmass = (10.**default_model.param_dict['logMmin'])*1.1
	defocc_highmass = default_model.mean_occupation(prim_haloprop=highmass)
	occ2_highmass = model2.mean_occupation(prim_haloprop=highmass)
	occ3_highmass = model3.mean_occupation(prim_haloprop=highmass)
	assert defocc_highmass > occ3_highmass 
	assert occ3_highmass > occ2_highmass
	### Verify that directly changing model parameters 
	# without a new instantiation also behaves properly
	default_model.param_dict['sigma_logM'] *= 2.
	updated_defocc_lowmass = default_model.mean_occupation(prim_haloprop=lowmass)
	updated_defocc_highmass = default_model.mean_occupation(prim_haloprop=highmass)
	assert updated_defocc_lowmass > defocc_lowmass
	assert updated_defocc_highmass < defocc_highmass
	# Check that updating parameters produces identical behavior 
	# to a new instantiatino with the same parameters
	assert updated_defocc_lowmass == occ3_lowmass
	assert updated_defocc_highmass == occ3_highmass
	

def test_Zheng07Sats():
	""" Function to test 
	`~halotools.empirical_models.Zheng07Sats`. 
	Here's a brief summary of the tests performed: 

		* The basic metadata of the model is correct, e.g., ``self._upper_occupation_bound = 1`` 

		* The `mean_occupation` function is bounded by zero and unity for the full range of reasonable input masses, :math:`0 <= \\langle N_{\mathrm{cen}}(M) \\rangle <=1` for :math:`\\log_{10}M/M_{\odot} \\in [10, 16]`

		* The `mean_occupation` function increases monotonically for the full range of reasonable input masses, :math:`\\langle N_{\mathrm{cen}}(M_{2}) \\rangle > \\langle N_{\mathrm{cen}}(M_{1}) \\rangle` for :math:`M_{2}>M_{1}`

		* The model correctly navigates having either array or halo catalog arguments, and returns the identical result regardless of how the inputs are bundled

		* The `mean_occupation` function scales properly as a function of variations in :math:`\\sigma_{\\mathrm{log}M}`, and also variations in :math:`\\log M_{\mathrm{min}}`, for both low and high halo masses. 


	"""

	def test_attributes(model):
		assert isinstance(model, hod_components.OccupationComponent)
		assert model.gal_type == 'satellites'

		assert hasattr(model, 'prim_haloprop_key')

		assert model._upper_occupation_bound == float("inf")

	def test_mean_occupation(model):

		assert hasattr(model, 'mean_occupation')

		mvir_array = np.logspace(10, 16, 10)
		mean_occ = model.mean_occupation(prim_haloprop=mvir_array) 

		# Check non-negative
		assert np.all(mean_occ >= 0)
		# The mean occupation should be monotonically increasing
		assert np.all(np.diff(mean_occ) >= 0)

	def test_mc_occupation(model):

		### Check the Monte Carlo realization method
		assert hasattr(model, 'mc_occupation')

		model.param_dict['alpha'] = 1
		model.param_dict['logM0'] = 11.25
		model.param_dict['logM1'] = model.param_dict['logM0'] + np.log10(20.)

		Npts = 1e3
		masses = np.ones(Npts)*10.**model.param_dict['logM1']
		mc_occ = model.mc_occupation(prim_haloprop=masses, seed=43)
		# We chose a specific seed that has been pre-tested, 
		# so we should always get the same result
		expected_result = 1.0
		np.testing.assert_allclose(mc_occ.mean(), expected_result, rtol=1e-2, atol=1.e-2)

	def test_ncen_inheritance():
		satmodel_nocens = zheng07_components.Zheng07Sats()
		cenmodel = zheng07_components.Zheng07Cens()
		satmodel_cens = zheng07_components.Zheng07Sats(modulate_with_cenocc=True)

		Npts = 1e2 
		masses = np.logspace(10, 15, Npts)
		mean_occ_satmodel_nocens = satmodel_nocens.mean_occupation(prim_haloprop=masses)
		mean_occ_satmodel_cens = satmodel_cens.mean_occupation(prim_haloprop=masses)
		assert np.all(mean_occ_satmodel_cens <= mean_occ_satmodel_nocens)

		diff = mean_occ_satmodel_cens - mean_occ_satmodel_nocens
		assert diff.sum() < 0

		mean_occ_cens = satmodel_cens.central_occupation_model.mean_occupation(prim_haloprop=masses)
		assert np.all(mean_occ_satmodel_cens == mean_occ_satmodel_nocens*mean_occ_cens)

	### First test the model with all default settings
	default_model = zheng07_components.Zheng07Sats()
	test_attributes(default_model)
	test_mean_occupation(default_model)
	test_mc_occupation(default_model)

	### Now test the various threshold settings
	for threshold in np.arange(-22, -17.5, 0.5):
		thresh_model = zheng07_components.Zheng07Sats(threshold = threshold)
		test_attributes(thresh_model)
		test_mean_occupation(thresh_model)

	test_ncen_inheritance()

	### Check that models scale reasonably with different param_dict values
	default_dict = default_model.param_dict

	###### power law slope ######
	# Increase steepness of high-mass-end power law
	model2_dict = copy(default_dict)
	model2_dict['alpha'] *= 1.25
	model2 = zheng07_components.Zheng07Sats()
	model2.param_dict = model2_dict

	logmass = model2.param_dict['logM1'] + np.log10(5)
	mass = 10.**logmass
	assert model2.mean_occupation(prim_haloprop=mass) > default_model.mean_occupation(prim_haloprop=mass)

	Npts = 1e3
	masses = np.ones(Npts)*mass
	assert model2.mc_occupation(prim_haloprop=masses,seed=43).mean() > default_model.mc_occupation(prim_haloprop=masses,seed=43).mean()

	default_model.param_dict['alpha'] = model2.param_dict['alpha']
	assert model2.mc_occupation(prim_haloprop=masses,seed=43).mean() == default_model.mc_occupation(prim_haloprop=masses,seed=43).mean()

	###### Increase in M0 ######
	model2_dict = copy(default_dict)
	model2_dict['logM0'] += np.log10(2)
	model2 = zheng07_components.Zheng07Sats()
	model2.param_dict = model2_dict

	# At very low mass, both models should have zero satellites 
	lowmass = 1e10
	assert model2.mean_occupation(prim_haloprop=lowmass) == default_model.mean_occupation(prim_haloprop=lowmass)	
	# At intermediate masses, there should be fewer satellites for larger M0
	midmass = 1e12
	assert model2.mean_occupation(prim_haloprop=midmass) < default_model.mean_occupation(prim_haloprop=midmass)
	# At high masses, the difference should be negligible
	highmass = 1e15
	np.testing.assert_allclose(
		model2.mean_occupation(prim_haloprop=highmass) , 
		default_model.mean_occupation(prim_haloprop=highmass), 
		rtol=1e-3, atol=1.e-3)

	###### Increase in M1 ######
	model2_dict = copy(default_dict)
	model2_dict['logM0'] += np.log10(2)
	model2 = zheng07_components.Zheng07Sats()
	model2.param_dict = model2_dict
	# At very low mass, both models should have zero satellites 
	lowmass = 1e10
	assert model2.mean_occupation(prim_haloprop=lowmass) == default_model.mean_occupation(prim_haloprop=lowmass)	
	# At intermediate masses, there should be fewer satellites for larger M1
	midmass = 1e12
	fracdiff_midmass = ((model2.mean_occupation(prim_haloprop=midmass) - default_model.mean_occupation(prim_haloprop=midmass)) / 
		default_model.mean_occupation(prim_haloprop=midmass))
	assert fracdiff_midmass < 0
	# At high masses, the difference should persist, and be fractionally greater 
	highmass = 1e14
	fracdiff_highmass = ((model2.mean_occupation(prim_haloprop=highmass) - default_model.mean_occupation(prim_haloprop=highmass)) / 
		default_model.mean_occupation(prim_haloprop=highmass))
	assert fracdiff_highmass < 0
	assert fracdiff_highmass > fracdiff_midmass












