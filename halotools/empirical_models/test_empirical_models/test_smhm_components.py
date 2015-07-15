#!/usr/bin/env python

import numpy as np 
from astropy.table import Table 

from .. import smhm_components
from .. import model_defaults


__all__ = ['test_Moster13SmHm_initialization', 'test_LogNormalScatterModel_initialization']

def test_Moster13SmHm_initialization():
	""" Function testing the initialization of 
	`~halotools.empirical_models.smhm_components.Moster13SmHm`. 
	Summary of tests:

		* Class successfully instantiates when called with no arguments. 

		* Class successfully instantiates when constructor is passed ``redshift``, ``prim_haloprop_key``. 

		* When the above arguments are passed to the constructor, the instance is correctly initialized with the input values.

		* The scatter model bound to Moster13SmHm correctly inherits each of the above arguments. 
	"""

	default_model = smhm_components.Moster13SmHm()
	assert default_model.prim_haloprop_key == model_defaults.default_smhm_haloprop
	assert default_model.scatter_model.prim_haloprop_key == model_defaults.default_smhm_haloprop
	assert hasattr(default_model, 'redshift') == False
	assert isinstance(default_model.scatter_model, smhm_components.LogNormalScatterModel)

	keys = ['m10', 'm11', 'n10', 'n11', 'beta10', 'beta11', 'gamma10', 'gamma11', 'scatter_model_param1']
	for key in keys:
		assert key in default_model.param_dict.keys()
	assert default_model.param_dict['scatter_model_param1'] == model_defaults.default_smhm_scatter

	default_scatter_dict = {'scatter_model_param1': model_defaults.default_smhm_scatter}
	assert default_model.scatter_model.param_dict == default_scatter_dict
	assert default_model.scatter_model.ordinates == [model_defaults.default_smhm_scatter]

	z0_model = smhm_components.Moster13SmHm(redshift=0)
	assert z0_model.redshift == 0
	z1_model = smhm_components.Moster13SmHm(redshift=1)
	assert z1_model.redshift == 1

	macc_model = smhm_components.Moster13SmHm(prim_haloprop_key='macc')
	assert macc_model.prim_haloprop_key == 'macc'
	assert macc_model.scatter_model.prim_haloprop_key == 'macc'


def test_LogNormalScatterModel_initialization():
	""" Function testing the initialization of 
	`~halotools.empirical_models.smhm_components.LogNormalScatterModel`. 
	Summary of tests:

		* Class successfully instantiates when called with no arguments. 

		* Class successfully instantiates when constructor is passed ``ordinates`` and ``abcissa``. 

		* When the above arguments are passed to the constructor, the instance is correctly initialized with the input values.

	"""
	default_scatter_model = smhm_components.LogNormalScatterModel()
	assert default_scatter_model.prim_haloprop_key == model_defaults.default_smhm_haloprop
	assert default_scatter_model.abcissa == [12]
	assert default_scatter_model.ordinates == [model_defaults.default_smhm_scatter]
	default_param_dict = {'scatter_model_param1': model_defaults.default_smhm_scatter}
	assert default_scatter_model.param_dict == default_param_dict

	input_abcissa = [12, 15]
	input_ordinates = [0.3, 0.1]
	scatter_model2 = smhm_components.LogNormalScatterModel(
		scatter_abcissa = input_abcissa, scatter_ordinates = input_ordinates)

	assert scatter_model2.abcissa == input_abcissa
	assert scatter_model2.ordinates == input_ordinates
	model2_param_dict = {'scatter_model_param1': 0.3, 'scatter_model_param2': 0.1}
	assert scatter_model2.param_dict == model2_param_dict


def test_LogNormalScatterModel_behavior():
	""" Function testing the behavior of 
	`~halotools.empirical_models.smhm_components.LogNormalScatterModel`. 

	Summary of tests:

		* The default model returns the default scatter, both the mean_scatter method and the scatter_realization method. 

		* A model defined by interpolation between 12 and 15 returns the input scatter at the input abcissa, both the mean_scatter method and the scatter_realization method. 

		* The 12-15 model returns the correct intermediate level of scatter at the halfway point between 12 and 15, both the mean_scatter method and the scatter_realization method. 

		* All the above results apply equally well to cases where ``mass`` or ``halos`` is used as input. 

		* When the param_dict of a model is updated (as it would be during an MCMC), the behavior is correctly adjusted. 
	"""

	testing_seed = 43

	default_scatter_model = smhm_components.LogNormalScatterModel()

	Npts = 1e4
	testmass12 = 1e12
	mass12 = np.zeros(Npts) + testmass12
	masskey = model_defaults.default_smhm_haloprop 
	d = {masskey: mass12}
	halos12 = Table(d)

	# Test the mean_scatter method of the default model
	scatter = default_scatter_model.mean_scatter(prim_haloprop = testmass12)
	assert np.allclose(scatter, model_defaults.default_smhm_scatter)
	scatter_array = default_scatter_model.mean_scatter(prim_haloprop = mass12)
	assert np.allclose(scatter_array, model_defaults.default_smhm_scatter)
	scatter_array = default_scatter_model.mean_scatter(halos = halos12)
	assert np.allclose(scatter_array, model_defaults.default_smhm_scatter)

	# Test the scatter_realization method of the default model
	scatter_realization = default_scatter_model.scatter_realization(seed=testing_seed, prim_haloprop=mass12)
	disp = np.std(scatter_realization)
	np.testing.assert_almost_equal(disp, model_defaults.default_smhm_scatter, decimal=2)
	scatter_realization = default_scatter_model.scatter_realization(seed=testing_seed, halos=halos12)
	disp = np.std(scatter_realization)
	np.testing.assert_almost_equal(disp, model_defaults.default_smhm_scatter, decimal=2)


	input_abcissa = [12, 15]
	input_ordinates = [0.3, 0.1]
	scatter_model2 = smhm_components.LogNormalScatterModel(
		scatter_abcissa = input_abcissa, scatter_ordinates = input_ordinates)

	assert len(scatter_model2.abcissa) == 2
	assert len(scatter_model2.param_dict) == 2
	assert set(scatter_model2.param_dict.keys()) == set(['scatter_model_param1', 'scatter_model_param2'])
	assert set(scatter_model2.param_dict.values()) == set(input_ordinates)

	# Test the mean_scatter method of a non-trivial model at the first abcissa
	scatter_array = scatter_model2.mean_scatter(prim_haloprop = mass12)
	assert np.allclose(scatter_array, 0.3)
	scatter_array = scatter_model2.mean_scatter(halos = halos12)
	assert np.allclose(scatter_array, 0.3)

	# Test the scatter_realization method of a non-trivial model at the first abcissa
	scatter_realization = scatter_model2.scatter_realization(seed=testing_seed, prim_haloprop=mass12)
	disp = np.std(scatter_realization)
	np.testing.assert_almost_equal(disp, 0.3, decimal=2)
	scatter_realization = scatter_model2.scatter_realization(seed=testing_seed, halos=halos12)
	disp = np.std(scatter_realization)
	np.testing.assert_almost_equal(disp, 0.3, decimal=2)


	# Test the mean_scatter method of a non-trivial model at the second abcissa
	testmass15 = 1e15
	mass15 = np.zeros(Npts) + testmass15
	masskey = model_defaults.default_smhm_haloprop 
	d = {masskey: mass15}
	halos15 = Table(d)

	scatter_array = scatter_model2.mean_scatter(prim_haloprop = mass15)
	assert np.allclose(scatter_array, 0.1)
	scatter_array = scatter_model2.mean_scatter(halos = halos15)
	assert np.allclose(scatter_array, 0.1)

	# Test the scatter_realization method of a non-trivial model at the second abcissa
	scatter_realization = scatter_model2.scatter_realization(seed=testing_seed, prim_haloprop=mass15)
	disp = np.std(scatter_realization)
	np.testing.assert_almost_equal(disp, 0.1, decimal=2)
	scatter_realization = scatter_model2.scatter_realization(seed=testing_seed, halos=halos15)
	disp = np.std(scatter_realization)
	np.testing.assert_almost_equal(disp, 0.1, decimal=2)

	# Test the mean_scatter method of a non-trivial model at an intermediate value
	testmass135 = 10.**13.5
	mass135 = np.zeros(Npts) + testmass135
	masskey = model_defaults.default_smhm_haloprop 
	d = {masskey: mass135}
	halos135 = Table(d)

	scatter_array = scatter_model2.mean_scatter(prim_haloprop = mass135)
	assert np.allclose(scatter_array, 0.2)
	scatter_array = scatter_model2.mean_scatter(halos = halos135)
	assert np.allclose(scatter_array, 0.2)

	# Test the scatter_realization method of a non-trivial model at an intermediate value
	scatter_realization = scatter_model2.scatter_realization(seed=testing_seed, prim_haloprop=mass135)
	disp = np.std(scatter_realization)
	np.testing.assert_almost_equal(disp, 0.2, decimal=2)
	scatter_realization = scatter_model2.scatter_realization(seed=testing_seed, halos=halos135)
	disp = np.std(scatter_realization)
	np.testing.assert_almost_equal(disp, 0.2, decimal=2)

	# Update the parameter dictionary that defines the non-trivial model
	scatter_model2.param_dict['scatter_model_param2'] = 0.5

	# Test the mean_scatter method of the updated non-trivial model 
	scatter_array = scatter_model2.mean_scatter(prim_haloprop = mass12)
	assert np.allclose(scatter_array, 0.3)
	scatter_array = scatter_model2.mean_scatter(prim_haloprop = mass15)
	assert np.allclose(scatter_array, 0.5)
	scatter_array = scatter_model2.mean_scatter(prim_haloprop = mass135)
	assert np.allclose(scatter_array, 0.4)

	# Test the scatter_realization method of the updated non-trivial model 
	scatter_realization = scatter_model2.scatter_realization(seed=testing_seed, prim_haloprop=mass15)
	disp = np.std(scatter_realization)
	np.testing.assert_almost_equal(disp, 0.5, decimal=2)
	scatter_realization = scatter_model2.scatter_realization(seed=testing_seed, prim_haloprop=mass135)
	disp = np.std(scatter_realization)
	np.testing.assert_almost_equal(disp, 0.4, decimal=2)
	scatter_realization = scatter_model2.scatter_realization(seed=testing_seed, prim_haloprop=mass12)
	disp = np.std(scatter_realization)
	np.testing.assert_almost_equal(disp, 0.3, decimal=2)














