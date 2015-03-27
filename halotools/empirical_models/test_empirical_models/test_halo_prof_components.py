#!/usr/bin/env python
from .. import halo_prof_components as hpc
from astropy import cosmology
import numpy as np
from copy import copy

__all__ = ['test_TrivialProfile','test_NFWProfile']

def test_HaloProfileModel():
	prof_model_list = hpc.__all__
	parent_class = hpc.HaloProfileModel

	# First create a list of all sub-classes to test
	component_models_to_test = []
	for clname in prof_model_list:
		cl = getattr(hpc, clname)

		if (issubclass(cl, parent_class)) & (cl != parent_class):
			component_models_to_test.append(cl)

	# Now we will test that all sub-classes inherit the correct behavior
	for model_class in component_models_to_test:
		model_instance = model_class()

		assert hasattr(model_instance, 'cosmology')
		assert isinstance(model_instance.cosmology, cosmology.FlatLambdaCDM)

		assert hasattr(model_instance, '_set_prof_param_table_dict')
		input_dict = {}
		model_instance._set_prof_param_table_dict(input_dict)
		input_dict = model_instance.prof_param_table_dict
		model_instance._set_prof_param_table_dict(input_dict)

		assert hasattr(model_instance, 'build_inv_cumu_lookup_table')
		model_instance.build_inv_cumu_lookup_table()
		assert hasattr(model_instance, 'cumu_inv_func_table')
		assert type(model_instance.cumu_inv_func_table) == np.ndarray
		assert hasattr(model_instance, 'cumu_inv_param_table_dict')
		assert type(model_instance.cumu_inv_param_table_dict) == dict
		assert hasattr(model_instance, 'func_table_indices')
		assert type(model_instance.func_table_indices) == np.ndarray


def test_TrivialProfile():
	""" Simple tests of `~halotools.empirical_models.halo_prof_components.TrivialProfile`. 

	Mostly this function checks that the each of the following attributes is present, 
	and is an empty array, list, or dictionary:

		* ``cumu_inv_func_table``

		* ``cumu_inv_func_table_dict``

		* ``cumu_inv_param_table``

		* ``cumu_inv_param_table_dict``

		* ``halo_prof_func_dict``

		* ``haloprop_key_dict``
	"""

	# Check that the initialized attributes are correct
	model_instance = hpc.TrivialProfile()
	assert model_instance.halo_prof_func_dict == {}
	assert model_instance.haloprop_key_dict == {}
	
	# Check that the lookup table attributes are correct
	model_instance.build_inv_cumu_lookup_table()
	assert len(model_instance.cumu_inv_func_table) == 0
	assert model_instance.cumu_inv_param_table_dict == {}
	assert len(model_instance.func_table_indices) == 0


def test_NFWProfile():
	""" Tests of `~halotools.empirical_models.halo_prof_components.NFWProfile`. 
	"""

	# Check that the initialized attributes are correct
	model_instance = hpc.NFWProfile()
	assert hasattr(model_instance, 'cosmology')
	assert isinstance(model_instance.cosmology, cosmology.FlatLambdaCDM)
	assert model_instance._conc_parname == 'NFWmodel_conc'

	# Check that the lookup table attributes are correct
	model_instance.build_inv_cumu_lookup_table()
	assert np.all(model_instance.cumu_inv_param_table_dict[model_instance._conc_parname] > 0)
	assert np.all(model_instance.cumu_inv_param_table_dict[model_instance._conc_parname] < 1000)
	assert len(model_instance.cumu_inv_param_table_dict[model_instance._conc_parname]) >= 10
	assert (len(model_instance.cumu_inv_func_table) == 
		len(model_instance.func_table_indices) )

	# The lookup table should adjust properly when passed an input_dict
	input_dict = copy(model_instance.prof_param_table_dict)
	input_dict[model_instance._conc_parname] = (1.0, 25.0, 0.04)
	model_instance._set_prof_param_table_dict(input_dict)
	assert model_instance.prof_param_table_dict == input_dict
	input_dict[model_instance._conc_parname] = (2.0, 20.0, 0.03)
	assert model_instance.prof_param_table_dict != input_dict
	model_instance.build_inv_cumu_lookup_table(
		prof_param_table_dict=input_dict)
	assert model_instance.prof_param_table_dict == input_dict
	dict_persistence_check = copy(model_instance.prof_param_table_dict)
	input_dict['some_irrelevant_key'] = 4
	model_instance.build_inv_cumu_lookup_table(
		prof_param_table_dict=input_dict)
	assert dict_persistence_check == model_instance.prof_param_table_dict














