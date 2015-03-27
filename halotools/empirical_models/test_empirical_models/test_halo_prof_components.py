#!/usr/bin/env python
from .. import halo_prof_components as hpc
from astropy import cosmology
import numpy as np

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

		assert hasattr(model_instance, 'cumu_inv_func_table')
		assert type(model_instance.cumu_inv_func_table) == np.ndarray

		assert hasattr(model_instance, 'cumu_inv_param_table_dict')
		assert type(model_instance.cumu_inv_param_table_dict) == dict

		assert hasattr(model_instance, 'build_inv_cumu_lookup_table')
		model_instance.build_inv_cumu_lookup_table()







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
	profile_model = hpc.TrivialProfile()
	
	assert len(profile_model.cumu_inv_func_table) == 0

	assert profile_model.cumu_inv_param_table_dict == {}

	assert profile_model.halo_prof_func_dict == {}

	assert profile_model.haloprop_key_dict == {}

	profile_model.build_inv_cumu_lookup_table()


def test_NFWProfile():
	""" Tests of `~halotools.empirical_models.halo_prof_components.NFWProfile`. 
	"""

	profile_model = hpc.NFWProfile()

	assert hasattr(profile_model, 'cosmology')
	assert isinstance(profile_model.cosmology, cosmology.FlatLambdaCDM)

	assert type(profile_model.cumu_inv_param_table_dict) == dict
	assert np.all(profile_model.cumu_inv_param_table_dict[profile_model._conc_parname] > 0)
	assert np.all(profile_model.cumu_inv_param_table_dict[profile_model._conc_parname] < 1000)
	assert len(profile_model.cumu_inv_param_table_dict[profile_model._conc_parname]) >= 10


	assert type(profile_model.cumu_inv_func_table) == np.ndarray

	assert profile_model._conc_parname == 'NFWmodel_conc'













