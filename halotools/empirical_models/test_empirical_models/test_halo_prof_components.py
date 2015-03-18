#!/usr/bin/env python
from .. import halo_prof_components as hpc
from astropy import cosmology
import numpy as np

__all__ = ['test_TrivialProfile','test_NFWProfile']

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

	assert hasattr(profile_model, 'cosmology')
	assert isinstance(profile_model.cosmology, cosmology.FlatLambdaCDM)

	assert type(profile_model.cumu_inv_func_table) == np.ndarray
	assert list(profile_model.cumu_inv_func_table) == []

	assert profile_model.cumu_inv_func_table_dict == {}

	assert type(profile_model.cumu_inv_param_table) == np.ndarray
	assert list(profile_model.cumu_inv_param_table) == []

	assert profile_model.cumu_inv_param_table_dict == {}

	assert profile_model.halo_prof_func_dict == {}

	assert profile_model.haloprop_key_dict == {}

	profile_model.build_inv_cumu_lookup_table()


def test_NFWProfile():
	profile_model = hpc.NFWProfile()

	assert hasattr(profile_model, 'cosmology')
	assert isinstance(profile_model.cosmology, cosmology.FlatLambdaCDM)

	assert type(profile_model.cumu_inv_param_table) == np.ndarray
	assert np.all(profile_model.cumu_inv_param_table > 0)
	assert np.all(profile_model.cumu_inv_param_table < 1000)


	assert type(profile_model.cumu_inv_func_table) == np.ndarray

	assert profile_model._conc_parname == 'NFWmodel_conc'













