#!/usr/bin/env python
from .. import halo_prof_components as hpc
from astropy import cosmology
import numpy as np

__all__ = ['test_NFWProfile','test_TrivialProfile']

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
	trivial = hpc.TrivialProfile()

	assert hasattr(trivial, 'cosmology')
	assert isinstance(trivial.cosmology, cosmology.FlatLambdaCDM)

	assert type(trivial.cumu_inv_func_table) == np.ndarray
	assert list(trivial.cumu_inv_func_table) == []

	assert trivial.cumu_inv_func_table_dict == {}

	assert type(trivial.cumu_inv_param_table) == np.ndarray
	assert list(trivial.cumu_inv_param_table) == []

	assert trivial.cumu_inv_param_table_dict == {}

	assert trivial.halo_prof_func_dict == {}

	assert trivial.haloprop_key_dict == {}

	trivial.build_inv_cumu_lookup_table()


def test_NFWProfile():
	nfw = hpc.NFWProfile()
	assert nfw._conc_parname == 'NFWmodel_conc'