#!/usr/bin/env python
import numpy as np

from .. import preloaded_hod_blueprints
from .. import model_defaults
from .. import hod_components
from .. import gal_prof_factory
from astropy.table import Table



def test_Kravtsov04Cens():

	def test_attributes(model):
		assert isinstance(model, hod_components.OccupationComponent)
		assert model.gal_type == 'centrals'

		correct_haloprops = {'halo_boundary', 'prim_haloprop_key'}
		assert set(model.haloprop_key_dict.keys()) == correct_haloprops

		assert model.num_haloprops == 1
		assert model.occupation_bound == 1
		assert model.prim_func_dict.keys() == [None]

	def test_mean_occupation(model):

		assert hasattr(model, 'mean_occupation')

		mvir_array = np.logspace(10, 15, 10)
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
		masses = np.ones(Npts)*1.e15
		mc_occ = model.mc_occupation(masses, seed=43)
		assert set(mc_occ).issubset([0,1])
		expected_result = 1.0
		np.testing.assert_allclose(mc_occ.mean(), expected_result, rtol=1e-3, atol=1.e-3)

		# Now check that the model is ~ 0.0 when evaluated for a tiny halo
		masses = np.ones(Npts)*1.e10
		mc_occ = model.mc_occupation(masses, seed=43)
		assert set(mc_occ).issubset([0,1])
		expected_result = 0.0
		np.testing.assert_allclose(mc_occ.mean(), expected_result, rtol=1e-3, atol=1.e-3)

	#default_model = hod_components.Kravtsov04Cens()
	#test_attributes(default_model)
	#test_mean_occupation(default_model)
	#test_mc_occupation(default_model)

	










