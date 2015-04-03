#!/usr/bin/env python

import numpy as np 
from .. import preloaded_models
from .. import hod_factory

__all__ = ['test_Kravtsov04_composite']

def test_Kravtsov04_composite():
	""" Method to test the basic behavior of 
	`~halotools.empirical_models.preloaded_models.Kravtsov04`, 
	a specific pre-loaded model of 
	`~halotools.empirical_models.hod_factory.HodModelFactory`. 

	The suite includes the following tests:

		* Changes to ``self.param_dict`` properly propagate through to occupation component models. 

		* Default behavior is recovered after calling the `~halotools.empirical_models.hod_factory.HodModelFactory.restore_init_param_dict` method. 
	"""
	model = preloaded_models.Kravtsov04(threshold = -18)

	# Verify that changes param_dict properly propagate
	testmass1 = 5.e11
	cenocc_orig = model.mean_occupation_centrals(testmass1)
	model.param_dict['logMmin_centrals'] = 11.5
	cenocc_new = model.mean_occupation_centrals(testmass1)
	assert cenocc_new < cenocc_orig

	testmass2 = 5.e12
	satocc_orig = model.mean_occupation_satellites(testmass2)
	model.param_dict['logM0_satellites'] = 11.4
	satocc_new = model.mean_occupation_satellites(testmass2)
	assert satocc_new < satocc_orig

	# Test that we can recover our initial behavior
	model.restore_init_param_dict()
	cenocc_restored = model.mean_occupation_centrals(testmass1)
	assert cenocc_restored == cenocc_orig
	satocc_restored = model.mean_occupation_satellites(testmass2)
	assert satocc_restored == satocc_orig