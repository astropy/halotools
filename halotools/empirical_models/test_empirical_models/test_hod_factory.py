#!/usr/bin/env python

import numpy as np 
from copy import copy 

from .. import preloaded_models
from .. import model_factories
from .. import hod_components

from ...sim_manager import FakeSim

__all__ = ['test_Zheng07_composite']

def test_Zheng07_composite():
	""" Method to test the basic behavior of 
	`~halotools.empirical_models.preloaded_models.Zheng07`, 
	a specific pre-loaded model of 
	`~halotools.empirical_models.model_factories.HodModelFactory`. 

	The suite includes the following tests:

		* Changes to ``self.param_dict`` properly propagate through to occupation component models. 

		* Default behavior is recovered after calling the `~halotools.empirical_models.model_factories.HodModelFactory.restore_init_param_dict` method. 
	"""
	model = preloaded_models.Zheng07(threshold = -18)

	# Verify that changes param_dict properly propagate
	testmass1 = 5.e11
	cenocc_orig = model.mean_occupation_centrals(prim_haloprop=testmass1)
	orig_logMmin_centrals = model.param_dict['logMmin_centrals']
	model.param_dict['logMmin_centrals'] = 11.5
	cenocc_new = model.mean_occupation_centrals(prim_haloprop=testmass1)
	assert cenocc_new < cenocc_orig

	testmass2 = 5.e12
	satocc_orig = model.mean_occupation_satellites(prim_haloprop=testmass2)
	model.param_dict['logM0_satellites'] = 11.4
	satocc_new = model.mean_occupation_satellites(prim_haloprop=testmass2)
	assert satocc_new < satocc_orig

	# Test that we can recover our initial behavior
	model.restore_init_param_dict()
	assert model.param_dict['logMmin_centrals'] == orig_logMmin_centrals
	cenocc_restored = model.mean_occupation_centrals(prim_haloprop=testmass1)
	assert cenocc_restored == cenocc_orig
	satocc_restored = model.mean_occupation_satellites(prim_haloprop=testmass2)
	assert satocc_restored == satocc_orig

	#######################################################
	fakesim = FakeSim()
	model.populate_mock(snapshot = fakesim)


def test_alt_Zheng07_composites():

	# First build two models that are identical except for the satellite occupations
	default_model = preloaded_models.Zheng07()
	default_model_blueprint = default_model._input_model_blueprint
	default_satocc_component = default_model_blueprint['satellites']['occupation']
	assert not hasattr(default_satocc_component, 'ancillary_model_dependencies')
	cenmod_satocc_compoent = hod_components.Zheng07Sats(
		threshold = default_satocc_component.threshold, modulate_with_cenocc = True, 
		gal_type_centrals = 'centrals')
	assert hasattr(cenmod_satocc_compoent, 'ancillary_model_dependencies')
	cenmod_model_blueprint = copy(default_model_blueprint)
	cenmod_model_blueprint['satellites']['occupation'] = cenmod_satocc_compoent
	cenmod_model = model_factories.HodModelFactory(cenmod_model_blueprint)

	# Now we test whether changes to the param_dict keys of the composite model 
	# that pertain to the centrals properly propagate through to the behavior 
	# of the satellites, only for cases where satellite occupations are modulated 
	# by central occupations 
	assert set(cenmod_model.param_dict) == set(default_model.param_dict)

	nsat1 = default_model.mean_occupation_satellites(prim_haloprop = 2.e12)
	nsat2 = cenmod_model.mean_occupation_satellites(prim_haloprop = 2.e12)
	assert nsat2 < nsat1

	cenmod_model.param_dict['logMmin_centrals'] *= 1.1
	nsat3 = cenmod_model.mean_occupation_satellites(prim_haloprop = 2.e12)
	assert nsat3 < nsat2

	nsat3 = default_model.mean_occupation_satellites(prim_haloprop = 2.e12)
	default_model.param_dict['logMmin_centrals'] *= 1.1
	nsat4 = default_model.mean_occupation_satellites(prim_haloprop = 2.e12)
	assert nsat3 == nsat4

	fakesim = FakeSim()
	cenmod_model.populate_mock(snapshot = fakesim)
	default_model.populate_mock(snapshot = fakesim)


def test_Leauthaud11_composite():
	""" Method to test the basic behavior of 
	`~halotools.empirical_models.preloaded_models.Zheng07`, 
	a specific pre-loaded model of 
	`~halotools.empirical_models.model_factories.HodModelFactory`. 

	The suite includes the following tests:

		* Changes to ``self.param_dict`` properly propagate through to occupation component models. 

		* Default behavior is recovered after calling the `~halotools.empirical_models.model_factories.HodModelFactory.restore_init_param_dict` method. 
	"""
	model = preloaded_models.Leauthaud11(threshold = 10.5)

	# Verify that changes param_dict properly propagate
	testmass1 = 5.e11
	ncen1 = model.mean_occupation_centrals(prim_haloprop=testmass1)
	nsat1 = model.mean_occupation_satellites(prim_haloprop=testmass1)
	model.param_dict['n10_centrals'] *= 1.1
	ncen2 = model.mean_occupation_centrals(prim_haloprop=testmass1)
	nsat2 = model.mean_occupation_satellites(prim_haloprop=testmass1)
	assert ncen2 > ncen1
	assert nsat2 > nsat1

	model.param_dict['n11_centrals'] *= 1.1
	ncen3 = model.mean_occupation_centrals(prim_haloprop=testmass1)
	nsat3 = model.mean_occupation_satellites(prim_haloprop=testmass1)
	assert ncen3 == ncen2
	assert nsat3 == nsat2

	fakesim = FakeSim()
	model.populate_mock(snapshot = fakesim)


	"""
	orig_logMmin_centrals = model.param_dict['logMmin_centrals']
	model.param_dict['logMmin_centrals'] = 11.5
	cenocc_new = model.mean_occupation_centrals(prim_haloprop=testmass1)
	assert cenocc_new < cenocc_orig

	testmass2 = 5.e12
	satocc_orig = model.mean_occupation_satellites(prim_haloprop=testmass2)
	model.param_dict['logM0_satellites'] = 11.4
	satocc_new = model.mean_occupation_satellites(prim_haloprop=testmass2)
	assert satocc_new < satocc_orig

	# Test that we can recover our initial behavior
	model.restore_init_param_dict()
	assert model.param_dict['logMmin_centrals'] == orig_logMmin_centrals
	cenocc_restored = model.mean_occupation_centrals(prim_haloprop=testmass1)
	assert cenocc_restored == cenocc_orig
	satocc_restored = model.mean_occupation_satellites(prim_haloprop=testmass2)
	assert satocc_restored == satocc_orig
	"""













