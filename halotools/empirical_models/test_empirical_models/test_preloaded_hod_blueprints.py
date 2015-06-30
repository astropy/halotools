#!/usr/bin/env python

import numpy as np

from .. import preloaded_hod_blueprints
from .. import model_defaults
from .. import hod_components
from .. import gal_prof_factory
from .. import halo_prof_components

__all__ = ['test_Kravtsov04_blueprint']


def get_gal_type_model(blueprint, gal_type):
	return blueprint[gal_type]

def get_component_models(gal_type_blueprint,feature_key):
	return gal_type_blueprint[feature_key]


def test_Kravtsov04_blueprint():
	""" Suite of tests to check the self-consistency of 
	`~halotools.empirical_models.Kravtsov04_blueprint`
	and its component models. 

	Bullet-point overview of the tests peformed is as follows:

	* Model is composed of two populations: ``centrals`` and ``satellites``

	* Both populations have the same luminosity threshold
	
	* Both populations have ``profile`` and ``occupation`` features
	
	* Satellite profiles are NFW, central profiles are trival

	Since the Kravtsov04 composite model derives all its behavior from 
	`~halotools.empirical_models.hod_components.Zheng07Cens` and 
	`~halotools.empirical_models.hod_components.Kravtsov04Sats`, 
	all further testing is relegated to 
	`~halotools.empirical_models.test_empirical_models.test_Zheng07Cens` and 
	`~halotools.empirical_models.test_empirical_models.test_Kravtsov04Sats`. 

	Examples 
	--------
	>>> from halotools.empirical_models import preloaded_hod_blueprints
	>>> blueprint  = preloaded_hod_blueprints.Kravtsov04_blueprint(threshold = -21)

	"""
	default_blueprint = preloaded_hod_blueprints.Kravtsov04_blueprint()
	assert {'satellites','centrals'}.issubset(set(default_blueprint.keys()))

	# Check thresholds are being self-consistently set
	for threshold in np.arange(-22, -17.5, 0.5):
		temp_blueprint = preloaded_hod_blueprints.Kravtsov04_blueprint(threshold=threshold)
		assert (
			temp_blueprint['satellites']['occupation'].threshold ==
			temp_blueprint['centrals']['occupation'].threshold 
			)

	gal_type_list = [key for key in default_blueprint.keys() if key != 'mock_factory']
	for gal_type in gal_type_list:
		gal_type_blueprint = get_gal_type_model(default_blueprint, gal_type)
		assert set(gal_type_blueprint.keys()) == {'profile', 'occupation'}

		# Test that the component models are subclasses of the correct abstract base class
		assert isinstance(gal_type_blueprint['occupation'], 
			hod_components.OccupationComponent)

		assert (
			isinstance(gal_type_blueprint['profile'], gal_prof_factory.SphericallySymmetricGalProf) or 
			isinstance(gal_type_blueprint['profile'], halo_prof_components.HaloProfileModel)
			)


		# Test the profile model component
		if isinstance(gal_type_blueprint['profile'], gal_prof_factory.SphericallySymmetricGalProf):
			component_prof = gal_type_blueprint['profile']
			assert component_prof.gal_type == gal_type
			assert set(component_prof.prof_param_keys).issubset(['NFWmodel_conc'])













