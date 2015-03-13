#!/usr/bin/env python

import numpy as np

from .. import preloaded_hod_blueprints
from .. import model_defaults
from .. import hod_components
from .. import gal_prof_factory


def get_gal_type_model(blueprint, gal_type):
	return blueprint[gal_type]

def get_component_models(gal_type_blueprint,feature_key):
	return gal_type_blueprint[feature_key]


def test_Kravtsov04_blueprint():
	default_blueprint = preloaded_hod_blueprints.Kravtsov04()
	assert set(default_blueprint.keys()) == {'satellites','centrals'} 

	# Check thresholds are being self-consistently set
	for threshold in np.arange(-22, -17.5, 0.5):
		temp_blueprint = preloaded_hod_blueprints.Kravtsov04(threshold=threshold)
		assert (
			temp_blueprint['satellites']['occupation'].threshold ==
			temp_blueprint['centrals']['occupation'].threshold 
			)

	for gal_type in default_blueprint.keys():
		gal_type_blueprint = get_gal_type_model(default_blueprint, gal_type)
		assert set(gal_type_blueprint.keys()) == {'profile', 'occupation'}

		# Test that the component models are subclasses of the correct abstract base class
		assert isinstance(gal_type_blueprint['occupation'], 
			hod_components.OccupationComponent)
		assert isinstance(gal_type_blueprint['profile'], 
			gal_prof_factory.GalProfModel)

		component_occ = gal_type_blueprint['occupation']
		assert component_occ.gal_type == gal_type
		correct_haloprops = {'halo_boundary', 'prim_haloprop_key'}
		assert set(component_occ.haloprop_key_dict.keys()) == correct_haloprops
		assert component_occ.num_haloprops == 1
		assert (component_occ.occupation_bound == 1) or (component_occ.occupation_bound == float("inf"))
		assert (component_occ.prim_func_dict.keys() == [None])
		assert hasattr(component_occ, 'mc_occupation')
		assert hasattr(component_occ, 'mean_occupation')

		# Test the profile model component
		component_prof = gal_type_blueprint['profile']
		assert component_prof.gal_type == gal_type
		assert set(component_prof.gal_prof_param_keys).issubset(['gal_NFWmodel_conc'])
		assert np.all(component_prof.cumu_inv_param_table > 0)
		assert np.all(component_prof.cumu_inv_param_table < 105)













