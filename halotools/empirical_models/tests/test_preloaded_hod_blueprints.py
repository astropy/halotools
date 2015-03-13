#!/usr/bin/env python
from .. import preloaded_hod_blueprints
from .. import model_defaults


def test_Kravtsov04_blueprint():
	default_blueprint = preloaded_hod_blueprints.Kravtsov04()
	assert set(default_blueprint.keys()) == {'satellites','centrals'} 

	# Test the centrals
	cen_model = default_blueprint['centrals']
	assert set(cen_model.keys()) == {'profile', 'occupation'}
	cen_occ_model = cen_model['occupation']
	cen_prof_model = cen_model['profile']

	# Test the satellites
	sat_model = default_blueprint['satellites']
	assert set(sat_model.keys()) == {'profile', 'occupation'}
	sat_occ_model = sat_model['occupation']
	sat_prof_model = sat_model['profile']
