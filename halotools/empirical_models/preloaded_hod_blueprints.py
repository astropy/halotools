# -*- coding: utf-8 -*-
"""

Module containing some commonly used composite HOD model blueprints.

"""

from . import model_defaults
from . import hod_components as hoc
from . import gal_prof_factory as gpf
from . import halo_prof_components as hpc
from . import gal_prof_components as gpc
from . import mock_factories

__all__ = ['Zheng07_blueprint']


def Zheng07_blueprint(**kwargs):
	""" Blueprint for the simplest pre-loaded HOD model. 
	There are two populations, 
	centrals and satellites, with occupation statistics, 
	positions and velocities based on Kravtsov et al. (2004). 

	Documentation of the test suite of this blueprint can be found at 
	`~halotools.empirical_models.test_empirical_models.test_Zheng07_blueprint`

	Parameters 
	----------
	threshold : float, optional 
		Luminosity threshold of the galaxy sample being modeled. 

	Returns 
	-------
	model_blueprint : dict 
		Dictionary containing instructions for how to build the model. 
		When model_blueprint is passed to `~halotools.empirical_models.HodModelFactory`, 
		the factory returns the Zheng07 model object. 

	Examples 
	--------
	>>> from halotools.empirical_models import preloaded_hod_blueprints
	>>> blueprint = preloaded_hod_blueprints.Zheng07_blueprint()
	>>> blueprint  = preloaded_hod_blueprints.Zheng07_blueprint(threshold = -19)
	"""

	if 'threshold' in kwargs.keys():
		threshold = kwargs['threshold']
	else:
		threshold = model_defaults.default_luminosity_threshold

	### Build model for centrals
	cen_key = 'centrals'
	cen_model_dict = {}
	# Build the occupation model
	dark_side_cen_model = hoc.Zheng07Cens(gal_type=cen_key, 
		threshold = threshold)
	cen_model_dict['occupation'] = dark_side_cen_model
	# Build the profile model
	cen_profile = gpf.IsotropicGalProf(
		gal_type=cen_key, halo_prof_model=hpc.TrivialProfile)
	cen_model_dict['profile'] = cen_profile

	### Build model for satellites
	sat_key = 'satellites'
	sat_model_dict = {}
	# Build the occupation model
	dark_side_sat_model = hoc.Zheng07Sats(gal_type=sat_key, 
		threshold = threshold)
	sat_model_dict['occupation'] = dark_side_sat_model
	# Build the profile model
	sat_profile = gpf.IsotropicGalProf(
		gal_type=sat_key, halo_prof_model=hpc.NFWProfile)
	sat_model_dict['profile'] = sat_profile

	model_blueprint = {
		dark_side_cen_model.gal_type : cen_model_dict,
		dark_side_sat_model.gal_type : sat_model_dict, 
		'mock_factory' : mock_factories.HodMockFactory
		}

	return model_blueprint















