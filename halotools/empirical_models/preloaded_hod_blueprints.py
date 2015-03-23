# -*- coding: utf-8 -*-
"""

Module containing some commonly used composite HOD model blueprints.

"""

import model_defaults
import hod_components as hoc
import gal_prof_factory as gpf
import halo_prof_components as hpc
import gal_prof_components as gpc

__all__ = ['Kravtsov04_blueprint']


def Kravtsov04_blueprint(**kwargs):
	""" Blueprint for the simplest pre-loaded HOD model. 
	There are two populations, 
	centrals and satellites, with occupation statistics, 
	positions and velocities based on Kravtsov et al. (2004). 

	Documentation of the test suite of this blueprint can be found at 
	`~halotools.empirical_models.test_empirical_models.test_Kravtsov04_blueprint`

	Parameters 
	----------
	threshold : float, optional 
		Luminosity threshold of the galaxy sample being modeled. 

	Returns 
	-------
	model_blueprint : dict 
		Dictionary containing instructions for how to build the model. 
		When model_blueprint is passed to `~halotools.empirical_models.HodModelFactory`, 
		the factory returns the Kravtsov04 model object. 

	Examples 
	--------
	>>> from halotools.empirical_models import preloaded_hod_blueprints
	>>> blueprint = preloaded_hod_blueprints.Kravtsov04_blueprint()
	>>> blueprint  = preloaded_hod_blueprints.Kravtsov04_blueprint(threshold = -19)
	"""

	if 'threshold' in kwargs.keys():
		threshold = kwargs['threshold']
	else:
		threshold = model_defaults.default_luminosity_threshold

	### Build model for centrals
	cen_key = 'centrals'
	cen_model_dict = {}
	# Build the occupation model
	dark_side_cen_model = hoc.Kravtsov04Cens(gal_type=cen_key, 
		threshold = threshold)
	cen_model_dict['occupation'] = dark_side_cen_model
	# Build the profile model
	halo_profile_model_cens = hpc.TrivialProfile()
	cen_profile = gpf.GalProfModel(cen_key, halo_profile_model_cens)
	cen_model_dict['profile'] = cen_profile

	### Build model for satellites
	sat_key = 'satellites'
	sat_model_dict = {}
	# Build the occupation model
	dark_side_sat_model = hoc.Kravtsov04Sats(gal_type=sat_key, 
		threshold = threshold)
	sat_model_dict['occupation'] = dark_side_sat_model
	# Build the profile model
	halo_profile_model_sats = hpc.NFWProfile()
	sat_profile = gpf.GalProfModel(sat_key, halo_profile_model_sats)
	sat_model_dict['profile'] = sat_profile


	model_blueprint = {
		dark_side_cen_model.gal_type : cen_model_dict,
		dark_side_sat_model.gal_type : sat_model_dict
		}

	return model_blueprint















