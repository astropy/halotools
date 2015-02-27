# -*- coding: utf-8 -*-
"""

Module containing some commonly used composite HOD models.

"""

import model_defaults
import hod_components as hoc
import gal_prof_factory as gpf
import halo_prof_components as hpc
import gal_prof_components as gpc


def Kravtsov04(**kwargs):

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
	sat_model_dict['profile'] = halo_profile_model_sats


	model_blueprint = {
		dark_side_cen_model.gal_type : cen_model_dict,
		dark_side_sat_model.gal_type : sat_model_dict
		}

	return model_blueprint















