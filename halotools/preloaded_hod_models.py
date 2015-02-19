# -*- coding: utf-8 -*-
"""

Module containing some commonly used composite HOD models.

"""

import defaults
import hod_components as hoc


def Kravtsov04(**kwargs):

	if 'threshold' in kwargs.keys():
		threshold = kwargs['threshold']
	else:
		threshold = defaults.default_luminosity_threshold

	cen_model_dict = {}
	cen_key = 'centrals'
	dark_side_cen_model = hoc.Kravtsov04Cens(gal_type=cen_key, 
		threshold = threshold)
	cen_model_dict['occupation_model'] = dark_side_cen_model

	sat_model_dict = {}
	sat_key = 'satellites'
	dark_side_sat_model = hoc.Kravtsov04Sats(gal_type=sat_key, 
		threshold = threshold)
	sat_model_dict['occupation_model'] = dark_side_sat_model


	vdB03_cen_quiescence_model = hoc.vdB03Quiescence(
		dark_side_cen_model.gal_type)
	cen_model_dict['quiescence_model'] = vdB03_cen_quiescence_model

	vdB03_sat_quiescence_model = hoc.vdB03Quiescence(
		dark_side_sat_model.gal_type)
	sat_model_dict['quiescence_model'] = vdB03_sat_quiescence_model


	cen_model_dict['profile'] = None
	sat_model_dict['profile'] = None


	model_blueprint = {
		dark_side_cen_model.gal_type : cen_model_dict,
		dark_side_sat_model.gal_type : sat_model_dict
		}

	return model_blueprint















