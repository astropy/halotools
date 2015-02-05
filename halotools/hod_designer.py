# -*- coding: utf-8 -*-
"""

This module provides a convenient interface 
for building composite models from a set of 
component features. 
`halotools.hod_designer` creates a set of instructions 
that is passed to `halotools.hod_factory`, 
in the form of a dictionary. The dictionary provides 
a blueprint telling the factory 
how to build an HOD model from the HOD components. 

"""

__all__ = ['HodModelDesigner']

import numpy as np


class HodModelDesigner(object):
	""" This class is used to create a blueprint to give to 
	`halotools.hod_factory`; the HOD factory will use the blueprint to 
	instantiate a composite HOD model; all of the behavior of the 
	composite model derives from a set of component models. 

	The blueprint created by `HodModelDesigner` is in the form of a dictionary. 
	The keys of this dictionary give the name of the gal_type population, 
	the values of the dictionary are themselves dictionaries giving a 
	correspondence between the type of behavior, e.g., 'occupation_model' or 
	'quiescence_model', and the class instance used to govern that behavior. 

	For clarity, consider a specific, simple example. 
	The blueprint dictionary is {'centrals':central_dict}. 
	So the blueprint has only one key, 'centrals', 
	meaning that the composite model 
	created with this  universe that will 

	This class provides a range of options for 
	how to draw up a composite model blueprint. 

	Option 1: keyword arguments are interpreted as the 
	string specifying the gal_type, and the values attached to 
	each keyword are either component model instances and/or 
	strings specifying class names of component models.  

	a set of HOD model components as input, 
	and composes their behavior in the appropriate fashion 
	for the HOD factory to understand. 
	In particular, the prime function of this class 
	is to bundle a set of input component models into a 
	model_blueprint, which is a dictionary 
	containing a set of instructions to pass to the HOD factory. 
	"""



	def __init__(self, *args, **kwargs):

		# Needs to be an attribute for halo_prof_model 
		# This should be passed to the profile_component  models 
		# to ensure that they are always passed the same underlying halo 
		# profile model

		# class instances need an attribute _example_attr_dict 
		# that provides keys for all relevant galaxy properties, 
		# such as 'stellar_mass', 'luminosity', 'quenched', etc., 
		# and values will be used to provide the information 
		# about the shape of the attribute
		# this should inherit example entries also from, 
		# for example, the halo profile model, so that the 
		# composite model directly knows the shape information 
		# of the halo_prof_model parameters. 
		
		pass





