# -*- coding: utf-8 -*-
"""

Module containing the primary class used to build 
composite HOD models from a set of components. 

"""

import numpy as np

class HOD_Model(object):
	""" The most basic HOD model object. 
	"""

	def __init__(self,component_model_dict):
		self.component_model_dict = component_model_dict
		self.gal_types = component_model_dict.keys()

	def mean_occupation(self,gal_type,*args):
		""" Method supplying the mean abundance of gal_type galaxies. 
		The behavior of this method is inherited from one of the component models. 
		"""

		self.test_component_consistency(gal_type,'occupation_model')

		# For galaxies of type gal_type, the behavior of this method 
		# will be set by the inherited occupation_model object 
		occupation_model = self.component_model_dict[gal_type]['occupation_model']
		inherited_method = occupation_model.mean_occupation
		output_occupation = self.retrieve_inherited_behavior(inherited_method,args)

		return output_occupation


	def mc_occupation(self,gal_type,*args):

		self.test_component_consistency(gal_type,'occupation_model')

		occupation_model = self.component_model_dict[gal_type]['occupation_model']
		inherited_method = occupation_model.mc_occupation
		output_mc_realization = self.retrieve_inherited_behavior(inherited_method,args)

		return output_mc_realization
		

	def mean_profile_parameters(self,gal_type,*args):

		self.test_component_consistency(gal_type,'profile_model')

		profile_model = self.component_model_dict[gal_type]['profile_model']
		inherited_method = occupation_model.mean_profile_parameters
		output_profiles = self.retrieve_inherited_behavior(inherited_method,args)

		return output_profiles

	def mc_profile(self,gal_type,*args):

		self.test_component_consistency(gal_type,'profile_model')

		profile_model = self.component_model_dict[gal_type]['profile_model']
		inherited_method = occupation_model.mc_profile
		output_mc_realization = self.retrieve_inherited_behavior(inherited_method,args)

		return output_mc_realization

	def test_component_consistency(self,gal_type,component_key):
		""" Simple tests to run to make sure that the desired behavior 
		can be found in the component model.
		"""

		if gal_type not in self.gal_types:
			raise KeyError("Input gal_type is not supported "
				"by any of the components of this composite model")			

		if component_key not in self.component_model_dict[gal_type]:
			raise KeyError("Could not find method to compute "
				" method in the provided component model")

	def retrieve_inherited_behavior(self,inherited_method,*args):

		if len(args)==1:
			primary_haloprop = args[0]
			output_occupation = inherited_method(primary_haloprop)
		elif len(args)==2:
			primary_haloprop, secondary_haloprop = args[0], args[1]
			output_occupation = inherited_method(primary_haloprop,secondary_haloprop)
		else:
			raise TypeError("Only one or two halo property inputs are supported by "
				"mean_occupation method")				














