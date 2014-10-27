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

		if gal_type not in self.gal_types:
			raise KeyError("Input gal_type is not supported "
				"by any of the components of this composite model")			

		if 'occupation_model' not in self.component_model_dict[gal_type]:
			raise KeyError("Could not find method to compute "
				" mean_occupation in the provided component model")

		# For galaxies of type gal_type, the behavior of this method 
		# will be set by the inherited occupation_model object 
		occupation_model = self.component_model_dict[gal_type]['occupation_model']

		if len(args)==1:
			primary_haloprop = args[0]
			output_occupation = occupation_model.mean_occupation(primary_haloprop)
		elif len(args)==2:
			primary_haloprop = args[0]
			secondary_haloprop = args[1]
			output_occupation = occupation_model.mean_occupation(
				primary_haloprop,secondary_haloprop)
		else:
			raise TypeError("Only one or two halo property inputs are supported by "
				"mean_occupation method")				

		return output_occupation
















