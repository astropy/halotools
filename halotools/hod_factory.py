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


	def mean_occupation(self,gal_type,**kwargs):
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
		# will be set by the occupation_model object 
		# stored in the dictionary of component models passed to the constructor
		occupation_model = self.component_model_dict[gal_type]['occupation_model']

		# This method has two options for the set of supported inputs.
		# First, you can simply pass a 'halos' object, in which case 
		# there should be no other arguments besides gal_type. In this case, 
		# the following lines will identify whether there is a secondary halo property 
		# being used, and call the appropriate method of occupation_model.

		if 'halos' in kwargs.keys():

			# Forbid passing both 'halos' object and arrays of halo properties
			if primary_haloprop in kwargs.keys():
				raise IOError("Pass either halos object, or ndarrays of halo properties, "
					"to component occupation model. Passing both is ambiguous, and not permitted.")

			# Get the primary halo property array, and the secondary array, if using.
			try:
				primary_haloprop = halos[occupation_model.primary_haloprop_key]
			except KeyError:
				print("primary_haloprop_key not found in "
					"input halos object")

			# Check to see if we are using an additional halo property in this model
			if hasattr(occupation_model,'secondary_haloprop_key'):
				# If so, pass both primary and secondary arrays to occupation_model
				try:
					secondary_haloprop = halos[occupation_model.secondary_haloprop_key]
					output_occupation = (
						occupation_model.mean_occupation(
							primary_haloprop,secondary_haloprop))
				except KeyError:
					print("secondary_haloprop_key not found in input halos object")
				# If not, just pass the primary halo property to occupation_model
			else:
				output_occupation = occupation_model.mean_occupation(primary_haloprop)

		# if 'halos' object has not been passed to this method, 
		# directly use the passed ndarrays
		else:
			try:
				primary_haloprop = kwargs['primary_haloprop']
			except KeyError:
				print("If not passing a halos object to mean_occupation, "
				" you must pass a primary_haloprop argument containing a numpy array.")

			if 'secondary_haloprop' in kwargs.keys():
				secondary_haloprop = kwargs['secondary_haloprop']
				output_occupation = (
					occupation_model.mean_occupation(
						primary_haloprop,secondary_haloprop))
			else:
				output_occupation = occupation_model.mean_occupation(primary_haloprop)


		return output_occupation
















