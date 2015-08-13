# -*- coding: utf-8 -*-
"""
Classes for all Halotools-specific exceptions. 
"""
__all__ = ['HalotoolsError', 'HalotoolsCacheError', 'UnsupportedSimError', 'CatalogTypeError']


class HalotoolsError(Exception):
	""" Base class of all Halotools-specific exceptions. 
	"""
	def __init__(self, message):
		super(HalotoolsError, self).__init__(message)



########################################



class HalotoolsCacheError(HalotoolsError):
	def __init__(self, message):
		super(HalotoolsCacheError, self).__init__(message)

class HalotoolsIOError(HalotoolsError):
	def __init__(self, message):
		super(HalotoolsIOError, self).__init__(message)


class UnsupportedSimError(HalotoolsCacheError):
	def __init__(self, simname):

		message = ("\nThe input simname " + simname + " is not recognized by Halotools.\n"
			"See the supported_sim_list defined at the top of halotools.sim_manager.cache_config\n")

		super(UnsupportedSimError, self).__init__(message)


class CatalogTypeError(HalotoolsCacheError):
	def __init__(self, catalog_type):

		message = "\nInput catalog_type = ``"+catalog_type+"``\n Must be either 'raw_halos', 'halos', or 'particles'.\n"
		
		super(CatalogTypeError, self).__init__(message)


########################################

class HalotoolsModelInputError(HalotoolsError):
	def __init__(self, function_name):
		message = ("\nMust pass one of the following keyword arguments to %s:\n"
                "``halo_table`` or  ``prim_haloprop``" % function_name)
		super(HalotoolsModelInputError, self).__init__(message)

class HalotoolsArgumentError(HalotoolsError):
	def __init__(self, function_name, required_input_list):
		"""
		Parameters 
		-----------
		function_name : string 

		required_input_list : list 
			List of strings 
		"""
		message = "\nMust pass each of the following keyword arguments to " + function_name + ":\n"
		for required_input in required_input_list:
			message = message + required_input + ', '
		message = message[:-2]
		super(HalotoolsArgumentError, self).__init__(message)




