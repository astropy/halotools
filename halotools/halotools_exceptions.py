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

		message = "Input catalog_type = ``"+catalog_type+"``\n Must be either 'raw_halos', 'halos', or 'particles'.\n"
		
		super(CatalogTypeError, self).__init__(message)


########################################

class ModelInputError(HalotoolsError):
	def __init__(self, function_name):
		message = ("Must pass one of the following keyword arguments to %s:\n"
                "``halo_table`` or  ``prim_haloprop``" % function_name)
		super(HalotoolsCacheError, self).__init__(message)





