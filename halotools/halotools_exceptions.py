# -*- coding: utf-8 -*-
"""
Classes for all package-specific exceptions. 
"""
__all__ = ['HalotoolsError', 'HalotoolsCacheError', 'UnsupportedSimError', 'CatalogTypeError']

class HalotoolsError(Exception):
	pass




class HalotoolsCacheError(HalotoolsError):
	pass

class UnsupportedSimError(HalotoolsCacheError):
	def __init__(self, simname):

		message = ("\nThe input simname " + simname + " is not recognized by Halotools.\n"
			"See the supported_sim_list defined at the top of halotools.sim_manager.cache_config\n")

		super(UnsupportedSimError, self).__init__(message)


class CatalogTypeError(HalotoolsCacheError):
	def __init__(self, catalog_type):

		message = "Input catalog_type = ``"+catalog_type+"``\n Must be either 'raw_halos', 'halos', or 'particles'.\n"
		
		super(CatalogTypeError, self).__init__(message)



