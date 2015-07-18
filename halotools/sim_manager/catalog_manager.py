# -*- coding: utf-8 -*-
"""
Methods and classes for halo catalog I/O and organization.

"""

class CatalogManager(object):
	""" Class used to scrape the web for simulation data  
	and manage the set of cached catalogs. 
	"""

	def __init__(self):
		pass

	def processed_halocats_in_cache(self, **kwargs):
		pass

	def processed_halocats_available_for_download(self, **kwargs):
		pass

	def raw_halocats_in_cache(self, **kwargs):
		pass

	def raw_halocats_available_for_download(self, **kwargs):
		pass

	def ptcl_cats_in_cache(self, **kwargs):
		pass

	def ptcl_cats_available_for_download(self, **kwargs):
		pass

	def closest_matching_catalog_in_cache(self, **kwargs):
		pass

	def download_raw_halocat(self, **kwargs):
		pass

	def download_processed_halocat(self, **kwargs):
		pass

	def download_ptcl_cat(self, **kwargs):
		pass

	def retrieve_ptcl_cat_from_cache(self, **kwargs):
		pass

	def retrieve_processed_halocat_from_cache(self, **kwargs):
		pass

	def retrieve_raw_halocat_from_cache(self, **kwargs):
		pass

	def store_newly_processed_halocat(self, **kwargs):
		pass



class HaloCatalogProcessor(object):
	""" Class used to read halo catalog ASCII data, 
	produce a value-added halo catalog, and store the catalog  
	in the cache directory or other desired location. 
	"""

	def __init__(self):
		pass

	def read_raw_halocat_ascii(self, **kwargs):
		pass




