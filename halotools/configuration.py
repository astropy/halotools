# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""This module standardizes access to
various files used throughout the halotools package. 
The only current functionality pertains to 
downloading and caching files containing halo catalog information, 
so having a dedicated config file is mostly serving as a 
scaffolding for later complexity. """

import os

class Config(object):
	""" Configuration object providing standardization of 
	a variety of cross-package settings. """

	def __init__(self):
		self.catalog_pathname = os.path.abspath('./') + '/CATALOGS/'
		self.hearin_url="http://www.astro.yale.edu/aphearin/Data_files/"








