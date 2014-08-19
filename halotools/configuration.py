# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""This module standardizes access to
various files used throughout the halotools package. 
Global scope functions have been modified from the 
paths methods of the astropy config sub-package.
"""
import os

class Config(object):
	""" Configuration object providing standardization of 
	a variety of cross-package settings. """

	def __init__(self):

		self.catalog_pathname = os.path.abspath('./') + '/CATALOGS/'
		self.hearin_url="http://www.astro.yale.edu/aphearin/Data_files/"










