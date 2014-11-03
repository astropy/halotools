# -*- coding: utf-8 -*-
"""

This module contains the components for 
the intra-halo spatial positions of galaxies 
used by `halotools.hod_designer` to build composite HOD models 
by composing the behavior of the components. 

"""

from astropy.extern import six
from abc import ABCMeta, abstractmethod
import numpy as np

@six.add_metaclass(ABCMeta)
class Halo_Profile_Model(object):
	""" Container class for any halo profile.
	"""

	def __init__(self):
		pass

	@abstractmethod
	def density_profile(self,x,*args):
		""" Intra-halo density profile. 

		Parameters 
		----------
		x : array_like
			Input value of the halo-centric distance, 
			scaled by the size of the halo so that :math:`0 < x < 1`.

		args : array_like
			Parameters of the profile. 

		Returns 
		-------
		density : array_like
			For a density profile whose behavior is determined by the input args, 
			the output is the value of that density profile evaluated at the input x. 
		"""
		pass

	def cumulative_profile(self,x,*args):
		""" Cumulative density profile. 

		Parameters 
		----------
		x : array_like
			Input value of the halo-centric distance, 
			scaled by the size of the halo so that :math:`0 < x < 1`.

		args : array_like
			Parameters of the profile. 

		Returns 
		-------
		cumulative_density : array_like
			For a density profile whose behavior is determined by the input args, 
			the output is the value of the cumulative density evaluated at the input x. 

		Notes 
		-----
		The generic behavior of this method derives from integrating the `density_profile` method. 
		However, this behavior should be over-ridden by subclasses of profiles 
		where the integral can be done analytically, such as for NFW profiles. 

		"""

		# (1) code for interpolating from lookup table
		# code for integrating self.density_profile
		# will be over-ridden 

		pass

	def lookup_cumulative_profile(self):
		""" Method for evaluating the cumulative profile 
		from a pre-computed lookup table.

		First, check the lookup table cache 
		directory to see if the function has been pre-computed. 
		If the lookup table is there, it is loaded into memory 
		and the cumulative profile is evaluated via interpolation. 
		If the lookup table is not there, a warning is issued and
		`cumulative_profile` is used to generate the table, 
		which is then stored in cache and evaluated.
		""" 
		pass

	def inverse_cumulative_profile(self):
		pass


































