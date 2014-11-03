# -*- coding: utf-8 -*-
"""

This module contains the components for 
the intra-halo spatial positions of galaxies 
used by `halotools.hod_designer` to build composite HOD models 
by composing the behavior of the components. 

"""

from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np

@six.add_metaclass(ABCMeta)
class Halo_Profile_Model(object):
	""" Container class for any halo profile.
	"""

	def __init__(self):
		pass

	def profile(self):
		pass

	def cumulative_profile(self):
		pass

	def inverse_cumulative_profile(self):
		pass


































