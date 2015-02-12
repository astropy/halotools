# -*- coding: utf-8 -*-
"""

Module containing the primary class used to build 
galaxy profiles from a set of components. 

"""

import numpy as np
from scipy.interpolate import UnivariateSpline as spline

import defaults
from utils.array_utils import array_like_length as aph_len
import occupation_helpers as occuhelp 
from functools import partial
import halo_prof_components as hpc
import gal_prof_components as gpc


class GalProfModel(object):

	def __init__(self, gal_type, halo_prof_model,
		spatial_bias_model = None,
		):

		self.gal_type = gal_type
		self.halo_prof_model = halo_prof_model
		self.cosmology = self.halo_prof_model.cosmology
		self.redshift = self.halo_prof_model.redshift




