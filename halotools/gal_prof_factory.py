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

		self.spatial_bias_model = spatial_bias_model

	def build_inv_cumu_lookup_table(self, prof_param_table_dict=None):
		self.halo_prof_model.build_inv_cumu_lookup_table(
			prof_param_table_dict=prof_param_table_dict)

		self.prof_param_table_dict = self.halo_prof_model.prof_param_table_dict
		self.cumu_inv_func_table = self.halo_prof_model.cumu_inv_func_table
		self.cumu_inv_param_table = self.halo_prof_model.cumu_inv_param_table

	def set_param_func_dict(self, input_dict):
		self.halo_prof_model.set_param_func_dict(input_dict)
		self.param_func_dict = self.halo_prof_model.param_func_dict

	def set_prof_param_table_dict(self,input_dict=None):
		self.halo_prof_model.set_prof_param_table_dict(input_dict)
		self.prof_param_table_dict = self.halo_prof_model.prof_param_table_dict

	def density_profile(self, *args):
		return self.halo_prof_model.density_profile(*args)

	def cumulative_mass_PDF(self, *args):
		return self.halo_prof_model.cumulative_mass_PDF(*args)

	def get_discretized_prof_funcs(self, params):
		return self.cumu_inv_func_table[np.digitize(params, self.cumu_inv_param_table)]

	







		





