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

    def set_prof_param_table_dict(self,input_dict=None):
        self.halo_prof_model.set_prof_param_table_dict(input_dict)
        self.prof_param_table_dict = self.halo_prof_model.prof_param_table_dict

    def density_profile(self, *args):
        return self.halo_prof_model.density_profile(*args)

    def cumulative_mass_PDF(self, *args):
        return self.halo_prof_model.cumulative_mass_PDF(*args)

    def get_prof_table_indices(self, params):
        return np.digitize(params, self.cumu_inv_param_table)

    def get_scaled_radii_from_func_table(self, rho, profile_params):
        func_table_indices = self.get_prof_table_indices(profile_params)
        prof_func_array = self.cumu_inv_func_table[func_table_indices]
        return occuhelp.call_func_table(
            self.cumu_inv_func_table, rho, func_table_indices)

    def mc_angles(self, Npts):
        """ Returns Npts random points on the unit sphere. 

        Parameters 
        ----------
        Npts : int 
            Number of desired points. 

        Returns 
        -------
        output_pos : array_like 
            3-D coordinates on the unit sphere. Output_pos has shape (Npts, 3). 
        """

        cos_t = np.random.uniform(-1.,1.,Npts)
        phi = np.random.uniform(0,2*np.pi,Npts)
        sin_t = np.sqrt((1.-cos_t**2))

        output_pos = np.zeros(Npts*3).reshape(Npts,3)
        output_pos[:,0] = sin_t * np.cos(phi)
        output_pos[:,1] = sin_t * np.sin(phi)
        output_pos[:,2] = cos_t

        return output_pos

    def mc_radii(self, *args):
        """ args is a tuple of profile parameter arrays. In the simplest case, 
        this is a one-element tuple of concentration values. 
        """
        rho = np.random.random(len(args[0]))
        return self.get_scaled_radii_from_func_table(rho, *args)

    def mc_pos(self, gals):
        Npts = len(gals.halo_rvir)
        angles = self.mc_angles(Npts)
        radii = None # need to know what the concentration attribute name is
















        





