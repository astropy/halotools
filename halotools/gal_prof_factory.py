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

        self.set_param_dict()

        self.set_gal_prof_func_dict()

    def set_param_dict(self):

        if self.spatial_bias_model == None:
            self.param_dict = {}
        else:
            self.param_dict = self.spatial_bias_model.param_dict

    def update_param_dict(self, new_param_dict):

        if self.spatial_bias_model == None:
            pass
        else:
            self.spatial_bias_model.update_param_dict(new_param_dict)
            self.param_dict = self.spatial_bias_model.param_dict

    def _set_prof_params(self):
        self.halo_prof_param_keys = (
            self.halo_prof_model.halo_prof_func_dict.keys()
            )
        self.gal_prof_param_keys = (
            [defaults.galprop_prefix+key for key in self.halo_prof_param_keys]
            )

    def build_inv_cumu_lookup_table(self, prof_param_table_dict=None):
        self.halo_prof_model.build_inv_cumu_lookup_table(
            prof_param_table_dict=prof_param_table_dict)

        self.prof_param_table_dict = self.halo_prof_model.prof_param_table_dict
        self.cumu_inv_func_table = self.halo_prof_model.cumu_inv_func_table
        self.cumu_inv_param_table = self.halo_prof_model.cumu_inv_param_table

    def set_gal_prof_func_dict(self):
        if self.spatial_bias_model == None:
            self.gal_prof_func_dict = {}
        else:
            self.gal_prof_func_dict = self.spatial_bias_model.prim_func_dict

    def set_halo_prof_func_dict(self, input_dict):
        self.halo_prof_model.set_halo_prof_func_dict(input_dict)

    def set_prof_param_table_dict(self,input_dict=None):
        self.halo_prof_model.set_prof_param_table_dict(input_dict)
        self.prof_param_table_dict = self.halo_prof_model.prof_param_table_dict
        
    def get_prof_table_indices(self, params):
        return np.digitize(params, self.cumu_inv_param_table)

    def get_scaled_radii_from_func_table(self, rho, profile_params):
        func_table_indices = self.get_prof_table_indices(profile_params)
        prof_func_array = self.cumu_inv_func_table[func_table_indices]
        return occuhelp.call_func_table(
            self.cumu_inv_func_table, rho, func_table_indices)

    def mc_radii(self, profile_params):
        """ args is a tuple of profile parameter arrays. In the simplest case, 
        this is a one-element tuple of concentration values. 
        """
        rho = np.random.random(len(profile_params))
        return self.get_scaled_radii_from_func_table(rho, profile_params)

    def mc_angles(self, Npts):
        """ Returns Npts random points on the unit sphere. 

        Parameters 
        ----------
        pos : array_like  
            Array with shape (Npts, 3) of points. 
            Method over-writes this array with points on the unit sphere. 

        """

        pos = np.zeros(Npts*3).reshape(Npts,3)
        cos_t = np.random.uniform(-1.,1.,Npts)
        phi = np.random.uniform(0,2*np.pi,Npts)
        sin_t = np.sqrt((1.-cos_t*cos_t))

        pos[:,0] = sin_t * np.cos(phi)
        pos[:,1] = sin_t * np.sin(phi)
        pos[:,2] = cos_t

        return pos

    def mc_pos(self, mock_galaxies):

        if isinstance(self.halo_prof_model, hpc.TrivialProfile) is True:
            return 0
        else:
            # get the appropriate slice
            gal_type_slice = mock_galaxies._gal_type_indices[self.gal_type]
            pos = getattr(mock_galaxies, 'pos')[gal_type_slice]
            # get angles
            Ngals = len(pos[:,0])
            pos = self.mc_angles(Ngals)

            # get radii
            # NOTE THE HARD-CODING OF A SINGLE HALO PROFILE PARAMETER
            profile_param_key = self.gal_prof_param_keys[0]
            scaled_mc_radii = self.mc_radii(getattr(mock_galaxies, profile_param_key)[gal_type_slice])
            # multiply radii by angles 
            for idim in range(3): pos[:,idim] *= scaled_mc_radii

        return pos

#################
### Include the following two methods for completeness
    def density_profile(self, *args):
        return self.halo_prof_model.density_profile(*args)

    def cumulative_mass_PDF(self, *args):
        return self.halo_prof_model.cumulative_mass_PDF(*args)
#################















        





