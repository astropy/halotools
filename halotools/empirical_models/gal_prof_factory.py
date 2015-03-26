# -*- coding: utf-8 -*-
"""

Module containing the primary class used to build 
galaxy profiles from a set of components. 

"""

__all__ = ['GalProfModel']

import numpy as np
from scipy.interpolate import UnivariateSpline as spline

import model_defaults
from ..utils.array_utils import array_like_length as aph_len
import occupation_helpers as occuhelp 
from functools import partial
import halo_prof_components as hpc
import gal_prof_components as gpc


class GalProfModel(object):
    """ Container class for the intra-halo spatial profile of a galaxy population. 

    This class derives the vast majority of its 
    behavior from external functions and classes. 
    The main purpose of the `GalProfModel` class is to provide a standardized 
    interface for the rest of the package, particularly model factories such as 
    `~halotools.empirical_models.HodModelFactory` 
    and mock factories such as `~halotools.empirical_models.HodMockFactory`. 
    """

    def __init__(self, gal_type, halo_prof_model, spatial_bias_model = None):
        """
        Parameters 
        ----------
        gal_type : string 
            User-supplied name of the galaxy population being modeled, 
            e.g., ``sats`` or ``orphans``. 

        halo_prof_model : object 
            Instance of a concrete sub-class of 
            `~halotools.empirical_models.halo_prof_components.HaloProfileModel`. 

        spatial_bias_model : object, optional
            Instance of the class `~halotools.empirical_models.gal_prof_components.SpatialBias`. 
            Default is None. 
        """

        # Bind the inputs to the instance 
        self.gal_type = gal_type
        self.halo_prof_model = halo_prof_model
        self.cosmology = self.halo_prof_model.cosmology
        self.redshift = self.halo_prof_model.redshift
        self.spatial_bias_model = spatial_bias_model

        # Bind param_dict to the class instance 
        # param_dict contains the parameters of the model 
        # for which posteriors can be computed in a likelihood analysis.
        self._initialize_param_dict()

        self.build_inv_cumu_lookup_table(
            prof_param_table_dict=self.prof_param_table_dict)

        self.publications = []

    @property 
    def haloprop_key_dict(self):
        """ Dictionary determining the halo properties used by the model (e.g., ``mvir``).
        `haloprop_key_dict` dictionary keys are, e.g., ``prim_haloprop_key``; 
        dictionary values are strings providing the column name 
        used to extract the relevant data from a halo catalog, e.g., ``mvir``.

        Notes 
        ----- 
        The `haloprop_key_dict` bound to `GalProfModel` derives entirely from 
        the `~halotools.empirical_models.HaloProfileModel.haloprop_key_dict` attribute 
        bound to `~halotools.empirical_models.HaloProfileModel`. 

        Implemented as a read-only getter method via the ``@property`` decorator syntax. 

        """
        return self.halo_prof_model.haloprop_key_dict

    def _initialize_param_dict(self):
        """ Method binding ``param_dict`` to the class instance. 

        If galaxy profile parameters are biased relative to the 
        dark matter halo, ``param_dict`` will be derived 
        directly from the instance of 
        `~halotools.empirical_models.gal_prof_components.SpatialBias` 
        bound to `GalProfModel`. 
        If the galaxy profile is unbiased, 
        then the galaxies exactly trace the potential well of their host halo; 
        in such a case there are no free parameters, 
        and so ``param_dict`` will be an empty dictionary. 
        """

        if self.spatial_bias_model == None:
            self.param_dict = {}
        else:
            self.param_dict = self.spatial_bias_model.param_dict

    def build_inv_cumu_lookup_table(self, prof_param_table_dict={}):

        self._set_prof_param_table_dict(input_dict=prof_param_table_dict)

        self.halo_prof_model.build_inv_cumu_lookup_table(
            prof_param_table_dict=prof_param_table_dict)

        self.cumu_inv_func_table_dict = self.halo_prof_model.cumu_inv_func_table_dict
        self.cumu_inv_param_table_dict = self.halo_prof_model.cumu_inv_param_table_dict

    @property 
    def gal_prof_func_dict(self):
        """ Dictionary used as a container for 
        the functions that map galaxy profile parameter values onto dark matter halos. 

        Each dict key of `gal_prof_func_dict` corresponds to 
        the name of a halo profile parameter, but pre-pended by ``gal_``, 
        e.g., ``gal_NFWmodel_conc``. 
        Each dict value attached is a function object
        providing the mapping between ``gal_type`` galaxies 
        and their halo profile parameter. 
        For example, for the case of an underlying NFW profile, 
        the function object would just be some concentration-mass function. 

        Galaxy profile parameters may systematically differ from their underlying 
        dark matter halo profile parameters, a phenomenon governed by the 
        `~halotools.empirical_models.SpatialBias` class. 
        The possibility of spatial bias is why the 
        `gal_prof_func_dict` has different keys than the `halo_prof_func_dict`. 
        For profile parameters that are unbiased, the function object in 
        `gal_prof_func_dict` is identical to the function object in `halo_prof_func_dict`. 

        Notes 
        ----- 
        Implemented as a read-only getter method via the ``@property`` decorator syntax. 
        """
        output_dict = {}
        if self.spatial_bias_model == None:

            halo_prof_dict = self.halo_prof_model.halo_prof_func_dict
            for key, func in halo_prof_dict.iteritems():
                newkey = model_defaults.galprop_prefix + key
                output_dict[newkey] = func

        else:
            raise SyntaxError("Never finished integrating SpatialBias into galaxy profile factory")

        return output_dict

    @property 
    def halo_prof_func_dict(self):
        """ Dictionary used as a container for 
        the functions that map profile parameter values onto dark matter halos. 

        Each dict key of ``halo_prof_func_dict`` corresponds to 
        the name of a halo profile parameter, e.g., 'NFWmodel_conc'. 
        The dict value attached to each dict key is a function object
        providing the mapping between halos and the halo profile parameter, 
        such as a concentration-mass function. 

        Notes 
        ----- 
        Implemented as a read-only getter method via the ``@property`` decorator syntax. 

        The `halo_prof_func_dict` bound to `GalProfModel` 
        is not defined within the `GalProfModel` class, but instead is defined in the getter method 
        `~halotools.empirical_models.halo_prof_components.halo_prof_func_dict.HaloProfileModel.halo_prof_func_dict`
        of `~halotools.empirical_models.halo_prof_components.halo_prof_func_dict.HaloProfileModel`. 
        """

        return self.halo_prof_model.halo_prof_func_dict

    def _set_prof_param_table_dict(self,input_dict={}):
        """ Create dictionary attribute providing instructions for how to discretize 
        halo profile parameter values. 

        After calling this method, 
        the `~halotools.empirical_models.halo_prof_components.HaloProfileModel` 
        sub-class instance bound to `GalProfModel` gets a 
        ``prof_param_table_dict`` attribute that is a dictionary. 
        Each dict key of ``prof_param_table_dict`` 
        is a profile parameter name, e.g., ``NFWmodel_conc``. 
        Each dict value is a 3-element tuple; 
        the tuple entries provide, respectively, the min, max, and linear 
        spacing used to discretize the profile parameter. 
        This discretization is used by `build_inv_cumu_lookup_table` to 
        create a lookup table associated with `HaloProfileModel`. 

        Notes 
        -----
        The ``prof_param_table_dict`` dictionary can be empty, 
        as is the case for `TrivialProfile`. 

        The behavior of `_set_prof_param_table_dict` is derived entirely from 
        `~halotools.empirical_models.halo_prof_components.HaloProfileModel`, 
        or one of its sub-classes. 

        """ 
        self.halo_prof_model._set_prof_param_table_dict(input_dict)

    @property 
    def prof_param_table_dict(self):
        """ Dictionary attribute providing instructions for how to discretize 
        halo profile parameter values. 

        Each dict key of ``prof_param_table_dict`` 
        is a profile parameter name, e.g., ``NFWmodel_conc``. 
        Each dict value is a 3-element tuple; 
        the tuple entries provide, respectively, the min, max, and linear 
        spacing used to discretize the profile parameter. 

        Notes 
        -----
        This discretization is used by `build_inv_cumu_lookup_table` to 
        create a lookup table associated with `HaloProfileModel`. 

        The ``prof_param_table_dict`` dictionary can be empty, 
        as is the case for 
        `~halotools.empirical_models.halo_prof_components.TrivialProfile`. 

        Keys and values are set by the `_set_prof_param_table_dict` method. 

        """ 
        return self.halo_prof_model.prof_param_table_dict
        
    def get_prof_table_indices(self, params):
        result = np.digitize(params, self.cumu_inv_param_table)
        return result

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
        # get the appropriate slice
        gal_type_slice = mock_galaxies._gal_type_indices[self.gal_type]
        pos = getattr(mock_galaxies, 'pos')[gal_type_slice]

        if isinstance(self.halo_prof_model, hpc.TrivialProfile) is True:
            return np.zeros_like(pos)
        else:
            # get angles
            Ngals = len(pos[:,0])
            pos = self.mc_angles(Ngals)

            # get radii
            # NOTE THE HARD-CODING OF A SINGLE HALO PROFILE PARAMETER
            profile_param_key = self.gal_prof_func_dict.keys()[0]
            scaled_mc_radii = self.mc_radii(
                getattr(mock_galaxies, profile_param_key)[gal_type_slice])            
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















        





