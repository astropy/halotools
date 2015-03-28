# -*- coding: utf-8 -*-
"""

Module containing the primary class used to build 
galaxy profiles from a set of components. 

"""

__all__ = ['GalProfFactory']

import numpy as np
from scipy.interpolate import UnivariateSpline as spline

import model_defaults
from ..utils.array_utils import array_like_length as aph_len
import occupation_helpers as occuhelp 
from functools import partial

import halo_prof_components
import gal_prof_components as gpc


class GalProfFactory(object):
    """ Class modeling the way galaxies are distributed 
    within their halos. 

    `GalProfFactory` can be thought of as a factory that produces 
    model objects for the intra-halo distribution of galaxies.  
    `GalProfFactory` derives most of its 
    behavior from external functions and classes. 
    The main purpose of the `GalProfFactory` class is to provide a standardized 
    interface for the rest of the package, particularly model factories such as 
    `~halotools.empirical_models.HodModelFactory`, 
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

        Examples 
        --------
        To build a centrals-like population, with galaxies residing at exactly 
        the halo center:

        >>> halo_prof_model = halo_prof_components.TrivialProfile()
        >>> gal_type_nickname = 'centrals'
        >>> gal_prof_model = GalProfFactory(gal_type_nickname, halo_prof_model)

        For a satellite-type population distributed according to the NFW profile of the parent halo:

        >>> halo_prof_model = halo_prof_components.NFWProfile()
        >>> gal_type_nickname = 'sats'
        >>> gal_prof_model = GalProfFactory(gal_type_nickname, halo_prof_model)

        """

        # Bind the inputs to the instance 
        self.gal_type = gal_type
        self.halo_prof_model = halo_prof_model
        self.spatial_bias_model = spatial_bias_model

        # Create convenience attributes deriving from halo_prof_model
        self.cosmology = self.halo_prof_model.cosmology
        self.redshift = self.halo_prof_model.redshift

        # Bind param_dict to the class instance 
        # param_dict contains the parameters of the model 
        # for which posteriors can be inferred using a likelihood analysis.
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
        The `haloprop_key_dict` bound to `GalProfFactory` derives entirely from 
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
        bound to ``self``. 
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
        """ Method building a lookup table used to 
        rapidly generate Monte Carlo realizations of radial positions 
        within the halo. 

        Parameters 
        ----------
        prof_param_table_dict : dict, optional
            Dictionary determining how the profile parameters are discretized 
            during the building of the lookup table. If no ``prof_param_table_dict`` 
            is passed, default values for the discretization will be chosen. 
            See `set_prof_param_table_dict` for details. 

        Notes 
        -----
        All of the behavior of this method is derived from 
        ``self.halo_prof_model``. For further documentation about 
        how this method works, see the 
        `~halotools.empirical_models.HaloProfileModel.build_inv_cumu_lookup_table`
        method of the `~halotools.empirical_models.HaloProfileModel` class. 
        """
        self.set_prof_param_table_dict(input_dict=prof_param_table_dict)

        self.halo_prof_model.build_inv_cumu_lookup_table(
            prof_param_table_dict=prof_param_table_dict)

        self.cumu_inv_func_table = self.halo_prof_model.cumu_inv_func_table
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

        The purpose of `gal_prof_func_dict` is primarily for use by the 
        `~halotools.empirical_models.mock_factory` module. For example, through the use of 
        `gal_prof_func_dict`, the `~halotools.empirical_models.mock_factory.HodMockFactory` 
        can create a ``gal_NFWmodel_conc`` attribute for the mock, 
        without knowing the name of the concentration-mass function used in the assignment, 
        nor knowledge of the ``gal_NFWmodel_conc`` keyword. 
        This is one of the tricks that permits   
        `~halotools.empirical_models.mock_factory.HodMockFactory` to call 
        its component models using a uniform syntax, regardless of the complexity 
        of the underlying model. 
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

        The `halo_prof_func_dict` bound to `GalProfFactory` 
        is not defined within the `GalProfFactory` class, but instead is defined in the getter method 
        `~halotools.empirical_models.halo_prof_components.halo_prof_func_dict.HaloProfileModel.halo_prof_func_dict`
        of `~halotools.empirical_models.halo_prof_components.halo_prof_func_dict.HaloProfileModel`. 
        """

        return self.halo_prof_model.halo_prof_func_dict

    def set_prof_param_table_dict(self,input_dict={}):
        """ Create dictionary attribute providing instructions for how to discretize 
        halo profile parameter values. 

        After calling this method, 
        the `~halotools.empirical_models.halo_prof_components.HaloProfileModel` 
        sub-class instance bound to `GalProfFactory` gets a 
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

        The behavior of `set_prof_param_table_dict` is derived entirely from 
        `~halotools.empirical_models.halo_prof_components.HaloProfileModel`, 
        or one of its sub-classes. 

        """ 
        self.halo_prof_model.set_prof_param_table_dict(input_dict)

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

        Keys and values are set by the `set_prof_param_table_dict` method. 

        """ 
        return self.halo_prof_model.prof_param_table_dict

    def mc_radii(self, *args):
        """ Method to generate Monte Carlo realizations of the profile model. 

        Parameters 
        ----------
        param_array : array_like, optional position argument(s)
            Array or arrays containing the input profile parameters. 
            In the simplest case, this is a single array of 
            NFW concentration values. 
            There should be an input ``param_array`` 
            for every parameter in the profile model, 
            all of the same length. 

        Returns 
        -------
        r : array 
            Contains the radial position of galaxies within their halos, 
            scaled by the size of the halo's boundary, 
            so that :math:`0 < r < 1`. 
        """
        rho = np.random.random(len(args[0]))

        digitized_param_list = []
        for param_index, param_key in enumerate(self.halo_prof_model.prof_param_keys):
            digitized_params = np.digitize(args[param_index], 
                self.cumu_inv_param_table_dict[param_key])
            digitized_param_list.append(digitized_params)

        func_table_indices = (
            self.halo_prof_model.func_table_indices[digitized_param_list]
            )

        return 10.**occuhelp.call_func_table(
            self.cumu_inv_func_table, np.log10(rho), func_table_indices)

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

        if isinstance(self.halo_prof_model, 
            halo_prof_components.TrivialProfile) is True:
            return np.zeros_like(pos)
        else:
            # get angles
            Ngals = len(pos[:,0])
            pos = self.mc_angles(Ngals)

            # get radii
            profile_params = (
                [getattr(mock_galaxies, 
                    model_defaults.host_haloprop_prefix+profile_param_key)[gal_type_slice] 
                for profile_param_key in self.halo_prof_model.prof_param_keys]
                )

            scaled_mc_radii = self.mc_radii(*profile_params) 

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















        





