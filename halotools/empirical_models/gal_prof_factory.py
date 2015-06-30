# -*- coding: utf-8 -*-
"""

Module containing the primary class used to build 
galaxy profiles from a set of components. 

"""

__all__ = ['SphericallySymmetricGalProf']

import numpy as np
from scipy.interpolate import UnivariateSpline as spline

import model_defaults
from ..utils.array_utils import array_like_length as custom_len
from . import model_helpers 
from functools import partial

import halo_prof_components
import gal_prof_components as gpc


class SphericallySymmetricGalProf(halo_prof_components.HaloProfileModel):
    """ Class modeling the way galaxies are distributed 
    within their halos. 

    `SphericallySymmetricGalProf` can be thought of as a factory that produces 
    model objects for the intra-halo distribution of galaxies.  
    `SphericallySymmetricGalProf` derives most of its 
    behavior from external functions and classes. 
    The main purpose of the `SphericallySymmetricGalProf` class is to provide a standardized 
    interface for the rest of the package, particularly model factories such as 
    `~halotools.empirical_models.HodModelFactory`, 
    and mock factories such as `~halotools.empirical_models.HodMockFactory`. 

    """

    def __init__(self, **kwargs):
        """
        Parameters 
        ----------
        halo_prof_model : class 
            Any sub-class of 
            `~halotools.empirical_models.halo_prof_components.HaloProfileModel`. 

        cosmology : object, optional keyword argument 
            Astropy cosmology object. Default cosmology is WMAP5. 

        redshift : float, optional keyword argument 
            Default redshift is 0.

        halo_boundary : string, optional keyword argument 
            String giving the column name of the halo catalog that stores the 
            boundary of the halo. Default is set in 
            the `~halotools.empirical_models.model_defaults` module. 

        conc_mass_model : string, optional keyword argument  
            String specifying which concentration-mass relation is used to paint model 
            concentrations onto simulated halos. 
            Default string/model is set in `~halotools.empirical_models.model_defaults`.

        gal_type : string 
            Name of the galaxy population being modeled, e.g., ``sats`` or ``orphans``. 

        spatial_bias_model : object, optional
            Instance of the class `~halotools.empirical_models.gal_prof_components.SpatialBias`. 
            Default is None. 

        Examples 
        --------
        To build a centrals-like population, with galaxies residing at exactly 
        the halo center:

        >>> halo_prof_model = halo_prof_components.TrivialProfile()
        >>> gal_type_nickname = 'centrals'
        >>> gal_prof_model = SphericallySymmetricGalProf(gal_type_nickname, halo_prof_model)

        For a satellite-type population distributed according to the NFW profile of the parent halo:

        >>> halo_prof_model = halo_prof_components.NFWProfile()
        >>> gal_type_nickname = 'sats'
        >>> gal_prof_model = SphericallySymmetricGalProf(gal_type_nickname, halo_prof_model)

        """

        self.halo_prof_model = halo_prof_model(**kwargs)

        super(SphericallySymmetricGalProf, self).__init__(
            prof_param_keys=self.halo_prof_model.prof_param_keys, **kwargs)

        required_kwargs = ['gal_type', 'halo_prof_model']
        model_helpers.bind_required_kwargs(required_kwargs, self, **kwargs)

        optional_attrs = ['cosmology', 'redshift']
        for attr in optional_attrs:
            if hasattr(self.halo_prof_model, attr):
                setattr(self, attr, getattr(self.halo_prof_model, attr))

        self.build_inv_cumu_lookup_table()

        self.publications = []

    def build_inv_cumu_lookup_table(self):
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

        self.halo_prof_model.build_inv_cumu_lookup_table()

    def mc_radii(self, *args):
        """ Method to generate Monte Carlo realizations of the profile model. 

        Parameters 
        ----------
        param_array : array_like, optional position argument(s)
            Array or arrays of length *Ngals* containing the input profile parameters. 
            In the simplest case, this is a single array of 
            NFW concentration values. 
            There should be an input ``param_array`` 
            for every parameter in the profile model, 
            all of the same length. 

        seed : int, optional keyword argument
            Random number seed used to generate Monte Carlo realization. 
            Default is None. 

        Returns 
        -------
        r : array 
            Length-*Ngals* array containing the 
            radial position of galaxies within their halos, 
            scaled by the size of the halo's boundary, 
            so that :math:`0 < r < 1`. 
        """
        # Draw random values for the cumulative mass PDF 
        # at the position of the satellites
        
        rho = np.random.random(len(args[0]))
        # These will be turned into random radial positions 
        # via the method of transformation of random variables

        # Discretize each profile parameter for every galaxy
        # Store the collection of arrays in digitized_param_list 
        digitized_param_list = []
        for param_index, param_key in enumerate(self.halo_prof_model.prof_param_keys):
            input_param_array = args[param_index]
            param_bins = getattr(self.halo_prof_model, param_key + '_cumu_inv_table')
            digitized_params = np.digitize(input_param_array, param_bins)
            digitized_param_list.append(digitized_params)
        # Each element of digitized_param_list is an array. 
        # The i^th element of each array contains the bin index of 
        # the discretized profile parameter array. 
        # So if self.cumu_inv_param_table_dict[concentration] = [4, 5, 6, 7,...], 
        # and the i^th entry of the the first array in param_array is 6.7, 
        # then the i^th entry of the 
        # first array in digitized_param_list will be 2

        # Now we have a collection of arrays storing indices of individual 
        # profile parameters, (A_0, A_1, A_2, ...), (B_0, B_1, B_2, ...), etc. 
        # For the combination of profile parameters (A_0, B_0, ...), we need 
        # the profile function object f_0, which we need to then evaluate 
        # on the randomly generated rho[0], and likewise for 
        # A_i, B_i, ...), f_i, and rho[i]. 
        # First determine the index in the profile function table 
        # where the relevant function object is stored:
        func_table_indices = (
            self.halo_prof_model.func_table_indices[digitized_param_list]
            )
        # Now we have an array of function objects, and we need to evaluate 
        # the i^th funcobj on the i^th element of rho. 
        # Call the model_helpers module to access generic code for doing this 
        return 10.**model_helpers.call_func_table(
            self.halo_prof_model.cumu_inv_func_table, np.log10(rho), func_table_indices)

    def mc_angles(self, Npts):
        """ Returns Npts random points on the unit sphere. 

        Parameters 
        ----------
        Npts : int 
            Number of 3d points to generate

        seed : int, optional keyword argument
            Random number seed. Default is None. 

        Returns 
        -------
        x, y, z : array_like  
            Length-Npts arrays of the coordinate positions. 

        """
        
        cos_t = np.random.uniform(-1.,1.,Npts)
        phi = np.random.uniform(0,2*np.pi,Npts)
        sin_t = np.sqrt((1.-cos_t*cos_t))

        x = sin_t * np.cos(phi)
        y = sin_t * np.sin(phi)
        z = cos_t

        return x, y, z

    def mc_pos(self, **kwargs):
        """ Method to generate random, three-dimensional, 
        halo-centric positions of galaxies. 

        Parameters 
        ----------
        galaxy_table : Astropy Table, required keyword argument
            Data table storing galaxy catalog. 

        seed : int, optional keyword argument 
            Random number seed used in Monte Carlo realization
        """
        galaxy_table = kwargs['galaxy_table']
        # get the appropriate slice for the gal_type of this component model
        x = galaxy_table['x']
        y = galaxy_table['y']
        z = galaxy_table['z']

        # For the case of a trivial profile model, return the trivial result
        if isinstance(self.halo_prof_model, 
            halo_prof_components.TrivialProfile) is True:
            return np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)
        else:
            # get angles
            Ngals = len(x)
            x, y, z = self.mc_angles(Ngals)

            # extract all relevant profile parameters from the mock
            profile_params = (
                [galaxy_table[model_defaults.host_haloprop_prefix+profile_param_key] 
                for profile_param_key in self.halo_prof_model.prof_param_keys]
                )

            # Get the radial positions of the gal_type galaxies
            scaled_mc_radii = self.mc_radii(*profile_params) 

            # multiply the random radial positions by the random points on the unit sphere 
            # to get random three-dimensional positions
            x *= scaled_mc_radii
            y *= scaled_mc_radii
            z *= scaled_mc_radii
           
        return x, y, z

















        





