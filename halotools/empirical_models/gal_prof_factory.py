# -*- coding: utf-8 -*-
"""

Module containing the primary class used to build 
galaxy profiles from a set of components. 

"""

__all__ = ['IsotropicGalProf']

from functools import partial
import numpy as np
from scipy.interpolate import UnivariateSpline as spline
from copy import copy

from . import model_defaults, model_helpers, halo_prof_components
from ..utils.array_utils import custom_len
from ..sim_manager import sim_defaults


class IsotropicGalProf(halo_prof_components.HaloProfileModel):
    """ Class modeling the intra-halo distribution of galaxies within their halos 
    under the assumption of spherical symmetry. 

    """

    def __init__(self, **kwargs):
        """
        Parameters 
        ----------
        halo_prof_model : class 
            Any sub-class of `~halotools.empirical_models.halo_prof_components.HaloProfileModel`. 
            All keyword arguments of `IsotropicGalProf` will be passed to the constructor of ``halo_prof_model``. 

        gal_type : string 
            Name of the galaxy population being modeled, e.g., ``sats`` or ``orphans``. 

        cosmology : object, optional 
            Astropy cosmology object. Default is None.

        redshift : float, optional  
            Default is None.

        halo_boundary : string, optional  
            String giving the column name of the halo catalog that stores the boundary of the halo. 
            Default is set in the `~halotools.empirical_models.model_defaults` module. 

        Examples 
        --------
        To build a centrals-like population, with galaxies residing at exactly 
        the halo center:

        >>> gal_prof_model = IsotropicGalProf(gal_type='centrals', halo_prof_model=halo_prof_components.TrivialProfile)

        For a satellite-type population distributed according to the NFW profile of the parent halo:

        >>> gal_prof_model = IsotropicGalProf(gal_type='sats', halo_prof_model=halo_prof_components.NFWProfile)

        """
        self.gal_type = kwargs['gal_type']

        kwargs_without_halo_prof_model = copy(kwargs)
        #print("printing kwargs keys & values during instantiation of IsotropicGalProf for gal_type %s" % self.gal_type)
        #for key, value in kwargs_without_halo_prof_model.iteritems():
        #    print key, value
        #print("\nDone printing arguments\n")
        del kwargs_without_halo_prof_model['halo_prof_model']
        #print "check 1"
        self.halo_prof_model = kwargs['halo_prof_model'](**kwargs_without_halo_prof_model)
        #print "check 2"

        super(IsotropicGalProf, self).__init__(
            prof_param_keys=self.halo_prof_model.prof_param_keys, **kwargs_without_halo_prof_model)
        #print "check 3"

        for key in self.prof_param_keys:
            setattr(self, key, getattr(self.halo_prof_model, key))

        self.build_inv_cumu_lookup_table()

        self.publications = self.halo_prof_model.publications

        ### NOT CORRECTLY IMPLEMENTED YET ###
        self.param_dict = {}

    def __getattr__(self, attr):
        """ Over-ride of the python built-in. Necessary because `IsotropicGalProf` is a sub-class 
        of `halo_prof_components.HaloProfileModel`, which is an abstract container class with 
        little functionality of its own. This override eliminates the need to explicitly call 
        ``setattr`` to inherit the needed attributes and methods of the ``halo_prof_model`` 
        passed to the constructor of `IsotropicGalProf`. 
        """
        return getattr(self.halo_prof_model, attr)

    def mc_radii(self, *args, **kwargs):
        """ Method to generate Monte Carlo realizations of the profile model. 

        Parameters 
        ----------
        param_array : array_like, positional argument(s)
            Array or arrays of length-Ngals containing the input profile parameters. 
            In the simplest case, this is a single array of, e.g., NFW concentration values. 
            There should be an input ``param_array`` for every parameter in the profile model, 
            all of the same length. 

        seed : int, optional 
            Random number seed used to generate Monte Carlo realization. 
            Default is None. 

        Returns 
        -------
        r : array 
            Length-Ngals array containing the radial position of galaxies within their halos, 
            scaled by the size of the halo's boundary, so that :math:`0 < r < 1`. 
        """
        # Draw random values for the cumulative mass PDF         
        # These will be turned into random radial positions 
        # via the method of transformation of random variables
        if 'seed' in kwargs.keys():
            np.random.seed(kwargs['seed'])
        rho = np.random.random(len(args[0]))

        # Discretize each profile parameter for every galaxy
        # Store the collection of arrays in digitized_param_list 
        # The number of elements of digitized_param_list is the number of profile parameters in the model
        digitized_param_list = []
        for param_index, param_key in enumerate(self.halo_prof_model.prof_param_keys):
            input_param_array = args[param_index]
            param_bins = getattr(self.halo_prof_model, param_key + '_lookup_table_bins')
            digitized_params = np.digitize(input_param_array, param_bins)
            digitized_param_list.append(digitized_params)
        # Each element of digitized_param_list is a length-Ngals array. 
        # The i^th element of each array contains the bin index of 
        # the discretized profile parameter of the galaxy. 
        # So if self.NFWmodel_conc_lookup_table_bins = [4, 5, 6, 7,...], 
        # and the i^th entry of the first argument in the input param_array is 6.7, 
        # then the i^th entry of the array stored in the 
        # first element in digitized_param_list will be 3. 

        # Now we have a collection of arrays storing indices of individual 
        # profile parameters, [A_0, A_1, A_2, ...], [B_0, B_1, B_2, ...], etc. 
        # For the combination of profile parameters [A_0, B_0, ...], we need 
        # the profile function object f_0, which we need to then evaluate 
        # on the randomly generated rho[0], and likewise for 
        # [A_i, B_i, ...], f_i, and rho[i], for i = 0, ..., Ngals-1.
        # To do this, we first determine the index in the profile function table 
        # where the relevant function object is stored:
        func_table_indices = (
            self.halo_prof_model.func_table_indices[digitized_param_list]
            )
        # Now we have an array of indices for our functions, and we need to evaluate 
        # the i^th function on the i^th element of rho. 
        # Call the model_helpers module to access generic code for doing this.
        # (Remember that the interpolation is being done in log-space)
        return 10.**model_helpers.call_func_table(
            self.halo_prof_model.cumu_inv_func_table, np.log10(rho), func_table_indices)

    def mc_angles(self, Npts, **kwargs):
        """ Returns Npts random points on the unit sphere. 

        Parameters 
        ----------
        Npts : int 
            Number of 3d points to generate

        seed : int, optional 
            Random number seed. Default is None. 

        Returns 
        -------
        x, y, z : array_like  
            Length-Npts arrays of the coordinate positions. 

        """
        if 'seed' in kwargs.keys():
            np.random.seed(kwargs['seed'])

        cos_t = np.random.uniform(-1.,1.,Npts)
        phi = np.random.uniform(0,2*np.pi,Npts)
        sin_t = np.sqrt((1.-cos_t*cos_t))

        x = sin_t * np.cos(phi)
        y = sin_t * np.sin(phi)
        z = cos_t

        return x, y, z

    def mc_pos(self, **kwargs):
        """ Method to generate random, three-dimensional, halo-centric positions of galaxies. 

        Parameters 
        ----------
        halo_table : Astropy Table, required keyword argument
            Data table storing a length-Ngals galaxy catalog. 

        seed : int, optional  
            Random number seed used in Monte Carlo realization

        Returns 
        -------
        x, y, z : arrays 
            Length-Ngals array storing a Monte Carlo realization of the galaxy positions. 
        """
        halo_table = kwargs['halo_table']
        x, y, z = halo_table['x'], halo_table['y'], halo_table['z']

        # For the case of a trivial profile model, return the trivial result
        if isinstance(self.halo_prof_model, 
            halo_prof_components.TrivialProfile) is True:
            return np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)
        else:
            # get angles
            Ngals = len(x)
            x, y, z = self.mc_angles(Ngals, **kwargs)

            # extract all relevant profile parameters from the mock
            profile_params = (
                [halo_table[profile_param_key] 
                for profile_param_key in self.halo_prof_model.prof_param_keys]
                )

            # Get the radial positions of the gal_type galaxies
            scaled_mc_radii = self.mc_radii(*profile_params, **kwargs) 

            # multiply the random radial positions by the random points on the unit sphere 
            # to get random three-dimensional positions
            x *= scaled_mc_radii
            y *= scaled_mc_radii
            z *= scaled_mc_radii
           
        return x, y, z

















        





