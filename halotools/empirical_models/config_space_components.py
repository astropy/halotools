# -*- coding: utf-8 -*-
"""
This module contains the components for 
the intra-halo spatial positions of galaxies 
within their halos. 
"""
__author__ = ['Andrew Hearin']

__all__ = ['AnalyticDensityProf']

import numpy as np
from abc import ABCMeta
from functools import partial
from itertools import product

from . import model_defaults


@six.add_metaclass(ABCMeta)
class AnalyticDensityProf(object):
    """ Container class for any halo profile model. 

    This is an abstract class, and cannot itself be instantiated. 
    Rather, `HaloProfileModel` provides a template for any model of 
    the radial profile of dark matter particles within their halos. 

    """

    def __init__(self, halo_boundary = model_defaults.halo_boundary, 
        prim_haloprop_key = model_defaults.prim_haloprop_key, **kwargs):
        """
        Parameters 
        ----------
        prof_param_keys : string, or list of strings
            Provides the names of the halo profile parameters of the model. 
            String entries are typically an underscore-concatenation 
            of the model nickname and parameter nickname, e.g., ``NFWmodel_conc``. 

        halo_boundary : string, optional  
            String giving the column name of the halo catalog that stores the boundary of the halo. 
            Default is set in the `~halotools.empirical_models.model_defaults` module. 

        prim_haloprop_key : string, optional  
            String giving the column name of the primary halo property governing 
            the occupation statistics of gal_type galaxies. 
            Default is set in `~halotools.empirical_models.sim_defaults`.

        cosmology : object, optional 
            Astropy cosmology object. Default is None.

        redshift : float, optional  
            Default is None. 

        """
        super(AnalyticDensityProf, self).__init__(galprop_key='pos')

        self.halo_boundary = halo_boundary
        self.prim_haloprop_key = prim_haloprop_key

        required_kwargs = ['prof_param_keys']
        model_helpers.bind_required_kwargs(required_kwargs, self, **kwargs)

        if 'redshift' in kwargs.keys():
            self.redshift = kwargs['redshift']

        if 'cosmology' in kwargs.keys():
            self.cosmology = kwargs['cosmology']
        

    def build_inv_cumu_lookup_table(self, 
        logrmin = model_defaults.default_lograd_min, 
        logrmax = model_defaults.default_lograd_max, 
        Npts_radius_table=model_defaults.Npts_radius_table):
        """ Method used to create a lookup table of inverse cumulative mass 
        profile functions. 

        Parameters 
        ----------
        logrmin : float, optional 
            Minimum radius used to build the spline table. 
            Default is set in `~halotools.empirical_models.model_defaults`. 

        logrmax : float, optional 
            Maximum radius used to build the spline table
            Default is set in `~halotools.empirical_models.model_defaults`. 

        Npts_radius_table : int, optional 
            Number of control points used in the spline. 
            Default is set in `~halotools.empirical_models.model_defaults`. 

        Notes 
        ----- 

            * Used by mock factories such as `~halotools.empirical_models.HodMockFactory` to rapidly generate Monte Carlo realizations of intra-halo positions. 

            * As tested in `~halotools.empirical_models.test_empirical_models.test_halo_prof_components`, for the case of a `~halotools.empirical_models.NFWProfile`, errors due to interpolation from the lookup table are below 0.1 percent at all relevant radii and concentration. 

            * The interpolation is done in log-space. Thus each function object stored in ``cumu_inv_func_table`` operates on :math:`\\log_{10}\\mathrm{P}`, and returns :math:`\\log_{10}r`, where :math:`\\mathrm{P} = \\mathrm{P}_{\\mathrm{NFW}}( < r | c )`, computed by the `cumulative_mass_PDF` method. 

        """
        
        radius_array = np.logspace(logrmin,logrmax,Npts_radius_table)
        logradius_array = np.log10(radius_array)

        param_array_list = []
        for prof_param_key in self.prof_param_keys:
            parmin = getattr(self, prof_param_key + '_lookup_table_min')
            parmax = getattr(self, prof_param_key + '_lookup_table_max')
            dpar = getattr(self, prof_param_key + '_lookup_table_spacing')
            npts_par = int(np.round((parmax-parmin)/dpar))
            param_array = np.linspace(parmin,parmax,npts_par)
            param_array_list.append(param_array)
            setattr(self, prof_param_key + '_lookup_table_bins', param_array)
        
        # Using the itertools product method requires 
        # special handling of the length-zero edge case
        if len(param_array_list) == 0:
            self.cumu_inv_func_table = np.array([])
            self.func_table_indices = np.array([])
        else:
            func_table = []
            for items in product(*param_array_list):
                table_ordinates = self.cumulative_mass_PDF(radius_array,*items)
                log_table_ordinates = np.log10(table_ordinates)
                funcobj = spline(log_table_ordinates, logradius_array, k=4)
                func_table.append(funcobj)

            param_array_dimensions = [len(param_array) for param_array in param_array_list]
            self.cumu_inv_func_table = np.array(func_table).reshape(param_array_dimensions)
            self.func_table_indices = (
                np.arange(np.prod(param_array_dimensions)).reshape(param_array_dimensions)
                )

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

    def mc_spherical_shell(self, Npts, **kwargs):
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

    def mc_solid_sphere(self, **kwargs):
        """ Returns Npts random points inside the unit solid sphere, 
        with a radial density profile governed by 
        the subclass of `AnalyticDensityProf`.

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

    def mc_pos(self, **kwargs):
        """ Method used to generate Monte Carlo realizations of galaxy positions. 

        This method re-scales points in the unit solid sphere centered at the origin 
        to the :math:`R_{\\rm vir}`-sphere centered at the host halo position. 

        Parameters 
        ----------
        halo_table : Astropy Table, required keyword argument
            Data table storing a length-Ngals galaxy catalog. 

        gal_type : string, required keyword argument
            Name of the galaxy population. 

        Returns 
        -------
        x, y, z : array_like 
            Length-Ngals arrays of coordinate positions.

        Notes 
        -----
        This method is not directly called by 
        `~halotools.empirical_models.mock_factories.HodMockFactory`. 
        Instead, the `_set_primary_behaviors` method calls functools.partial 
        to create a ``mc_pos_gal_type`` method for each ``gal_type`` in the model. 

        """
        halo_table = kwargs['halo_table']
        x, y, z = self.mc_solid_sphere(halo_table=halo_table)

        # Re-scale the halo-centric distance by the halo boundary
        halo_boundary_key = self.halo_boundary
        x *= halo_table[halo_boundary_key]
        y *= halo_table[halo_boundary_key]
        z *= halo_table[halo_boundary_key]

        # Re-center the positions by the host halo location
        halo_xpos_key = model_defaults.host_haloprop_prefix+'x'
        halo_ypos_key = model_defaults.host_haloprop_prefix+'y'
        halo_zpos_key = model_defaults.host_haloprop_prefix+'z'
        x += halo_table[halo_xpos_key]
        y += halo_table[halo_ypos_key]
        z += halo_table[halo_zpos_key]

        return x, y, z


