# -*- coding: utf-8 -*-
"""
Module composes the behavior of the profile models 
and the velocity models to produce models for the 
full phase space distribution of galaxies within their halos. 
"""

__author__ = ['Andrew Hearin']

__all__ = ['MonteCarloGalProf']

import numpy as np 

from functools import partial
from itertools import product

from .model_helpers import custom_spline 
from . import model_defaults

class MonteCarloGalProf(object):
    """ Orthogonal mix-in class used to turn an analytical 
    phase space model into a class that can be used 
    to generate the phase space distribution 
    of a mock galaxy population. 
    """

    def _setup_lookup_tables(self, *args):
        """
        Private method used to set up the lookup table grid 

        Parameters 
        ----------
        args : sequence 
            Length-Nparams list, with one entry per radial profile parameter. 
            Each entry must be a 3-element tuple. The first entry will be the minimum 
            value of the profile parameter, the second entry the maxium, the third entry 
            the linear spacing of the grid. The i^th element of the input ``args`` 
            is assumed to correspond to the i^th element of ``self.prom_param_keys``. 
        """
        for ipar, prof_param_key in enumerate(self.prof_param_keys):
            setattr(self, '_' + prof_param_key + '_lookup_table_min', args[ipar][0])
            setattr(self, '_' + prof_param_key + '_lookup_table_max', args[ipar][1])
            setattr(self, '_' + prof_param_key + '_lookup_table_spacing', args[ipar][2])

    def build_profile_lookup_tables(self, 
        logrmin = model_defaults.default_lograd_min, 
        logrmax = model_defaults.default_lograd_max, 
        Npts_radius_table=model_defaults.Npts_radius_table):
        """ Method used to create a lookup table of the radial profile 
        and velocity profile.  

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

        """
        
        radius_array = np.logspace(logrmin,logrmax,Npts_radius_table)
        self.logradius_array = np.log10(radius_array)

        param_array_list = []
        for prof_param_key in self.prof_param_keys:
            parmin = getattr(self, '_' + prof_param_key + '_lookup_table_min')
            parmax = getattr(self, '_' + prof_param_key + '_lookup_table_max')
            dpar = getattr(self, '_' + prof_param_key + '_lookup_table_spacing')
            npts_par = int(np.round((parmax-parmin)/dpar))
            param_array = np.linspace(parmin,parmax,npts_par)
            param_array_list.append(param_array)
            setattr(self, '_' + prof_param_key + '_lookup_table_bins', param_array)
        
        # Using the itertools product method requires 
        # special handling of the length-zero edge case
        if len(param_array_list) == 0:
            self.rad_prof_func_table = np.array([])
            self.rad_prof_func_table_indices = np.array([])
        else:
            func_table = []
            velocity_func_table = []
            for items in product(*param_array_list):
                table_ordinates = self.cumulative_mass_PDF(radius_array,*items)
                log_table_ordinates = np.log10(table_ordinates)
                funcobj = custom_spline(log_table_ordinates, self.logradius_array, k=4)
                func_table.append(funcobj)

                velocity_table_ordinates = self.dimensionless_velocity_dispersion(radius_array,*items)
                velocity_funcobj = custom_spline(velocity_table_ordinates, self.logradius_array, k=4)
                velocity_func_table.append(velocity_funcobj)

            param_array_dimensions = [len(param_array) for param_array in param_array_list]
            self.rad_prof_func_table = np.array(func_table).reshape(param_array_dimensions)
            self.vel_prof_func_table = np.array(velocity_func_table).reshape(param_array_dimensions)

            self.rad_prof_func_table_indices = (
                np.arange(np.prod(param_array_dimensions)).reshape(param_array_dimensions)
                )

    def mc_dimensionless_radial_distance(self, *args, **kwargs):
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
        for param_index, param_key in enumerate(self.prof_param_keys):
            input_param_array = args[param_index]
            param_bins = getattr(self, '_' + param_key + '_lookup_table_bins')
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
        rad_prof_func_table_indices = (
            self.rad_prof_func_table_indices[digitized_param_list]
            )
        # Now we have an array of indices for our functions, and we need to evaluate 
        # the i^th function on the i^th element of rho. 
        # Call the model_helpers module to access generic code for doing this.
        # (Remember that the interpolation is being done in log-space)
        return 10.**model_helpers.call_func_table(
            self.rad_prof_func_table, np.log10(rho), rad_prof_func_table_indices)

    def mc_unit_sphere(self, Npts, seed = None):
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
        np.random.seed(seed)

        cos_t = np.random.uniform(-1.,1.,Npts)
        phi = np.random.uniform(0,2*np.pi,Npts)
        sin_t = np.sqrt((1.-cos_t*cos_t))

        x = sin_t * np.cos(phi)
        y = sin_t * np.sin(phi)
        z = cos_t

        return x, y, z

    def mc_solid_sphere(self, profile_params, **kwargs):
        """ Method to generate random, three-dimensional, halo-centric positions of galaxies. 

        Parameters 
        ----------
        profile_params : list 
            Length-Nparams list, where Nparams is the number of 
            parameters specifying the profile model. Each list 
            entry must be a length-Ngals array storing the values of 
            the profile parameter for each galaxy. 

        seed : int, optional  
            Random number seed used in Monte Carlo realization

        Returns 
        -------
        x, y, z : arrays 
            Length-Ngals array storing a Monte Carlo realization of the galaxy positions. 
        """
        Ngals = len(profile_params[0])

        # get angles
        x, y, z = self.mc_unit_sphere(Ngals, **kwargs)

        # Get the radial positions of the galaxies scaled by the halo radius
        dimensionless_radial_distance = self.mc_dimensionless_radial_distance(
            *profile_params, **kwargs) 

        # get random positions within the solid sphere
        x *= dimensionless_radial_distance
        y *= dimensionless_radial_distance
        z *= dimensionless_radial_distance
           
        return x, y, z

    def mc_halo_centric_pos(self, profile_params, halo_radius, **kwargs):
        """ Method to generate random, three-dimensional 
        halo-centric positions of galaxies. 

        Parameters 
        ----------
        profile_params : list 
            Length-Nparams list, where Nparams is the number of 
            parameters specifying the profile model. Each list 
            entry must be a length-Ngals array storing the values of 
            the profile parameter for each galaxy. 

        halo_radius : array_like 
            Length-Ngals array storing the radial boundary of the halo 
            hosting each galaxy. 
            Units assumed to be in Mpc/h. 

        seed : int, optional  
            Random number seed used in Monte Carlo realization

        Returns 
        -------
        x, y, z : arrays 
            Length-Ngals array storing a Monte Carlo realization of the galaxy positions. 
        """
        x, y, z = self.mc_solid_sphere(profile_params, **kwargs)
        x *= halo_radius 
        y *= halo_radius 
        z *= halo_radius 

        return x, y, z


    def mc_pos(self, **kwargs):
        """ Method to generate random, three-dimensional positions of galaxies. 

        Parameters 
        ----------
        halo_table : data table, optional 
            Astropy Table storing a length-Ngals galaxy catalog. 
            If ``halo_table`` is not provided, 
            then both ``profile_params`` and ``halo_radius`` must be provided. 

        profile_params : list, optional 
            Length-Nparams list, where Nparams is the number of 
            parameters specifying the profile model. Each list 
            entry must be a length-Ngals array storing the values of 
            the profile parameter for each galaxy. 
            If ``halo_table`` is not provided, 
            then both ``profile_params`` and ``halo_radius`` must be provided. 

        halo_radius : array_like, optional 
            Length-Ngals array storing the radial boundary of the halo 
            hosting each galaxy. 
            Units assumed to be in Mpc/h. 
            If ``halo_table`` is not provided, 
            then both ``profile_params`` and ``halo_radius`` must be provided. 

        seed : int, optional  
            Random number seed used in Monte Carlo realization

        Returns 
        -------
        x, y, z : arrays, optional 
            If no ``halo_table`` is passed as an argument, 
            method will return x, y and z points distributed about the 
            origin according to the profile model. 

            If ``halo_table`` is passed as an argument, 
            the ``x``, ``y``, and ``z`` columns will be over-written with the 
            existing values plus the values that would otherwise be 
            returned by the method. Thus the ``halo_table`` mode of operation 
            assumes that the ``x``, ``y``, and ``z`` columns already store 
            the position of the host halo center. 
        """

        if 'halo_table' in kwargs:
            halo_table = kwargs['halo_table']
            profile_params = ([halo_table[profile_param_key] 
                for profile_param_key in self.prof_param_keys])
            halo_radius = halo_table[self.halo_boundary_key]
            x, y, z = self.mc_halo_centric_pos(
                profile_params, halo_radius, **kwargs)
            halo_table['x'] += x
            halo_table['y'] += y
            halo_table['z'] += z
        else:
            profile_params = kwargs['profile_params']
            halo_radius = kwargs['halo_radius']
            x, y, z = self.mc_halo_centric_pos(
                profile_params, halo_radius, **kwargs)
            return x, y, z


    def mc_dimensionless_radial_velocity_dispersion(self, x, *args, **kwargs):
        """ Method to generate Monte Carlo realizations of the profile model. 

        Parameters 
        ----------
        x : array_like 
            Halo-centric distance scaled by the halo boundary, so that 
            :math:`0 <= x <= 1`. Can be a scalar or length-Ngals numpy array

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
        sigma_vr : array 
            Length-Ngals array containing the radial velocity dispersion 
            of galaxies within their halos, 
            scaled by the size of the halo's virial velocity. 
        """
        # Discretize each profile parameter for every galaxy
        # Store the collection of arrays in digitized_param_list 
        # The number of elements of digitized_param_list is the number of profile parameters in the model
        digitized_param_list = []
        for param_index, param_key in enumerate(self.prof_param_keys):
            input_param_array = args[param_index]
            param_bins = getattr(self, '_' + param_key + '_lookup_table_bins')
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
        vel_prof_func_table_indices = (
            self.vel_prof_func_table_indices[digitized_param_list]
            )
        # Now we have an array of indices for our functions, and we need to evaluate 
        # the i^th function on the i^th element of rho. 
        # Call the model_helpers module to access generic code for doing this.
        return model_helpers.call_func_table(
            self.vel_prof_func_table, np.log10(x), vel_prof_func_table_indices)










        
