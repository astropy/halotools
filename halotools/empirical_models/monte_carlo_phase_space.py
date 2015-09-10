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
from time import time

from .model_helpers import custom_spline, call_func_table
from ..utils.array_utils import custom_len, convert_to_ndarray
from ..custom_exceptions import HalotoolsError 

from . import model_defaults

class MonteCarloGalProf(object):
    """ Orthogonal mix-in class used to turn an analytical 
    phase space model into a class that can be used 
    to generate the phase space distribution 
    of a mock galaxy population. 
    """

    def __init__(self):
        """
        """
        # For each function computing a profile parameter, 
        # add it to new_haloprop_func_dict so that the profile parameter 
        # will be pre-computed for each halo prior to mock population
        self.new_haloprop_func_dict = {}
        for key in self.prof_param_keys:
            self.new_haloprop_func_dict[key] = getattr(self, key)

        self._galprop_dtypes_to_allocate = np.dtype([
            ('host_centric_distance', 'f8'), 
            ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), 
            ('vx', 'f8'), ('vy', 'f8'), ('vz', 'f8'), 
            ])


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

    def build_lookup_tables(self, 
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
        key = self.prof_param_keys[0]
        if not hasattr(self, '_' + key + '_lookup_table_min'):
            raise HalotoolsError("You must first call _setup_lookup_tables"
                "to determine the grids before building the lookup tables")

        modelname = self.__class__.__name__
        print("\n...Building lookup tables for the %s radial profile." % modelname)
        
        radius_array = np.logspace(logrmin,logrmax,Npts_radius_table)
        self.logradius_array = np.log10(radius_array)

        profile_params_list = []
        for prof_param_key in self.prof_param_keys:
            parmin = getattr(self, '_' + prof_param_key + '_lookup_table_min')
            parmax = getattr(self, '_' + prof_param_key + '_lookup_table_max')
            dpar = getattr(self, '_' + prof_param_key + '_lookup_table_spacing')
            npts_par = int(np.round((parmax-parmin)/dpar))
            profile_params = np.linspace(parmin,parmax,npts_par)
            profile_params_list.append(profile_params)
            setattr(self, '_' + prof_param_key + '_lookup_table_bins', profile_params)
        
        # Using the itertools product method requires 
        # special handling of the length-zero edge case
        if len(profile_params_list) == 0:
            self.rad_prof_func_table = np.array([])
            self.rad_prof_func_table_indices = np.array([])
        else:
            func_table = []
            velocity_func_table = []
            start = time()
            for ii, items in enumerate(product(*profile_params_list)):
                table_ordinates = self.cumulative_mass_PDF(radius_array,*items)
                log_table_ordinates = np.log10(table_ordinates)
                funcobj = custom_spline(log_table_ordinates, self.logradius_array, k=4)
                func_table.append(funcobj)

                velocity_table_ordinates = self.dimensionless_velocity_dispersion(
                    radius_array, *items)
                velocity_funcobj = custom_spline(self.logradius_array, velocity_table_ordinates)
                velocity_func_table.append(velocity_funcobj)
                if ii == 9:
                    current_lookup_time = time() - start
                    runtime = (
                        current_lookup_time*
                        len(list(product(*profile_params_list)))/(2.*float(ii)+1.)
                        )
                    print("    (This will take about %.0f seconds)" % runtime)

            profile_params_dimensions = [len(profile_params) for profile_params in profile_params_list]
            self.rad_prof_func_table = np.array(func_table).reshape(profile_params_dimensions)
            self.vel_prof_func_table = np.array(velocity_func_table).reshape(profile_params_dimensions)

            self.rad_prof_func_table_indices = (
                np.arange(np.prod(profile_params_dimensions)).reshape(profile_params_dimensions)
                )

    def _mc_dimensionless_radial_distance(self, **kwargs):
        """ Method to generate Monte Carlo realizations of the profile model. 

        Parameters 
        ----------
        profile_params : list
            List of length-Ngals array(s) containing the input profile parameter(s). 
            In the simplest case, this list has a single element, 
            e.g. a single array of the NFW concentration values. 
            There should be a ``profile_params`` list item for 
            every parameter in the profile model, each item a length-Ngals array.

        seed : int, optional 
            Random number seed used to generate Monte Carlo realization. 
            Default is None. 

        Returns 
        -------
        r : array 
            Length-Ngals array containing the radial position of galaxies within their halos, 
            scaled by the size of the halo's boundary, so that :math:`0 < r < 1`. 
        """
        profile_params = kwargs['profile_params']

        if not hasattr(self, 'rad_prof_func_table'):
            self.build_lookup_tables()

        # Draw random values for the cumulative mass PDF         
        # These will be turned into random radial positions 
        # via the method of transformation of random variables
        if 'seed' in kwargs.keys():
            np.random.seed(kwargs['seed'])
        rho = np.random.random(len(profile_params[0]))

        # Discretize each profile parameter for every galaxy
        # Store the collection of arrays in digitized_param_list 
        # The number of elements of digitized_param_list is the number of profile parameters in the model
        digitized_param_list = []
        for param_index, param_key in enumerate(self.prof_param_keys):
            input_profile_params = convert_to_ndarray(profile_params[param_index])
            param_bins = getattr(self, '_' + param_key + '_lookup_table_bins')
            digitized_params = np.digitize(input_profile_params, param_bins)
            digitized_param_list.append(digitized_params)
        # Each element of digitized_param_list is a length-Ngals array. 
        # The i^th element of each array contains the bin index of 
        # the discretized profile parameter of the galaxy. 
        # So if self.NFWmodel_conc_lookup_table_bins = [4, 5, 6, 7,...], 
        # and the i^th entry of the first argument in the input profile_params is 6.7, 
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
        return 10.**call_func_table(
            self.rad_prof_func_table, np.log10(rho), rad_prof_func_table_indices)

    def mc_unit_sphere(self, Npts, **kwargs):
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
        if 'seed' in kwargs:
            np.random.seed(kwargs['seed'])

        cos_t = np.random.uniform(-1.,1.,Npts)
        phi = np.random.uniform(0,2*np.pi,Npts)
        sin_t = np.sqrt((1.-cos_t*cos_t))

        x = sin_t * np.cos(phi)
        y = sin_t * np.sin(phi)
        z = cos_t

        return x, y, z

    def mc_solid_sphere(self, **kwargs):
        """ Method to generate random, three-dimensional, halo-centric positions of galaxies. 

        Parameters 
        ----------
        profile_params : list, optional 
            List of length-Ngals array(s) containing the input profile parameter(s). 
            In the simplest case, this list has a single element, 
            e.g. a single array of the NFW concentration values. 
            There should be a ``profile_params`` list item for 
            every parameter in the profile model, each item a length-Ngals array.
            If ``profile_params`` is not passed, ``halo_table`` must be passed. 

        halo_table : data table, optional 
            Astropy Table storing a length-Ngals galaxy catalog. 
            If ``halo_table`` is not passed, ``profile_params`` must be passed. 

        seed : int, optional  
            Random number seed used in Monte Carlo realization

        Returns 
        -------
        x, y, z : arrays 
            Length-Ngals array storing a Monte Carlo realization of the galaxy positions. 
        """
        # Retrieve the list of profile_params
        if 'halo_table' in kwargs:
            halo_table = kwargs['halo_table']
            profile_params = ([halo_table[profile_param_key] 
                for profile_param_key in self.prof_param_keys])
            halo_radius = halo_table[self.halo_boundary_key]
        else:
            try:
                profile_params = kwargs['profile_params']
            except KeyError:
                raise HalotoolsError("If not passing an input ``halo_table`` "
                    "keyword argument to mc_solid_sphere,\n"
                    "must pass a ``profile_params`` keyword argument")

        # get random angles
        Ngals = len(profile_params[0])
        x, y, z = self.mc_unit_sphere(Ngals, **kwargs)

        # Get the radial positions of the galaxies scaled by the halo radius
        if 'seed' in kwargs:
            seed = kwargs['seed']
        else:
            seed = None
        dimensionless_radial_distance = self._mc_dimensionless_radial_distance(
            profile_params = profile_params, seed = seed) 

        # get random positions within the solid sphere
        x *= dimensionless_radial_distance
        y *= dimensionless_radial_distance
        z *= dimensionless_radial_distance
            
        # Assign the value of the host_centric_distance halo_table column
        if 'halo_table' in kwargs:    
            try:
                halo_table['host_centric_distance'][:] = dimensionless_radial_distance
                halo_table['host_centric_distance'][:] *= halo_radius
            except KeyError:
                msg = ("The mc_solid_sphere method of the MonteCarloGalProf class "
                    "requires a halo_table key ``host_centric_distance`` to be pre-allocated ")
                raise HalotoolsError(msg)
           
        return x, y, z

    def mc_halo_centric_pos(self, **kwargs):
        """ Method to generate random, three-dimensional 
        halo-centric positions of galaxies. 

        Parameters 
        ----------
        halo_radius : array_like 
            Length-Ngals array storing the radial boundary of the halo 
            hosting each galaxy. Units assumed to be in Mpc/h. 

        profile_params : list 
            List of length-Ngals array(s) containing the input profile parameter(s). 
            In the simplest case, this list has a single element, 
            e.g. a single array of the NFW concentration values. 
            There should be a ``profile_params`` list item for 
            every parameter in the profile model, each item a length-Ngals array.

        seed : int, optional  
            Random number seed used in Monte Carlo realization

        Returns 
        -------
        x, y, z : arrays 
            Length-Ngals array storing a Monte Carlo realization of the galaxy positions. 
        """

        x, y, z = self.mc_solid_sphere(**kwargs)

        ### Retrieve the halo_radius
        if 'halo_table' in kwargs:    
            halo_table = kwargs['halo_table']
            halo_radius = halo_table[self.halo_boundary_key]
        else:
            try:
                halo_radius = convert_to_ndarray(kwargs['halo_radius'])
            except KeyError:
                raise HalotoolsError("If not passing an input ``halo_table`` "
                    "keyword argument to mc_halo_centric_pos,\n"
                    "must pass the following keyword arguments:\n"
                    "``halo_radius``, ``profile_params``.")

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
            List of length-Ngals array(s) containing the input profile parameter(s). 
            In the simplest case, this list has a single element, 
            e.g. a single array of the NFW concentration values. 
            There should be a ``profile_params`` list item for 
            every parameter in the profile model, each item a length-Ngals array.
            If ``halo_table`` is not provided, 
            then both ``profile_params`` and ``halo_radius`` must be provided. 

        halo_radius : array_like, optional 
            Length-Ngals array storing the radial boundary of the halo 
            hosting each galaxy. 
            Units assumed to be in Mpc/h. 
            If ``halo_table`` is not provided, 
            then both ``profile_params`` and ``halo_radius`` must be provided. 
            If ``halo_table`` is not provided, 
            then both ``profile_params`` and ``halo_radius`` must be provided. 

        seed : int, optional  
            Random number seed used in Monte Carlo realization. Default is None. 

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
            x, y, z = self.mc_halo_centric_pos(**kwargs)
            halo_table['x'][:] += x
            halo_table['y'][:] += y
            halo_table['z'][:] += z
        else:
            try:
                profile_params = kwargs['profile_params']
                halo_radius = convert_to_ndarray(kwargs['halo_radius'])
            except KeyError:
                raise HalotoolsError("\nIf not passing a ``halo_table`` keyword argument "
                    "to mc_pos, must pass the following keyword arguments:\n"
                    "``profile_params``, ``halo_radius``.")
            x, y, z = self.mc_halo_centric_pos(**kwargs)
            return x, y, z


    def _vrad_disp_from_lookup(self, **kwargs):
        """ Method to generate Monte Carlo realizations of the profile model. 

        Parameters 
        ----------
        x : array_like 
            Halo-centric distance scaled by the halo boundary, so that 
            :math:`0 <= x <= 1`. Can be a scalar or length-Ngals numpy array

        profile_params : list
            List of length-Ngals array(s) containing the input profile parameter(s). 
            In the simplest case, this list has a single element, 
            e.g. a single array of the NFW concentration values. 
            There should be a ``profile_params`` list item for 
            every parameter in the profile model, each item a length-Ngals array.

        Returns 
        -------
        sigma_vr : array 
            Length-Ngals array containing the radial velocity dispersion 
            of galaxies within their halos, 
            scaled by the size of the halo's virial velocity. 
        """
        x = convert_to_ndarray(kwargs['x'])
        x = x.astype(float)
        profile_params = kwargs['profile_params']

        if not hasattr(self, 'vel_prof_func_table'):
            self.build_lookup_tables()
        # Discretize each profile parameter for every galaxy
        # Store the collection of arrays in digitized_param_list 
        # The number of elements of digitized_param_list is the number of profile parameters in the model
        digitized_param_list = []
        for param_index, param_key in enumerate(self.prof_param_keys):
            input_profile_params = convert_to_ndarray(profile_params[param_index])
            param_bins = getattr(self, '_' + param_key + '_lookup_table_bins')
            digitized_params = np.digitize(input_profile_params, param_bins)
            digitized_param_list.append(digitized_params)
        # Each element of digitized_param_list is a length-Ngals array. 
        # The i^th element of each array contains the bin index of 
        # the discretized profile parameter of the galaxy. 
        # So if self.NFWmodel_conc_lookup_table_bins = [4, 5, 6, 7,...], 
        # and the i^th entry of the first argument in the input profile_params is 6.7, 
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
            self.rad_prof_func_table_indices[digitized_param_list]
            )
        # Now we have an array of indices for our functions, and we need to evaluate 
        # the i^th function on the i^th element of rho. 
        # Call the model_helpers module to access generic code for doing this.
        dimensionless_radial_dispersions = call_func_table(
            self.vel_prof_func_table, np.log10(x), vel_prof_func_table_indices)

        return dimensionless_radial_dispersions

    def mc_radial_velocity(self, **kwargs):
        """
        Parameters 
        ----------
        x : array_like 
            Halo-centric distance scaled by the halo boundary, so that 
            :math:`0 <= x <= 1`. Can be a scalar or length-Ngals numpy array

        virial_velocities : array_like 
            Array storing the virial velocity of the halos hosting the galaxies. 

        profile_params : list
            List of length-Ngals array(s) containing the input profile parameter(s). 
            In the simplest case, this list has a single element, 
            e.g. a single array of the NFW concentration values. 
            There should be a ``profile_params`` list item for 
            every parameter in the profile model, each item a length-Ngals array.

        seed : int, optional  
            Random number seed used in Monte Carlo realization. Default is None. 

        Returns 
        -------
        radial_velocities : array_like 
            Array of radial velocities drawn from Gaussians with a width determined by the 
            solution to the Jeans equation. 
        """

        dimensionless_radial_dispersions = (
            self._vrad_disp_from_lookup(**kwargs))

        virial_velocities = convert_to_ndarray(kwargs['virial_velocities'])
        radial_dispersions = virial_velocities*dimensionless_radial_dispersions

        if 'seed' in kwargs.keys():
            np.random.seed(kwargs['seed'])

        radial_velocities = np.random.normal(scale = radial_dispersions)

        return radial_velocities

    def mc_vel(self, halo_table):
        """
        """
        try:
            d = halo_table['host_centric_distance']
        except KeyError:
            raise HalotoolsError("The mc_vel method requires ``host_centric_distance`` "
                "to be an existing column of the input halo_table")
        try:
            rhalo = halo_table[self.halo_boundary_key]
        except KeyError:
            msg = ("halo_boundary_key = %s must be a key of the input halo catalog")
            raise HalotoolsError(msg % self.halo_boundary_key)
        x = d/rhalo

        profile_params = [halo_table[key] for key in self.prof_param_keys]
        try:
            virial_velocities = halo_table['halo_vvir']
        except KeyError:
            virial_velocities = self.virial_velocity(
                total_mass = halo_table[self.halo_mass_key])

        if 'velbias_satellites' in self.param_dict:
            virial_velocities *= self.param_dict['velbias_satellites']
    
        vx = self.mc_radial_velocity(
            virial_velocities = virial_velocities, 
            x = x, profile_params = profile_params)
        vy = self.mc_radial_velocity(
            virial_velocities = virial_velocities, 
            x = x, profile_params = profile_params)
        vz = self.mc_radial_velocity(
            virial_velocities = virial_velocities, 
            x = x, profile_params = profile_params)


        halo_table['vx'][:] = halo_table['halo_vx'] + vx
        halo_table['vy'][:] = halo_table['halo_vy'] + vy
        halo_table['vz'][:] = halo_table['halo_vz'] + vz


















        
