# -*- coding: utf-8 -*-
"""
The `MonteCarloGalProf` class defined in this module is 
used as an orthogonal mix-in class to supplement the behavior of 
the analytic profile and velocity models. 
The result of using `MonteCarloGalProf` as an orthogonal mix-in class 
is a composite class that can be used to generate Monte Carlo realizations 
of the full phase space distribution of galaxies within their halos. 

Testing for this module is done in `~halotools.empirical_models.test_phase_space` module. 
"""

__author__ = ['Andrew Hearin']
__all__ = ['MonteCarloGalProf']

import numpy as np 

from itertools import product
from time import time

from ..model_helpers import custom_spline, call_func_table
from .. import model_defaults

from ...utils.array_utils import custom_len, convert_to_ndarray
from ...custom_exceptions import HalotoolsError 


class MonteCarloGalProf(object):
    """ Orthogonal mix-in class used to turn an analytical 
    phase space model (e.g., `~halotools.empirical_models.NFWPhaseSpace`)
    into a class that can generate the phase space distribution 
    of a mock galaxy population. 

    Testing for this module is done in `~halotools.empirical_models.test_phase_space.TestNFWPhaseSpace` class. 

    """

    def __init__(self):
        """
        """
        # For each function computing a profile parameter, 
        # add it to new_haloprop_func_dict so that the profile parameter 
        # will be pre-computed for each halo prior to mock population
        if not hasattr(self, 'new_haloprop_func_dict'):
            self.new_haloprop_func_dict = {}
        for key in self.prof_param_keys:
            self.new_haloprop_func_dict[key] = getattr(self, key)

        self._galprop_dtypes_to_allocate = np.dtype([
            ('host_centric_distance', 'f8'), 
            ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), 
            ('vx', 'f8'), ('vy', 'f8'), ('vz', 'f8'), 
            ])


    def setup_prof_lookup_tables(self, *lookup_table_binning):
        """
        Method used to set up the lookup table grid. 

        Each analytical profile has profile parameters associated with it. This method 
        sets up how we will digitize the value of each such parameter for the purposes of 
        mock population. 

        As an example, the `~halotools.empirical_models.NFWPhaseSpace` 
        model has a single profile parameter, ``conc_NFWmodel``. After calling the 
        `setup_prof_lookup_tables` method, there will be three new attributes bound to the 
        `~halotools.empirical_models.NFWPhaseSpace` instance:

        * ``_conc_NFWmodel_lookup_table_min``

        * ``_conc_NFWmodel_lookup_table_max``

        * ``_conc_NFWmodel_lookup_table_spacing``

        These three attributes define the linearly spacing of the ``conc_NFWmodel`` parameter 
        lookup table created by the `build_lookup_tables` method.

        Parameters 
        ----------
        *lookup_table_binning : sequence 
            Sequence of tuples, with one tuple per radial profile parameter, 
            and each tuple having three elements. 
            The first entry of each tuple will be the minimum 
            value of the profile parameter in the lookup table, 
            the second entry the maximum, the third entry 
            the linear spacing of the grid. The i^th element of 
            the input ``lookup_table_binning`` sequence 
            is assumed to correspond to the i^th element of ``self.prof_param_keys``, 
            which is defined in the 
            `~halotools.empirical_models.profile_params.profile_model_template.AnalyticalDensityProf` sub-class. 
        """
        for ipar, prof_param_key in enumerate(self.prof_param_keys):
            setattr(self, '_' + prof_param_key + '_lookup_table_min', lookup_table_binning[ipar][0])
            setattr(self, '_' + prof_param_key + '_lookup_table_max', lookup_table_binning[ipar][1])
            setattr(self, '_' + prof_param_key + '_lookup_table_spacing', lookup_table_binning[ipar][2])

    def build_lookup_tables(self, 
        logrmin = model_defaults.default_lograd_min, 
        logrmax = model_defaults.default_lograd_max, 
        Npts_radius_table = model_defaults.Npts_radius_table):
        """ Method used to create a lookup table of the spatial and velocity radial profiles.  

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
            raise HalotoolsError("You must first call setup_prof_lookup_tables"
                "to determine the grids before building the lookup tables")
        
        radius_array = np.logspace(logrmin, logrmax, Npts_radius_table)
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
                funcobj = custom_spline(log_table_ordinates, self.logradius_array, k=3)
                func_table.append(funcobj)

                velocity_table_ordinates = self.dimensionless_radial_velocity_dispersion(
                    radius_array, *items)
                velocity_funcobj = custom_spline(self.logradius_array, velocity_table_ordinates, k=3)
                velocity_func_table.append(velocity_funcobj)
                # Print a message for the expected runtime of the table build
                if ii == 9:
                    current_lookup_time = time() - start
                    runtime = (
                        current_lookup_time*
                        len(list(product(*profile_params_list)))/(2.*float(ii)+1.)
                        )
                    if runtime > 5:
                        modelname = self.__class__.__name__
                        print(("\n...Building lookup tables for the %s radial profile." % modelname))
                        print(("    (This will take about %.0f seconds, and only needs to be done once)" % runtime))

            profile_params_dimensions = [len(profile_params) for profile_params in profile_params_list]
            self.rad_prof_func_table = np.array(func_table).reshape(profile_params_dimensions)
            self.vel_prof_func_table = np.array(velocity_func_table).reshape(profile_params_dimensions)

            self.rad_prof_func_table_indices = (
                np.arange(np.prod(profile_params_dimensions)).reshape(profile_params_dimensions)
                )

    def _mc_dimensionless_radial_distance(self, *profile_params, **kwargs):
        """ Method to generate Monte Carlo realizations of the profile model. 

        Parameters 
        ----------
        *profile_params : Sequence of arrays
            Sequence of length-Ngals array(s) containing the input profile parameter(s). 
            In the simplest case, this sequence has a single element, 
            e.g. a single array storing values of the NFW concentrations of the Ngals galaxies. 
            More generally, there should be a ``profile_params`` sequence item for 
            every parameter in the profile model, each item a length-Ngals array.
            The sequence must have the same order as ``self.prof_param_keys``. 

        seed : int, optional  
            Random number seed used in Monte Carlo realization. Default is None. 

        Returns 
        -------
        scaled_radius : array_like 
            Length-Ngals array storing the halo-centric distance *r* scaled 
            by the halo boundary :math:`R_{\\Delta}`, so that 
            :math:`0 <= \\tilde{r} \\equiv r/R_{\\Delta} <= 1`.

        Notes 
        ------
        This method is tested by the `~halotools.empirical_models.test_phase_space.TestNFWPhaseSpace.test_mc_dimensionless_radial_distance` function. 
        """

        if not hasattr(self, 'rad_prof_func_table'):
            self.build_lookup_tables()

        # Draw random values for the cumulative mass PDF         
        # These will be turned into random radial positions 
        # by inverting the tabulated cumulative_mass_PDF
        if 'seed' in kwargs:
            np.random.seed(kwargs['seed'])
        rho = np.random.random(len(profile_params[0]))

        # Discretize each profile parameter for every galaxy
        # Store the collection of arrays in digitized_param_list 
        # The number of elements of digitized_param_list is the number of profile parameters in the model
        digitized_param_list = []
        for param_index, param_key in enumerate(self.prof_param_keys):
            input_profile_params = convert_to_ndarray(profile_params[param_index])
            param_bins = getattr(self, '_' + param_key + '_lookup_table_bins')
            digitized_params = np.digitize(input_profile_params, param_bins, right=True)
            digitized_params[digitized_params==len(param_bins)] -= 1
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
            Random number seed used in Monte Carlo realization. Default is None. 

        Returns 
        -------
        x, y, z : array_like  
            Length-Npts arrays of the coordinate positions. 

        Notes 
        ------
        This method is tested by the `~halotools.empirical_models.test_phase_space.TestNFWPhaseSpace.test_mc_unit_sphere` function. 

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

    def mc_solid_sphere(self, *profile_params, **kwargs):
        """ Method to generate random, three-dimensional, halo-centric positions of galaxies. 

        Parameters 
        ----------
        *profile_params : Sequence of arrays
            Sequence of length-Ngals array(s) containing the input profile parameter(s). 
            In the simplest case, this sequence has a single element, 
            e.g. a single array storing values of the NFW concentrations of the Ngals galaxies. 
            More generally, there should be a ``profile_params`` sequence item for 
            every parameter in the profile model, each item a length-Ngals array.
            The sequence must have the same order as ``self.prof_param_keys``. 

        table : data table, optional 
            Astropy Table storing a length-Ngals galaxy catalog. 
            If ``table`` is not passed, ``profile_params`` must be passed. 

        seed : int, optional  
            Random number seed used in Monte Carlo realization. Default is None. 

        Returns 
        -------
        x, y, z : arrays 
            Length-Ngals array storing a Monte Carlo realization of the galaxy positions. 

        Notes 
        ------
        This method is tested by the `~halotools.empirical_models.test_phase_space.TestNFWPhaseSpace.test_mc_solid_sphere` function. 
        """
        # Retrieve the list of profile_params
        if 'table' in kwargs:
            table = kwargs['table']
            profile_params = ([table[profile_param_key] 
                for profile_param_key in self.prof_param_keys])
            halo_radius = table[self.halo_boundary_key]
        else:
            try:
                # profile_params = kwargs['profile_params']
                assert len(profile_params) > 0
            except AssertionError:
                raise HalotoolsError("If not passing an input ``table`` "
                    "keyword argument to mc_solid_sphere,\n"
                    "must pass a ``profile_params`` keyword argument")

        # get random angles
        Ngals = len(profile_params[0])
        if Ngals == 0: 
            return None, None, None
        x, y, z = self.mc_unit_sphere(Ngals, **kwargs)

        # Get the radial positions of the galaxies scaled by the halo radius
        if 'seed' in kwargs:
            seed = kwargs['seed']
        else:
            seed = None
        dimensionless_radial_distance = self._mc_dimensionless_radial_distance(
            *profile_params, seed = seed) 

        # get random positions within the solid sphere
        x *= dimensionless_radial_distance
        y *= dimensionless_radial_distance
        z *= dimensionless_radial_distance
            
        # Assign the value of the host_centric_distance table column
        if 'table' in kwargs:    
            try:
                table['host_centric_distance'][:] = dimensionless_radial_distance
                table['host_centric_distance'][:] *= halo_radius
            except KeyError:
                msg = ("The mc_solid_sphere method of the MonteCarloGalProf class "
                    "requires a table key ``host_centric_distance`` to be pre-allocated ")
                raise HalotoolsError(msg)
           
        return x, y, z

    def mc_halo_centric_pos(self, *profile_params, **kwargs):
        """ Method to generate random, three-dimensional 
        halo-centric positions of galaxies. 

        Parameters 
        ----------
        table : data table, optional 
            Astropy Table storing a length-Ngals galaxy catalog. 
            If ``table`` is not passed, ``profile_params`` and 
            keyword argument ``halo_radius`` must be passed. 

        *profile_params : Sequence of arrays
            Sequence of length-Ngals array(s) containing the input profile parameter(s). 
            In the simplest case, this sequence has a single element, 
            e.g. a single array storing values of the NFW concentrations of the Ngals galaxies. 
            More generally, there should be a ``profile_params`` sequence item for 
            every parameter in the profile model, each item a length-Ngals array.
            If ``profile_params`` is passed, ``halo_radius`` must be passed as a keyword argument.
            The sequence must have the same order as ``self.prof_param_keys``. 

        halo_radius : array_like, optional 
            Length-Ngals array storing the radial boundary of the halo 
            hosting each galaxy. Units assumed to be in Mpc/h. 
            If ``profile_params`` and ``halo_radius`` are not passed, 
            ``table`` must be passed. 

        seed : int, optional  
            Random number seed used in Monte Carlo realization. Default is None. 

        Returns 
        -------
        x, y, z : arrays 
            Length-Ngals array storing a Monte Carlo realization of the galaxy positions. 

        Notes 
        ------
        This method is tested by the `~halotools.empirical_models.test_phase_space.TestNFWPhaseSpace.test_mc_halo_centric_pos` function. 
        """

        x, y, z = self.mc_solid_sphere(*profile_params, **kwargs)
        if x is None:
            return None, None, None

        ### Retrieve the halo_radius
        if 'table' in kwargs:    
            table = kwargs['table']
            halo_radius = table[self.halo_boundary_key]
        else:
            try:
                halo_radius = convert_to_ndarray(kwargs['halo_radius'])
            except KeyError:
                raise HalotoolsError("If not passing an input ``table`` "
                    "keyword argument to mc_halo_centric_pos,\n"
                    "must pass the following keyword arguments:\n"
                    "``halo_radius``, ``profile_params``.")

        x *= halo_radius 
        y *= halo_radius 
        z *= halo_radius 
        return x, y, z


    def mc_pos(self, *profile_params, **kwargs):
        """ Method to generate random, three-dimensional positions of galaxies. 

        Parameters 
        ----------
        table : data table, optional 
            Astropy Table storing a length-Ngals galaxy catalog. 
            If ``table`` is not passed, ``profile_params`` and ``halo_radius`` must be passed. 

        *profile_params : Sequence of arrays
            Sequence of length-Ngals array(s) containing the input profile parameter(s). 
            In the simplest case, this sequence has a single element, 
            e.g. a single array storing values of the NFW concentrations of the Ngals galaxies. 
            More generally, there should be a ``profile_params`` sequence item for 
            every parameter in the profile model, each item a length-Ngals array.
            If ``profile_params`` is passed, ``halo_radius`` must be passed as a keyword argument.
            The sequence must have the same order as ``self.prof_param_keys``. 

        halo_radius : array_like, optional 
            Length-Ngals array storing the radial boundary of the halo 
            hosting each galaxy. Units assumed to be in Mpc/h. 
            If ``profile_params`` and ``halo_radius`` are not passed, 
            ``table`` must be passed. 

        overwrite_table_pos : bool, optional 
            If True, the `mc_pos` method will over-write the existing values of 
            the ``x``, ``y`` and ``z`` table columns. Default is True

        return_pos : bool, optional 
            If True, method will return the computed host-centric 
            values of ``x``, ``y`` and ``z``. Default is False.

        seed : int, optional  
            Random number seed used in Monte Carlo realization. Default is None. 

        Returns 
        -------
        x, y, z : arrays, optional 
            For the case where no ``table`` is passed as an argument, 
            method will return x, y and z points distributed about the 
            origin according to the profile model. 

            For the case where ``table`` is passed as an argument 
            (this is the use case of populating halos with mock galaxies), 
            the ``x``, ``y``, and ``z`` columns of the table will be over-written.
            When ``table`` is passed as an argument, the method 
            assumes that the ``x``, ``y``, and ``z`` columns already store 
            the position of the host halo center. 

        Notes 
        ------
        This method is tested by the `~halotools.empirical_models.test_phase_space.TestNFWPhaseSpace.test_mc_pos` function. 
        """
        try:
            overwrite_table_pos = kwargs['overwrite_table_pos']
        except KeyError:
            overwrite_table_pos = True

        try:
            return_pos = kwargs['return_pos']
        except KeyError:
            return_pos = False

        if 'table' in kwargs:
            table = kwargs['table']
            x, y, z = self.mc_halo_centric_pos(*profile_params, **kwargs)
            if x is None:
                return None
            if overwrite_table_pos is True:
                table['x'][:] += x
                table['y'][:] += y
                table['z'][:] += z
            if return_pos is True:
                return x, y, z
        else:
            try:
                # profile_params = kwargs['profile_params']
                halo_radius = convert_to_ndarray(kwargs['halo_radius'])
                assert len(halo_radius) == len(profile_params[0])
            except KeyError as AssertionError:
                raise HalotoolsError("\nIf not passing a ``table`` keyword argument "
                    "to mc_pos, must pass the following keyword arguments:\n"
                    "``profile_params``, ``halo_radius``.")
            x, y, z = self.mc_halo_centric_pos(*profile_params, **kwargs)
            if x is None:
                return None
            else:
                return x, y, z


    def _vrad_disp_from_lookup(self, scaled_radius, *profile_params, **kwargs):
        """ Method to generate Monte Carlo realizations of the profile model. 

        Parameters 
        ----------
        scaled_radius : array_like 
            Halo-centric distance *r* scaled by the halo boundary :math:`R_{\\Delta}`, so that 
            :math:`0 <= \\tilde{r} \\equiv r/R_{\\Delta} <= 1`. Can be a scalar or numpy array. 

        *profile_params : Sequence of arrays
            Sequence of length-Ngals array(s) containing the input profile parameter(s). 
            In the simplest case, this sequence has a single element, 
            e.g. a single array storing values of the NFW concentrations of the Ngals galaxies. 
            More generally, there should be a ``profile_params`` sequence item for 
            every parameter in the profile model, each item a length-Ngals array.
            The sequence must have the same order as ``self.prof_param_keys``. 

        Returns 
        -------
        sigma_vr : array 
            Length-Ngals array containing the radial velocity dispersion 
            of galaxies within their halos, 
            scaled by the size of the halo's virial velocity. 
        """
        scaled_radius = convert_to_ndarray(scaled_radius, dt = np.float64)
        # x = convert_to_ndarray(kwargs['x'])
        # x = x.astype(float)
        # profile_params = kwargs['profile_params']

        if not hasattr(self, 'vel_prof_func_table'):
            self.build_lookup_tables()
        # Discretize each profile parameter for every galaxy
        # Store the collection of arrays in digitized_param_list 
        # The number of elements of digitized_param_list is the number of profile parameters in the model
        digitized_param_list = []
        for param_index, param_key in enumerate(self.prof_param_keys):
            input_profile_params = convert_to_ndarray(profile_params[param_index])
            param_bins = getattr(self, '_' + param_key + '_lookup_table_bins')
            digitized_params = np.digitize(input_profile_params, param_bins, right=True)
            digitized_params[digitized_params==len(param_bins)] -= 1
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
            self.vel_prof_func_table, np.log10(scaled_radius), vel_prof_func_table_indices)

        return dimensionless_radial_dispersions

    def mc_radial_velocity(self, scaled_radius, total_mass, *profile_params, **kwargs):
        """
        Method returns a Monte Carlo realization of radial velocities drawn from Gaussians 
        with a width determined by the solution to the isotropic Jeans equation. 

        Parameters 
        ----------
        scaled_radius : array_like 
            Halo-centric distance *r* scaled by the halo boundary :math:`R_{\\Delta}`, so that 
            :math:`0 <= \\tilde{r} \\equiv r/R_{\\Delta} <= 1`. Can be a scalar or numpy array. 

        total_mass: array_like
            Length-Ngals numpy array storing the halo mass in :math:`M_{\odot}/h`. 

        *profile_params : Sequence of arrays
            Sequence of length-Ngals array(s) containing the input profile parameter(s). 
            In the simplest case, this sequence has a single element, 
            e.g. a single array storing values of the NFW concentrations of the Ngals galaxies. 
            More generally, there should be a ``profile_params`` sequence item for 
            every parameter in the profile model, each item a length-Ngals array. 
            The sequence must have the same order as ``self.prof_param_keys``. 

        seed : int, optional  
            Random number seed used in Monte Carlo realization. Default is None. 

        Returns 
        -------
        radial_velocities : array_like 
            Array of radial velocities drawn from Gaussians with a width determined by the 
            solution to the isotropic Jeans equation. 
        """

        dimensionless_radial_dispersions = (
            self._vrad_disp_from_lookup(scaled_radius, *profile_params, **kwargs))

        virial_velocities = self.virial_velocity(total_mass)
        radial_dispersions = virial_velocities*dimensionless_radial_dispersions

        if 'seed' in list(kwargs.keys()):
            np.random.seed(kwargs['seed'])

        radial_velocities = np.random.normal(scale = radial_dispersions)

        return radial_velocities

    def mc_vel(self, table, overwrite_table_velocities = True, 
        return_velocities = False):
        """ Method assigns a Monte Carlo realization of the Jeans velocity 
        solution to the halos in the input ``table``. 

        Parameters 
        -----------
        table : Astropy Table 
            `astropy.table.Table` object storing the halo catalog. 

        overwrite_table_velocities : bool, optional 
            If True, the `mc_vel` method will over-write the existing values of 
            the ``vx``, ``vy`` and ``vz`` columns. Default is True

        return_velocities : bool, optional 
            If True, method will return the computed values of ``vx``, ``vy`` and ``vz``. 
            Default is False.
        
        Notes 
        -------
        The method assumes that the ``vx``, ``vy``, and ``vz`` columns already store 
        the position of the host halo center. 

        """
        try:
            d = table['host_centric_distance']
        except KeyError:
            raise HalotoolsError("The mc_vel method requires ``host_centric_distance`` "
                "to be an existing column of the input table")
        try:
            rhalo = table[self.halo_boundary_key]
        except KeyError:
            msg = ("halo_boundary_key = %s must be a key of the input halo catalog")
            raise HalotoolsError(msg % self.halo_boundary_key)
        scaled_radius = d/rhalo

        profile_params = [table[key] for key in self.prof_param_keys]

        Ngals = len(profile_params[0])
        if Ngals == 0:
            return None, None, None

        total_mass = table[self.prim_haloprop_key]

        vx = self.mc_radial_velocity(scaled_radius, total_mass, *profile_params)
        vy = self.mc_radial_velocity(scaled_radius, total_mass, *profile_params)
        vz = self.mc_radial_velocity(scaled_radius, total_mass, *profile_params)

        if overwrite_table_velocities is True:
            table['vx'][:] += vx
            table['vy'][:] += vy
            table['vz'][:] += vz

        if return_velocities is True:
            return vx, vy, vz


















        
