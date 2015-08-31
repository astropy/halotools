# -*- coding: utf-8 -*-
"""
Module composes the behavior of the profile models 
and the velocity models to produce models for the 
full phase space distribution of galaxies within their halos. 
"""

__author__ = ['Andrew Hearin']

class MonteCarloGalProf(object):
    """ Orthogonal mix-in class used to turn an analytical 
    phase space model into a class that can be used 
    to generate the phase space distribution 
    of a mock galaxy population. 
    """
    def __init__(self, **kwargs):
        """
        """
        pass

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
            param_bins = getattr(self, param_key + '_lookup_table_bins')
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
            self.func_table_indices[digitized_param_list]
            )
        # Now we have an array of indices for our functions, and we need to evaluate 
        # the i^th function on the i^th element of rho. 
        # Call the model_helpers module to access generic code for doing this.
        # (Remember that the interpolation is being done in log-space)
        return 10.**model_helpers.call_func_table(
            self.cumu_inv_func_table, np.log10(rho), func_table_indices)

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









        
