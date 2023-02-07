"""
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from astropy.table import Table
from astropy.utils.misc import NumpyRNGContext

from halotools.empirical_models import NFWProfile, MonteCarloGalProf, NFWPhaseSpace
from halotools.utils import angles_between_list_of_vectors, vectors_normal_to_planes
from halotools.utils.rotations3d import rotation_matrices_from_angles
from halotools.utils import rotate_vector_collection
from halotools.mock_observables import relative_positions_and_velocities

from halotools.utils.mcrotations import random_unit_vectors_3d
from halotools.utils.rotations3d import rotation_matrices_from_basis


__author__ = ['Andrew Hearin', 'Duncan Campbell']
__all__ = ['AnisotropicNFWPhaseSpace', 'MonteCarloAnisotropicGalProf']


class MonteCarloAnisotropicGalProf(MonteCarloGalProf):
    r"""
    sub-class of MonteCarloGalProf
    """

    def __init__(self):
        r"""
        """

        super(MonteCarloAnisotropicGalProf, self).__init__()

    def mc_pos(self, *profile_params, **kwargs):
        r""" Method to generate random, three-dimensional positions of galaxies.
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
            The sequence must have the same order as ``self.gal_prof_param_keys``.
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
            Random number seed used in the Monte Carlo realization.
            Default is None, which will produce stochastic results.
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
                halo_radius = np.atleast_1d(kwargs['halo_radius'])
                assert len(halo_radius) == len(np.atleast_1d(profile_params[0]))
            except KeyError:
                raise HalotoolsError("\nIf not passing a ``table`` keyword argument "
                    "to mc_pos, must pass the following keyword arguments:\n"
                    "``profile_params``, ``halo_radius``.")
            x, y, z = self.mc_halo_centric_pos(*profile_params, **kwargs)
            if x is None:
                return None
            else:
                return x, y, z

    def mc_halo_centric_pos(self, *profile_params, **kwargs):
        r""" Method to generate random, three-dimensional
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
            The sequence must have the same order as ``self.gal_prof_param_keys``.
        halo_radius : array_like, optional
            Length-Ngals array storing the radial boundary of the halo
            hosting each galaxy. Units assumed to be in Mpc/h.
            If ``profile_params`` and ``halo_radius`` are not passed,
            ``table`` must be passed.
        seed : int, optional
            Random number seed used in the Monte Carlo realization.
            Default is None, which will produce stochastic results.
        Returns
        -------
        x, y, z : arrays
            Length-Ngals array storing a Monte Carlo realization of the galaxy positions.
        """

        x, y, z = self.mc_solid_sphere(*profile_params, **kwargs)
        if x is None:
            return None, None, None

        # Retrieve the halo_radius
        if 'table' in kwargs:
            table = kwargs['table']
            halo_radius = table[self.halo_boundary_key]
        else:
            try:
                halo_radius = np.atleast_1d(kwargs['halo_radius'])
            except KeyError:
                raise HalotoolsError("If not passing an input ``table`` "
                    "keyword argument to mc_halo_centric_pos,\n"
                    "must pass the following keyword arguments:\n"
                    "``halo_radius``, ``profile_params``.")

        x *= halo_radius
        y *= halo_radius
        z *= halo_radius
        return x, y, z

    def mc_unit_sphere(self, Npts, **kwargs):
        r"""
        Returns Npts anisotropically distributed points on the unit sphere.
        Parameters
        ----------
        Npts : int
            Number of 3d points to generate
        seed : int, optional
            Random number seed used in the Monte Carlo realization.
            Default is None, which will produce stochastic results.
        Returns
        -------
        x, y, z : array_like
            Length-Npts arrays of the coordinate positions.
        """

        seed = kwargs.get('seed', None)

        if 'table' in kwargs:
            table = kwargs['table']
            try:
                b_to_a = table['halo_b_to_a']
            except KeyError:
                b_to_a = 1.0
            try:
                c_to_a = table['halo_c_to_a']
            except KeyError:
                c_to_a = 1.0
            try:
                halo_axisA_x = table['halo_axisA_x']
                halo_axisA_y = table['halo_axisA_y']
                halo_axisA_z = table['halo_axisA_z']
            except KeyError:
                with NumpyRNGContext(seed):
                    v = random_unit_vectors_3d(len(table))
                    halo_axisA_x = v[:,0]
                    halo_axisA_y = v[:,1]
                    halo_axisA_z = v[:,2]
            try:
                halo_axisC_x = table['halo_axisC_x']
                halo_axisC_y = table['halo_axisC_y']
                halo_axisC_z = table['halo_axisC_z']
            except KeyError:
                with NumpyRNGContext(seed):
                    v = random_unit_vectors_3d(len(table))
                    halo_axisC_x = v[:,0]
                    halo_axisC_y = v[:,1]
                    halo_axisC_z = v[:,2]
        else:
            try:
                b_to_a = np.atleast_1d(kwargs['b_to_a'])
            except KeyError:
                b_to_a = 1.0
            try:
                c_to_a = np.atleast_1d(kwargs['c_to_a'])
            except KeyError:
                c_to_a = 1.0
            try:
                halo_axisA_x = np.atleast_1d(kwargs['halo_axisA_x'])
                halo_axisA_y = np.atleast_1d(kwargs['halo_axisA_y'])
                halo_axisA_z = np.atleast_1d(kwargs['halo_axisA_z'])
            except KeyError:
                with NumpyRNGContext(seed):
                    v = random_unit_vectors_3d(1)
                    halo_axisC_x = v[:,0]
                    halo_axisC_y = v[:,1]
                    halo_axisC_z = v[:,2]
            try:
                halo_axisC_x = np.atleast_1d(kwargs['halo_axisC_x'])
                halo_axisC_y = np.atleast_1d(kwargs['halo_axisC_y'])
                halo_axisC_z = np.atleast_1d(kwargs['halo_axisC_z'])
            except KeyError:
                with NumpyRNGContext(seed):
                    v = random_unit_vectors_3d(len(halo_axisA_x))
                    halo_axisC_x = v[:,0]
                    halo_axisC_y = v[:,1]
                    halo_axisC_z = v[:,2]

        v1 = np.vstack((halo_axisA_x, halo_axisA_y, halo_axisA_z)).T
        v3 = np.vstack((halo_axisC_x, halo_axisC_y, halo_axisC_z)).T
        v2 = np.cross(v1,v3)

        with NumpyRNGContext(seed):
            phi = np.random.uniform(0, 2*np.pi, Npts)
            uran = np.random.rand(Npts)*2 - 1

        cos_t = uran
        sin_t = np.sqrt((1.-cos_t*cos_t))

        b_to_a, c_to_a = self.anisotropy_bias_response(b_to_a, c_to_a)

        c_to_b = c_to_a/b_to_a

        # temporarily use x-axis as the major axis
        x = 1.0/c_to_a*sin_t * np.cos(phi)
        y = 1.0/c_to_b*sin_t * np.sin(phi)
        z = cos_t
        x_correlated_axes = np.vstack((x, y, z)).T

        x_axes = np.tile((1, 0, 0), Npts).reshape((Npts, 3))
        major_axes = v1

        matrices = rotation_matrices_from_basis(v1,v2,v3)

        # rotate x-axis into the major axis
        #angles = angles_between_list_of_vectors(x_axes, major_axes)
        #rotation_axes = vectors_normal_to_planes(x_axes, major_axes)
        #matrices = rotation_matrices_from_angles(angles, rotation_axes)

        correlated_axes = rotate_vector_collection(matrices, x_correlated_axes)
        #correlated_axes = x_correlated_axes

        return correlated_axes[:, 0], correlated_axes[:, 1], correlated_axes[:, 2]

    def mc_solid_sphere(self, *profile_params, **kwargs):
        r""" Method to generate random, three-dimensional, halo-centric positions of galaxies.
        Parameters
        ----------
        *profile_params : Sequence of arrays
            Sequence of length-Ngals array(s) containing the input profile parameter(s).
            In the simplest case, this sequence has a single element,
            e.g. a single array storing values of the NFW concentrations of the Ngals galaxies.
            More generally, there should be a ``profile_params`` sequence item for
            every parameter in the profile model, each item a length-Ngals array.
            The sequence must have the same order as ``self.gal_prof_param_keys``.
        table : data table, optional
            Astropy Table storing a length-Ngals galaxy catalog.
            If ``table`` is not passed, ``profile_params`` must be passed.
        seed : int, optional
            Random number seed used in the Monte Carlo realization.
            Default is None, which will produce stochastic results.
        Returns
        -------
        x, y, z : arrays
            Length-Ngals array storing a Monte Carlo realization of the galaxy positions.
        """

        # Retrieve the list of profile_params
        if 'table' in kwargs:
            table = kwargs['table']
            profile_params = ([table[profile_param_key]
                for profile_param_key in self.gal_prof_param_keys])
            halo_radius = table[self.halo_boundary_key]
            b_to_a = table['halo_b_to_a']
            c_to_a = table['halo_c_to_a']
        else:
            try:
                assert len(profile_params) > 0
            except AssertionError:
                raise ValueError("If not passing an input ``table`` "
                    "keyword argument to mc_solid_sphere,\n"
                    "must pass a ``profile_params`` keyword argument")

            b_to_a = kwargs['b_to_a']
            c_to_a = kwargs['c_to_a']

        # get random angles
        Ngals = len(np.atleast_1d(profile_params[0]))
        if Ngals == 0:
            return None, None, None

        x, y, z = self.mc_unit_sphere(Ngals, **kwargs)

        # Get the radial positions of the galaxies scaled by the halo radius
        seed = kwargs.get('seed', None)
        if seed is not None:
            seed += 1
        dimensionless_radial_distance = self._mc_dimensionless_radial_distance(
                *profile_params, seed=seed)

        # get random positions within the solid sphere
        x *= dimensionless_radial_distance
        y *= dimensionless_radial_distance
        z *= dimensionless_radial_distance

        a = 1
        b = b_to_a * a
        c = c_to_a * a
        T = (c**2-b**2)/(c**2-a**2)
        q = b/a
        s = c/a

        x *= np.sqrt(q*s)
        y *= np.sqrt(q*s)
        z *= np.sqrt(q*s)

        # Assign the value of the host_centric_distance table column
        if 'table' in kwargs:
            try:
                table['host_centric_distance'][:] = dimensionless_radial_distance
                table['host_centric_distance'][:] *= halo_radius
            except KeyError:
                msg = ("The mc_solid_sphere method of the MonteCarloGalProf class "
                    "requires a table key ``host_centric_distance`` to be pre-allocated ")
                raise ValueError(msg)

        return x, y, z


class AnisotropicNFWPhaseSpace(MonteCarloAnisotropicGalProf, NFWPhaseSpace):
    r"""
    sub-class of NFWPhaseSpace
    """
    def __init__(self, anisotropy_bias=1.0, **kwargs):
        r"""
        Parameters
        ----------
        conc_mass_model : string or callable, optional
            Specifies the function used to model the relation between
            NFW concentration and halo mass.
            Can either be a custom-built callable function,
            or one of the following strings:
            ``dutton_maccio14``, ``direct_from_halo_catalog``.
        cosmology : object, optional
            Instance of an astropy `~astropy.cosmology`.
            Default cosmology is set in
            `~halotools.sim_manager.sim_defaults`.
        redshift : float, optional
            Default is set in `~halotools.sim_manager.sim_defaults`.
        mdef: str, optional
            String specifying the halo mass definition, e.g., 'vir' or '200m'.
            Default is set in `~halotools.empirical_models.model_defaults`.
        concentration_key : string, optional
            Column name of the halo catalog storing NFW concentration.
            This argument is only relevant when ``conc_mass_model``
            is set to ``direct_from_halo_catalog``. In such a case,
            the default value is ``halo_nfw_conc``,
            which is consistent with all halo catalogs provided by Halotools
            but may differ from the convention adopted in custom halo catalogs.
        concentration_bins : ndarray, optional
            Array storing how halo concentrations will be digitized when building
            a lookup table for mock-population purposes.
            The spacing of this array sets a limit on how accurately the
            concentration parameter can be recovered in a likelihood analysis.
        anisotropy_bias : np.float, optional
            a float between math:`[0,\infty]` indicating the axis ratio response of the satellite distribution
            relative to the halo axis ratios.
        Examples
        --------
        >>> model = AnisotropicNFWPhaseSpace()
        """

        super(AnisotropicNFWPhaseSpace, self).__init__(**kwargs)
        self.list_of_haloprops_needed = ['halo_b_to_a', 'halo_c_to_a',
                                         'halo_axisA_x', 'halo_axisA_y', 'halo_axisA_z']

        self.param_dict = ({
            'anisotropy_bias': anisotropy_bias})

    def assign_phase_space(self, table, seed=None):
        r""" Primary method of the `NFWPhaseSpace` class
        called during the mock-population sequence.
        Parameters
        -----------
        table : object
            `~astropy.table.Table` storing halo catalog.
            After calling the `assign_phase_space` method,
            the `x`, `y`, `z`, `vx`, `vy`, and `vz`
            columns of the input ``table`` will be over-written
            with their host-centric values.
        seed : int, optional
            Random number seed used in the Monte Carlo realization.
            Default is None, which will produce stochastic results.
        """
        self.mc_pos(self, table=table, seed=seed)
        if seed is not None:
            seed += 1
        MonteCarloGalProf.mc_vel(self, table=table, seed=seed)


    def mc_generate_nfw_phase_space_points(self, Ngals=int(1e4),
            conc=5, mass=1e12, b_to_a=0.7, c_to_a=0.5,
            halo_axisA_x=1.0, halo_axisA_y=0.0, halo_axisA_z=0.0,
            halo_axisC_x=1.0, halo_axisC_y=0.0, halo_axisC_z=0.0,
             verbose=True, seed=None):
        r""" Return a Monte Carlo realization of points
        in the phase space of an NFW halo in isotropic Jeans equilibrium.
        Parameters
        -----------
        Ngals : int, optional
            Number of galaxies in the Monte Carlo realization of the
            phase space distribution. Default is 1e4.
        conc : float, optional
            Concentration of the NFW profile being realized.
            Default is 5.
        mass : float, optional
            Mass of the halo whose phase space distribution is being realized
            in units of Msun/h. Default is 1e12.
        verbose : bool, optional
            If True, a message prints with an estimate of the build time.
            Default is True.
        seed : int, optional
            Random number seed used in the Monte Carlo realization.
            Default is None, which will produce stochastic results.
        Returns
        --------
        t : table
            `~astropy.table.Table` containing the Monte Carlo realization of the
            phase space distribution.
            Keys are 'x', 'y', 'z', 'vx', 'vy', 'vz', 'radial_position', 'radial_velocity'.
            Length units in Mpc/h, velocity units in km/s.
        Examples
        ---------
        >>> nfw = AnisotropicNFWPhaseSpace()
        >>> mass, conc, b_to_a, c_to_a = 1e13, 8., 0.9, 0.6
        >>> data = nfw.mc_generate_nfw_phase_space_points(Ngals=100, mass=mass, conc=conc, b_to_a=b_to_a, c_to_a=c_to_a, verbose=False)
        Now suppose you wish to compute the radial velocity dispersion of all the returned points:
        >>> vrad_disp = np.std(data['radial_velocity'])
        If you wish to do the same calculation but for points in a specific range of radius:
        >>> mask = data['radial_position'] < 0.1
        >>> vrad_disp_inner_points = np.std(data['radial_velocity'][mask])
        You may also wish to select points according to their distance to the halo center
        in units of the virial radius. In such as case, you can use the
        `~halotools.empirical_models.NFWPhaseSpace.halo_mass_to_halo_radius`
        method to scale the halo-centric distances. Here is an example
        of how to compute the velocity dispersion in the z-dimension of all points
        residing within :math:`R_{\rm vir}/2`:
        >>> halo_radius = nfw.halo_mass_to_halo_radius(mass)
        >>> scaled_radial_positions = data['radial_position']/halo_radius
        >>> mask = scaled_radial_positions < 0.5
        >>> vz_disp_inner_half = np.std(data['vz'][mask])
        """

        m = np.zeros(Ngals) + mass
        c = np.zeros(Ngals) + conc
        halo_axisA_x = np.zeros(Ngals) + halo_axisA_x
        halo_axisA_y = np.zeros(Ngals) + halo_axisA_y
        halo_axisA_z = np.zeros(Ngals) + halo_axisA_z
        halo_axisC_x = np.zeros(Ngals) + halo_axisC_x
        halo_axisC_y = np.zeros(Ngals) + halo_axisC_y
        halo_axisC_z = np.zeros(Ngals) + halo_axisC_z
        rvir = NFWProfile.halo_mass_to_halo_radius(self, total_mass=m)


        new_b_to_a, new_c_to_a = self.anisotropy_bias_response(b_to_a, c_to_a)

        print('here 1:')
        x, y, z = self.mc_halo_centric_pos(c,
            halo_radius=rvir,
            b_to_a=new_b_to_a,
            c_to_a=new_c_to_a,
            halo_axisA_x=halo_axisA_x,
            halo_axisA_y=halo_axisA_y,
            halo_axisA_z=halo_axisA_z,
            halo_axisC_x=halo_axisC_x,
            halo_axisC_y=halo_axisC_y,
            halo_axisC_z=halo_axisC_z,
            seed=seed)
        r = np.sqrt(x**2 + y**2 + z**2)
        scaled_radius = r/rvir

        if seed is not None:
            seed += 1
        vx = self.mc_radial_velocity(scaled_radius, m, c, seed=seed)
        if seed is not None:
            seed += 1
        vy = self.mc_radial_velocity(scaled_radius, m, c, seed=seed)
        if seed is not None:
            seed += 1
        vz = self.mc_radial_velocity(scaled_radius, m, c, seed=seed)

        xrel, vxrel = relative_positions_and_velocities(x, 0, v1=vx, v2=0)
        yrel, vyrel = relative_positions_and_velocities(y, 0, v1=vy, v2=0)
        zrel, vzrel = relative_positions_and_velocities(z, 0, v1=vz, v2=0)

        vrad = (xrel*vxrel + yrel*vyrel + zrel*vzrel)/r

        t = Table({'x': x, 'y': y, 'z': z,
            'vx': vx, 'vy': vy, 'vz': vz,
            'radial_position': r, 'radial_velocity': vrad})

        return t

    def anisotropy_bias_response(self, b_to_a, c_to_a):
        """
        return new axis ratios
        """
        beta = self.param_dict['anisotropy_bias']
        return b_to_a**beta, c_to_a**beta
