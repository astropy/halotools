"""
Module defining the `~halotools.empirical_models.NFWPhaseSpace` class
governing the phase space distribution of massless tracers of an NFW potential.
"""
from __future__ import (
    division, print_function, absolute_import)

import numpy as np
from astropy.table import Table

from .profile_models import NFWProfile
from .velocity_models import NFWJeansVelocity
from .monte_carlo_helpers import MonteCarloGalProf

from .. import model_defaults

__author__ = ['Andrew Hearin']
__all__ = ['NFWPhaseSpace']


class NFWPhaseSpace(NFWProfile, NFWJeansVelocity, MonteCarloGalProf):
    """ Model for the phase space distribution of mass and/or galaxies
    in isotropic Jeans equilibrium in an NFW halo profile, based on Navarro, Frenk and White (1995).

    For a review of the mathematics underlying the NFW profile,
    including descriptions of how the relevant equations are
    implemented in the Halotools code base, see :ref:`nfw_profile_tutorial`.

    Testing for this class is done in the
    `~halotools.empirical_models.TestNFWPhaseSpace` class.

    """

    def __init__(self, high_precision=False, **kwargs):
        """
        Parameters
        ----------
        conc_mass_model : string, optional
            Specifies the calibrated fitting function used to model the concentration-mass relation.
            Default is set in `~halotools.sim_manager.sim_defaults`.

        cosmology : object, optional
            Astropy cosmology object. Default is set in `~halotools.sim_manager.sim_defaults`.

        redshift : float, optional
            Default is set in `~halotools.sim_manager.sim_defaults`.

        mdef: str
            String specifying the halo mass definition, e.g., 'vir' or '200m'.
            Default is set in `~halotools.empirical_models.model_defaults`.

        concentration_binning : tuple, optional
            Three-element tuple. The first entry will be the minimum
            value of the concentration in the lookup table,
            the second entry the maximum, the third entry
            the linear spacing of the grid.

        high_precision : bool, optional
            If set to True, concentration binning width is equal to
            to ``default_high_prec_dconc`` in `~halotools.empirical_models.model_defaults`.
            If False, spacing is 0.5. Default is False.

        Notes
        ------
        This model is tested by `~halotools.empirical_models.TestNFWPhaseSpace`.

        """
        NFWProfile.__init__(self, **kwargs)
        NFWJeansVelocity.__init__(self, **kwargs)
        MonteCarloGalProf.__init__(self)

        if 'concentration_binning' in kwargs:
            cmin, cmax, dc = kwargs['concentration_binning']
        elif high_precision is True:
            cmin, cmax, dc = (
                model_defaults.min_permitted_conc,
                model_defaults.max_permitted_conc,
                model_defaults.default_high_prec_dconc
                )
        else:
            cmin, cmax, dc = (
                model_defaults.min_permitted_conc, model_defaults.max_permitted_conc, 0.5
                )
        MonteCarloGalProf.setup_prof_lookup_tables(self, (cmin, cmax, dc))

        self._mock_generation_calling_sequence = ['assign_phase_space']

    def assign_phase_space(self, table):
        """ Primary method of the `NFWPhaseSpace` class called during the mock-population sequence.

        Parameters
        -----------
        table : object, optional
            Data table storing halo catalog.
            After calling the `assign_phase_space` method, the `x`, `y`, `z`, `vx`, `vy`, and `vz`
            columns of the input ``table`` will be over-written.

        Notes
        ------
        The behavior of this method is actually defined in the following two methods of the
        `~halotools.empirical_models.monte_carlo_helpers.MonteCarloGalProf` class:

        * `~halotools.empirical_models.monte_carlo_helpers.MonteCarloGalProf.mc_pos`

        * `~halotools.empirical_models.monte_carlo_helpers.MonteCarloGalProf.mc_vel`

        """
        MonteCarloGalProf.mc_pos(self, table=table)
        MonteCarloGalProf.mc_vel(self, table=table)

    def mc_generate_nfw_phase_space_points(self, Ngals=int(1e4), conc=5, mass=1e12, verbose=True):
        """ Stand-alone convenience function for returning
        a Monte Carlo realization of points in the phase space of an NFW halo in isotropic Jeans equilibrium.

        Parameters
        -----------
        Ngals : int, optional
            Number of galaxies in the Monte Carlo realization of the
            phase space distribution. Default is 1e4.

        conc : float, optional
            Concentration of the NFW profile being realized.
            Default is 5.

        mass : float, optional
            Mass of the halo whose phase space distribution is being realized.
            Default is 1e12.

        verbose : bool, optional
            If True, a message prints with an estimate of the build time.
            Default is True.

        Returns
        --------
        t : table
            `~astropy.table.Table` containing the Monte Carlo realization of the
            phase space distribution.
            Keys are 'x', 'y', 'z', 'vx', 'vy', 'vz', 'radial_position', 'radial_velocity'.

        Examples
        ---------
        >>> nfw = NFWPhaseSpace()
        >>> mass, conc = 1e13, 8.
        >>> data = nfw.mc_generate_nfw_phase_space_points(Ngals = 100, mass = mass, conc = conc, verbose=False)

        Now suppose you wish to compute the radial velocity dispersion of all the returned points:

        >>> vrad_disp = np.std(data['radial_velocity'])

        If you wish to do the same calculation but for points in a specific range of radius:

        >>> mask = data['radial_position'] < 0.1
        >>> vrad_disp_inner_points = np.std(data['radial_velocity'][mask])

        You may also wish to select points according to their distance to the halo center
        in units of the virial radius. In such as case, you can use the
        `halo_mass_to_halo_radius` method to scale the halo-centric distances. Here is an example
        of how to compute the velocity dispersion in the z-dimension of all points
        residing within :math:`R_{\\rm vir}/2`:

        >>> halo_radius = nfw.halo_mass_to_halo_radius(mass)
        >>> scaled_radial_positions = data['radial_position']/halo_radius
        >>> mask = scaled_radial_positions < 0.5
        >>> vz_disp_inner_half = np.std(data['vz'][mask])

        """

        m = np.zeros(Ngals) + mass
        c = np.zeros(Ngals) + conc
        rvir = NFWProfile.halo_mass_to_halo_radius(self, total_mass=m)

        x, y, z = MonteCarloGalProf.mc_halo_centric_pos(self, c,
            halo_radius=rvir)
        r = np.sqrt(x**2 + y**2 + z**2)
        scaled_radius = r/rvir

        vrad = MonteCarloGalProf.mc_radial_velocity(self, scaled_radius, m, c)
        vx = MonteCarloGalProf.mc_radial_velocity(self, scaled_radius, m, c)
        vy = MonteCarloGalProf.mc_radial_velocity(self, scaled_radius, m, c)
        vz = MonteCarloGalProf.mc_radial_velocity(self, scaled_radius, m, c)

        t = Table({'x': x, 'y': y, 'z': z,
            'vx': vx, 'vy': vy, 'vz': vz,
            'radial_position': r, 'radial_velocity': vrad})

        return t

    def conc_NFWmodel(self, **kwargs):
        """ Method computes the NFW concentration
        as a function of the input halos according to the
        ``conc_mass_model`` bound to the `NFWProfile` instance.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which
            occupation statistics are based.
            If ``prim_haloprop`` is not passed,
            then ``table`` keyword argument must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed,
            then ``prim_haloprop`` keyword argument must be passed.

        Returns
        -------
        c : array_like
            Concentrations of the input halos.

        Notes
        ------
        The behavior of this function is not defined here, but in the
        `~halotools.empirical_models.ConcMass` class.

        This method is tested by `~halotools.empirical_models.test_conc_mass.TestConcMass` class.

        """
        return NFWProfile.compute_concentration(self, **kwargs)

    def dimensionless_mass_density(self, scaled_radius, conc):
        """
        Physical density of the NFW halo scaled by the density threshold of the mass definition:

        The `dimensionless_mass_density` is defined as
        :math:`\\tilde{\\rho}_{\\rm prof}(\\tilde{r}) \\equiv \\rho_{\\rm prof}(\\tilde{r}) / \\rho_{\\rm thresh}`,
        where :math:`\\tilde{r}\\equiv r/R_{\\Delta}`.

        For an NFW halo,
        :math:`\\tilde{\\rho}_{\\rm NFW}(\\tilde{r}, c) = \\frac{c^{3}}{3g(c)}\\times\\frac{1}{c\\tilde{r}(1 + c\\tilde{r})^{2}},`

        where :math:`g(x) \\equiv \\int_{0}^{x}dy\\frac{y}{(1+y)^{2}} = \\log(1+x) - x / (1+x)` is computed using the `g` function.

        The quantity :math:`\\rho_{\\rm thresh}` is a function of
        the halo mass definition, cosmology and redshift,
        and is computed via the
        `~halotools.empirical_models.profile_helpers.density_threshold` function.
        The quantity :math:`\\rho_{\\rm prof}` is the physical mass density of the
        halo profile and is computed via the `mass_density` function.

        Parameters
        -----------
        scaled_radius : array_like
            Halo-centric distance *r* scaled by the halo boundary :math:`R_{\\Delta}`, so that
            :math:`0 <= \\tilde{r} \\equiv r/R_{\\Delta} <= 1`. Can be a scalar or numpy array.

        conc : array_like
            Value of the halo concentration. Can either be a scalar, or a numpy array
            of the same dimension as the input ``scaled_radius``.

        Returns
        -------
        dimensionless_density: array_like
            Dimensionless density of a dark matter halo
            at the input ``scaled_radius``, normalized by the
            `~halotools.empirical_models.profile_helpers.density_threshold`
            :math:`\\rho_{\\rm thresh}` for the
            halo mass definition, cosmology, and redshift.
            Result is an array of the dimension as the input ``scaled_radius``.

        Notes
        -----

        This method is tested by
        `~halotools.empirical_models.test_nfw_profile.TestNFWProfile.test_mass_density` function.

        """
        return NFWProfile.dimensionless_mass_density(self, scaled_radius, conc)

    def mass_density(self, radius, mass, conc):
        """
        Physical density of the halo at the input radius,
        given in units of :math:`h^{3}/{\\rm Mpc}^{3}`.

        Parameters
        -----------
        radius : array_like
            Halo-centric distance in Mpc/h units; can be a scalar or numpy array

        mass : array_like
            Total mass of the halo; can be a scalar or numpy array of the same
            dimension as the input ``radius``.

        conc : array_like
            Value of the halo concentration. Can either be a scalar, or a numpy array
            of the same dimension as the input ``radius``.

        Returns
        -------
        density: array_like
            Physical density of a dark matter halo of the input ``mass``
            at the input ``radius``. Result is an array of the
            dimension as the input ``radius``, reported in units of :math:`h^{3}/Mpc^{3}`.

        Examples
        --------
        >>> model = NFWProfile()
        >>> Npts = 100
        >>> radius = np.logspace(-2, -1, Npts)
        >>> mass = np.zeros(Npts) + 1e12
        >>> conc = 5
        >>> result = model.mass_density(radius, mass, conc)
        >>> concarr = np.linspace(1, 100, Npts)
        >>> result = model.mass_density(radius, mass, concarr)

        Notes
        ------
        See :ref:`halo_profile_definitions` for derivations and implementation details.

        This method is tested by
        `~halotools.empirical_models.test_nfw_profile.TestNFWProfile.test_mass_density` function.

        """
        return NFWProfile.mass_density(self, radius, mass, conc)

    def g(self, x):
        """ Convenience function used to evaluate the profile.

            :math:`g(x) \\equiv \\int_{0}^{x}dy\\frac{y}{(1+y)^{2}} = \\log(1+x) - x / (1+x)`

        Parameters
        ----------
        x : array_like

        Returns
        -------
        g : array_like

        Examples
        --------
        >>> model = NFWProfile()
        >>> result = model.g(1)
        >>> Npts = 25
        >>> result = model.g(np.logspace(-1, 1, Npts))
        """
        return NFWProfile.g(self, x)

    def cumulative_mass_PDF(self, scaled_radius, conc):
        """
        Analytical result for the fraction of the total mass
        enclosed within dimensionless radius of an NFW halo,

        :math:`P_{\\rm NFW}(<\\tilde{r}) \equiv M_{\\Delta}(<\\tilde{r}) / M_{\\Delta} = g(c\\tilde{r})/g(\\tilde{r}),`

        where :math:`g(x) \\equiv \\int_{0}^{x}dy\\frac{y}{(1+y)^{2}} = \\log(1+x) - x / (1+x)` is computed
        using `g`, and where :math:`\\tilde{r} \\equiv r / R_{\\Delta}`.

        Parameters
        -------------
        scaled_radius : array_like
            Halo-centric distance *r* scaled by the halo boundary :math:`R_{\\Delta}`, so that
            :math:`0 <= \\tilde{r} \\equiv r/R_{\\Delta} <= 1`. Can be a scalar or numpy array.

        conc : array_like
            Value of the halo concentration. Can either be a scalar, or a numpy array
            of the same dimension as the input ``scaled_radius``.

        Returns
        -------------
        p: array_like
            The fraction of the total mass enclosed
            within the input ``scaled_radius``, in :math:`M_{\odot}/h`;
            has the same dimensions as the input ``scaled_radius``.

        Examples
        --------
        >>> model = NFWProfile()
        >>> Npts = 100
        >>> scaled_radius = np.logspace(-2, 0, Npts)
        >>> conc = 5
        >>> result = model.cumulative_mass_PDF(scaled_radius, conc)
        >>> concarr = np.linspace(1, 100, Npts)
        >>> result = model.cumulative_mass_PDF(scaled_radius, concarr)

        Notes
        ------
        See :ref:`halo_profile_definitions` for derivations and implementation details.

        This method is tested by
        `~halotools.empirical_models.test_nfw_profile.TestNFWProfile.test_cumulative_mass_PDF` function.

        """
        return NFWProfile.cumulative_mass_PDF(self, scaled_radius, conc)

    def enclosed_mass(self, radius, total_mass, conc):
        """
        The mass enclosed within the input radius, :math:`M(<r) = 4\\pi\\int_{0}^{r}dr'r'^{2}\\rho(r)`.

        Parameters
        -----------
        radius : array_like
            Halo-centric distance in Mpc/h units; can be a scalar or numpy array

        total_mass : array_like
            Total mass of the halo; can be a scalar or numpy array of the same
            dimension as the input ``radius``.

        conc : array_like
            Value of the halo concentration. Can either be a scalar, or a numpy array
            of the same dimension as the input ``radius``.

        Returns
        ----------
        enclosed_mass: array_like
            The mass enclosed within radius r, in :math:`M_{\odot}/h`;
            has the same dimensions as the input ``radius``.

        Examples
        --------
        >>> model = NFWProfile()
        >>> Npts = 100
        >>> radius = np.logspace(-2, -1, Npts)
        >>> total_mass = np.zeros(Npts) + 1e12
        >>> conc = 5
        >>> result = model.enclosed_mass(radius, total_mass, conc)
        >>> concarr = np.linspace(1, 100, Npts)
        >>> result = model.enclosed_mass(radius, total_mass, concarr)

        Notes
        ------
        See :ref:`halo_profile_definitions` for derivations and implementation details.

        This method is tested by
        `~halotools.empirical_models.test_nfw_profile.TestNFWProfile.test_cumulative_mass_PDF` function.

        """
        return NFWProfile.enclosed_mass(self, radius, total_mass, conc)

    def virial_velocity(self, total_mass):
        """ The circular velocity evaluated at the halo boundary,
        :math:`V_{\\rm vir} \\equiv \\sqrt{GM_{\\rm halo}/R_{\\rm halo}}`.

        Parameters
        --------------
        total_mass : array_like
            Total mass of the halo; can be a scalar or numpy array.

        Returns
        --------
        vvir : array_like
            Virial velocity in km/s.

        Examples
        --------
        >>> model = NFWProfile()
        >>> Npts = 100
        >>> mass_array = np.logspace(11, 15, Npts)
        >>> vvir_array = model.virial_velocity(mass_array)

        Notes
        ------
        See :ref:`halo_profile_definitions` for derivations and implementation details.

        """
        return NFWProfile.virial_velocity(self, total_mass)

    def circular_velocity(self, radius, total_mass, conc):
        """
        The circular velocity, :math:`V_{\\rm cir} \\equiv \\sqrt{GM(<r)/r}`,
        as a function of halo-centric distance r.

        Parameters
        --------------
        radius : array_like
            Halo-centric distance in Mpc/h units; can be a scalar or numpy array

        total_mass : array_like
            Total mass of the halo; can be a scalar or numpy array of the same
            dimension as the input ``radius``.

        conc : array_like
            Value of the halo concentration. Can either be a scalar, or a numpy array
            of the same dimension as the input ``radius``.

        Returns
        ----------
        vc: array_like
            The circular velocity in km/s; has the same dimensions as the input ``radius``.

        Examples
        --------
        >>> model = NFWProfile()
        >>> Npts = 100
        >>> radius = np.logspace(-2, -1, Npts)
        >>> total_mass = np.zeros(Npts) + 1e12
        >>> conc = 5
        >>> result = model.circular_velocity(radius, total_mass, conc)
        >>> concarr = np.linspace(1, 100, Npts)
        >>> result = model.circular_velocity(radius, total_mass, concarr)

        Notes
        ------
        See :ref:`halo_profile_definitions` for derivations and implementation details.

        This method is tested by
        `~halotools.empirical_models.test_nfw_profile.TestNFWProfile.test_vmax` function.
        """
        return NFWProfile.circular_velocity(self, radius, total_mass, conc)

    def vmax(self, total_mass, conc):
        """ Maximum circular velocity of the halo profile.

        Parameters
        ----------
        total_mass: array_like
            Total halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.

        conc : array_like
            Value of the halo concentration. Can either be a scalar, or a numpy array
            of the same dimension as the input ``total_mass``.

        Returns
        --------
        vmax : array_like
            :math:`V_{\\rm max}` in km/s.

        Examples
        --------
        >>> model = NFWProfile()
        >>> Npts = 100
        >>> total_mass = np.zeros(Npts) + 1e12
        >>> conc = 5
        >>> result = model.vmax(total_mass, conc)
        >>> concarr = np.linspace(1, 100, Npts)
        >>> result = model.vmax(total_mass, concarr)

        Notes
        ------
        See :ref:`halo_profile_definitions` for derivations and implementation details.

        This method is tested by
        `~halotools.empirical_models.test_nfw_profile.TestNFWProfile.test_vmax` function,
        and also the
        `~halotools.empirical_models.test_halo_catalog_nfw_consistency.TestHaloCatNFWConsistency.test_vmax_consistency` function.

        """
        return NFWProfile.vmax(self, total_mass, conc)

    def halo_mass_to_halo_radius(self, total_mass):
        """
        Spherical overdensity radius as a function of the input mass.

        Note that this function is independent of the form of the density profile.

        Parameters
        ----------
        total_mass: array_like
            Total halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.

        Returns
        -------
        radius : array_like
            Radius of the halo in Mpc/h units.
            Will have the same dimension as the input ``total_mass``.

        Examples
        --------
        >>> model = NFWProfile()
        >>> halo_radius = model.halo_mass_to_halo_radius(1e13)

        Notes
        ------
        This function is tested with the
        `~halotools.empirical_models.test_profile_helpers.TestProfileHelpers.test_halo_mass_to_halo_radius` function.

        """
        return NFWProfile.halo_mass_to_halo_radius(self, total_mass)

    def halo_radius_to_halo_mass(self, radius):
        """
        Spherical overdensity mass as a function of the input radius.

        Note that this function is independent of the form of the density profile.

        Parameters
        ------------
        radius : array_like
            Radius of the halo in Mpc/h units; can be a number or a numpy array.

        Returns
        ----------
        total_mass: array_like
            Total halo mass in :math:`M_{\odot}/h`.
            Will have the same dimension as the input ``radius``.

        Examples
        --------
        >>> model = NFWProfile()
        >>> halo_mass = model.halo_mass_to_halo_radius(500.)

        Notes
        ------
        This function is tested with the
        `~halotools.empirical_models.test_profile_helpers.TestProfileHelpers.test_halo_radius_to_halo_mass` function.

        """
        return NFWProfile.halo_radius_to_halo_mass(self, radius)

    def dimensionless_radial_velocity_dispersion(self, scaled_radius, *conc):
        """
        Analytical solution to the isotropic jeans equation for an NFW potential,
        rendered dimensionless via scaling by the virial velocity.

        :math:`\\tilde{\\sigma}^{2}_{r}(\\tilde{r})\\equiv\\sigma^{2}_{r}(\\tilde{r})/V_{\\rm vir}^{2} = \\frac{c^{2}\\tilde{r}(1 + c\\tilde{r})^{2}}{g(c)}\int_{c\\tilde{r}}^{\infty}{\\rm d}y\\frac{g(y)}{y^{3}(1 + y)^{2}}`

        See :ref:`nfw_jeans_velocity_profile_derivations` for derivations and implementation details.

        Parameters
        -----------
        scaled_radius : array_like
            Length-Ngals numpy array storing the halo-centric distance
            *r* scaled by the halo boundary :math:`R_{\\Delta}`, so that
            :math:`0 <= \\tilde{r} \\equiv r/R_{\\Delta} <= 1`.

        total_mass: array_like
            Length-Ngals numpy array storing the halo mass in :math:`M_{\odot}/h`.

        conc : float
            Concentration of the halo.

        Returns
        -------
        result : array_like
            Radial velocity dispersion profile scaled by the virial velocity.
            The returned result has the same dimension as the input ``scaled_radius``.
        """
        return NFWJeansVelocity.dimensionless_radial_velocity_dispersion(self, scaled_radius, *conc)

    def setup_prof_lookup_tables(self, *concentration_binning):
        """
        This method sets up how we will digitize halo concentrations during mock population.

        After calling the `setup_prof_lookup_tables` method, the
        `NFWPhaseSpace` instance will have three new private attributes bound to it:

        * ``_conc_NFWmodel_lookup_table_min``

        * ``_conc_NFWmodel_lookup_table_max``

        * ``_conc_NFWmodel_lookup_table_spacing``

        These three attributes define the linear spacing of the ``conc_NFWmodel`` parameter
        lookup table created by the `build_lookup_tables` method.

        Parameters
        ----------
        *concentration_binning : sequence
            Sequence of three inputs determining the binning.
            The first entry will be the minimum
            value of the concentration in the lookup table,
            the second entry the maximum, the third entry
            the linear spacing of the grid.

        """

        MonteCarloGalProf.setup_prof_lookup_tables(self, *concentration_binning)

    def build_lookup_tables(self,
            logrmin=model_defaults.default_lograd_min,
            logrmax=model_defaults.default_lograd_max,
            Npts_radius_table=model_defaults.Npts_radius_table):
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
        MonteCarloGalProf.build_lookup_tables(self, logrmin, logrmax, Npts_radius_table)

    def _mc_dimensionless_radial_distance(self, concentration_array, **kwargs):
        """ Method to generate Monte Carlo realizations of the profile model.

        Parameters
        ----------
        concentration_array : array_like
            Length-Ngals numpy array storing the concentrations of the mock galaxies.

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
        This method is tested by the
        `~halotools.empirical_models.test_phase_space.TestNFWPhaseSpace.test_mc_dimensionless_radial_distance` function.
        """
        return MonteCarloGalProf._mc_dimensionless_radial_distance(
            self, concentration_array, **kwargs)

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
        This method is tested by the
        `~halotools.empirical_models.test_phase_space.TestNFWPhaseSpace.test_mc_unit_sphere` function.

        """
        return MonteCarloGalProf.mc_unit_sphere(self, Npts, **kwargs)

    def mc_solid_sphere(self, *concentration_array, **kwargs):
        """ Method to generate random, three-dimensional, halo-centric positions of galaxies.

        Parameters
        ----------
        concentration_array : array_like, optional
            Length-Ngals numpy array storing the concentrations of the mock galaxies.

        table : data table, optional
            Astropy Table storing a length-Ngals galaxy catalog.
            If ``table`` is not passed, ``concentration_array`` must be passed.

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
        return MonteCarloGalProf.mc_solid_sphere(self, *concentration_array, **kwargs)

    def mc_halo_centric_pos(self, *concentration_array, **kwargs):
        """ Method to generate random, three-dimensional
        halo-centric positions of galaxies.

        Parameters
        ----------
        table : data table, optional
            Astropy Table storing a length-Ngals galaxy catalog.
            If ``table`` is not passed, ``concentration_array`` and
            keyword argument ``halo_radius`` must be passed.

        concentration_array : array_like, optional
            Length-Ngals numpy array storing the concentrations of the mock galaxies.
            If ``table`` is not passed, ``concentration_array`` and
            keyword argument ``halo_radius`` must be passed.

        halo_radius : array_like, optional
            Length-Ngals array storing the radial boundary of the halo
            hosting each galaxy. Units assumed to be in Mpc/h.
            If ``concentration_array`` and ``halo_radius`` are not passed,
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
        return MonteCarloGalProf.mc_halo_centric_pos(self, *concentration_array, **kwargs)

    def mc_pos(self, *concentration_array, **kwargs):
        """ Method to generate random, three-dimensional positions of galaxies.

        Parameters
        ----------
        table : data table, optional
            Astropy Table storing a length-Ngals galaxy catalog.
            If ``table`` is not passed, ``concentration_array`` and ``halo_radius`` must be passed.

        concentration_array : array_like, optional
            Length-Ngals numpy array storing the concentrations of the mock galaxies.
            If ``table`` is not passed, ``concentration_array`` and
            keyword argument ``halo_radius`` must be passed.
            If ``concentration_array`` is passed, ``halo_radius`` must be passed as a keyword argument.
            The sequence must have the same order as ``self.prof_param_keys``.

        halo_radius : array_like, optional
            Length-Ngals array storing the radial boundary of the halo
            hosting each galaxy. Units assumed to be in Mpc/h.
            If ``concentration_array`` and ``halo_radius`` are not passed,
            ``table`` must be passed.

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
        return MonteCarloGalProf.mc_pos(self, *concentration_array, **kwargs)

    def _vrad_disp_from_lookup(self, scaled_radius, *concentration_array, **kwargs):
        """ Method to generate Monte Carlo realizations of the profile model.

        Parameters
        ----------
        scaled_radius : array_like
            Halo-centric distance *r* scaled by the halo boundary :math:`R_{\\Delta}`, so that
            :math:`0 <= \\tilde{r} \\equiv r/R_{\\Delta} <= 1`. Can be a scalar or numpy array.

        concentration_array : array_like
            Length-Ngals numpy array storing the concentrations of the mock galaxies.

        Returns
        -------
        sigma_vr : array
            Length-Ngals array containing the radial velocity dispersion
            of galaxies within their halos,
            scaled by the size of the halo's virial velocity.

        Notes
        ------
        This method is tested by the `~halotools.empirical_models.test_phase_space.TestNFWPhaseSpace.test_vrad_disp_from_lookup` function.
        """
        return MonteCarloGalProf._vrad_disp_from_lookup(self,
            scaled_radius, *concentration_array, **kwargs)

    def mc_radial_velocity(self, scaled_radius, total_mass, *concentration_array, **kwargs):
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

        concentration_array : array_like
            Length-Ngals numpy array storing the concentrations of the mock galaxies.

        seed : int, optional
            Random number seed used in Monte Carlo realization. Default is None.

        Returns
        -------
        radial_velocities : array_like
            Array of radial velocities drawn from Gaussians with a width determined by the
            solution to the Jeans equation.

        Notes
        ------
        This method is tested by the `~halotools.empirical_models.test_phase_space.TestNFWPhaseSpace.test_mc_radial_velocity` function.
        """
        return MonteCarloGalProf.mc_radial_velocity(self,
            scaled_radius, total_mass, *concentration_array, **kwargs)

    def mc_vel(self, table):
        """ Method assigns a Monte Carlo realization of the Jeans velocity
        solution to the halos in the input ``table``.

        Parameters
        -----------
        table : Astropy Table
            `astropy.table.Table` object storing the halo catalog.
            Calling the `mc_vel` method will over-write the existing values of
            the ``vx``, ``vy`` and ``vz`` columns.

        Notes
        ------
        This method is tested by the `~halotools.empirical_models.test_phase_space.TestNFWPhaseSpace.test_mc_vel` function.
        """
        MonteCarloGalProf.mc_vel(self, table)
