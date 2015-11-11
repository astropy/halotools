# -*- coding: utf-8 -*-
"""
Module defining the `~halotools.empirical_models.phase_space_models.NFWPhaseSpace` class 
governing the phase space distribution of massless tracers of an NFW potential. 
"""
from __future__ import (
    division, print_function, absolute_import)

__author__ = ['Andrew Hearin']
__all__ = ['NFWPhaseSpace']

import numpy as np
from astropy.table import Table

from .profile_models import NFWProfile
from .velocity_models import NFWJeansVelocity
from .monte_carlo_helpers import MonteCarloGalProf

from .. import model_defaults

from ...sim_manager import sim_defaults


class NFWPhaseSpace(NFWProfile, NFWJeansVelocity, MonteCarloGalProf):
    """ Model for the phase space distribution of mass and/or galaxies in isotropic Jeans equilibrium in an NFW halo profile, based on Navarro, Frenk and White (1999).

    For a review of the mathematics underlying the NFW profile, 
    including descriptions of how the relevant equations are 
    implemented in the Halotools code base, see :ref:`nfw_profile_tutorial`. 

    Testing for this class is done in the 
    `~halotools.empirical_models.phase_space_models.tests.TestNFWPhaseSpace` class. 

    """

    def __init__(self, high_precision = False, **kwargs):
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
        This model is tested by `~halotools.empirical_models.phase_space_models.tests.TestNFWPhaseSpace`. 

        """        
        NFWProfile.__init__(self, **kwargs)
        NFWJeansVelocity.__init__(self, **kwargs)
        MonteCarloGalProf.__init__(self)

        if 'concentration_binning' in kwargs:
            cmin, cmax, dc = kwargs['concentration_binning']
        elif high_precision == True:
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

    def assign_phase_space(self, halo_table):
        """ Primary method of the `NFWPhaseSpace` class called during the mock-population sequence. 

        Parameters 
        -----------
        halo_table : object, optional  
            Data table storing halo catalog. 
            After calling the `assign_phase_space` method, the `x`, `y`, `z`, `vx`, `vy`, and `vz` 
            columns of the input ``halo_table`` will be over-written. 

        Notes 
        ------
        The behavior of this method is actually defined in the following two methods of the 
        `~halotools.empirical_models.phase_space_models.monte_carlo_helpers.MonteCarloGalProf` class: 

        * `~halotools.empirical_models.phase_space_models.monte_carlo_helpers.MonteCarloGalProf.mc_pos`

        * `~halotools.empirical_models.phase_space_models.monte_carlo_helpers.MonteCarloGalProf.mc_vel`

        """
        MonteCarloGalProf.mc_pos(self, halo_table = halo_table)
        MonteCarloGalProf.mc_vel(self, halo_table = halo_table)


    def mc_generate_phase_space_points(self, Ngals = 1e4, conc=5, mass = 1e12):
        """ Stand-alone convenience function for returning a Monte Carlo realization of points in the phase space of an NFW halo in isotropic Jeans equilibrium.

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
        >>> data = nfw.mc_generate_phase_space_points(Ngals = 1e2, mass = mass, conc = conc) 

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
        vvir = NFWProfile.virial_velocity(self, total_mass = m)
        rvir = NFWProfile.halo_mass_to_halo_radius(self, total_mass = m)

        x, y, z = MonteCarloGalProf.mc_halo_centric_pos(self, 
            profile_params = [c], halo_radius = rvir)
        r = np.sqrt(x**2 + y**2 + z**2)

        vrad = MonteCarloGalProf.mc_radial_velocity(self, x = r/rvir, 
            virial_velocities = vvir, profile_params = [c])
        vx = MonteCarloGalProf.mc_radial_velocity(self, x = r/rvir, 
            virial_velocities = vvir, profile_params = [c])
        vy = MonteCarloGalProf.mc_radial_velocity(self, x = r/rvir, 
            virial_velocities = vvir, profile_params = [c])
        vz = MonteCarloGalProf.mc_radial_velocity(self, x = r/rvir, 
            virial_velocities = vvir, profile_params = [c])

        t = Table()
        t['x'] = x 
        t['y'] = y
        t['z'] = z 
        t['vx'] = vx 
        t['vy'] = vy
        t['vz'] = vz 

        t['radial_position'] = r 
        t['radial_velocity'] = vrad 
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
            then ``halo_table`` keyword argument must be passed. 

        halo_table : object, optional  
            Data table storing halo catalog. 
            If ``halo_table`` is not passed, 
            then ``prim_haloprop`` keyword argument must be passed. 

        Returns 
        -------
        c : array_like
            Concentrations of the input halos. 

        Notes 
        ------
        The behavior of this function is not defined here, but in the 
        `~halotools.empirical_models.phase_space_models.profile_models.ConcMass` class.

        This method is tested by `~halotools.empirical_models.phase_space_models.profile_models.tests.test_conc_mass.TestConcMass` class. 

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
        `~halotools.empirical_models.phase_space_models.profile_models.profile_helpers.density_threshold` function. 
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
            `~halotools.empirical_models.phase_space_models.profile_models.profile_helpers.density_threshold` 
            :math:`\\rho_{\\rm thresh}` for the 
            halo mass definition, cosmology, and redshift. 
            Result is an array of the dimension as the input ``scaled_radius``. 

        Notes 
        -----

        This method is tested by `~halotools.empirical_models.phase_space_models.profile_models.tests.test_nfw_profile.TestNFWProfile.test_mass_density` function. 

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

        This method is tested by `~halotools.empirical_models.phase_space_models.profile_models.tests.test_nfw_profile.TestNFWProfile.test_mass_density` function. 

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
        Analytical result for the fraction of the total mass enclosed within dimensionless radius of an NFW halo, 

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

        This method is tested by `~halotools.empirical_models.phase_space_models.profile_models.tests.test_nfw_profile.TestNFWProfile.test_cumulative_mass_PDF` function. 

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

        This method is tested by `~halotools.empirical_models.phase_space_models.profile_models.tests.test_nfw_profile.TestNFWProfile.test_cumulative_mass_PDF` function. 

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

        This method is tested by `~halotools.empirical_models.phase_space_models.profile_models.tests.test_nfw_profile.TestNFWProfile.test_vmax` function. 
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

        This method is tested by `~halotools.empirical_models.phase_space_models.profile_models.tests.test_nfw_profile.TestNFWProfile.test_vmax` function, 
        and also the `~halotools.empirical_models.phase_space_models.profile_models.tests.test_halo_catalog_nfw_consistency.TestHaloCatalogNFWConsistency.test_vmax_consistency` function. 

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
        `~halotools.empirical_models.phase_space_models.profile_models.tests.test_profile_helpers.TestProfileHelpers.test_halo_mass_to_halo_radius` function. 

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
        `~halotools.empirical_models.phase_space_models.profile_models.tests.test_profile_helpers.TestProfileHelpers.test_halo_radius_to_halo_mass` function. 

        """
        return NFWProfile.halo_radius_to_halo_mass(self, radius)









