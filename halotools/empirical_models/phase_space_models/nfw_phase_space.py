# -*- coding: utf-8 -*-
"""
Module composes the behavior of the profile models 
and the velocity models to produce models for the 
full phase space distribution of galaxies within their halos. 
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
    """ NFW halo profile, based on Navarro, Frenk and White (1999).

    """

    def __init__(self, high_precision = False, **kwargs):
        """
        Parameters 
        ----------
        conc_mass_model : string, optional  
            Specifies the calibrated fitting function used to model the concentration-mass relation. 
             Default is set in `~halotools.empirical_models.sim_defaults`.

        cosmology : object, optional 
            Astropy cosmology object. Default is set in `~halotools.empirical_models.sim_defaults`.

        redshift : float, optional  
            Default is set in `~halotools.empirical_models.sim_defaults`.

        mdef: str
            String specifying the halo mass definition, e.g., 'vir' or '200m'. 
            Default is set in `~halotools.empirical_models.model_defaults`.  

        velocity_bias : bool, optional 
            Boolean specifying whether the galaxy velocities are biased 
            with respect to the halo velocities. If True, ``param_dict`` will have a 
            parameter called ``velbias_satellites`` that multiplies the underlying 
            Jeans solution for the halo radial velocity dispersion by an overall factor. 
            Default is False. 

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

        MonteCarloGalProf._setup_lookup_tables(self, (cmin, cmax, dc))

        self._mock_generation_calling_sequence = ['assign_phase_space']

    def assign_phase_space(self, halo_table):
        """
        """
        MonteCarloGalProf.mc_pos(self, halo_table = halo_table)
        MonteCarloGalProf.mc_vel(self, halo_table = halo_table)


    def mc_generate_phase_space_points(self, Ngals = 1e4, conc=5, mass = 1e12):
        """ Stand-alone convenience function for returning a Monte Carlo 
        realization of NFW phase space.

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
        >>> data = nfw.mc_generate_phase_space_points(Ngals = 1e2) # doctest: +SKIP
        >>> data = nfw.mc_generate_phase_space_points(Ngals = 1e3, mass = 1e15) # doctest: +SKIP

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

        """
        return NFWProfile.dimensionless_mass_density(self, scaled_radius, conc)

    def mc_vel(self, halo_table):
        """ Method assigns a Monte Carlo realization of the Jeans velocity 
        solution to the halos in the input ``halo_table``. 

        Parameters 
        -----------
        halo_table : Astropy Table 
            `astropy.table.Table` object storing the halo catalog. 
            Calling the `mc_vel` method will over-write the existing values of 
            the ``vx``, ``vy`` and ``vz`` columns. 
        """
        return MonteCarloGalProf.mc_vel(self, halo_table)










