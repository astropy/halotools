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

        self._setup_lookup_tables((cmin, cmax, dc))

        self._mock_generation_calling_sequence = ['assign_phase_space']

    def assign_phase_space(self, halo_table):
        """
        """
        self.mc_pos(halo_table = halo_table)
        self.mc_vel(halo_table = halo_table)


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
        vvir = self.virial_velocity(total_mass = m)
        rvir = self.halo_mass_to_halo_radius(total_mass = m)

        x, y, z = self.mc_halo_centric_pos(
            profile_params = [c], halo_radius = rvir)
        r = np.sqrt(x**2 + y**2 + z**2)

        vrad = self.mc_radial_velocity(x = r/rvir, 
            virial_velocities = vvir, profile_params = [c])
        vx = self.mc_radial_velocity(x = r/rvir, 
            virial_velocities = vvir, profile_params = [c])
        vy = self.mc_radial_velocity(x = r/rvir, 
            virial_velocities = vvir, profile_params = [c])
        vz = self.mc_radial_velocity(x = r/rvir, 
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


