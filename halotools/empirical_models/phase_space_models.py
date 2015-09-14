# -*- coding: utf-8 -*-
"""
Module composes the behavior of the profile models 
and the velocity models to produce models for the 
full phase space distribution of galaxies within their halos. 
"""

__author__ = ['Andrew Hearin']
__all__ = ['NFWPhaseSpace', 'TrivialPhaseSpace']

import numpy as np
from .profile_models import *
from .velocity_models import *
from .monte_carlo_phase_space import *
from . import model_defaults
from ..sim_manager import sim_defaults

class NFWPhaseSpace(NFWProfile, NFWJeansVelocity, MonteCarloGalProf):
    """ NFW halo profile, based on Navarro, Frenk and White (1999).

    """

    def __init__(self, **kwargs):
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
            Default is set in `~halotools.empirical_models.model_defaults`.  

        """        
        NFWProfile.__init__(self, **kwargs)
        NFWJeansVelocity.__init__(self, **kwargs)
        MonteCarloGalProf.__init__(self)

        if 'concentration_binning' in kwargs:
            cmin, cmax, dc = kwargs['concentration_binning']
        else:
            cmin, cmax, dc = (
                model_defaults.min_permitted_conc, 
                model_defaults.max_permitted_conc,
                model_defaults.default_dconc
                )
        self._setup_lookup_tables((cmin, cmax, dc))

        self._mock_generation_calling_sequence = ['assign_phase_space']

    def assign_phase_space(self, halo_table):
        """
        """
        self.mc_pos(halo_table = halo_table)
        self.mc_vel(halo_table = halo_table)


class TrivialPhaseSpace(object):
    """
    """
    def __init__(self, velocity_bias = False, 
        cosmology = sim_defaults.default_cosmology, 
        redshift = sim_defaults.default_redshift, 
        mdef = model_defaults.halo_mass_definition, 
        **kwargs):
        """
        Parameters 
        ----------
        velocity_bias : bool, optional 
            Boolean specifying whether the galaxy velocities are biased 
            with respect to the halo velocities. If True, ``param_dict`` will have a 
            parameter called ``velbias_centrals`` that multiplies the underlying 
            halo velocity by an overall factor. Default is False. 

        cosmology : object, optional 
            Astropy cosmology object. Default is set in `~halotools.empirical_models.sim_defaults`.

        redshift : float, optional  
            Default is set in `~halotools.empirical_models.sim_defaults`.

        mdef: str
            String specifying the halo mass definition, e.g., 'vir' or '200m'. 
            Default is set in `~halotools.empirical_models.model_defaults`.  
        """
        self._mock_generation_calling_sequence = ['assign_phase_space']
        self._galprop_dtypes_to_allocate = np.dtype([
            ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), 
            ('vx', 'f8'), ('vy', 'f8'), ('vz', 'f8'), 
            ])

        self.param_dict = {}
        if velocity_bias is True:
            self.param_dict['velbias_centrals'] = 1.

        self.cosmology = cosmology
        self.redshift = redshift 
        self.mdef = mdef 
        self.halo_boundary_key = model_defaults.get_halo_boundary_key(self.mdef)

    def assign_phase_space(self, halo_table):
        """
        """
        phase_space_keys = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        for key in phase_space_keys:
            halo_table[key] = halo_table['halo_'+key]
            if 'velbias_centrals' in self.param_dict:
                halo_table[key] *= self.param_dict['velbias_centrals']










        
