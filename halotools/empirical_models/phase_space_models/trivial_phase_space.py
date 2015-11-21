# -*- coding: utf-8 -*-
"""
Module composes the behavior of the profile models 
and the velocity models to produce models for the 
full phase space distribution of galaxies within their halos. 
"""
from __future__ import (
    division, print_function, absolute_import)

__author__ = ['Andrew Hearin']
__all__ = ['TrivialPhaseSpace']

import numpy as np
from astropy.table import Table

from .profile_models import NFWProfile
from .velocity_models import NFWJeansVelocity
from .monte_carlo_helpers import MonteCarloGalProf

from .. import model_defaults

from ...sim_manager import sim_defaults

class TrivialPhaseSpace(object):
    """ Profile of central galaxies residing at the exact center of their 
    host halo with the exact same velocity as the halo velocity. 
    """
    def __init__(self, 
        cosmology = sim_defaults.default_cosmology, 
        redshift = sim_defaults.default_redshift, 
        mdef = model_defaults.halo_mass_definition, 
        **kwargs):
        """
        Parameters 
        ----------
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

        self.cosmology = cosmology
        self.redshift = redshift 
        self.mdef = mdef 
        self.halo_boundary_key = model_defaults.get_halo_boundary_key(self.mdef)

    def assign_phase_space(self, table):
        """
        """
        phase_space_keys = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        for key in phase_space_keys:
            table[key] = table['halo_'+key]










        
