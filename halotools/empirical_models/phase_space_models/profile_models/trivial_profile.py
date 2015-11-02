# -*- coding: utf-8 -*-
"""
This module contains the components for 
the radial profiles of galaxies 
inside their halos. 
"""
from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np 
from astropy.extern import six 
from abc import ABCMeta, abstractmethod
from scipy.integrate import quad as quad_integration
from scipy.optimize import minimize as scipy_minimize
from astropy import units as u
from astropy.constants import G
newtonG = G.to(u.km*u.km*u.Mpc/(u.Msun*u.s*u.s))

from .. import model_defaults
from .conc_mass_models import ConcMass
from .profile_helpers import *

from ...utils.array_utils import convert_to_ndarray
from ...custom_exceptions import *
from ...sim_manager import sim_defaults 


__author__ = ['Andrew Hearin', 'Benedikt Diemer']

__all__ = ['TrivialProfile', 'NFWProfile']

class TrivialProfile(AnalyticDensityProf):
    """ Profile of dark matter halos with all their mass concentrated at exactly the halo center. 

    """
    def __init__(self, 
        cosmology=sim_defaults.default_cosmology, 
        redshift=sim_defaults.default_redshift,
        mdef = model_defaults.halo_mass_definition,
        **kwargs):
        """
        Notes 
        -----
        Testing done by `~halotools.empirical_models.test_empirical_models.test_TrivialProfile`

        Examples 
        --------
        You can load a trivial profile model with the default settings simply by calling 
        the class constructor with no arguments:

        >>> trivial_halo_prof_model = TrivialProfile() 

        """

        super(TrivialProfile, self).__init__(cosmology, redshift, mdef)


    def dimensionless_mass_density(self, x, total_mass):
        """
        Parameters 
        -----------
        x: array_like
            Halo-centric distance scaled by the halo boundary, such that :math:`0 < x < 1`. 
            Can be a scalar or a numpy array.

        total_mass: array_like
            Total halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.

        """
        volume = (4*np.pi/3)*x**3
        return total_mass/volume

    def enclosed_mass(self, radius, total_mass):
        return total_mass

