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

__all__ = ['NFWProfile']

class NFWProfile(AnalyticDensityProf, ConcMass):
    """ NFW halo profile, based on Navarro, Frenk and White (1999).

    """

    def __init__(self, 
        cosmology=sim_defaults.default_cosmology, 
        redshift=sim_defaults.default_redshift,
        mdef = model_defaults.halo_mass_definition,
        **kwargs):
        """
        Parameters 
        ----------
        cosmology : object, optional 
            Astropy cosmology object. Default is set in `~halotools.empirical_models.sim_defaults`.

        redshift : float, optional  
            Default is set in `~halotools.empirical_models.sim_defaults`.

        mdef: str, optional 
            String specifying the halo mass definition, e.g., 'vir' or '200m'. 
            Default is set in `~halotools.empirical_models.model_defaults`.

        conc_mass_model : string, optional  
            Specifies the calibrated fitting function used to model the concentration-mass relation. 
            Default is set in `~halotools.empirical_models.model_defaults`.

        Examples 
        --------
        You can load a NFW profile model with the default settings simply by calling 
        the class constructor with no arguments:

        >>> nfw_halo_prof_model = NFWProfile() # doctest: +SKIP 

        For an NFW profile with an alternative cosmology and redshift:

        >>> from astropy.cosmology import WMAP9
        >>> nfw_halo_prof_model = NFWProfile(cosmology = WMAP9, redshift = 2) # doctest: +SKIP 
        """

        super(NFWProfile, self).__init__(cosmology, redshift, mdef)
        ConcMass.__init__(self, **kwargs)

        self.prof_param_keys = ['conc_NFWmodel']

        self.publications = ['arXiv:9611107', 'arXiv:0002395', 'arXiv:1402.7073']

        # Grab the docstring of the ConcMass function and bind it to the conc_NFWmodel function
        self.conc_NFWmodel.__func__.__doc__ = getattr(self, self.conc_mass_model).__doc__

    def conc_NFWmodel(self, **kwargs):
        """ Method computes the NFW concentration as a function of the input halos according to the 
        ``conc_mass_model`` bound to the `NFWProfile` instance. 

        Parameters
        ----------        
        prim_haloprop : array, optional  
            Array of mass-like variable upon which occupation statistics are based. 
            If ``prim_haloprop`` is not passed, then ``halo_table`` keyword argument must be passed. 

        halo_table : object, optional  
            Data table storing halo catalog. 
            If ``halo_table`` is not passed, then ``prim_haloprop`` keyword argument must be passed. 

        Returns 
        -------
        c : array_like
            Concentrations of the input halos. 

        Notes 
        ------
        The behavior of this function is not defined here, but in the 
        `~halotools.empirical_models.ConcMass` class.
        """
        return self.compute_concentration(**kwargs)

    def dimensionless_mass_density(self, x, conc):
        """
        Physical density of the halo scaled by the density threshold of the 
        mass definition:

        `dimensionless_mass_density` :math:`\\equiv \\rho(x) / \\rho_{\\rm thresh}`, 
        where :math:`x\\equiv r/R_{\\rm vir}`, and :math:`\\rho_{\\rm thresh}` is 
        a function of the halo mass definition, cosmology and redshift, 
        and is computed via the 
        `~halotools.empirical_models.phase_space_models.profile_helpers.density_threshold` function. 

        Parameters 
        -----------
        x : array_like 
            Halo-centric distance scaled by the halo boundary, so that 
            :math:`0 <= x <= 1`. Can be a scalar or numpy array

        conc : array_like
            Concentrations of the input halos. 

        Returns 
        -------
        dimensionless_density: array_like 
            Dimensionless density of a dark matter halo 
            at the input ``x``, scaled by the density threshold for this 
            halo mass definition, cosmology, and redshift. 
            Result is an array of the dimension as the input ``x``. 
            The physical `mass_density` is simply the `dimensionless_mass_density` 
            multiplied by the appropriate physical density threshold. 

        """
        numerator = conc**3/(3.*self.g(conc))
        denominator = conc*x*(1 + conc*x)**2
        return numerator/denominator

    def g(self, x):
        """ Convenience function used to evaluate the profile. 

        Parameters 
        ----------
        x : array_like 

        Returns 
        -------
        g : array_like 
            :math:`g(x) \\equiv \\int_{0}^{x}dy\\frac{y}{(1+y)^{2}} = \\log(1+x) - x / (1+x)`

        Examples 
        --------
        >>> model = NFWProfile() # doctest: +SKIP 
        >>> g = model.g(1) # doctest: +SKIP 
        >>> Npts = 25 # doctest: +SKIP 
        >>> g = model.g(np.logspace(-1, 1, Npts)) # doctest: +SKIP 
        """
        return np.log(1.0+x) - (x/(1.0+x))

    def cumulative_mass_PDF(self, x, conc):
        """
        The fraction of the total mass enclosed within 
        dimensionless radius :math:`x = r / R_{\\rm halo}`.

        Parameters
        -------------
        x: array_like
            Halo-centric distance scaled by the halo boundary, such that :math:`0 < x < 1`. 
            Can be a scalar or a numpy array.

        conc : array_like 
            Value of the halo concentration. Can either be a scalar, or a numpy array 
            of the same dimension as the input ``x``. 
            
        Returns
        -------------
        p: array_like
            The fraction of the total mass enclosed 
            within radius x, in :math:`M_{\odot}/h`; 
            has the same dimensions as the input ``x``.
        """     
        x = np.where(x > 1, 1, x)
        return self.g(conc*x) / self.g(conc)

