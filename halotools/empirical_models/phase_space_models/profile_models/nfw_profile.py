# -*- coding: utf-8 -*-
"""
This module contains the `NFWProfile` class, 
a sub-class of `~halotools.empirical_models.phase_space_models.AnalyticDensityProf`. 
The `NFWProfile` class is used to model the distribution of mass and/or galaxies 
inside dark matter halos according to the fitting function introduced in 
Navarry, Frenk and White (1999). 
"""
from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np 
from astropy import units as u

from .conc_mass_models import ConcMass
from .profile_helpers import *
from .profile_model_template import AnalyticDensityProf

from ... import model_defaults

from ....utils.array_utils import convert_to_ndarray
from ....custom_exceptions import *
from ....sim_manager import sim_defaults 


__author__ = ['Andrew Hearin', 'Benedikt Diemer']

__all__ = ['NFWProfile']

class NFWProfile(AnalyticDensityProf, ConcMass):
    """ NFW halo profile, based on Navarro, Frenk and White (1999).

    For a review of the mathematics underlying the NFW profile, 
    including descriptions of how the relevant equations are 
    implemented in the Halotools code base, see 

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
            Instance of an astropy `~astropy.cosmology`. 
            Default cosmology is set in 
            `~halotools.empirical_models.sim_manager.sim_defaults`.

        redshift : float, optional  
            Default is set in `~halotools.empirical_models.sim_defaults`.

        mdef: str, optional 
            String specifying the halo mass definition, e.g., 'vir' or '200m'. 
            Default is set in `~halotools.empirical_models.model_defaults`.

        conc_mass_model : string, optional  
            Specifies the calibrated fitting function used 
            to model the concentration-mass relation. 
            Default is set in `~halotools.empirical_models.model_defaults`.

        Examples 
        --------
        You can load a NFW profile model with the default settings simply by calling 
        the class constructor with no arguments:

        >>> nfw_halo_prof_model = NFWProfile() 

        For an NFW profile with an alternative cosmology and redshift:

        >>> from astropy.cosmology import WMAP9
        >>> nfw_halo_prof_model = NFWProfile(cosmology = WMAP9, redshift = 2) 

        For a profile based on an alternative mass definition:

        >>> nfw_halo_prof_model = NFWProfile(mdef = '2500c') 

        """

        super(NFWProfile, self).__init__(cosmology, redshift, mdef)
        ConcMass.__init__(self, **kwargs)

        self.prof_param_keys = ['conc_NFWmodel']

        self.publications = ['arXiv:9611107', 'arXiv:0002395', 'arXiv:1402.7073']

        # Grab the docstring of the ConcMass function and bind it to the conc_NFWmodel function
        self.conc_NFWmodel.__func__.__doc__ = getattr(self, self.conc_mass_model).__doc__

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
        `~halotools.empirical_models.ConcMass` class.
        """
        return ConcMass.compute_concentration(self, **kwargs)

    def dimensionless_mass_density(self, scaled_radius, conc):
        """
        Physical density of the halo scaled by the density threshold of the 
        mass definition:

        `dimensionless_mass_density` :math:`\\equiv \\rho(\\tilde{r}) / \\rho_{\\rm thresh}`, 
        where :math:`\\tilde{r}\\equiv r/R_{\\rm vir}`. 
        The quantity:math:`\\rho_{\\rm thresh}` is a function of 
        the halo mass definition, cosmology and redshift, 
        and is computed via the 
        `~halotools.empirical_models.phase_space_models.profile_helpers.density_threshold` function. 

        Parameters 
        -----------
        scaled_radius : array_like 
            Halo-centric distance *r* scaled by the halo boundary :math:`R_{\\Delta}`, so that 
            :math:`0 <= \\tilde{r} \\equiv r/R_{\\Delta} <= 1`. Can be a scalar or numpy array. 

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
        denominator = conc*scaled_radius*(1 + conc*scaled_radius)**2
        return numerator/denominator

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
            Concentrations of the input halos. 

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

        """
        return AnalyticDensityProf.mass_density(self, radius, mass, conc)

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
        >>> model = NFWProfile() 
        >>> result = model.g(1) 
        >>> Npts = 25 
        >>> result = model.g(np.logspace(-1, 1, Npts)) 
        """
        x = convert_to_ndarray(x, dt = np.float64)
        return np.log(1.0+x) - (x/(1.0+x))

    def cumulative_mass_PDF(self, scaled_radius, conc):
        """
        The fraction of the total mass enclosed within 
        dimensionless radius :math:`\\tilde{r} \\equiv r / R_{\\rm halo}`.

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

        """     
        scaled_radius = np.where(scaled_radius > 1, 1, scaled_radius)
        scaled_radius = np.where(scaled_radius < 0, 0, scaled_radius)
        return self.g(conc*scaled_radius) / self.g(conc)

