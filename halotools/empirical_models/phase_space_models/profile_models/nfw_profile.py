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
from .profile_model_template import AnalyticDensityProf

from ... import model_defaults

from ....utils.array_utils import convert_to_ndarray
from ....custom_exceptions import HalotoolsError
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
        You can load an NFW profile model with the default settings simply by calling 
        the class constructor with no arguments:

        >>> nfw_halo_prof_model = NFWProfile() 

        The density threshold used to define the halo boundary depends on cosmology, 
        redshift, and the chosen halo mass definition. If you are using 
        the `NFWProfile` class to build a mock, you should choose values for these 
        inputs that are consistent with the halo catalog you are populating. 
        For an NFW profile with an alternative cosmology and redshift:

        >>> from astropy.cosmology import WMAP9
        >>> nfw_halo_prof_model = NFWProfile(cosmology = WMAP9, redshift = 2) 

        For a profile based on an alternative mass definition:

        >>> nfw_halo_prof_model = NFWProfile(mdef = '2500c') 

        """

        AnalyticDensityProf.__init__(self, cosmology, redshift, mdef)
        ConcMass.__init__(self, **kwargs)

        self.prof_param_keys = ['conc_NFWmodel']

        self.publications = ['arXiv:9611107', 'arXiv:0002395']

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
        `~halotools.empirical_models.phase_space_models.profile_models.ConcMass` class.
        """
        return ConcMass.compute_concentration(self, **kwargs)

    def dimensionless_mass_density(self, scaled_radius, conc):
        """
        Physical density of the halo scaled by the density threshold of the mass definition:

        The `dimensionless_mass_density` is defined as 
        :math:`\\equiv \\rho_{\\rm prof}(\\tilde{r}) / \\rho_{\\rm thresh}`, 
        where :math:`\\tilde{r}\\equiv r/R_{\\Delta}`. 
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

    def enclosed_mass(self, radius, total_mass, conc):
        """
        The mass enclosed within the input radius. 

        :math:`M(<r) = 4\\pi\\int_{0}^{r}dr'r'^{2}\\rho(r)`. 

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
        """
        return AnalyticDensityProf.enclosed_mass(self, radius, total_mass, conc)

    def virial_velocity(self, total_mass):
        """ The circular velocity evaluated at the halo boundary, 
        :math:`V_{\\rm vir} \\equiv \\sqrt{GM_{\\rm halo}/R_{\\rm halo}}`.

        Parameters
        --------------
        total_mass : array_like 
            Total mass of the halo; can be a scalar or numpy array of the same 
            dimension as the input ``radius``. 

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
        return AnalyticDensityProf.virial_velocity(self, total_mass)

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

        """    
        return AnalyticDensityProf.circular_velocity(self, radius, total_mass, conc)

    def vmax(self, total_mass, conc):
        """ Maximum circular velocity of the halo profile. 

        Parameters 
        ----------
        total_mass: array_like
            Total halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.

        conc : array_like 
            Value of the halo concentration. Can either be a scalar, or a numpy array 
            of the same dimension as the input ``radius``. 

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

        """
        halo_radius = self.halo_mass_to_halo_radius(total_mass)
        scale_radius = halo_radius/conc

        rmax = 2.16258 * scale_radius
        vmax = self.circular_velocity(rmax, total_mass, conc)
        return vmax

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

        """
        return AnalyticDensityProf.halo_mass_to_halo_radius(self, total_mass)

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

        """
        return AnalyticDensityProf.halo_radius_to_halo_mass(self, radius)



