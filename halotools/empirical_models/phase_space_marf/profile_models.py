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

__all__ = ['AnalyticDensityProf', 'TrivialProfile', 'NFWProfile']

@six.add_metaclass(ABCMeta)
class AnalyticDensityProf(object):
    """ Container class for any analytical radial profile model. 

    Notes 
    -----
    The primary behavior of the `AnalyticDensityProf` class is governed by the 
    `dimensionless_mass_density`  method. The `AnalyticDensityProf` class has no 
    implementation of its own of `dimensionless_mass_density`, but does implement 
    all other behaviors that derive from `dimensionless_mass_density`. Thus for users 
    who wish to define their own profile class, defining the `dimensionless_mass_density` of 
    the profile is the necessary and sufficient ingredient. 
    """

    def __init__(self, cosmology, redshift, mdef, **kwargs):
        """
        Parameters 
        -----------
        cosmology : object 
            Instance of an `~astropy.cosmology` object. 

        redshift: array_like
            Can be a scalar or a numpy array.

        mdef: str
            String specifying the halo mass definition, e.g., 'vir' or '200m'. 

        """
        self.cosmology = cosmology
        self.redshift = redshift
        self.mdef = mdef

        self.halo_boundary_key = model_defaults.get_halo_boundary_key(self.mdef)
        self.halo_mass_key = model_defaults.get_halo_mass_key(self.mdef)
        self.prim_haloprop_key = self.halo_mass_key 

        self.density_threshold = density_threshold(
            cosmology = self.cosmology, 
            redshift = self.redshift, mdef = self.mdef)

        self.prof_param_keys = []
        self.publications = []
        self.param_dict = {}

    @abstractmethod
    def dimensionless_mass_density(self, x, *args):
        """
        Physical density of the halo scaled by the density threshold of the 
        mass definition:

        `dimensionless_mass_density` :math:`\\equiv \\rho(x) / \\rho_{\\rm thresh}`, 
        where :math:`x\\equiv r/R_{\\rm vir}`, and :math:`\\rho_{\\rm thresh}` is 
        a function of the halo mass definition, cosmology and redshift, 
        and is computed via the 
        `~halotools.empirical_models.phase_space_marf.profile_helpers.density_threshold` function. 

        Parameters 
        -----------
        x : array_like 
            Halo-centric distance scaled by the halo boundary, so that 
            :math:`0 <= x <= 1`. Can be a scalar or numpy array

        args : array_like, optional 
            Any additional array(s) necessary to specify the shape of the radial profile, 
            e.g., halo concentration. 

        Returns 
        -------
        dimensionless_density: array_like 
            Dimensionless density of a dark matter halo 
            at the input ``x``, scaled by the density threshold for this 
            halo mass definition, cosmology, and redshift. 
            Result is an array of the dimension as the input ``x``. 
            The physical `mass_density` is simply the `dimensionless_mass_density` 
            multiplied by the appropriate physical density threshold. 

        Notes 
        -----
        All of the behavior of a subclass of `AnalyticDensityProf` is determined by 
        `dimensionless_mass_density`. This is numerically convenient, because mass densities 
        in physical units are astronomically large numbers, whereas `dimensionless_mass_density` 
        is of order :math:`\\mathcal{O}(1-100)`. This also saves users writing their own subclass 
        from having to worry over factors of little h, how profile normalization scales 
        with the mass definition, etc. Once a model's `dimensionless_mass_density` is specified, 
        all the other functionality is derived from this definition. 

        """
        pass

    def mass_density(self, radius, mass, *args):
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

        args : array_like, optional 
            Any additional array(s) necessary to specify the shape of the radial profile, 
            e.g., halo concentration. 

        Returns 
        -------
        density: array_like 
            Physical density of a dark matter halo of the input ``mass`` 
            at the input ``radius``. Result is an array of the 
            dimension as the input ``radius``, reported in units of :math:`h^{3}/Mpc^{3}`. 
        """
        halo_radius = self.halo_mass_to_halo_radius(mass)
        x = radius/halo_radius

        dimensionless_mass = self.dimensionless_mass_density(x, *args)

        density = self.density_threshold*dimensionless_mass
        return density

    def _enclosed_dimensionless_mass_integrand(self, x, *args):
        """
        Integrand used when computing `cumulative_mass_PDF`. 
        Parameters 
        -----------
        x : array_like 
            Halo-centric distance scaled by the halo boundary, so that 
            :math:`0 <= x <= 1`. Can be a scalar or numpy array

        args : array_like, optional 
            Any additional array(s) necessary to specify the shape of the radial profile, 
            e.g., halo concentration. 

        Returns 
        -------
        integrand: array_like 
            function to be integrated to yield the amount of enclosed mass.
        """
        dimensionless_density = self.dimensionless_mass_density(x, *args)
        return dimensionless_density*4*np.pi*x**2

    def cumulative_mass_PDF(self, x, *args):
        """
        The fraction of the total mass enclosed within 
        dimensionless radius :math:`x = r / R_{\\rm halo}`.

        Parameters 
        -----------
        x : array_like 
            Halo-centric distance scaled by the halo boundary, so that 
            :math:`0 <= x <= 1`. Can be a scalar or numpy array

        args : array_like, optional 
            Any additional array(s) necessary to specify the shape of the radial profile, 
            e.g., halo concentration.         
            
        Returns
        -------------
        p: array_like
            The fraction of the total mass enclosed 
            within radius x, in :math:`M_{\odot}/h`; 
            has the same dimensions as the input ``x``.
        """
        x = convert_to_ndarray(x)
        x = x.astype(np.float64)
        enclosed_mass = np.zeros_like(x)

        for i in range(len(x)):
            enclosed_mass[i], _ = quad_integration(
                self._enclosed_dimensionless_mass_integrand, 0., x[i], epsrel = 1e-5, 
                args = args)
    
        total, _ = quad_integration(
                self._enclosed_dimensionless_mass_integrand, 0., 1.0, epsrel = 1e-5, 
                args = args)

        return enclosed_mass / total

    def enclosed_mass(self, radius, total_mass, *args):
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

        args : array_like, optional 
            Any additional array(s) necessary to specify the shape of the radial profile, 
            e.g., halo concentration.         
            
        Returns
        ----------
        enclosed_mass: array_like
            The mass enclosed within radius r, in :math:`M_{\odot}/h`; 
            has the same dimensions as the input ``radius``.
        """
        radius = convert_to_ndarray(radius)
        x = radius / self.halo_mass_to_halo_radius(total_mass)
        mass = self.cumulative_mass_PDF(x, *args)*total_mass

        return mass

    def dimensionless_circular_velocity(self, x, *args):
        """ Circular velocity scaled by the virial velocity, 
        :math:`V_{\\rm cir}(x) / V_{\\rm vir}`, as a function of 
        dimensionless position :math:`x = r / R_{\\rm vir}`.

        Parameters 
        -----------
        x : array_like 
            Halo-centric distance scaled by the halo boundary, so that 
            :math:`0 <= x <= 1`. Can be a scalar or numpy array

        args : array_like, optional 
            Any additional array(s) necessary to specify the shape of the radial profile, 
            e.g., halo concentration. 

        Returns 
        -------
        vcir : array_like 
            Circular velocity scaled by the virial velocity, 
            :math:`V_{\\rm cir}(x) / V_{\\rm vir}`.         

        """
        return np.sqrt(self.cumulative_mass_PDF(x, *args)/x)


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
        """
        halo_radius = self.halo_mass_to_halo_radius(total_mass)
        return np.sqrt(newtonG.value*total_mass/halo_radius)

    def circular_velocity(self, radius, total_mass, *args):
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

        args : array_like, optional 
            Any additional array(s) necessary to specify the shape of the radial profile, 
            e.g., halo concentration.         

        Returns
        ----------
        vc: array_like
            The circular velocity in km/s; has the same dimensions as the input ``radius``.

        """     
        halo_radius = self.halo_mass_to_halo_radius(total_mass)
        x = convert_to_ndarray(radius) / halo_radius
        return self.dimensionless_circular_velocity(x, *args)*self.virial_velocity(total_mass)

    def _vmax_helper(self, x, *args):
        """ Helper function used to calculate `vmax` and `rmax`. 
        """
        encl = self.cumulative_mass_PDF(x, *args)
        return -1.*encl/x

    def rmax(self, total_mass, *args):
        """ Radius at which the halo attains its maximum circular velocity.

        Parameters 
        ----------
        total_mass: array_like
            Total halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.

        args : array_like 
            Any additional array(s) necessary to specify the shape of the radial profile, 
            e.g., halo concentration.         

        Returns 
        --------
        rmax : array_like 
            :math:`R_{\\rm max}` in Mpc/h.
        """
        halo_radius = self.halo_mass_to_halo_radius(total_mass)

        guess = 0.25

        result = scipy_minimize(self._vmax_helper, guess, args=args)

        return result.x[0]*halo_radius

    def vmax(self, total_mass, *args):
        """ Maximum circular velocity of the halo profile. 

        Parameters 
        ----------
        total_mass: array_like
            Total halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.

        args : array_like 
            Any additional array(s) necessary to specify the shape of the radial profile, 
            e.g., halo concentration.         

        Returns 
        --------
        vmax : array_like 
            :math:`V_{\\rm max}` in km/s.
        """

        guess = 0.25
        result = scipy_minimize(self._vmax_helper, guess, args=args)
        halo_radius = self.halo_mass_to_halo_radius(total_mass)

        return self.circular_velocity(result.x[0]*halo_radius, total_mass, *args)

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
        """
        return halo_mass_to_halo_radius(total_mass, cosmology = self.cosmology, 
            redshift = self.redshift, mdef = self.mdef)

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

        """
        return halo_radius_to_halo_mass(radius, cosmology = self.cosmology, 
            redshift = self.redshift, mdef = self.mdef)


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
        `~halotools.empirical_models.phase_space_marf.profile_helpers.density_threshold` function. 

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

### The current implementation of the Jeans solutions will not be correct for the BiasedNFWProfile class
# class BiasedNFWProfile(NFWProfile):
#     """ NFW halo profile, based on Navarro, Frenk and White (1999), 
#     allowing galaxies to have distinct concentrations from their underlying 
#     dark matter halos.

#     """

#     def __init__(self, **kwargs):
#         """
#         Parameters 
#         ----------
#         cosmology : object, optional 
#             Astropy cosmology object. Default is set in `~halotools.empirical_models.sim_defaults`.

#         redshift : float, optional  
#             Default is set in `~halotools.empirical_models.sim_defaults`.

#         mdef: str, optional 
#             String specifying the halo mass definition, e.g., 'vir' or '200m'. 
#              Default is set in `~halotools.empirical_models.model_defaults`.

#         conc_mass_model : string, optional  
#             Specifies the calibrated fitting function used to model the concentration-mass relation. 
#              Default is set in `~halotools.empirical_models.model_defaults`.

#         """

#         super(BiasedNFWProfile, self).__init__(**kwargs)

#         self.param_dict['conc_NFWmodel_bias'] = 1.

    # def conc_NFWmodel(self, **kwargs):
    #     """
    #     """
    #     result = (self.param_dict['conc_NFWmodel_bias']*
    #         super(BiasedNFWProfile, self).conc_NFWmodel(**kwargs)
    #         )
    #     return result



##################################################################################











