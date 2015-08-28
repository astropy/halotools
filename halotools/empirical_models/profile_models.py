# -*- coding: utf-8 -*-
"""
This module contains the components for 
the radial profiles of galaxies 
inside their halos. 
"""

import numpy as np 
from astropy.extern import six 
from abc import ABCMeta, abstractmethod
from ..sim_manager import sim_defaults 
from . import model_defaults
from .conc_mass_models import ConcMass
from profile_helpers import *
from ..utils.array_utils import convert_to_ndarray
from ..custom_exceptions import *
from scipy.integrate import quad as quad_integration
from scipy.optimize import minimize as scipy_minimize

from astropy import units as u
from astropy.constants import G
newtonG = G.to(u.km*u.km*u.Mpc/(u.Msun*u.s*u.s))

__author__ = ['Andrew Hearin']

__all__ = ['TrivialProfile', 'NFWProfile', 'BiasedNFWProfile']

@six.add_metaclass(ABCMeta)
class AnalyticDensityProf(object):
    """ Container class for any radial profile model. 
    """

    def __init__(self, cosmology, redshift, mdef, **kwargs):
        """
        """
        self.cosmology = cosmology
        self.redshift = redshift
        self.mdef = mdef

        self.halo_boundary = model_defaults.halo_boundary_key(self.mdef)

        self.density_threshold = density_threshold(
            cosmology = self.cosmology, 
            redshift = self.redshift, mdef = self.mdef)

        self.prof_param_keys = []
        self.publications = []
        self.param_dict = {}

    def dimensionless_mass_density(self, x, *args):
        """
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
            Density of a dark matter halo of the input ``mass`` 
            at the input ``radius``, scaled by the density threshold for this 
            halo mass definition, cosmology, and redshift. 
            Result is an array of the dimension as the input ``x``, which equals the 
            physical `mass_density` when multiplied by the appropriate `density_threshold`. 

        """
        pass

    def mass_density(self, radius, mass, *args):
        """
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

    def circular_velocity(self, radius, total_mass, *args):
        """
        The circular velocity, :math:`v_c \\equiv \\sqrt{GM(<r)/r}`.

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
            The circular velocity in km / s; has the same dimensions as r.

        See also
        ------------
        Vmax: The maximum circular velocity, and the radius where it occurs.
        """     
        halo_radius = self.halo_mass_to_halo_radius(total_mass)
        x = convert_to_ndarray(radius) / halo_radius
        return self.dimensionless_circular_velocity(x, *args)*self.virial_velocity(total_mass)
        
    def dimensionless_circular_velocity(self, x, *args):
        """
        """
        return np.sqrt(self.cumulative_mass_PDF(x, *args)/x)

    def virial_velocity(self, total_mass):
        """
        """
        halo_radius = self.halo_mass_to_halo_radius(total_mass)
        return np.sqrt(newtonG.value*total_mass/halo_radius)

    def _vmax_helper(self, x, *args):
        """ Helper function used to calculate `vmax` and `rmax`. 
        """
        encl = self.cumulative_mass_PDF(x, *args)
        return -1.*encl/x

    def rmax(self, total_mass, *args):
        """ Radius at which the halo attains its maximum circular velocity.
        """
        halo_radius = self.halo_mass_to_halo_radius(total_mass)

        guess = 0.25

        result = scipy_minimize(self._vmax_helper, guess, args=args)

        return result.x[0]*halo_radius

    def vmax(self, total_mass, *args):
        """ Maximum circular velocity of the halo profile. 
        """
        halo_radius = self.halo_mass_to_halo_radius(total_mass)

        guess = 0.25

        result = scipy_minimize(self._vmax_helper, guess, args=args)

        return self.circular_velocity(result.x[0]*halo_radius, total_mass, *args)


    def gravitational_potential_radial_gradient(self, radius, total_mass, *args):
        """
        """
        pass

    def halo_mass_to_halo_radius(self, mass):
        return halo_mass_to_halo_radius(mass, cosmology = self.cosmology, 
            redshift = self.redshift, mdef = self.mdef)

    def halo_radius_to_halo_mass(self, radius):
        return halo_radius_to_halo_mass(radius, cosmology = self.cosmology, 
            redshift = self.redshift, mdef = self.mdef)


class TrivialProfile(AnalyticDensityProf):
    """ Profile of dark matter halos with all their mass concentrated at exactly the halo center. 

    """
    def __init__(self, 
        cosmology=sim_defaults.default_cosmology, 
        redshift=sim_defaults.default_redshift,
        mdef = model_defaults.halo_mass_definition,
        halo_boundary=model_defaults.halo_boundary, 
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

        super(TrivialProfile, self).__init__(cosmology, redshift, mdef, **kwargs)

    def mass_density(self, radius, mass):
        """
        Parameters 
        -----------
        radius: array_like
            Halo radius in physical Mpc/h; can be a scalar or a numpy array.

        mass: array_like
            Total halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.

        """
        volume = (4*np.pi/3)*radius**3
        return mass/volume

    def enclosed_mass(self, radius, mass):
        return mass

class NFWProfile(AnalyticDensityProf, ConcMass):
    """ NFW halo profile, based on Navarro, Frenk and White (1999).

    """

    def __init__(self, 
        cosmology=sim_defaults.default_cosmology, 
        redshift=sim_defaults.default_redshift,
        mdef = model_defaults.halo_mass_definition,
        halo_boundary=model_defaults.halo_boundary, 
        **kwargs):
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

        halo_boundary : string, optional  
            String giving the column name of the halo catalog that stores the boundary of the halo. 
            Default is set in the `~halotools.empirical_models.model_defaults` module. 

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

    def conc_NFWmodel(self, **kwargs):
        """
        """
        return self.compute_concentration(**kwargs)

    def dimensionless_mass_density(self, x, conc):
        """
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
        denominator = np.log(1.0+x) - (x/(1.0+x))
        return 1./denominator

    def cumulative_mass_PDF_override(self, x, conc):
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
        x = convert_to_ndarray(x)
        x = np.where(x > 1, 1, x)

        conc = convert_to_ndarray(conc)

        if len(x) != len(conc):
            raise HalotoolsError("If passing an array of concentrations to "
                "cumulative_mass_PDF, the array must have the same length "
                "as the input array of radial positions")
        else:
            return self.g(conc) / self.g(x*conc)

    def enclosed_mass_override(self, radius, mass, conc):
        """
        The mass enclosed within dimensionless radius :math:`x = r / R_{\\rm halo}`.

        Parameters
        -----------------
        radius: array_like
            Halo radius in physical Mpc/h; can be a scalar or a numpy array.

        mass: array_like
            Total halo mass. Can either be a scalar, or a numpy array with
            the same dimensions as the input ``radius``.

        conc : array_like 
            Value of the halo concentration. Can either be a scalar, or a numpy array 
            of the same dimension as the input ``radius``. 
            
        Returns
        ----------
        mass_encl: array_like
            The mass enclosed within the input ``radius``, in :math:`M_{\odot}/h`; 
            has the same dimensions as the input ``radius``.
        """   
        radius = convert_to_ndarray(radius)  
        mass = convert_to_ndarray(mass)  
        halo_boundary = self.halo_mass_to_halo_radius(mass)
        x = radius/halo_boundary
        conc = convert_to_ndarray(conc)  
        return mass*self.cumulative_mass_PDF(x, conc)



class BiasedNFWProfile(NFWProfile):
    """ NFW halo profile, based on Navarro, Frenk and White (1999), 
    allowing galaxies to have distinct concentrations from their underlying 
    dark matter halos.

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

        halo_boundary : string, optional  
            String giving the column name of the halo catalog that stores the boundary of the halo. 
            Default is set in the `~halotools.empirical_models.model_defaults` module. 

        """

        super(BiasedNFWProfile, self).__init__(**kwargs)

        self.param_dict['conc_NFWmodel_bias'] = 1.

    def conc_NFWmodel(self, **kwargs):
        """
        """
        result = (self.param_dict['conc_NFWmodel_bias']*
            super(BiasedNFWProfile, self).conc_NFWmodel(**kwargs)
            )
        return result



##################################################################################

@six.add_metaclass(ABCMeta)
class IsotropicJeansVelocity(object):
    """ Orthogonal mixin class used to transform a configuration 
    space model for the 1-halo term into a phase space model in which 
    velocities solve the Jeans equation of the underlying potential. 
    """
    def __init__(self, **kwargs):
        pass

    def _unscaled_radial_velocity_dispersion(self, x):
        """
        Method returns the radial velocity dispersion as a function of the 
        halo-centric distance. 
        """
        pass

class NFWPhaseSpace(NFWProfile, IsotropicJeansVelocity):
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

        halo_boundary : string, optional  
            String giving the column name of the halo catalog that stores the boundary of the halo. 
            Default is set in the `~halotools.empirical_models.model_defaults` module. 

        """

        super(NFWPhaseSpace, self).__init__(**kwargs)









