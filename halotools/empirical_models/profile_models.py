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


    @abstractmethod
    def mass_density(self, r, *args, **kwargs):
        """
        """
        pass

    def _enclosed_mass_integrand(self, y):
        """
        """
        return self.mass_density(z) * 4.0 * np.pi * z**2

    def enclosed_mass(self, x, rtol = 1E-5):
        """
        The mass enclosed within dimensionless radius :math:`x = r / R_{\\rm halo}`.

        Parameters
        -----------------
        x: array_like
            Halo-centric distance scaled by the halo boundary, such that :math:`0 < x < 1`. 
            Can be a scalar or a numpy array.

        rtol: float, optional
            Relative tolerance of the integration accuracy. Default is 1e-5. 
            
        Returns
        ----------
        mass: array_like
            The mass enclosed within radius r, in :math:`M_{\odot}/h`; 
            has the same dimensions as the input ``x``.
        """                
        x = convert_to_ndarray(x)
        mass = np.zeros_like(x)
        for i in range(len(x)):
            mass[i], _ = quad_integration(self._enclosed_mass_integrand, 0., x[i], epsrel = rtol)
    
        return mass

    def cumulative_mass_PDF(self, x, rtol = 1E-5):
        """
        The fraction of the total mass enclosed within 
        dimensionless radius :math:`x = r / R_{\\rm halo}`.

        Parameters
        -------------
        x: array_like
            Halo-centric distance scaled by the halo boundary, such that :math:`0 < x < 1`. 
            Can be a scalar or a numpy array.

        rtol: float, optional
            Relative tolerance of the integration accuracy. Default is 1e-5. 
            
        Returns
        -------------
        p: array_like
            The fraction of the total mass enclosed 
            within radius x, in :math:`M_{\odot}/h`; 
            has the same dimensions as the input ``x``.
        """     
        p = self.enclosed_mass(x)/self.enclosed_mass(1.0)
        return p

    def circular_velocity(self, x):
        """
        The circular velocity, :math:`v_c \\equiv \\sqrt{GM(<r)/r}`.

        Parameters
        --------------
        x: array_like
            Halo-centric distance scaled by the halo boundary, such that :math:`0 < x < 1`. 
            Can be a scalar or a numpy array.
            
        Returns
        ----------
        vc: float
            The circular velocity in km / s; has the same dimensions as r.

        See also
        ------------
        Vmax: The maximum circular velocity, and the radius where it occurs.
        """     
    
        M = self.enclosed_mass(x)
        v = numpy.sqrt(newtonG.value * M / x)
        
        return v

    def gravitational_potential_radial_gradient(self, x):
        """
        """
        return newtonG.value * self.enclosed_mass(x) / x**2


    ###############################################################################################

    def vmax(self):
        """
        The maximum circular velocity, and the radius where it occurs.
            
        Returns
        -----------------
        vmax: float
            The maximum circular velocity in km / s.
        rmax: float
            The radius where fmax occurs, in physical kpc/h.

        """     
        def _circular_velocity_negative(x):
            return -self.circular_velocity(x)

        x_guess = 0.1
        res = scipy_minimize(_circular_velocity_negative, x_guess)
        rmax = res.x[0]
        vmax = self.circular_velocity(rmax)
        
        return vmax, rmax

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

    def mass_density(self, r, mass, conc):
        """
        """
        halo_radius = halo_mass_to_halo_radius(mass=mass, 
            cosmology=self.cosmology, redshift=self.redshift, mdef=self.mdef)
        x = r/halo_radius
        physical_density = (self.density_threshold*
            self.dimensionless_mass_density(x, conc)
            )
        return physical_density

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
        x = convert_to_ndarray(x)
        x = np.where(x > 1, 1, x)

        conc = convert_to_ndarray(conc)

        if len(x) != len(conc):
            raise HalotoolsError("If passing an array of concentrations to "
                "cumulative_mass_PDF, the array must have the same length "
                "as the input array of radial positions")
        else:
            return self.g(conc) / self.g(x*conc)

    def enclosed_mass(self, radius, mass, conc):
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
        halo_boundary = halo_mass_to_halo_radius(mass, 
            self.cosmology, self.redshift, self.mdef)
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









