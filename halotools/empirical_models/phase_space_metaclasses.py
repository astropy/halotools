# -*- coding: utf-8 -*-
"""
This module contains the components for 
the intra-halo spatial positions of galaxies 
within their halos. 
"""
__author__ = ['Andrew Hearin']

__all__ = ['AnalyticDensityProf', 'IsotropicJeansVelocity']

import numpy as np
from abc import ABCMeta, abstractmethod
from functools import partial
from itertools import product

from ..utils import convert_to_ndarray

from . import model_defaults

from scipy.integrate import quad as quad_integration
from scipy.optimize import minimize as scipy_minimize
from astropy.extern import six

from astropy import units as u
from astropy.constants import G
newtonG = G.to(u.Mpc*u.km*u.km/(u.Msun*u.s*u.s)) 

@six.add_metaclass(ABCMeta)
class AnalyticDensityProf(object):
    """ Container class for any radial profile model. 
    """

    def __init__(self):
        """
        """
        pass


    @abstractmethod
    def density(self, x):
        """
        """
        pass

    def enclosed_mass(self, x, rtol = 1E-5):
        """
        The mass enclosed within dimensionless radius :math:`x = r / R_{\\rm halo}`.

        Parameters
        -------------------------------------------------------------------------------------------
        x: array_like
            Halo-centric distance scaled by the halo boundary, such that :math:`0 < x < 1`. 
            Can be a scalar or a numpy array.

        rtol: float, optional
            Relative tolerance of the integration accuracy. Default is 1e-5. 
            
        Returns
        -------------------------------------------------------------------------------------------
        M: array_like
            The mass enclosed within radius r, in :math:`M_{\odot}/h`; 
            has the same dimensions as the input ``x``.
        """     

        def integrand(z):
            return self.density(z) * 4.0 * np.pi * z**2

        x = convert_to_ndarray(x)
        M = np.zeros_like(x)
        for i in range(len(x)):
            M[i], _ = quad_integration(integrand, 0., x[i], epsrel = rtol)
    
        return M

    def enclosed_mass_cumulative_pdf(self, x, rtol = 1E-5):
        """
        The fraction of the total mass enclosed within 
        dimensionless radius :math:`x = r / R_{\\rm halo}`.

        Parameters
        -------------------------------------------------------------------------------------------
        x: array_like
            Halo-centric distance scaled by the halo boundary, such that :math:`0 < x < 1`. 
            Can be a scalar or a numpy array.

        rtol: float, optional
            Relative tolerance of the integration accuracy. Default is 1e-5. 
            
        Returns
        -------------------------------------------------------------------------------------------
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
        -------------------------------------------------------------------------------------------
        x: array_like
            Halo-centric distance scaled by the halo boundary, such that :math:`0 < x < 1`. 
            Can be a scalar or a numpy array.
            
        Returns
        -------------------------------------------------------------------------------------------
        vc: float
            The circular velocity in km / s; has the same dimensions as r.

        See also
        -------------------------------------------------------------------------------------------
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
        -------------------------------------------------------------------------------------------
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


@six.add_metaclass(ABCMeta)
class IsotropicJeansVelocity(object):
    """ Orthogonal mixin class used to transform a configuration 
    space model for the 1-halo term into a phase space model in which 
    velocities solve the Jeans equation of the underlying potential. 
    """
    def __init__(self):
        pass

    def _unscaled_radial_velocity_dispersion(self, x):
        """
        Method returns the radial velocity dispersion as a function of the 
        halo-centric distance. 
        """
        pass






















