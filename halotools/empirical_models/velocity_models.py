# -*- coding: utf-8 -*-
"""
This module contains the components for 
the velocities of galaxies 
relative to their halos. 
"""

import numpy as np 
from scipy.integrate import quad as quad_integration
from scipy.special import spence 

from astropy.extern import six 
from abc import ABCMeta, abstractmethod

from ..utils.array_utils import convert_to_ndarray

__author__ = ['Andrew Hearin']

__all__ = ['IsotropicJeansVelocity', 'NFWJeansVelocity']



@six.add_metaclass(ABCMeta)
class IsotropicJeansVelocity(object):
    """ Orthogonal mix-in class used to transform a configuration 
    space model for the 1-halo term into a phase space model  
    by solving the Jeans equation of the underlying potential. 
    """

    @abstractmethod
    def dimensionless_velocity_dispersion(self, x):
        """
        Method returns the radial velocity dispersion scaled by 
        the virial velocity, as a function of the 
        halo-centric distance scaled by the halo radius.
        """
        pass



class NFWJeansVelocity(IsotropicJeansVelocity):
    """ Orthogonal mix-in class providing the solution to the Jeans equation 
    for galaxies orbiting in an isotropic NFW profile with no spatial bias. 
    """

    def _jeans_integrand_term1(self, y):
        """
        """
        return np.log(1+y)/(y**3*(1+y)**2)

    def _jeans_integrand_term2(self, y):
        """
        """
        return 1/(y**2*(1+y)**3)

    def _jeans_integral(self, x):
        """
        """
        term1 = 6.*spence(1. + x)
        term2 = -np.log(1. + x)/x**2
        term3 = 1./x
        term4 = 6./(1. + x)
        term5 = 1./(1. + x)**2
        term6 = -3.*(np.log(1. + x))**2
        term7 = 4.*np.log(1. + x)/x
        term8 = 2.*np.log(1. + x)/(1. + x)
        term9 = -1*np.log(1. + x)
        term10 = np.log(x)

        sum_of_terms = (term1 + term2 + term3 + term4 + term5 + 
            term6 + term7 + term8 + term9 + term10)

        return 0.5*sum_of_terms

    def dimensionless_velocity_dispersion(self, x, conc):
        """
        Parameters 
        -----------
        x : array_like 
            Halo-centric distance scaled by the halo boundary, so that 
            :math:`0 <= x <= 1`. Can be a scalar or numpy array

        conc : float 
            Concentration of the halo.

        Returns 
        -------
        result : array_like 
            Radial velocity dispersion profile scaled by the virial velocity. 
            The returned result has the same dimension as the input ``x``. 
        """
        x = convert_to_ndarray(x)
        result = np.zeros_like(x)

        prefactor = conc*(conc*x)*(1 + conc*x)**2/self.g(conc)

        lower_limit = conc*x
        upper_limit = float("inf")
        for i in range(len(x)):
            term1, _ = quad_integration(self._jeans_integrand_term1, 
                lower_limit[i], upper_limit, epsrel=1e-5)
            term2, _ = quad_integration(self._jeans_integrand_term2, 
                lower_limit[i], upper_limit, epsrel=1e-5)
            result[i] = term1 - term2 

        return result, prefactor, result*prefactor












