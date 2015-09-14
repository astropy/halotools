# -*- coding: utf-8 -*-
"""
Module contains the classes used to model the velocities 
of galaxies within their halos. 
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

    def __init__(self, velocity_bias = False, **kwargs):
        """
        Parameters 
        -----------
        velocity_bias : bool, optional 
            Boolean specifying whether the galaxy velocities are biased 
            with respect to the halo velocities. If True, ``param_dict`` will have a 
            parameter called ``velbias_satellites`` that multiplies the underlying 
            Jeans solution for the halo radial velocity dispersion by an overall factor. 
            Default is False. 
        """
        if velocity_bias is True:
            self.param_dict['velbias_satellites'] = 1.

    @abstractmethod
    def dimensionless_velocity_dispersion(self, x, *args):
        """
        Method returns the radial velocity dispersion scaled by 
        the virial velocity, as a function of the 
        halo-centric distance scaled by the halo radius.

        Parameters 
        ----------
        x : array_like 
            Halo-centric distance scaled by the halo boundary, so that 
            :math:`0 <= x <= 1`. Can be a scalar or numpy array

        args : sequence, optional 
            Any additional parameters necessary to specify the shape of the radial profile, 
            e.g., halo concentration.         

        Returns 
        -------
        result : array_like 
            Radial velocity dispersion profile scaled by the virial velocity. 
            The returned result has the same dimension as the input ``x``. 

        """
        pass


class NFWJeansVelocity(IsotropicJeansVelocity):
    """ Orthogonal mix-in class providing the solution to the Jeans equation 
    for galaxies orbiting in an isotropic NFW profile with no spatial bias. 
    """

    def __init__(self, **kwargs):
        """
        Parameters 
        -----------
        velocity_bias : bool, optional 
            Boolean specifying whether the galaxy velocities are biased 
            with respect to the halo velocities. If True, ``param_dict`` will have a 
            parameter called ``velbias_satellites`` that multiplies the underlying 
            Jeans solution for the halo radial velocity dispersion by an overall factor. 
            Default is False. 
        """
        IsotropicJeansVelocity.__init__(self, **kwargs)

    def _jeans_integrand_term1(self, y):
        """
        """
        return np.log(1+y)/(y**3*(1+y)**2)

    def _jeans_integrand_term2(self, y):
        """
        """
        return 1/(y**2*(1+y)**3)

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
        x = x.astype(float)
        result = np.zeros_like(x)

        prefactor = conc*(conc*x)*(1. + conc*x)**2/self.g(conc)

        lower_limit = conc*x
        upper_limit = float("inf")
        for i in range(len(x)):
            term1, _ = quad_integration(self._jeans_integrand_term1, 
                lower_limit[i], upper_limit, epsrel=1e-5)
            term2, _ = quad_integration(self._jeans_integrand_term2, 
                lower_limit[i], upper_limit, epsrel=1e-5)
            result[i] = term1 - term2 

        return np.sqrt(result*prefactor)












