# -*- coding: utf-8 -*-
"""
Module contains the classes used to model the velocities 
of galaxies within their halos. 
"""
from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np 
from scipy.integrate import quad as quad_integration
from scipy.special import spence 

from astropy.extern import six 
from abc import ABCMeta, abstractmethod

from ....utils.array_utils import convert_to_ndarray

__author__ = ['Andrew Hearin']

__all__ = ['IsotropicJeansVelocity']



@six.add_metaclass(ABCMeta)
class IsotropicJeansVelocity(object):
    """ Orthogonal mix-in class used to transform a configuration 
    space model for the 1-halo term into a phase space model  
    by solving the Jeans equation of the underlying potential. 
    """

    def __init__(self, **kwargs):
        """
        """

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

