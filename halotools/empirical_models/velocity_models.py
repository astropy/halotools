# -*- coding: utf-8 -*-
"""
This module contains the components for 
the velocities of galaxies 
relative to their halos. 
"""

__author__ = ['Andrew Hearin']

__all__ = ['IsotropicJeansVelocity']

@six.add_metaclass(ABCMeta)
class IsotropicJeansVelocity(object):
    """ Orthogonal mixin class used to transform a configuration 
    space model for the 1-halo term into a phase space model  
    by solving the Jeans equation of the underlying potential. 
    """
    def __init__(self, **kwargs):
        pass

    def _unscaled_radial_velocity_dispersion(self, x):
        """
        Method returns the radial velocity dispersion as a function of the 
        halo-centric distance. 
        """
        pass
