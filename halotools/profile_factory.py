# -*- coding: utf-8 -*-
"""

Classes for halo profile objects. 

"""


from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.interpolate import interp1d as interp1d

from astropy.extern import six

from utils.array_utils import array_like_length as aph_len

import profile_components



class Galaxy_Profile(object):
    """ Container class for the radial profile of any galaxy population. 
    The behavior of the profile parameter model 
    (e.g., a c-M relation such as Bullock 2001 or DK14)
    and the actual profile itself (e.g., NFW, DK14, etc.) 
    are derived externally. The Halo_Profile container class 
    therefore only composes these behaviors into a composite object. 
    """

    def __init__(self,parameter_function_dict=None):
        """ 
        Parameters 
        ----------
        parameter_function_dict : dict 
            Dictionary of functions. Keys are the names of the 
            parameters governing the behavior of the profile. 
            Values are function objects governing how the profile 
            parameters vary as a function of halo properties such as 
            mass and accretion rate. 

        """
        self.parameter_function_dict = parameter_function_dict

    def profile_parameter(self,profile_parameter_key,*args):
        """ Method to compute the value of the profile parameter 
        as a function of the halo properties. 
        The behavior of this method is inherited by the function objects 
        passed to the constructor of Halo_Profile. 

        Parameters 
        ----------
        profile_parameter_key : string
            Specifies the name of the profile parameter, 
            e.g., 'conc', or 'gamma'.

        args : array_like
            Array of halo properties such as Mvir, 
            or M200b and mass accretion rate. 

        Returns 
        -------
        parameters : array_like
            Array of profile parameters. 

        """
        if self.parameter_function_dict is None:
            return None
        else:
            return self.parameter_function_dict[profile_parameter_key](args)

    def cumulative_profile(self,x,*args):
        """ Cumulative density profile. 

        Parameters 
        ----------
        x : array_like
            Input value of the halo-centric distance, 
            scaled by the size of the halo so that :math:`0 < x < 1`.

        args : array_like
            Parameters of the profile. 

        Returns 
        -------
        cumulative_density : array_like
            For a density profile whose behavior is determined by the input args, 
            the output is the value of the cumulative density evaluated at the input x. 

        Notes 
        -----
        The generic behavior of this method derives from either 
        1) numerically integrating the `density_profile` method, 
        or, preferably, 2) an explicit algebraic expression, in cases where 
        `density_profile` can be integrated analytically. 
        """
        pass

    def inverse_cumulative_profile(self, input_cumulative_density, abcissa, *args):
        """ Numerical inversion of the cumulative density profile. 

        Parameters 
        ---------- 
        cumulative_density : array_like 
            Input values of the cumulative_profile at which 
            the inverse function is being calculated.

        abcissa : array_like 
            1d array of values of halo-centric distance 
            scaled by the size of the halo. This array is passed to 
            cumulative_profile to construct a lookup table, which is then inverted. 

        args : list 
            Each element of the list is a len(input_cumulative_density)-array 
            of halo profile parameters. For an NFW profile, len(args)=1, and 
            the len(input_cumulative_density)-array contains 
            values of the concentration parameter. 

        Notes 
        -----
        Primarily used to generate a Monte Carlo realization of the profile.

        """

        inverse_function = interp1d(self.cumulative_profile(abcissa, args),abcissa)
        return inverse_function(input_cumulative_density)





