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

    def __init__(self,parameter_function_dict=None,
        cumulative_profile_function=None,
        galaxy_profile_component_model=None):
        """ 
        Parameters 
        ----------
        parameter_function_dict : dict, optional 
            Dictionary of functions. Keys are the names of the 
            parameters governing the behavior of the profile. 
            Values are function objects governing how the profile 
            parameters vary as a function of halo properties such as 
            mass and accretion rate. 
            If None, all galaxies will reside at the center of their host halo. 

        cumulative_profile_function : external function object
            Function used to evaluate the cumulative profile. 


        """
        self.parameter_function_dict = parameter_function_dict
        self.cumulative_profile_function = cumulative_profile_function
        self.galaxy_profile_component_model = galaxy_profile_component_model


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
        output_parameters : array_like
            Array of profile parameters. 

        """

        # For galaxy population types with trivial profiles, 
        # such as centrals with no spatial bias, return None
        if self.parameter_function_dict is None:
            output_parameters = None
        # For other cases such as satellites, orphans, biased centrals, etc., 
        # retrieve the profile parameter-halo relation passed to the constructor
        else:
            output_parameters = (
                self.parameter_function_dict[profile_parameter_key](args))

        # For cases where galaxies do not exactly trace the dark matter, 
        # modulate the halo profile parameters via the input galaxy profile model component
        if self.galaxy_profile_component_model is not None:
            profile_modulating_function = (
                self.galaxy_profile_model_component.profile_modulating_function[profile_parameter_key])
            output_parameters = (output_parameters*
                profile_modulating_function(args))

        return output_parameters
                

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






