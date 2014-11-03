# -*- coding: utf-8 -*-
"""

Classes for halo profile objects. 

"""


from astropy.extern import six
from abc import ABCMeta, abstractmethod
import numpy as np
from utils.array_utils import array_like_length as aph_len




def anatoly_concentration(logM):
    """ Power law fitting formula for the concentration-mass relation of Bolshoi host halos at z=0
    Taken from Klypin et al. 2011, arXiv:1002.3660v4, Eqn. 12.

    :math:`c(M) = c_{0}(M/M_{piv})^{\\alpha}`

    Parameters
    ----------
    logM : array 
        array of :math:`log_{10}M_{vir}` of halos in catalog

    Returns
    -------
    concentrations : array

    Notes 
    -----
    This is currently the only concentration-mass relation implemented. This will later be 
    bundled up into a class with a bunch of different radial profile methods, NFW and non-.

    Values are currently hard-coded to Anatoly's best-fit values:

    :math:`c_{0} = 9.6`

    :math:`\\alpha = -0.075`

    :math:`M_{piv} = 10^{12}M_{\odot}/h`

    """
    
    masses = np.zeros(aph_len(logM)) + 10.**np.array(logM)
    c0 = 9.6
    Mpiv = 1.e12
    a = -0.075
    concentrations = c0*(masses/Mpiv)**a
    return concentrations


@six.add_metaclass(ABCMeta)
class Halo_Profile(object):
    """ Container class for any halo profile.
    """

    def __init__(self,parameter_function_dict):
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
            Specifies the name of the profile parameter. 

        args : array_like
            Array of halo properties

        Returns 
        -------
        parameters : array_like
            Array of profile parameters. 

        """
        return self.parameter_function_dict[profile_parameter_key](args)


    @abstractmethod
    def density_profile(self,x,*args):
        """ Intra-halo density profile. 

        Parameters 
        ----------
        x : array_like
            Input value of the halo-centric distance, 
            scaled by the size of the halo so that :math:`0 < x < 1`.

        args : array_like
            Parameters of the profile. 

        Returns 
        -------
        density : array_like
            For a density profile whose behavior is determined by the input args, 
            the output is the value of that density profile evaluated at the input x. 
        """
        pass

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
        The generic behavior of this method derives from integrating the `density_profile` method. 
        However, this behavior should be over-ridden by subclasses of profiles 
        where the integral can be done analytically, such as for NFW profiles. 

        """

        # (1) code for interpolating from lookup table
        # code for integrating self.density_profile
        # will be over-ridden 

        pass

    def lookup_cumulative_profile(self):
        """ Method for evaluating the cumulative profile 
        from a pre-computed lookup table.

        First, check the lookup table cache 
        directory to see if the function has been pre-computed. 
        If the lookup table is there, it is loaded into memory 
        and the cumulative profile is evaluated via interpolation. 
        If the lookup table is not there, a warning is issued and
        `cumulative_profile` is used to generate the table, 
        which is then stored in cache and evaluated.
        """ 
        pass

    def inverse_cumulative_profile(self):
        pass





class NFW_Profile(Halo_Profile):

    def __init__(self,
        parameter_function_dict = {'conc':anatoly_concentration}):
        Halo_Profile.__init__(self,parameter_function_dict)

    def density_profile(self,x,c):
        """ Intra-halo density profile. 

        Parameters 
        ----------
        x : array_like
            Input value of the halo-centric distance, 
            scaled by the size of the halo so that :math:`0 < x < 1`.

        c : array_like
            Input value of the halo concentration. 

        Returns 
        -------
        normalized_density : array_like
            For a density profile whose behavior is determined by the input args, 
            the output is the value of that density profile evaluated at the input x. 

        Notes 
        -----
        Function is normalized to unity.
        """

        normalized_density = 1./(c*x*(1 + c*x)*(1 + c*x))

        return normalized_density


    def cumulative_profile(self,x,c):
        """ Unit-normalized integral of an NFW profile with concentration c.

        :math:`F(x,c) = \\frac{ln(1+xc) - \\frac{xc}{1+xc}} 
        {ln(1+c) - \\frac{c}{1+c}}`

        Parameters
        ----------
        x : array_like
            Values are in the range (0,1).
            Elements x = r/Rvir specify host-centric distances in the range 0 < r/Rvir < 1.

        c : array_like
            Concentration of halo whose profile is being tabulated.

        Returns
        -------
        F : array 
            Array of floats in the range 0 < x < 1 corresponding to the 
            cumulative mass of an NFW profile at x = r/Rhalo.

        """
        c = np.array(c)
        x = np.array(x)
        norm=np.log(1.+c)-c/(1.+c)
        F = (np.log(1.+x*c) - x*c/(1.+x*c))/norm
        return F


        






























