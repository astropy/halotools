# -*- coding: utf-8 -*-
"""

This module contains the components for 
the intra-halo spatial positions of galaxies 
used by `halotools.hod_designer` to build composite HOD models 
by composing the behavior of the components. 

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


        







