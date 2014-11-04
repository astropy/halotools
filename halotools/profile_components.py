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
from scipy.interpolate import UnivariateSpline as spline

from utils.array_utils import array_like_length as aph_len
import occupation_helpers as occuhelp 
import defaults

##################################################################################
######## Currently only in this module for temporary development purposes ########
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
##################################################################################



##################################################################################
######## Currently only in this module for temporary development purposes ########

def cumulative_NFW_profile(x,c):
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
##################################################################################



class ClassicalSpatialBias(object):
    """ Conventional model for the spatial bias of satellite galaxies. 
    The profile parameters governing the satellite distribution are set to be 
    a scalar multiple of the profile parameters of their host halo. 

    Traditionally applied to the NFW case, where the only profile parameter is 
    halo concentration, and the scalar multiple is mass-independent. This 
    traditional model is a special case of this class, 
    which encompasses halo-dependent spatial bias, non-NFW profiles, 
    as well as (mass-dependent) quenching gradients. 
    """

    def __init__(self, gal_type, 
        input_parameter_dict={'conc':defaults.default_profile_dict}, 
        interpol_method='spline',input_spline_degree=3):
        """ 
        Parameters 
        ----------
        gal_type : string, optional
            Sets the key value used by `halotools.hod_designer` and 
            `~halotools.profile_factory` to access the behavior of the methods 
            of this class. 

        input_parameter_dict : dictionary, optional 
            Dictionary specifying how each profile parameter should be modulated. 
            Keys are names of the profile parameter, e.g., 'conc'. 
            Values are dictionaries of abcissa and ordinates. 
            Thus parameter_dict is a dictionary of dictionaries. 

        interpol_method : string, optional 
            Keyword specifying how `profile_modulating_function` 
            evaluates input values that differ from the small number of values 
            in self.parameter_dict. 
            The default spline option interpolates the 
            model's abcissa and ordinates. 
            The polynomial option uses the unique, degree N polynomial 
            passing through the ordinates, where N is the number of supplied ordinates. 

        input_spline_degree : int, optional
            Degree of the spline interpolation for the case of interpol_method='spline'. 
            If there are k abcissa values specifying the model, input_spline_degree 
            is ensured to never exceed k-1, nor exceed 5. 
        """

        self.gal_type = gal_type

        self.parameter_dict = {}
        self.abcissa_key = {}
        self.ordinates_key = {}

        correct_keys = defaults.default_profile_dict.keys()
        # Loop over all profile parameters that are being modulated 
        for profile_parameter, profile_parameter_dict in input_parameter_dict.iteritems():
            # Test that the dictionary associated with profile_parameter has the right keys
            occuhelp.test_correct_keys(profile_parameter_dict, correct_keys)
            # Append the table_dictionary of each profile parameter 
            # to self.parameter_dict 
            new_dict_to_append = occuhelp.format_parameter_keys(profile_parameter_dict,
                correct_keys, self.gal_type, key_prefix=profile_parameter)
            self.parameter_dict = dict(
                self.parameter_dict.items() + 
                new_dict_to_append.items() 
                )
            # The profile_modulating_function method needs to access the ordinates and abcissa
            # This is accomplished by binding the key to an attribute of the model object
            # This binding is done via a dictionary, where each key of the dictionary 
            # corresponds to a profile parameter that is being modulated.
            self.abcissa_key[profile_parameter] = (
                profile_parameter+'_profile_abcissa_'+self.gal_type )
            self.ordinates_key[profile_parameter] = (
                profile_parameter+'_profile_ordinates_'+self.gal_type )

        # Set the interpolation scheme 
        if interpol_method not in ['spline', 'polynomial']:
            raise IOError("Input interpol_method must be 'polynomial' or 'spline'.")
        self.interpol_method = interpol_method

        # Set the degree of the spline
        if self.interpol_method=='spline':
            scipy_maxdegree = 5
            self.spline_degree ={}
            self.spline_function = {}

            for profile_parameter, profile_parameter_dict in input_parameter_dict.iteritems():
                self.spline_degree[profile_parameter] = (
                    np.min(
                [scipy_maxdegree, input_spline_degree, 
                aph_len(self.parameter_dict[self.abcissa_key[profile_parameter]])-1])
                    )
                self.spline_function[profile_parameter] = occuhelp.aph_spline(
                    self.parameter_dict[self.abcissa_key[profile_parameter]],
                    self.parameter_dict[self.ordinates_key[profile_parameter]],
                    k=self.spline_degree[profile_parameter])

    def profile_modulating_function(self,input_abcissa):
        """
        Factor by which gal_type galaxies differ from are quiescent 
        as a function of the primary halo property.

        Parameters 
        ----------
        input_abcissa : array_like
            array of primary halo property at which the quiescent fraction 
            is being computed. 

        Returns 
        -------
        output_profile_parameters : array_like
            Values of the profile parameters evaluated at input_abcissa. 

        Notes 
        -----
        Either assumes the profile parameters are modulated from those 
        of the underlying dark matter halo by a polynomial function 
        of the primary halo property, or is interpolated from a grid. 
        Either way, the behavior of this method is fully determined by 
        its values at the model abcissa, as specified in parameter_dict. 
        """
        pass

























        







