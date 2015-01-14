# -*- coding: utf-8 -*-
"""

This module contains the components for 
the intra-halo spatial positions of galaxies 
used by `halotools.hod_designer` to build composite HOD models 
by composing the behavior of the components. 

"""

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
from scipy.interpolate import UnivariateSpline as spline

from utils.array_utils import array_like_length as aph_len
import occupation_helpers as occuhelp 
import defaults

##################################################################################

class TrivialCenProfile(object):
    """ Profile assigning central galaxies to reside at exactly the halo center."""

    def __init__(self, gal_type):
        self.gal_type = gal_type

    def mc_coords(self, *args,**kwargs):

        too_many_args = (occuhelp.aph_len(args) > 0) & 'mock_galaxies' in kwargs.keys()
        if too_many_args == True:
            raise TypeError("TrivialCenProfile can be passed an array, or a mock, but not both")

        # If we are running in testmode, require that all galaxies 
        # passed to mc_coords are actually the same type
        runtest = ( (defaults.testmode_string in kwargs.keys()) & 
            (kwargs[defaults.testmode_string]==True) & 
            ('mock_galaxies' in kwargs.keys()) )
        if runtest == True:
            assert np.all(mock_galaxies.gal_type == self.gal_type)
        ###

        return 0

##################################################################################

class IsotropicSats(object):
    """ Classical satellite profile. """

    def __init__(self, gal_type):
        self.gal_type = gal_type

    def mc_angles(self,coords):
        """
        Generate a list of Ngals random points on the unit sphere. 
        The coords array is passed as input to save memory, 
        speeding up satellite position assignment when building mocks.

        """
        Ngals = aph_len(coords[:,0])
        cos_t = np.random.uniform(-1.,1.,Ngals)
        phi = np.random.uniform(0,2*np.pi,Ngals)
        sin_t = np.sqrt((1.-(cos_t*cos_t)))
        
        coords[:,0] = sin_t * np.cos(phi)
        coords[:,1] = sin_t * np.sin(phi)
        coords[:,2] = cos_t

        return coords

    def mc_coords(self,coords,inv_cumu_prof_func,system_center,host_Rvir):
        Ngals = aph_len(coords[:,0])
        random_cumu_prof_vals = np.random.random(Ngals)

        r_random = inv_cumu_prof_func(random_cumu_prof_vals)*host_Rvir

        coords *= r_random.reshape(Ngals,1)
        coords += system_center.reshape(1,3)

        return coords


##################################################################################
class RadProfBias(object):
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
            Keys are names of the profile parameter, e.g., 'conc', or 'gamma'. 
            Values are dictionaries of abcissa and ordinates. 
            Thus parameter_dict is a dictionary of dictionaries. 

        interpol_method : string, optional 
            Keyword specifying how `radprof_modfunc` 
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

        Notes 
        -----
        The initialization constructor will use input_parameter_dict to create a 
        new dictionary, prepend/append to the  
        input_parameter_dict keys to avoid potential key duplication 
        when using this class as a component of a composite model. 

        """

        self.gal_type = gal_type

        self.parameter_dict = {}
        self.abcissa_key = {}
        self.ordinates_key = {}

        # The correct keys are strings for the abcissa and ordinate arrays
        # with a naming convention set in the defaults module
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
            # The radprof_modfunc method needs to access the ordinates and abcissa
            # This is accomplished by binding the key to an attribute of the model object
            # This binding is done via a dictionary, where each key of the dictionary 
            # corresponds to a profile parameter that is being modulated.
            self.abcissa_key[profile_parameter] = (
                profile_parameter+'_model_abcissa_'+self.gal_type )
            self.ordinates_key[profile_parameter] = (
                profile_parameter+'_model_ordinates_'+self.gal_type )

        # Set the interpolation scheme 
        if interpol_method not in ['spline', 'polynomial']:
            raise IOError("Input interpol_method must be 'polynomial' or 'spline'.")
        self.interpol_method = interpol_method

        # If using spline interpolation, configure its settings 
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

    def radprof_modfunc(self,profile_parameter_key,input_abcissa):
        """
        Factor by which gal_type galaxies differ from are quiescent 
        as a function of the primary halo property.

        Parameters 
        ----------
        input_abcissa : array_like
            array of primary halo property 

        profile_parameter_key : string
            Dictionary key of the profile parameter being modulated, e.g., 'conc'. 

        Returns 
        -------
        output_profile_modulation : array_like
            Values of the profile parameters evaluated at input_abcissa. 

        Notes 
        -----
        Either assumes the profile parameters are modulated from those 
        of the underlying dark matter halo by a polynomial function 
        of the primary halo property, or is interpolated from a grid. 
        Either way, the behavior of this method is fully determined by 
        its values at the model abcissa, as specified in parameter_dict. 
        """

        model_abcissa = self.parameter_dict[self.abcissa_key[profile_parameter_key]]
        model_ordinates = self.parameter_dict[self.ordinates_key[profile_parameter_key]]

        if self.interpol_method=='polynomial':
            output_profile_modulation = occuhelp.polynomial_from_table(
                model_abcissa,model_ordinates,input_abcissa)
        elif self.interpol_method=='spline':
            modulating_function = self.spline_function[profile_parameter_key]
            output_profile_modulation = modulating_function(input_abcissa)
        else:
            raise IOError("Input interpol_method must be 'polynomial' or 'spline'.")

        return output_profile_modulation


















        







