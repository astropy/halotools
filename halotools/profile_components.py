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

class TrivialCenProfile(object):
    """ Profile assigning central galaxies to reside at exactly the halo center."""

    def __init__(self, gal_type):
        self.gal_type = gal_type

        # The following few lines may indicate that TrivialCenProfile 
        # should actually be a child class, fine for now as is 
        #self.halo_prof_model = None
        #self.sec_haloprop_bool = False
        #self.spatial_bias_model = None

    def mc_coords(self, coords, occupations, *args):
        host_centers = args[0]
        if np.all(occupations==1):
            coords = host_centers
        else:
            raise("Only occupied halos should be passed to mc_coords method")

        return coords

##################################################################################

class IsotropicSats(object):
    """ Classical satellite profile. """

    def __init__(self, gal_type, halo_prof_model, spatial_bias_model=None):
        self.gal_type = gal_type

        self.halo_prof_model = halo_prof_model
        self.sec_haloprop_bool = self.halo_prof_model.sec_haloprop_bool
        self.inv_cumu_prof_funcs = self.halo_prof_model.inv_cumu_prof_funcs
        self.host_prof_param_bins = self.halo_prof_model.prof_param_bins

        self.spatial_bias_model = spatial_bias_model



    def mc_coords(self, coords, occupations, *args):

        if np.any(occupations==0):
            raise("Only occupied halos should be passed to mc_coords method")

        host_centers = args[0]
        host_Rvirs = args[1]
        prim_haloprops = args[2]
        if self.sec_haloprop_bool is True:
            if aph_len(args) <= 3:
                raise("Input halo_prof_model requires two halo properties, only one was passed to mc_coords")
            else:
                sec_haloprops = args[3]
                host_prof_params = self.halo_prof_model(prim_haloprops, sec_haloprops)
        else:
            host_prof_params = (
                self.halo_prof_model(prim_haloprops)
                )

        if self.spatial_bias_model is None:
            satsys_prof_params = host_prof_params
        else:
            # Spatial bias model not yet integrated
            pass

        inv_cumu_prof_func_indices = np.digitize(satsys_prof_params, self.host_prof_param_bins)

        coords = self.mc_angles(coords)

        satsys_first_index = 0
        for host_index, Nsatsys in enumerate(occupations):
            satsys_coords = coords[satsys_first_index:satsys_first_index+Nsatsys]
            host_center = host_centers[host_index]
            host_Rvir = host_Rvirs[host_index]
            inv_cumu_prof_func = (self.inv_cumu_prof_funcs[
                inv_cumu_prof_func_indices[host_index]])

            satsys_coords = (self.mc_coords_singlesys(
                satsys_coords, inv_cumu_prof_func, host_center, host_Rvir)
                )
            satsys_first_index += Nsatsys

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

    def mc_coords_singlesys(self,coords,inv_cumu_prof_func,system_center,host_Rvir):
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
            Keys are names of the profile parameter, e.g., 'conc'. 
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
            # The radprof_modfunc method needs to access the ordinates and abcissa
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

    def radprof_modfunc(self,profile_parameter_key,input_abcissa):
        """
        Factor by which gal_type galaxies differ from are quiescent 
        as a function of the primary halo property.

        Parameters 
        ----------
        input_abcissa : array_like
            array of primary halo property at which the quiescent fraction 
            is being computed. 

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























        







