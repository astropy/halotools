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

        too_many_args = (occuhelp.aph_len(args) > 0) & ('mock_galaxies' in kwargs.keys())
        if too_many_args == True:
            raise TypeError("TrivialCenProfile can be passed an array, or a mock, but not both")

        # If we are running in testmode, require that all galaxies 
        # passed to mc_coords are actually the same type
        runtest = ( (defaults.testmode == True) & 
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
    halo concentration, and the scalar multiple is mass-independent, so that 
    :math:`c_{\\mathrm{gal}} = F*c_{\\mathrm{halo}}`. 
    That traditional model is a special case of this class, 
    which encompasses halo-dependence to the multiplicatively biased parameters, 
    as well as support for any profile model with any number of parameters. 
    """

    def __init__(self, gal_type, halo_prof_model, 
        input_prof_params=[], input_abcissa_dict={}, input_ordinates_dict={}, 
        interpol_method='spline',input_spline_degree=3):
        """ 
        Parameters 
        ----------
        gal_type : string, optional
            Used to set the key value of the galaxy population being modeled.  

        halo_prof_model : object 
            `~halotools.HaloProfileModel` class instance. Determines the 
            underlying dark matter halo profile to which gal_type galaxies respond.

        input_prof_params : array_like, optional
            String array specifying the halo profile parameters to be modulated. 
            Values of this array must equal one of halo_prof_model.param_keys, e.g., 'halo_NFW_conc'.
            If input_prof_params is passed to the constructor, 
            input_abcissa_dict and input_ordinates_dict should not be passed, and 
            the abcissa and ordinates defining the modulation of the halo profile parameters 
            will be set according to the default_profile_dict dict in `~halotools.defaults`

        input_abcissa_dict : dictionary, optional 
            Dictionary whose keys are halo profile parameters and values 
            are the abcissa used to define the profile parameter modulating function. 
            Default values are set according to default_profile_dict in `~halotools.defaults`
            If input_abcissa_dict is passed to the constructor, 
            input_ordinates_dict must also be passed, and the input_prof_params list must not.

        input_ordinates_dict : dictionary, optional 
            Dictionary whose keys are halo profile parameters and values 
            are the ordinates used to define the profile parameter modulating function. 
            Default values are set according to default_profile_dict in `~halotools.defaults`
            If input_ordinates_dict is passed to the constructor, 
            input_abcissa_dict must also be passed, and the input_prof_params list must not.

        interpol_method : string, optional 
            Keyword specifying the method used to interpolate continuous behavior of the function  
            `radprof_modfunc` from only knowledge of its values at a finite 
            number of points. The default spline option interpolates 
            the model's abcissa and ordinates. The polynomial option uses the unique, 
            degree N polynomial passing through (abcissa, ordinates), 
            where N = len(abcissa) = len(ordinates). 

        input_spline_degree : int, optional
            Degree of the spline interpolation for the case of interpol_method='spline'. 
            If there are k abcissa values specifying the model, input_spline_degree 
            is ensured to never exceed k-1, 
            nor exceed the maximum value of 5 supported by scipy. 

        """

        self.gal_type = gal_type
        self.halo_prof_model = halo_prof_model

        self.set_parameter_dict(input_prof_params,input_abcissa_dict,input_ordinates_dict)

        self._setup_interpol(interpol_method, input_spline_degree)


    def get_modulated_prof_params(self, prof_param_key, *args, **kwargs):
        """ Primary method used by the outside world. 
        Used to assign new values of halo profile parameters to gal_type galaxies. 
        The new values will differ from the 
        profile parameters of the galaxies' underlying halo 
        by a (possibly halo-dependent) multiplicative factor, governed by `radprof_modfunc`. 

        Parameters 
        ----------
        prof_param_key : string
            Specifies the halo profile parameter being modulated. 

        input_prim_haloprops : array_like, optional positional argument
            Array of the primary halo property of the mock galaxies.

        input_halo_prof_params : array_like, optional positional argument
            Array of the underlying dark matter halo profile 
            parameters of the mock galaxies.

        mock_galaxies : object, optional keyword argument 

        Returns 
        ------- 
        output_prof_params : array_like

        """

        kwargs['prof_param_key'] = prof_param_key
        input_prim_haloprops, input_halo_prof_params = (
            self._retrieve_input_halo_data(*args, **kwargs)
            )

        multiplicative_modulation = (
            self.radprof_modfunc(prof_param_key, input_prim_haloprops)
            )

        output_prof_params = multiplicative_modulation*input_halo_prof_params

        return output_prof_params


    def _retrieve_input_halo_data(self, *args, **kwargs):
        """ Private method to retrieve an array of the primary halo property (e.g., Mvir), 
        and an array of the halo profile parameter values, associated with 
        the mock galaxies. Mostly used for convenient API. This method allows 
        us to pass either two input arrays, or an entire mock galaxy population, 
        to get_modulated_prof_params. 

        Parameters 
        ----------
        input_prim_haloprops : array_like, optional positional argument
            Array of the primary halo property of the mock galaxies.

        input_halo_prof_params : array_like, optional positional argument
            Array of the underlying dark matter halo profile 
            parameters of the mock galaxies.

        mock_galaxies : object, optional keyword argument 

        prof_param_key : string, optional keyword argument
            Used to access the correct halo profile parameter. 

        Returns 
        ------- 
        input_halo_prof_params : array_like

        """

        ###
        if occuhelp.aph_len(args) > 0:
            # We were passed an array of profile parameters, 
            # so we should not have also been passed a galaxy sample
            if 'mock_galaxies' in kwargs.keys():
                raise TypeError("RadProfBias can be passed an array, "
                    "or a mock, but not both")
            input_prim_haloprops = args[0]
            input_halo_prof_params = args[1]

        elif (occuhelp.aph_len(args) == 0) & ('mock_galaxies' in kwargs.keys()):
            # We were passed a collection of galaxies
            mock_galaxies = kwargs['mock_galaxies']
            halo_prof_param_key = kwargs['prof_param_key']
            prim_haloprop_key = mock_galaxies.model.prim_haloprop_key
            input_prim_haloprops = mock_galaxies[prim_haloprop_key]
            input_halo_prof_params = mock_galaxies[halo_prof_param_key]
        else:
            raise SyntaxError("get_modified_prof_params was called with "
                " incorrect inputs. Method accepts a positional argument that is an array "
                "storing the initial profile parameters to be modulated, "
                "or alternatively a mock galaxy object with the same array"
                " stored in the mock_galaxies.prof_param_keys attribute")

        return input_prim_haloprops, input_halo_prof_params


    def radprof_modfunc(self,profile_parameter_key,input_abcissa):
        """
        Factor by which the halo profile parameters of gal_type galaxies 
        differ from the profile parameter of their underlying dark matter halo. 

        Parameters 
        ----------
        profile_parameter_key : string
            Dictionary key of the profile parameter being modulated, e.g., 'halo_NFW_conc'. 

        input_abcissa : array_like
            array of primary halo property 

        Returns 
        -------
        output_profile_modulation : array_like
            Values of the multiplicative factor that will be used to 
            modulate the halo profile parameters. 

        Notes 
        -----
        Either assumes the profile parameters are modulated from those 
        of the underlying dark matter halo by a polynomial function 
        of the primary halo property, or is interpolated from a grid. 
        Either way, the behavior of this method is fully determined by 
        its values at the model's (abcissa, ordinates), specified by 
        self.abcissa_dict and self.ordinates_dict.
        """

        model_abcissa, model_ordinates = (
            self._retrieve_model_abcissa_ordinates(profile_parameter_key)
            )

        if self.interpol_method=='polynomial':
            output_profile_modulation = occuhelp.polynomial_from_table(
                model_abcissa,model_ordinates,input_abcissa)
        elif self.interpol_method=='spline':
            modulating_function = self.spline_function[profile_parameter_key]
            output_profile_modulation = modulating_function(input_abcissa)
        else:
            raise IOError("Input interpol_method must be 'polynomial' or 'spline'.")

        return output_profile_modulation

    def _retrieve_model_abcissa_ordinates(self, profile_parameter_key):
        """ Private method used to make API convenient. 
        Used to pass the correct (abcissa, ordinates) pair to radprof_modfunc. 

        Parameters 
        ----------
        profile_parameter_key : string 
            Specifies the halo profile parameter to be modulated by the model.

        Returns 
        -------
        abcissa : array_like 
            Array at which the values of the modulating function are anchored. 

        ordinates : array_like
            Array of values of the modulating function when evaulated at the abcissa. 

        Notes 
        -----
        Retrieving the ordinates requires more complicated bookkeeping than 
        retrieving the abcissa. 
        This is because abcissa values will never vary in an MCMC, whereas 
        ordinate values will. All halotools models are set up so that 
        all model parameters varied by an MCMC walker have their values stored 
        in a parameter_dict dictionary. Thus the ordinate values 
        that actually govern the behavior of `get_modulated_prof_params` 
        must be stored in RadProfBias.parameter_dict, and when those values 
        are updated the behavior of `get_modulated_prof_params` needs to vary accordingly. 
        The primary purpose of this private method is to produce that behavior. 
        """

        abcissa = self.abcissa_dict[profile_parameter_key]

        # We need to access the up-to-date ordinate values 
        # through self.parameter_dict, which is how the outside world modifies the 
        # model parameters. The keys to this dictionary are strings such as 
        # 'halo_NFW_conc_pari_gal_type', whose value is the i^th ordinate. 
        # However, dictionaries have no intrinsic ordering, so in order to 
        # construct a properly sequenced and up-to-date ordinates list, 
        # we have to jump through the following hoop. 
        ordinates = []
        for ipar in range(len(self.ordinates_dict)):
            key_ipar = self._get_parameter_key(profile_parameter_key, ipar)
            value_ipar = self.parameter_dict[key_ipar]
            ordinates.extend([value_ipar])

 
        return abcissa, ordinates

    def set_parameter_dict(self, 
        input_prof_params, input_abcissa_dict, input_ordinates_dict):
        """ Method used to set up dictionaries governing the behavior of the 
        profile modulating function. 

        Parameters 
        ---------- 
        input_prof_params : array_like 
            String array of keys of the halo profile parameter being modulated. 

        input_abcissa_dict : dict 
            keys are halo profile parameter keys, e.g., 'halo_NFW_conc', 
            values are abcissa defining the behavior of the modulating function 
            on that halo profile parameter. 

        input_ordinates_dict : dict  
            keys are halo profile parameter keys, e.g., 'halo_NFW_conc', 
            values are ordinates defining the behavior of the modulating function 
            on that halo profile parameter. 
        """

        input_prof_params = list(input_prof_params)
        self._test_sensible_inputs(input_prof_params, input_abcissa_dict, input_ordinates_dict)

        self.abcissa_dict={}
        self.ordinates_dict={}

        if input_prof_params is not []:
            for prof_param_key in input_prof_params:
                self.abcissa_dict[prof_param_key] = defaults.default_profile_dict['profile_abcissa']
                self.ordinates_dict[prof_param_key] = defaults.default_profile_dict['profile_ordinates']
        else:
            self.abcissa_dict = input_abcissa_dict
            self.ordinates_dict = input_ordinates_dict

        self.parameter_dict={}
        for prof_param_key, ordinates in self.ordinates_dict.iteritems():
            for ii, val in enumerate(ordinates):
                key = self._get_parameter_key(prof_param_key, ii)
                self.parameter_dict[key] = val

        self.param_keys = self.abcissa_dict.keys()

    def _test_sensible_inputs(self, 
        input_prof_params, input_abcissa_dict, input_ordinates_dict):
        """ Private method to verify that `set_parameter_dict` was passed 
        a reasonable set of inputs. 
        """

        if input_prof_params != []:
            try:
                assert input_abcissa_dict == {}
            except:
                raise SyntaxError("If passing input_prof_params to the constructor,"
                    " do not pass input_abcissa_dict")
            try:
                assert input_ordinates_dict == {}
            except:
                raise SyntaxError("If passing input_prof_params to the constructor,"
                    " do not pass input_ordinates_dict")
            try:
                assert set(input_prof_params).issubset(
                    set(self.halo_prof_model.param_keys))
            except:
                raise SyntaxError("Entries of input_prof_params must be keys of halo_prof_model")
        else:
            try:
                assert input_abcissa_dict != {}
            except:
                raise SyntaxError("If not passing input_prof_params to the constructor,"
                    "must pass input_abcissa_dict")
            try:
                assert input_ordinates_dict != {}
            except:
                raise SyntaxError("If not passing input_ordinates_dict to the constructor,"
                    "must pass input_abcissa_dict")

    def _setup_interpol(self, interpol_method, input_spline_degree):
        """ Private method used to configure the behavior of `radprof_modfunc`. 
        """

        if interpol_method not in ['spline', 'polynomial']:
            raise IOError("Input interpol_method must be 'polynomial' or 'spline'.")
        self.interpol_method = interpol_method

        def _setup_spline(self):
        # If using spline interpolation, configure its settings 
        
            scipy_maxdegree = 5
            self.spline_degree ={}
            self.spline_function = {}

            for prof_param_key in self.abcissa_dict.keys():
                self.spline_degree[prof_param_key] = (
                    np.min(
                [scipy_maxdegree, self.input_spline_degree, 
                aph_len(self.abcissa_dict[prof_param_key])-1])
                    )
                self.spline_function[prof_param_key] = occuhelp.aph_spline(
                    self.abcissa_dict[prof_param_key], 
                    self.ordinates_dict[prof_param_key], 
                    k=self.spline_degree[prof_param_key])

        if self.interpol_method=='spline':
            self.input_spline_degree=input_spline_degree
            _setup_spline(self)

    def _get_parameter_key(self, profile_parameter_key, ipar):
        """ Private method used to retrieve the key of self.parameter_dict 
        that corresponds to the appropriately selected i^th ordinate defining 
        `radprof_modfunc`. 
        """
        return profile_parameter_key+'_biasfunc_par'+str(ipar+1)+'_'+self.gal_type




        
            
        
        
            
            












        







