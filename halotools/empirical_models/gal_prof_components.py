# -*- coding: utf-8 -*-
"""

This module contains the components for 
the intra-halo spatial positions of galaxies 
used by `halotools.hod_designer` to build composite HOD models 
by composing the behavior of the components. 

"""

__all__ = ['SpatialBias']

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
from scipy.interpolate import UnivariateSpline as spline

from ..utils.array_utils import custom_len
import model_helpers as model_helpers 
import model_defaults

from functools import partial

##################################################################################

##################################################################################
class SpatialBias(object):
    """ Classical method to model how the radial profile of a galaxy population 
    can systematically differ from the profile of the underlying dark matter halo. 
    Accomplished by keeping the form of the profile fixed, 
    but allowing for halo and galaxy profile parameters to be distinct. 

    Traditionally applied to the NFW case, where the only profile parameter is 
    halo concentration, and the galaxy concentration is a mass-independent 
    scalar multiple of its dark matter halo profile, so that 
    :math:`c_{\\mathrm{gal}} = F\\times c_{\\mathrm{halo}}`. 
    That traditional model is a special case of this class, 
    which encompasses possible dependence of 
    the multiplicative bias on whatever the primary halo property is, 
    as well as support for any arbitrary profile model with any number of parameters. 
    """

    def __init__(self, gal_type, halo_prof_model, 
        input_prof_params='all', input_abcissa_dict={}, input_ordinates_dict={}, 
        interpol_method='spline',input_spline_degree=3, 
        multiplicative_bias = True):
        """ 
        Parameters 
        ----------
        gal_type : string
            Name of the galaxy population being modeled, 
            e.g., ``satellites`` or ``orphans``.  

        halo_prof_model : object 
            `~halotools.empirical_models.HaloProfileModel` sub-class instance. 
            Determines the underlying dark matter halo profile 
            to which gal_type galaxies respond.
            
            Used *only* to ensure self-consistency between the galaxy and halo profiles, 
            accomplished by verifying that the parameters set to be modulated are 
            actually parameters of the underlying halo profile. 

        input_prof_params : array_like, optional
            Array of strings specifying the halo profile parameters to be modulated. 
            Values of this array must equal one of 
            halo_prof_model.prof_param_keys, e.g., 'halo_NFW_conc'.
            If input_prof_params is passed to the constructor, 
            input_abcissa_dict and input_ordinates_dict should not be passed, and 
            the abcissa and ordinates defining the modulation of the halo profile parameters 
            will be set according to the default_profile_dict, 
            located in `~halotools.empirical_models.model_defaults`

        input_abcissa_dict : dictionary, optional 
            Dictionary whose keys are halo profile parameter names and values 
            are the abcissa used to define the profile parameter modulating function. 
            See the Notes section below for examples. 

            Default values are set according to ``default_profile_dict``, 
            located in `~halotools.empirical_models.model_defaults`. 
            Default interpretation of the abcissa values is as the base-10 logarithm 
            of whatever is being used for the primary halo property. 

            If `input_abcissa_dict` is passed to the constructor, 
            `input_ordinates_dict` must also be passed, and the `input_prof_params` list must not.

        input_ordinates_dict : dictionary, optional 
            Dictionary whose keys are halo profile parameter names and values 
            are the ordinates used to define the profile parameter modulating function. 
            See the Notes section below for examples. 

            Default values are set according to ``default_profile_dict``, 
            located in `~halotools.empirical_models.model_defaults`. 

            If `input_abcissa_dict` is passed to the constructor, 
            `input_ordinates_dict` must also be passed, and the `input_prof_params` list must not.

        interpol_method : string, optional 
            Keyword specifying the method used to interpolate 
            continuous behavior of the function  `radprof_modfunc` from 
            only knowledge of its values at the finite 
            number of points in the abcissa. 

            The default option, ``spline``, interpolates 
            the model's abcissa and ordinates. The polynomial option uses the unique, 
            degree *N* polynomial passing through (abcissa, ordinates), 
            where :math:`N = \\mathrm{len}(abcissa) = \\mathrm{len}(ordinates)`. 

        input_spline_degree : int, optional
            Degree of the spline interpolation for the case where 
            the chosen interpol_method is `spline`. 

            If there are *k* abcissa values specifying the model, input_spline_degree 
            is ensured to never exceed *k-1*, 
            nor exceed the maximum value of *5* supported by scipy. 

        multiplicative_bias : boolean, optional 
            If *True* (the default setting), the galaxy profile parameters are set to be the 
            multiplication of `radprof_modfunc` and the underlying halo profile parameter. 
            If *False*, the galaxy profile parameters will be the actual value 
            returned by `radprof_modfunc`. 

        """

        # Bind the inputs to the instance
        self.gal_type = gal_type
        self.halo_prof_model = halo_prof_model

        self.multiplicative_bias = multiplicative_bias

        # The following call to _set_param_dict primarily does two things:
        # 1. Creates attributes self.abcissa_dict and self.ordinates_dict, 
        # each with one key per biased galaxy profile parameter
        # 2. Creates an attribute self.param_dict. This is the dictionary 
        # that actually governs the behavior of the model. Its keys have names 
        # such as 'NFWmodel_conc_biasfunc_par1_satellites'
        self._set_param_dict(
            input_prof_params,input_abcissa_dict,input_ordinates_dict)

        # Create the following two convenience lists:
        # self.halo_prof_param_keys and self.gal_prof_param_keys. 
        # halo_prof_param_keys has entries such as 'NFWmodel_conc'. 
        # These keys are simply the keys of self.abcissa_dict, 
        # so that ONLY SPATIALLY BIASED PARAMETERS ARE IN THE LIST. 
        # gal_prof_param_keys is identical to halo_prof_param_keys, 
        # but each entry has been prepended with 'gal_'. 
        self._set_prof_params()

        # Configure the settings of scipy's spline interpolation routine
        self._setup_interpol(interpol_method, input_spline_degree)

        self._set_primary_behaviors()


    def _set_primary_behaviors(self):
        """ Bind new methods to the `SpatialBias` instance that 
        will be used as the primary functions assigning biased 
        profile parameters to the gal_type population.  
        """
        # self.halo_prof_param_keys is a list set in self._set_prof_params
        # Only keys of biased profile parameters appear in the list
        for halokey in self.halo_prof_param_keys:
            galkey = self._get_gal_prof_param_key(halokey)
            new_method_name = galkey
            function = partial(self.get_modulated_prof_params, halokey)
            setattr(self, new_method_name, function)
        
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

        # Create a new dictionary key to conveniently pass 
        # 'prof_param_key' to _retrieve_input_halo_data via **kwargs
        kwargs['prof_param_key'] = prof_param_key
        input_prim_haloprops, input_halo_prof_params = (
            self._retrieve_input_halo_data(*args, **kwargs)
            )

        parameter_modulation = (
            self.radprof_modfunc(prof_param_key, input_prim_haloprops)
            )

        if self.multiplicative_bias==True:
            output_prof_params = parameter_modulation*input_halo_prof_params
        else:
            output_prof_params = parameter_modulation

        return output_prof_params

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
            output_profile_modulation = model_helpers.polynomial_from_table(
                model_abcissa,model_ordinates,input_abcissa)
        elif self.interpol_method=='spline':
            modulating_function = self.spline_function[profile_parameter_key]
            output_profile_modulation = modulating_function(input_abcissa)
        else:
            raise IOError("Input interpol_method must be 'polynomial' or 'spline'.")

        return output_profile_modulation

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
        if model_helpers.custom_len(args) > 0:
            # We were passed an array of profile parameters, 
            # so we should not have also been passed a galaxy sample
            if 'mock_galaxies' in kwargs.keys():
                raise TypeError("SpatialBias can be passed an array, "
                    "or a mock, but not both")
            return args[0], args[1]

        elif (model_helpers.custom_len(args) == 0) & ('mock_galaxies' in kwargs.keys()):
            # We were passed a collection of galaxies
            mock_galaxies = kwargs['mock_galaxies']
            halo_prof_param_key = kwargs['prof_param_key']
            prim_haloprop_key = mock_galaxies.model.prim_haloprop_key

            return mock_galaxies[prim_haloprop_key], mock_galaxies[halo_prof_param_key]

        else:
            raise SyntaxError("get_modified_prof_params was called with "
                " incorrect inputs. Method accepts a positional argument that is an array "
                "storing the initial profile parameters to be modulated, "
                "or alternatively a mock galaxy object with the same array"
                " stored in the mock_galaxies.prof_param_keys attribute")


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
        in a param_dict dictionary. Thus the ordinate values 
        that actually govern the behavior of `get_modulated_prof_params` 
        must be stored in SpatialBias.param_dict, and when those values 
        are updated the behavior of `get_modulated_prof_params` needs to vary accordingly. 
        The primary purpose of this private method is to produce that behavior. 
        """

        abcissa = self.abcissa_dict[profile_parameter_key]

        # We need to access the up-to-date ordinate values 
        # through self.param_dict, which is how the outside world modifies the 
        # model parameters. The keys to this dictionary are strings such as 
        # 'halo_NFW_conc_pari_gal_type', whose value is the i^th ordinate. 
        # However, dictionaries have no intrinsic ordering, so in order to 
        # construct a properly sequenced and up-to-date ordinates list, 
        # we have to jump through the following hoop. 
        ordinates = []
        for ipar in range(len(self.ordinates_dict)):
            key_ipar = self._get_parameter_key(profile_parameter_key, ipar)
            value_ipar = self.param_dict[key_ipar]
            ordinates.extend([value_ipar])
 
        return abcissa, ordinates

    def update_param_dict(self, new_param_dict):
        for key in self.param_dict.keys():
            self.param_dict[key] = new_param_dict[key]

    def _set_param_dict(self, 
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

        if input_prof_params=='all':
            input_prof_params = self.halo_prof_model.prof_param_keys
        else:
            input_prof_params = list(input_prof_params)
        self._test_sensible_inputs(input_prof_params, input_abcissa_dict, input_ordinates_dict)

        self.abcissa_dict={}
        self.ordinates_dict={}

        if input_prof_params is not []:
            for prof_param_key in input_prof_params:
                self.abcissa_dict[prof_param_key] = (
                    model_defaults.default_profile_dict['profile_abcissa'])
                self.ordinates_dict[prof_param_key] = (
                    model_defaults.default_profile_dict['profile_ordinates'])
        else:
            self.abcissa_dict = input_abcissa_dict
            self.ordinates_dict = input_ordinates_dict

        self.param_dict={}
        for prof_param_key, ordinates in self.ordinates_dict.iteritems():
            for ii, val in enumerate(ordinates):
                key = self._get_parameter_key(prof_param_key, ii)
                self.param_dict[key] = val

    def _set_prof_params(self):

        self.halo_prof_param_keys = self.abcissa_dict.keys()
        self.gal_prof_param_keys = (
            [self._get_gal_prof_param_key(key) for key in self.halo_prof_param_keys]
            )

    def _test_sensible_inputs(self, 
        input_prof_params, input_abcissa_dict, input_ordinates_dict):
        """ Private method to verify that `_set_param_dict` was passed 
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
            if self.multiplicative_bias==True:
                try:
                    input_keyset = set(input_prof_params)
                    halo_keyset = set(self.halo_prof_model.prof_param_keys)
                    assert input_keyset.issubset(halo_keyset)
                except:
                    errant_key = list(input_keyset-halo_keyset)[0]
                    raise SyntaxError("If multiplicative_bias is True, "
                        " input_prof_params must be keys of halo_prof_model\n"
                        "For input_prof_param %s, found no matching key "
                        " in the halo_prof_model" % errant_key) 
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
                custom_len(self.abcissa_dict[prof_param_key])-1])
                    )
                self.spline_function[prof_param_key] = model_helpers.custom_spline(
                    self.abcissa_dict[prof_param_key], 
                    self.ordinates_dict[prof_param_key], 
                    k=self.spline_degree[prof_param_key])

        if self.interpol_method=='spline':
            self.input_spline_degree=input_spline_degree
            _setup_spline(self)

    def _get_parameter_key(self, profile_parameter_key, ipar):
        """ Private method used to retrieve the key of self.param_dict 
        that corresponds to the appropriately selected i^th ordinate defining 
        `radprof_modfunc`. 
        """
        return profile_parameter_key+'_biasfunc_par'+str(ipar+1)+'_'+self.gal_type

    def _get_gal_prof_param_key(self, halo_prof_param_key):
        return model_defaults.galprop_prefix+halo_prof_param_key





        
            
        
        
            
            












        







