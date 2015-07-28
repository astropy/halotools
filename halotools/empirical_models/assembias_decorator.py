# -*- coding: utf-8 -*-
"""
Decorator class for implementing generalized assembly bias
"""

import numpy as np 
from . import model_defaults 

from ..halotools_exceptions import HalotoolsError

class HeavisideAssembiasComponent(object):
    """
    """
    def __init__(self, sec_haloprop_key = model_defaults.sec_haloprop_key, 
        loginterp = True, split = 0.5, assembias_strength = 0.5, **kwargs):
        """
        Parameters 
        ----------
        baseline_model : Class object 
            Class governing the underlying behavior which is being 
            decorated with assembly bias. An instance of this class will 
            be built by passing on the full set of keyword arguments that 
            were passed to `HeavisideAssembiasComponent`. 

        method_name_to_decorate : string 
            Name of the method bound instances of ``baseline_model`` that 
            we are decorating with assembly bias.

        lower_bound : float 
            Smallest physically meaningful value for the property being modeled. 

            For example, if modeling the first occupation moment of any galaxy type, 
            ``lower_bound`` = 0; the same value would be used if modeling 
            the quenched fraction. 

            Can be set to -float("inf") provided that 
            the ``upper_bound`` keyword argument is set to a bounded value. 

        upper_bound : float 
            Largest physically meaningful value for the property being modeled. 

            For example, if modeling the first occupation moment of any central galaxies, 
            ``upper_bound`` = 1, whereas for satellites we would set ``upper_bound`` = float("inf"). 

            Can be set to float("inf") provided that 
            the ``lower_bound`` keyword argument is set to a bounded value. 

        split : float, optional 
            Fraction between 0 and 1 defining how we split halos into two groupings based on 
            their conditional secondary percentiles. Default is 0.5

        assembias_strength : float, optional 
            Fraction between -1 and 1 defining the assembly bias correlation strength. 
            Default is 0.5. 

        sec_haloprop_key : string, optional 
            String giving the column name of the secondary halo property 
            governing the assembly bias. Must be a key in the halo_table 
            passed to the methods of `HeavisideAssembiasComponent`. 
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        loginterp : bool, optional
            If set to True, the interpolation will be done 
            in the logarithm of the primary halo property, 
            rather than linearly. Default is True. 

        """
        try:
            baseline_model_instance = kwargs['baseline_model'](**kwargs)
            self.baseline_model_instance = baseline_model_instance
            self._method_name_to_decorate = kwargs['method_name_to_decorate']
            self._lower_bound = kwargs['lower_bound']
            self._upper_bound = kwargs['upper_bound']
        except KeyError:
            msg = ("The constructor to the HeavisideAssembiasComponent class "
                "must be called with the following keyword arguments:\n" 
                "``%s``, ``%s``, ``%s``, ``%s``")
            raise HalotoolsError(msg % ('lower_bound', 'upper_bound', 
                'method_name_to_decorate', 'baseline_model_instance'))

        self._loginterp = loginterp
        self.sec_haloprop_key = sec_haloprop_key
        self.prim_haloprop_key = self.baseline_model_instance.prim_haloprop_key

        if 'split_abcissa' and 'split_ordinates' in kwargs:
            self._setup_percentile_splitting(split_abcissa=kwargs['split_abcissa'], 
                split_ordinates=kwargs['split_ordinates'])
        else:
            self._setup_percentile_splitting(split = split)


        if 'assembias_strength_abcissa' and 'assembias_strength_ordinates' in kwargs:
            self._initialize_param_dict(split_abcissa=kwargs['assembias_strength_abcissa'], 
                split_ordinates=kwargs['assembias_strength_abcissa'])
        else:
            self._initialize_param_dict(assembias_strength=assembias_strength)

        
    def __getattr__(self, attr):
        return getattr(self.baseline_model_instance, attr)

    def _setup_percentile_splitting(self, **kwargs):
        """
        """
        if 'split' in kwargs.keys():
            self._split_abcissa = [0]
            self._split_ordinates = [kwargs['split']]
        elif ('split_ordinates' in kwargs.keys()) & ('split_abcissa' in kwargs.keys()):
            self._split_abcissa = kwargs['split_abcissa']
            self._split_ordinates = kwargs['split_ordinates']
        else:
            msg = ("The constructor to the HeavisideAssembiasComponent class "
                "must be called with either the ``split`` keyword argument,\n"
                " or both the ``split_abcissa`` and ``split_ordinates`` keyword arguments" )
            raise HalotoolsError(msg)

    def percentile_splitting_function(self, halo_table):
        """
        """
        try:
            prim_haloprop = halo_table[self.prim_haloprop_key]
        except KeyError:
            raise HalotoolsError("prim_haloprop_key = %s is not a column of the input halo_table" % self.prim_haloprop_key)

        model_ordinates = [self.param_dict[self._get_param_dict_key(ipar)] for ipar in range(len(model_abcissa))]

        if self._loginterp is True:
            spline_function = model_helpers.custom_spline(
                np.log10(self._abcissa), model_ordinates)
            return spline_function(np.log10(prim_haloprop))
        else:
            model_abcissa = self._abcissa
            spline_function = model_helpers.custom_spline(
                self._abcissa, model_ordinates)
            return spline_function(prim_haloprop)

    def _initialize_param_dict(self, **kwargs):
        """
        """
        if hasattr(self.baseline_model_instance, 'param_dict'):
            self.param_dict = self.baseline_model_instance.param_dict
        else:
            self.param_dict = {}

        if 'assembias_strength' in kwargs.keys():
            self._assembias_strength_abcissa = [1]
            self.param_dict[self._get_param_dict_key(0)] = kwargs['assembias_strength']
        elif 'assembias_strength_ordinates' and 'assembias_strength_abcissa' in kwargs:
            self._assembias_strength_abcissa = kwargs['assembias_strength_abcissa']
            for ipar, val in enumerate(kwargs['assembias_strength_ordinates']):
                self.param_dict[self._get_param_dict_key(ipar)] = val
        else:
            msg = ("The constructor to the HeavisideAssembiasComponent class "
                "must be called with either the ``assembias_strength`` keyword argument,\n"
                " or both the ``assembias_strength_abcissa`` and ``assembias_strength_ordinates`` keyword arguments" )
            raise HalotoolsError(msg)


    def _get_param_dict_key(self, ipar):
        """
        """
        return self._method_name_to_decorate + '_assembias_param' + str(ipar+1)

    def bound_decfunc(self, bound):

        def decorator(func):
            def wrapper(*args, **kwargs):
                return bound - func(*args, **kwargs)
            return wrapper

        return decorator

    def complementary_bound_decfunc(self, split_func):

        def decorator(func):
            def wrapper(*args, **kwargs):
                prefactor = (1 - split_func(*args, **kwargs))/split_func(*args, **kwargs)
                return prefactor*func(*args, **kwargs)
            return wrapper

        return decorator

    def lower_bound(self, *args, **kwargs):

        lower_bound1_func = self.bound_decfunc(self.lower_bound)(self.main_func)
        lower_bound2_func = self.complementary_bound_decfunc(self.split_func)(upper_bound1_func)

        return np.max(lower_bound1_func(*args, **kwargs), lower_bound2_func(*args, **kwargs))

    def upper_bound(self, *args, **kwargs):

        upper_bound1_func = self.bound_decfunc(self.upper)(self.main_func)
        upper_bound2_func = self.complementary_bound_decfunc(self.split_func)(lower_bound1_func)

        return np.min(upper_bound1_func(*args, **kwargs), upper_bound2_func(*args, **kwargs))

    def dx1(self, *args, **kwargs):

        result = np.zeros(len(args(0)))

        strength = self.strength(*args, **kwargs)
        positive_strength_idx = strength > 0
        negative_strength_idx = np.invert(positive_strength)

        result[positive_strength_idx] = strength[positive_strength_idx]*upper_bound(*args, **kwargs)
        result[negative_strength_idx] = strength[negative_strength_idx]*lower_bound(*args, **kwargs)

        return result

    def dx2(self, *args, **kwargs):
        split = self.split_func(*args, **kwargs)
        dx1 = self.dx1(*args, **kwargs)
        return -split*dx1/(1-split)

    def new_main(self, *args, **kwargs):

        split = self.split_func(*args, **kwargs)
        result = self.main_func(*args, **kwargs)

        no_edge_mask = (split > 0) & (split < 1) & (result > lower_bound) & (result < upper_bound)
        no_edge_result = result[no_edge_mask]
        no_edge_halos = halos[no_edge_mask]

        case1_mask = no_edge_halos['case'] == 1
        dx1 = self.dx1(no_edge_halos[case1_mask])
        no_edge_result[case1_mask] += dx1
        dx2 = self.dx1(no_edge_halos[np.invert(case1_mask)])
        no_edge_result[np.invert(case1_mask)] += dx2

        result[no_edge_mask] = no_edge_result

        return result







