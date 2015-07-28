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
        loginterp = True, 
        **kwargs):
        """
        """
        try:
            self.baseline_model = kwargs['baseline_model']
            self._method_name_to_decorate = kwargs['method_name_to_decorate']
            self._lower_bound = kwargs['lower_bound']
            self._upper_bound = kwargs['upper_bound']
        except KeyError:
            msg = ("The constructor to the HeavisideAssembiasComponent class "
                "must be called with the following keyword arguments:\n" 
                "``%s``, ``%s``, ``%s``, ``%s``")
            raise HalotoolsError(msg % ('lower_bound', 'upper_bound', 
                'method_name_to_decorate', 'baseline_model'))

        self._loginterp = loginterp
        self.sec_haloprop_key = sec_haloprop_key
        self.prim_haloprop_key = self.baseline_model.prim_haloprop_key

        self._initialize_param_dict(**kwargs)
        self._setup_percentile_splitting(**kwargs)

    def __getattr__(self, attr):
        return getattr(self.baseline_model, attr)

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
        if hasattr(self.baseline_model, 'param_dict'):
            self.param_dict = self.baseline_model.param_dict
        else:
            self.param_dict = {}

        if 'assembias_strength' in kwargs.keys():
            self._assembias_strength_abcissa = [0]
            self.param_dict[self._get_param_dict_key(0)] = kwargs['assembias_strength']
        elif ('assembias_strength_ordinates' in kwargs.keys()) & ('assembias_strength_abcissa' in kwargs.keys()):
            self._assembias_strength_abcissa = kwargs['assembias_strength_abcissa']
            for ipar, val in enumerate(kwargs['assembias_strength_ordinates']):
                self.param_dict[self._get_param_dict_key(ipar)] = val
        else:
            msg = ("The constructor to the HeavisideAssembiasComponent class "
                "must be called with either the ``split`` keyword argument,\n"
                " or both the ``split_abcissa`` and ``split_ordinates`` keyword arguments" )
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







