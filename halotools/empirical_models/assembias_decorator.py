# -*- coding: utf-8 -*-
"""
Decorator class for implementing generalized assembly bias
"""

import numpy as np 
from . import hod_components 
from . import model_defaults 

from ..halotools_exceptions import HalotoolsError

class HeavisideAssembiasDecorator(object):
    """
    """
    def __init__(self, lower_bound, upper_bound, split_func, main_func):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.split_func = split_func
        self.main_func = main_func

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







