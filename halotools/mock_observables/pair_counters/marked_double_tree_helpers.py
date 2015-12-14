# -*- coding: utf-8 -*-

"""
This module contains private helper functions used in 
`~halotools.mock_observables.pair_counters.marked_double_tree_pairs` module to 
perform control flow on function arguments, bounds-checking and exception-handling. 
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from warnings import warn 
from copy import copy 
from ...custom_exceptions import *
from ...utils.array_utils import convert_to_ndarray, array_is_monotonic

__all__ = ['_marked_npairs_process_weights','_velocity_marked_npairs_process_weights']

def _marked_npairs_process_weights(data1, data2, weights1, weights2, wfunc):
    """
    process weights and associated arguments for
    `~halotools.mock_observables.pair_counters.marked_double_tree_pairs.marked_npairs`
    """
    
    correct_num_weights = _func_signature_int_from_wfunc(wfunc)
    npts_data1 = np.shape(data1)[0]
    npts_data2 = np.shape(data2)[0]
    correct_shape1 = (npts_data1, correct_num_weights)
    correct_shape2 = (npts_data2, correct_num_weights)
    
    ### Process the input weights1
    _converted_to_2d_from_1d = False
    # First convert weights1 into a 2-d ndarray
    if weights1 is None:
        weights1 = np.ones((npts_data1, 1), dtype = np.float64)
    else:
        weights1 = convert_to_ndarray(weights1)
        weights1 = weights1.astype("float64")
        if weights1.ndim == 1:
            _converted_to_2d_from_1d = True
            npts1 = len(weights1)
            weights1 = weights1.reshape((npts1, 1))
        elif weights1.ndim == 2:
            pass
        else:
            ndim1 = weights1.ndim
            msg = ("\n You must either pass in a 1-D or 2-D array \n"
                   "for the input `weights1`. Instead, an array of \n"
                   "dimension %i was received.")
            raise HalotoolsError(msg % ndim1)
    
    npts_weights1 = np.shape(weights1)[0]
    num_weights1 = np.shape(weights1)[1]
    # At this point, weights1 is guaranteed to be a 2-d ndarray
    ### now we check its shape
    if np.shape(weights1) != correct_shape1:
        if _converted_to_2d_from_1d is True:
            msg = ("\n You passed in a 1-D array for `weights1` that \n"
                   "does not have the correct length. The number of \n"
                   "points in `data1` = %i, while the number of points \n"
                   "in your input 1-D `weights1` array = %i")
            raise HalotoolsError(msg % (npts_data1, npts_weights1))
        else:
            msg = ("\n You passed in a 2-D array for `weights1` that \n"
                   "does not have a consistent shape with `data1`. \n"
                   "`data1` has length %i. The input value of `wfunc` = %i \n"
                   "For this value of `wfunc`, there should be %i weights \n"
                   "per point. The shape of your input `weights1` is (%i, %i)\n")
            raise HalotoolsError(msg % 
                (npts_data1, wfunc, correct_num_weights, npts_weights1, num_weights1))
    
    ### Process the input weights2
    _converted_to_2d_from_1d = False
    # Now convert weights2 into a 2-d ndarray
    if weights2 is None:
        weights2 = np.ones((npts_data2, 1), dtype = np.float64)
    else:
        weights2 = convert_to_ndarray(weights2)
        weights2 = weights2.astype("float64")
        if weights2.ndim == 1:
            _converted_to_2d_from_1d = True
            npts2 = len(weights2)
            weights2 = weights2.reshape((npts2, 1))
        elif weights2.ndim == 2:
            pass
        else:
            ndim2 = weights2.ndim
            msg = ("\n You must either pass in a 1-D or 2-D array \n"
                   "for the input `weights2`. Instead, an array of \n"
                   "dimension %i was received.")
            raise HalotoolsError(msg % ndim2)
    
    npts_weights2 = np.shape(weights2)[0]
    num_weights2 = np.shape(weights2)[1]
    # At this point, weights2 is guaranteed to be a 2-d ndarray
    ### now we check its shape
    if np.shape(weights2) != correct_shape2:
        if _converted_to_2d_from_1d is True:
            msg = ("\n You passed in a 1-D array for `weights2` that \n"
                   "does not have the correct length. The number of \n"
                   "points in `data2` = %i, while the number of points \n"
                   "in your input 1-D `weights2` array = %i")
            raise HalotoolsError(msg % (npts_data2, npts_weights2))
        else:
            msg = ("\n You passed in a 2-D array for `weights2` that \n"
                   "does not have a consistent shape with `data2`. \n"
                   "`data2` has length %i. The input value of `wfunc` = %i \n"
                   "For this value of `wfunc`, there should be %i weights \n"
                   "per point. The shape of your input `weights2` is (%i, %i)\n")
            raise HalotoolsError(msg % 
                (npts_data2, wfunc, correct_num_weights, npts_weights2, num_weights2))
    
    return weights1, weights2


def _velocity_marked_npairs_process_weights(data1, data2, weights1, weights2, wfunc):
    """
    process weights and associated arguments for
    `~halotools.mock_observables.pair_counters.marked_double_tree_pairs.velocity_marked_npairs`
    """
    
    correct_num_weights = _func_signature_int_from_vel_wfunc(wfunc)
    npts_data1 = np.shape(data1)[0]
    npts_data2 = np.shape(data2)[0]
    correct_shape1 = (npts_data1, correct_num_weights)
    correct_shape2 = (npts_data2, correct_num_weights)
    
    ### Process the input weights1
    _converted_to_2d_from_1d = False
    # First convert weights1 into a 2-d ndarray
    if weights1 is None:
        weights1 = np.ones((npts_data1, 1), dtype = np.float64)
    else:
        weights1 = convert_to_ndarray(weights1)
        weights1 = weights1.astype("float64")
        if weights1.ndim == 1:
            _converted_to_2d_from_1d = True
            npts1 = len(weights1)
            weights1 = weights1.reshape((npts1, 1))
        elif weights1.ndim == 2:
            pass
        else:
            ndim1 = weights1.ndim
            msg = ("\n You must either pass in a 1-D or 2-D array \n"
                   "for the input `weights1`. Instead, an array of \n"
                   "dimension %i was received.")
            raise HalotoolsError(msg % ndim1)
    
    npts_weights1 = np.shape(weights1)[0]
    num_weights1 = np.shape(weights1)[1]
    # At this point, weights1 is guaranteed to be a 2-d ndarray
    ### now we check its shape
    if np.shape(weights1) != correct_shape1:
        if _converted_to_2d_from_1d is True:
            msg = ("\n You passed in a 1-D array for `weights1` that \n"
                   "does not have the correct length. The number of \n"
                   "points in `data1` = %i, while the number of points \n"
                   "in your input 1-D `weights1` array = %i")
            raise HalotoolsError(msg % (npts_data1, npts_weights1))
        else:
            msg = ("\n You passed in a 2-D array for `weights1` that \n"
                   "does not have a consistent shape with `data1`. \n"
                   "`data1` has length %i. The input value of `wfunc` = %i \n"
                   "For this value of `wfunc`, there should be %i weights \n"
                   "per point. The shape of your input `weights1` is (%i, %i)\n")
            raise HalotoolsError(msg % 
                (npts_data1, wfunc, correct_num_weights, npts_weights1, num_weights1))
    
    ### Process the input weights2
    _converted_to_2d_from_1d = False
    # Now convert weights2 into a 2-d ndarray
    if weights2 is None:
        weights2 = np.ones((npts_data2, 1), dtype = np.float64)
    else:
        weights2 = convert_to_ndarray(weights2)
        weights2 = weights2.astype("float64")
        if weights2.ndim == 1:
            _converted_to_2d_from_1d = True
            npts2 = len(weights2)
            weights2 = weights2.reshape((npts2, 1))
        elif weights2.ndim == 2:
            pass
        else:
            ndim2 = weights2.ndim
            msg = ("\n You must either pass in a 1-D or 2-D array \n"
                   "for the input `weights2`. Instead, an array of \n"
                   "dimension %i was received.")
            raise HalotoolsError(msg % ndim2)
    
    npts_weights2 = np.shape(weights2)[0]
    num_weights2 = np.shape(weights2)[1]
    # At this point, weights2 is guaranteed to be a 2-d ndarray
    ### now we check its shape
    if np.shape(weights2) != correct_shape2:
        if _converted_to_2d_from_1d is True:
            msg = ("\n You passed in a 1-D array for `weights2` that \n"
                   "does not have the correct length. The number of \n"
                   "points in `data2` = %i, while the number of points \n"
                   "in your input 1-D `weights2` array = %i")
            raise HalotoolsError(msg % (npts_data2, npts_weights2))
        else:
            msg = ("\n You passed in a 2-D array for `weights2` that \n"
                   "does not have a consistent shape with `data2`. \n"
                   "`data2` has length %i. The input value of `wfunc` = %i \n"
                   "For this value of `wfunc`, there should be %i weights \n"
                   "per point. The shape of your input `weights2` is (%i, %i)\n")
            raise HalotoolsError(msg % 
                (npts_data2, wfunc, correct_num_weights, npts_weights2, num_weights2))
    
    return weights1, weights2


def _func_signature_int_from_wfunc(wfunc):
    """
    return the function signiture available weighting functions
    """
    
    if type(wfunc) != int:
        msg = "\n wfunc parameter must be an integer ID of a weighting function."
        raise HalotoolsError(msg)
    
    if wfunc == 1:
        return 1
    elif wfunc == 2:
        return 1
    elif wfunc == 3:
        return 2
    elif wfunc == 4:
        return 2
    elif wfunc == 5:
        return 2
    elif wfunc == 6:
        return 2
    elif wfunc == 7:
        return 2
    elif wfunc == 8:
        return 2
    elif wfunc == 9:
        return 2
    elif wfunc == 10:
        return 2
    else:
        msg = ("The value ``wfunc`` = %i is not recognized")
        raise HalotoolsError(msg % wfunc)


def _func_signature_int_from_vel_wfunc(wfunc):
    """
    return the function signiture available velocity weighting functions.
    """
    if type(wfunc) != int:
        msg = "\n wfunc parameter must be an integer ID of a weighting function."
        raise HalotoolsError(msg)
    
    elif wfunc == 11:
        return 6
    elif wfunc == 12:
        return 6
    elif wfunc == 13:
        return 7
    elif wfunc == 14:
        return 1
    elif wfunc == 15:
        return 1
    elif wfunc == 16:
        return 2
    else:
        msg = ("The value ``wfunc`` = %i is not recognized")
        raise HalotoolsError(msg % wfunc)

