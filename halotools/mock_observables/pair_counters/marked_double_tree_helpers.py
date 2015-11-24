# -*- coding: utf-8 -*-

"""
rectangular Cuboid Pair Counter. 
This module contains pair counting functions used to count the number of pairs with 
separations less than or equal to r, optimized for simulation boxes.
This module also contains a 'main' function which runs speed tests.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from time import time
from warnings import warn 
from copy import copy 
import sys
import multiprocessing
from functools import partial

from ...custom_exceptions import *
from ...utils.array_utils import convert_to_ndarray, array_is_monotonic

__all__ = (
    ['_wnpairs_process_weights']
    )

def _wnpairs_process_weights(data1, data2, weights1, weights2, wfunc):
    """
    """

    if type(wfunc) != int:
        raise HalotoolsError("wfunc parameter must be an integer ID of a weighting function")

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
            msg = ("You must either pass in a 1-d or 2-d array for the input ``weights1``\n"
                "The ``_wnpairs_process_weights`` function received a ``weights1`` array of dimension %i")
            raise HalotoolsError(msg % ndim1)

    npts_weights1 = np.shape(weights1)[0]
    num_weights1 = np.shape(weights1)[1]
    # At this point, weights1 is guaranteed to be a 2-d ndarray
    ### now we check its shape
    if np.shape(weights1) != correct_shape1:
        if _converted_to_2d_from_1d is True:
            msg = ("\nYou passed in a 1-d array for ``weights1`` that "
                "does not have the correct length.\n"
                "The number of points in ``data1`` = %i,\n"
                "while the number of points in your input 1-d ``weights1`` array = %i")
            raise HalotoolsError(msg % (npts_data1, npts_weights1))
        else:
            msg = ("\nYou passed in a 2-d array for ``weights1`` that \n"
                "does not have the consistent shape with the %i number of points in ``data1`` \n"
                "and the input value of ``wfunc`` = %i.\n"
                "For this value of ``wfunc``, there should be %i weights per point.\n"
                "The shape of your input ``weights1`` is (%i, %i)\n")
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
            msg = ("You must either pass in a 1-d or 2-d array for the input ``weights2``\n"
                "The ``_wnpairs_process_weights`` function received a ``weights2`` array of dimension %i")
            raise HalotoolsError(msg % ndim2)

    npts_weights2 = np.shape(weights2)[0]
    num_weights2 = np.shape(weights2)[1]
    # At this point, weights2 is guaranteed to be a 2-d ndarray
    ### now we check its shape
    if np.shape(weights2) != correct_shape2:
        if _converted_to_2d_from_1d is True:
            msg = ("\nYou passed in a 1-d array for ``weights2`` that "
                "does not have the correct length.\n"
                "The number of points in ``data2`` = %i,\n"
                "while the number of points in your input 1-d ``weights2`` array = %i")
            raise HalotoolsError(msg % (npts_data2, npts_weights2))
        else:
            msg = ("\nYou passed in a 2-d array for ``weights2`` that \n"
                "does not have the consistent shape with the %i number of points in ``data2`` \n"
                "and the input value of ``wfunc`` = %i.\n"
                "For this value of ``wfunc``, there should be %i weights per point.\n"
                "The shape of your input ``weights2`` is (%i, %i)\n")
            raise HalotoolsError(msg % 
                (npts_data2, wfunc, correct_num_weights, npts_weights2, num_weights2))

    return weights1, weights2


def _func_signature_int_from_wfunc(wfunc):
    """
    """
    if type(wfunc) != int:
        raise HalotoolsError("wfunc parameter must be an integer ID of a weighting function")

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
        return 6
    elif wfunc == 11:
        return 3
    elif wfunc == 12:
        return 3
    elif wfunc == 13:
        return 2
    else:
        msg = ("The value ``wfunc`` = %i is not recognized")
        raise HalotoolsError(msg % wfunc)
        

