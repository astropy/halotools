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

def _wnpairs_process_weights(data1, data2, weights1, weights2, aux1, aux2, wfunc):
    """
    """

    #Process weights1 entry and check for consistency.
    if weights1 is None:
        weights1 = np.array([1.0]*np.shape(data1)[0], dtype=np.float64)
    else:
        weights1 = np.asarray(weights1).astype("float64")
        if np.shape(weights1)[0] != np.shape(data1)[0]:
            raise HalotoolsError("weights1 should have same len as data1")
    #Process weights2 entry and check for consistency.
    if weights2 is None:
        weights2 = np.array([1.0]*np.shape(data2)[0], dtype=np.float64)
    else:
        weights2 = np.asarray(weights2).astype("float64")
        if np.shape(weights2)[0] != np.shape(data2)[0]:
            raise HalotoolsError("weights2 should have same len as data2")
    
    #Process weights1 entry and check for consistency.
    if aux1 is None:
        aux1 = np.array([1.0]*np.shape(data1)[0], dtype=np.float64)
    else:
        aux1 = np.asarray(aux1).astype("float64")
        if np.shape(aux1)[0] != np.shape(data1)[0]:
            raise HalotoolsError("aux1 should have same len as data1")
    #Process weights2 entry and check for consistency.
    if aux2 is None:
        aux2 = np.array([1.0]*np.shape(data2)[0], dtype=np.float64)
    else:
        aux2 = np.asarray(aux2).astype("float64")
        if np.shape(aux2)[0] != np.shape(data2)[0]:
            raise HalotoolsError("aux2 should have same len as data2")
    
    if type(wfunc) != int:
        raise HalotoolsError("wfunc parameter must be an integer ID of a weighting function")

    return weights1, weights2, aux1, aux2


