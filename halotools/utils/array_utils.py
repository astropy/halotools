# -*- coding: utf-8 -*-
"""

Modules performing small, commonly used tasks throughout the package.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['array_like_length', 'find_idx_nearest_val']

import numpy as np

import collections

def array_like_length(x):
    """ Simple method to return a zero-valued 1-D numpy array 
    with the length of the input x. 

    Parameters 
    ----------
    x : array_like
        Can be an iterable such as a list or non-iterable such as a float. 

    Returns 
    -------
    array_length : int 
        length of x

    Notes 
    ----- 
    Simple workaround of an awkward feature of numpy. When evaluating 
    the built-in len() function on non-iterables such as a 
    float or int, len() returns a TypeError, rather than unity. 
    Most annoyingly, the same is true on an object such as x=numpy.array(4), 
    even though such an object formally counts as an Iterable, being an ndarray. 
    This nuisance is so ubiquitous that it's convenient to have a single 
    line of code that replaces the default python len() function with sensible behavior.
    """

    if x is None:
        return 0
    try:
        array_length = len(x)
    except TypeError:
        array_length = 1

    return array_length

def find_idx_nearest_val(array, value):
    """ Method returns the index where the input array is closest 
    to the input value. 
    
    Parameters 
    ----------
    array : array_like 
    
    value : float or int
    
    Returns 
    -------
    idx_nearest : int
    """
    if len(array) == 0:
        return None

    idx_sorted = np.argsort(array)
    sorted_array = np.array(array[idx_sorted])
    idx = np.searchsorted(sorted_array, value, side="left")
    if idx >= len(array):
        idx_nearest = idx_sorted[len(array)-1]
        return idx_nearest
    elif idx == 0:
        idx_nearest = idx_sorted[0]
        return idx_nearest
    else:
        if abs(value - sorted_array[idx-1]) < abs(value - sorted_array[idx]):
            idx_nearest = idx_sorted[idx-1]
            return idx_nearest
        else:
            idx_nearest = idx_sorted[idx]
            return idx_nearest

