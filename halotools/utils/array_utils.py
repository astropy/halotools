# -*- coding: utf-8 -*-
"""

Modules performing small, commonly used tasks throughout the package.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['array_like_length', 'find_idx_nearest_val', 'randomly_downsample_data']

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

    Examples 
    --------
    >>> x = np.zeros(5)
    >>> xlen = array_like_length(x)
    >>> y = 4
    >>> ylen = array_like_length(y)
    >>> z = None 
    >>> zlen = array_like_length(z)
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

    Examples 
    --------
    >>> x = np.linspace(0, 1000, num=1e5)
    >>> val = 45.5
    >>> idx_nearest_val = find_idx_nearest_val(x, val)
    >>> nearest_val = x[idx_nearest_val]
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


def randomly_downsample_data(array, num_downsample):
    """ Method returns a length-num_downsample random downsampling of the input array.

    Parameters 
    ----------
    array : array

    num_downsample : int 
        Size of the desired downsampled version of the data

    Returns 
    -------
    downsampled_array : array or Astropy Table
        Random downsampling of the input array

    Examples 
    --------
    >>> x = np.linspace(0, 1000, num=1e5)
    >>> desired_sample_size = 1e3
    >>> downsampled_x = randomly_downsample_data(x, desired_sample_size)
    """

    input_array_length = array_like_length(array) 
    if num_downsample > input_array_length:
        raise SyntaxError("Length of the desired downsampling = %i, "
            "which exceeds input array length = %i " % (num_downsample, input_array_length))
    else:
        randomizer = np.random.random(input_array_length)
        idx_sorted = np.argsort(randomizer)
        return array[idx_sorted[0:num_downsample]]








