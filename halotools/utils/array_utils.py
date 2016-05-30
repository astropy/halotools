# -*- coding: utf-8 -*-
"""

Modules performing small, commonly used tasks throughout the package.

"""

import numpy as np
from astropy.table import Table

from ..custom_exceptions import HalotoolsError

__all__ = (['custom_len', 'find_idx_nearest_val', 'randomly_downsample_data',
            'array_is_monotonic', 'convert_to_ndarray'])


def convert_to_ndarray(x, dt=None):
    """ Method checks to see in the input array x is an ndarray
    or an Astropy Table. If not, returns an array version of x.

    Parameters
    -----------
    x : array_like
        Any sequence or scalar.

    dt : numpy dtype, optional
        np.dtype of the returned array.
        Default is to return the same dtype as the input ``x``

    Returns
    -------
    y : array
        Numpy ndarray

    Examples
    --------
    >>> x, y, z  = 0, [0], None
    >>> xarr, yarr, zarr = convert_to_ndarray(x), convert_to_ndarray(y), convert_to_ndarray(z)
    >>> assert len(xarr) == len(yarr) == len(zarr) == 1

    >>> t, u, v = np.array(1), np.array([1]), np.array('abc')
    >>> tarr, uarr, varr = convert_to_ndarray(t), convert_to_ndarray(u), convert_to_ndarray(v)
    >>> assert len(tarr) == len(uarr) == len(varr) == 1

    """
    if dt is not None:
        if type(dt) is not type(np.dtype):
            raise HalotoolsError("The input dt must be a numpy dtype object")

    if type(x) is np.ndarray:
        try:
            iterator = iter(x)
            if len(x) == 0:
                return np.array([], dtype=dt)
            if dt is None:
                return x.astype(type(x.flatten()[0]))
            else:
                return x.astype(dt)
        except TypeError:
            try:
                x = x.reshape(1)
            except ValueError:
                return np.array([], dtype=dt)
            if dt is None:
                return x.astype(type(x.flatten()[0]))
            else:
                return x.astype(dt)
    elif type(x) is Table:
        return x
    elif (type(x) is str) or (type(x) is str):
        x = np.array([x])
        if dt is None:
            return x.astype(type(x.flatten()[0]))
        else:
            return x.astype(dt)
    else:
        try:
            l = len(x)
            x = np.array(x)
            if len(x) == 0:
                return np.array([])
            if dt is None:
                return x.astype(type(x.flatten()[0]))
            else:
                return x.astype(dt)
        except TypeError:
            x = np.array([x])
            x = x.flatten()
            if dt is None:
                if len(x) == 0:
                    return np.array([])
                else:
                    return x.astype(type(x.flatten()[0]))
            else:
                if len(x) == 0:
                    return np.array([], dtype=dt)
                else:
                    return x.astype(dt)


def custom_len(x):
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
    >>> x, y, z  = 0, [0], None
    >>> xlen, ylen, zlen = custom_len(x), custom_len(y), custom_len(z)
    """

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
    if custom_len(array) == 0:
        msg = "find_idx_nearest_val method was passed an empty array"
        raise HalotoolsError(msg)

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
    >>> x = np.linspace(0, 1000, num=int(1e5))
    >>> desired_sample_size = int(1e3)
    >>> downsampled_x = randomly_downsample_data(x, desired_sample_size)

    """

    input_array_length = custom_len(array)
    if num_downsample > input_array_length:
        raise SyntaxError("Length of the desired downsampling = %i, "
            "which exceeds input array length = %i " % (num_downsample, input_array_length))
    else:
        randomizer = np.random.random(input_array_length)
        idx_sorted = np.argsort(randomizer)
        return array[idx_sorted[0:num_downsample]]


def array_is_monotonic(array, strict=False):
    """
    Method determines whether an input array is monotonic.

    Parameters
    -----------
    array : array_like

    strict : bool, optional
        If set to True, an array must be strictly monotonic to pass the criteria.
        Default is False.

    Returns
    -------
    flag : int
        If input ``array`` is monotonically increasing, the returned flag = 1;
        if ``array`` is monotonically decreasing, flag = -1. Otherwise, flag = 0.

    Examples
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> assert array_is_monotonic(x) == 1
    >>> assert array_is_monotonic(x[::-1]) == -1

    >>> y = np.ones(100)
    >>> assert array_is_monotonic(y) == 1
    >>> assert array_is_monotonic(y, strict=True) == 0

    Notes
    -----
    If the input ``array`` is constant-valued, method returns flag = 1.

    """
    if custom_len(array) < 3:
        msg = ("Input array to the array_is_monotonic method has less then 3 elements")
        raise HalotoolsError(msg)
    d = np.diff(array)

    if strict is True:
        if np.all(d > 0):
            return 1
        elif np.all(d < 0):
            return -1
        else:
            return 0
    else:
        if np.all(d >= 0):
            return 1
        elif np.all(d <= 0):
            return -1
        else:
            return 0
