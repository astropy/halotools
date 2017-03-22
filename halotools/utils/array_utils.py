"""
Modules performing small, commonly used tasks throughout the package.
"""

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from ..custom_exceptions import HalotoolsError

__all__ = ('custom_len', 'find_idx_nearest_val', 'randomly_downsample_data',
           'array_is_monotonic', 'unsorting_indices')


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
    >>> x = np.linspace(0, 1000, num=int(1e5))
    >>> val = 45.5
    >>> idx_nearest_val = find_idx_nearest_val(x, val)
    >>> nearest_val = x[idx_nearest_val]

    Notes
    ------
    Use of this function is deprecated. The `numpy.argmin` function provides
    equivalent behavior.
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


def randomly_downsample_data(array, num_downsample, seed=None):
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
    num_downsample = int(num_downsample)

    input_array_length = custom_len(array)
    if num_downsample > input_array_length:
        raise SyntaxError("Length of the desired downsampling = %i, "
            "which exceeds input array length = %i " % (num_downsample, input_array_length))
    else:
        with NumpyRNGContext(seed):
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


def unsorting_indices(sorting_indices):
    """ Return the indexing array that inverts `numpy.argsort`.

    Parameters
    ----------
    sorting_indices : array_like
        Length-Npts array - the output of `numpy.argsort`

    Returns
    -------
    unsorting_indices : array_like
        Length-Npts array

    Examples
    --------
    >>> x = np.random.rand(100)
    >>> idx_sorted = np.argsort(x)
    >>> x_sorted = x[idx_sorted]
    >>> idx_unsorted = unsorting_indices(idx_sorted)
    >>> assert np.all(x == x_sorted[idx_unsorted])

    """
    npts = len(sorting_indices)
    unsorting_indices = np.zeros_like(sorting_indices, dtype=int)
    unsorting_indices[sorting_indices] = np.arange(npts).astype(int)
    return unsorting_indices
