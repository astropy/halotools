from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from .array_utils import unsorting_indices
from .engines import cython_conditional_rank_kernel

__all__ = ('sliding_conditional_percentile', )


def sliding_conditional_percentile(x, y, window_length):
    """
    Examples
    --------
    >>> x = np.random.rand(100)
    >>> y = np.random.rand(100)
    >>> window_length = 5
    >>> result = sliding_conditional_percentile(x, y, window_length)
    """
    rank_orders = cython_sliding_rank(x, y, window_length)
    rank_order_percentiles = (1. + rank_orders)/float(window_length+1)
    return rank_order_percentiles


def rank_order_function(x):
    """ Calculate the rank-order of each element in an input array.

    Parameters
    ----------
    x : ndarray
        Array of shape (npts, )

    Results
    -------
    rank_orders : ndarray
        Integer array of shape (npts, ) storing values between 0 and npts-1
    """
    x = np.atleast_1d(x)
    assert x.ndim == 1, "x must be a 1-d sequence"
    assert len(x) > 1, "x must have more than one element"

    return unsorting_indices(np.argsort(x))


def cython_sliding_rank(x, y, window_length):
    """
    Examples
    --------
    >>> x = np.random.rand(100)
    >>> y = np.random.rand(100)
    >>> window_length = 5
    >>> result = cython_sliding_rank(x, y, window_length)
    """
    x, y, nwin = _check_xyn_bounds(x, y, window_length)
    nhalfwin = int(nwin/2)

    indx_x_sorted = np.argsort(x)
    indx_x_unsorted = unsorting_indices(indx_x_sorted)
    y_sorted = y[indx_x_sorted]

    result = np.array(cython_conditional_rank_kernel(y_sorted, nwin))

    leftmost_window_ranks = rank_order_function(y_sorted[:nwin])
    result[:nhalfwin+1] = leftmost_window_ranks[:nhalfwin+1]

    rightmost_window_ranks = rank_order_function(y_sorted[-nwin:])
    result[-nhalfwin-1:] = rightmost_window_ranks[-nhalfwin-1:]

    return result[indx_x_unsorted].astype(int)


def _check_xyn_bounds(x, y, n):
    """ Enforce bounds checks on the inputs and return 1-d Numpy arrays
    """
    x = np.atleast_1d(x).astype('f8')
    assert x.ndim == 1, "x must be a 1-d array"
    y = np.atleast_1d(y).astype('f8')
    assert y.ndim == 1, "y must be a 1-d array"

    assert len(x) == len(y), "x and y must have the same length"

    msg = "Window length = {0} must be an odd integer"
    try:
        assert n % 2 == 1, msg.format(n)
    except AssertionError:
        raise ValueError(msg.format(n))

    msg2 = "Window length = {0} must satisfy 1 < n < len(x)"
    assert 1 < n < len(x), msg2.format(n)

    return x, y, int(n)
