"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from .array_utils import unsorting_indices
from .engines import cython_conditional_rank_kernel

__all__ = ('sliding_conditional_percentile', )


def sliding_conditional_percentile(x, y, window_length, assume_x_is_sorted=False,
            add_subgrid_noise=True, seed=None):
    r""" Estimate the cumulative distribution function Prob(< y | x).

    Parameters
    ----------
    x : ndarray
        Array of shape (npts, )

    y : ndarray
        Array of shape (npts, )

    window_length : int
        Integer must be odd and less than ``npts``

    assume_x_is_sorted : bool, optional
        Performance enhancement flag that can be used for cases where input `x`
        has already been sorted. Default is False.

    add_subgrid_noise : bool, optional
        Flag determines whether random uniform noise will be added to fill in
        the gaps at the sub-grid level determined by `window_length`. Default is True.

    seed : int, optional
        Random number seed used together with the `add_subgrid_noise` argument
        to minimize discreteness effects due to the finite window size over which
        Prob(< y | x) is estimated. Default is None, for stochastic results.

    Returns
    -------
    rank_order_percentiles : ndarray
        Numpy array of shape (npts, ) storing values in the open interval (0, 1).
        Larger values of the returned array correspond to values of ``y``
        that are larger-than-average for the corresponding value of ``x``.

    Notes
    -----
    The ``window_length`` argument controls the precision of the calculation,
    and also the performance. For estimations of Prob(< y | x) with sub-percent accuracy,
    values of ``window_length`` must exceed 100.

    See :ref:`cam_tutorial` demonstrating how to use this
    function in galaxy-halo modeling with several worked examples.

    Examples
    --------
    >>> x = np.random.rand(100)
    >>> y = np.random.rand(100)
    >>> window_length = 5
    >>> result = sliding_conditional_percentile(x, y, window_length)
    """
    rank_orders = cython_sliding_rank(x, y, window_length,
                assume_x_is_sorted=assume_x_is_sorted)
    rank_order_percentiles = (1. + rank_orders)/float(window_length+1)

    if add_subgrid_noise:
        dp = 1./float(window_length+1)
        low = rank_order_percentiles - dp
        high = rank_order_percentiles + dp
        npts = len(rank_order_percentiles)
        with NumpyRNGContext(seed):
            rank_order_percentiles = np.random.uniform(low, high, npts)

    return rank_order_percentiles


def rank_order_function(x):
    r""" Calculate the rank-order of each element in an input array.

    Parameters
    ----------
    x : ndarray
        Array of shape (npts, )

    Results
    -------
    rank_orders : ndarray
        Integer array of shape (npts, ) storing values in the interval [0, npts-1]
    """
    x = np.atleast_1d(x)
    assert x.ndim == 1, "x must be a 1-d sequence"
    assert len(x) > 1, "x must have more than one element"

    return unsorting_indices(np.argsort(x))


def cython_sliding_rank(x, y, window_length, assume_x_is_sorted=False):
    r"""
    Return an array storing the rank-order of each element element in y
    computed over a fixed window length at each x

    This function is the kernel of calculation of Prob(< y | x).

    Parameters
    ----------
    x : ndarray
        Array of shape (npts, )

    y : ndarray
        Array of shape (npts, )

    window_length : int
        Integer must be odd and less than ``npts``

    assume_x_is_sorted : bool, optional
        Performance enhancement flag that can be used for cases where input `x`
        has already been sorted. Default is False.

    Returns
    -------
    sliding_rank_orders : ndarray
        Integer array of shape (npts, ) storing values between 0 and window_length-1

    Examples
    --------
    >>> x = np.random.rand(100)
    >>> y = np.random.rand(100)
    >>> window_length = 5
    >>> result = cython_sliding_rank(x, y, window_length)
    """
    x, y, nwin = _check_xyn_bounds(x, y, window_length)
    nhalfwin = int(nwin/2)

    if assume_x_is_sorted:
        y_sorted = y
    else:
        indx_x_sorted = np.argsort(x)
        indx_x_unsorted = unsorting_indices(indx_x_sorted)
        y_sorted = y[indx_x_sorted]

    result = np.array(cython_conditional_rank_kernel(y_sorted, nwin))

    leftmost_window_ranks = rank_order_function(y_sorted[:nwin])
    result[:nhalfwin+1] = leftmost_window_ranks[:nhalfwin+1]

    rightmost_window_ranks = rank_order_function(y_sorted[-nwin:])
    result[-nhalfwin-1:] = rightmost_window_ranks[-nhalfwin-1:]

    if assume_x_is_sorted:
        return result.astype(int)
    else:
        return result[indx_x_unsorted].astype(int)


def _check_xyn_bounds(x, y, n):
    r""" Enforce bounds checks on the inputs
    and return 1-d Numpy arrays with appropriate dtype
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
