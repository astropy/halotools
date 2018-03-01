"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
from copy import deepcopy

from ..conditional_percentile import cython_sliding_rank, sliding_conditional_percentile
from ..conditional_percentile import rank_order_function, _check_xyn_bounds
from ..array_utils import unsorting_indices


__all__ = ('test_brute_force_python_rank_comparison', )

fixed_seed = 43


def python_sliding_rank(x, y, window_length):
    r"""
    Return an array storing the rank-order of each element element in y
    computed over a fixed window length at each x

    This function is the kernel of calculation of P(y | x).

    Parameters
    ----------
    x : ndarray
        Array of shape (npts, )

    y : ndarray
        Array of shape (npts, )

    window_length : int

    Returns
    -------
    sliding_rank_orders : ndarray
        Integer array of shape (npts, ) storing values between 0 and window_length-1

    Examples
    --------
    >>> x = np.random.rand(100)
    >>> y = np.random.rand(100)
    >>> window_length = 5
    >>> result = python_sliding_rank(x, y, window_length)
    """
    x, y, nwin = _check_xyn_bounds(x, y, window_length)
    npts = len(x)

    indx_x_sorted = np.argsort(x)
    indx_x_unsorted = unsorting_indices(indx_x_sorted)

    y_sorted = y[indx_x_sorted]

    result = np.zeros(len(x), dtype='f4') + np.nan

    nhalfwin = int(nwin/2)
    for iy in range(nhalfwin, npts-nhalfwin):
        ifirst = iy - nhalfwin
        ilast = iy + nhalfwin + 1
        window_ranks = rank_order_function(y_sorted[ifirst:ilast])
        result[iy] = window_ranks[nhalfwin]

    leftmost_window_ranks = rank_order_function(y_sorted[:nwin])
    result[:nhalfwin+1] = leftmost_window_ranks[:nhalfwin+1]
    rightmost_window_ranks = rank_order_function(y_sorted[-nwin:])
    result[-nhalfwin-1:] = rightmost_window_ranks[-nhalfwin-1:]

    return result[indx_x_unsorted].astype(int)


def test_brute_force_python_rank_comparison():
    r""" Generate some longer datasets for brute force comparison.
    """
    npts = 300
    window_length_options = (5, 35, 77, 101, 203)
    num_tests = 20
    num_failures = 0
    for seed in range(num_tests):
        with NumpyRNGContext(seed):
            x = np.random.rand(npts)
        with NumpyRNGContext(seed+1):
            y = np.random.rand(npts)
        with NumpyRNGContext(seed+2):
            window_length = np.random.choice(window_length_options)

        result = python_sliding_rank(deepcopy(x), deepcopy(y), window_length)
        result2 = cython_sliding_rank(deepcopy(x), deepcopy(y), window_length)

        if not np.all(result == result2):
            num_failures += 1

    msg = "Failed brute force comparison on {0} of {1} random tests"
    assert num_failures == 0, msg.format(num_failures, num_tests)


def test1_assume_x_is_sorted():
    npts = 1000
    with NumpyRNGContext(fixed_seed):
        x = np.sort(np.random.random(npts))
        y = np.random.random(npts)
    window_length = 101
    p1 = sliding_conditional_percentile(deepcopy(x), deepcopy(y), window_length,
            assume_x_is_sorted=True, add_subgrid_noise=False)
    p2 = sliding_conditional_percentile(deepcopy(x), deepcopy(y), window_length,
            assume_x_is_sorted=False, add_subgrid_noise=False)
    assert np.allclose(p1, p2)


def test2_assume_x_is_sorted():
    npts = 1000
    with NumpyRNGContext(fixed_seed):
        x = np.random.random(npts)
        y = np.random.random(npts)
    window_length = 101
    p1 = sliding_conditional_percentile(deepcopy(x), deepcopy(y), window_length,
            assume_x_is_sorted=True, add_subgrid_noise=False)
    p2 = sliding_conditional_percentile(deepcopy(x), deepcopy(y), window_length,
            assume_x_is_sorted=False, add_subgrid_noise=False)
    assert not np.allclose(p1, p2)


def test3_assume_x_is_sorted():
    npts = 1000
    with NumpyRNGContext(fixed_seed):
        x = np.random.random(npts)
        y = np.random.random(npts)
    window_length = 101
    p1 = sliding_conditional_percentile(deepcopy(x), deepcopy(y), window_length,
            assume_x_is_sorted=False, add_subgrid_noise=False)

    indx_x_sorted = np.argsort(x)
    indx_x_unsorted = unsorting_indices(indx_x_sorted)
    x_sorted = x[indx_x_sorted]
    y_sorted = y[indx_x_sorted]
    p2_sorted = sliding_conditional_percentile(x_sorted, y_sorted, window_length,
            assume_x_is_sorted=True, add_subgrid_noise=False)
    p2 = p2_sorted[indx_x_unsorted]

    assert np.allclose(p1, p2)


def test_subgrid_noise():
    npts = 1000
    with NumpyRNGContext(fixed_seed):
        x = np.random.random(npts)
        y = np.random.random(npts)
    window_length = 51
    p1 = sliding_conditional_percentile(deepcopy(x), deepcopy(y), window_length,
            add_subgrid_noise=True)
    p2 = sliding_conditional_percentile(deepcopy(x), deepcopy(y), window_length,
            add_subgrid_noise=False)
    assert not np.allclose(p1, p2, atol=0.01)
    assert np.allclose(p1, p2, atol=0.05)
