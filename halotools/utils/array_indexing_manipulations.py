"""
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np

__all__ = ('calculate_first_idx_unique_array_vals',
    'calculate_last_idx_unique_array_vals', 'sum_in_bins')


def calculate_first_idx_unique_array_vals(sorted_array):
    """ Given an integer array with possibly repeated entries in ascending order,
    return the indices of the first appearance of each unique value.

    Parameters
    ----------
    sorted_array : array
        Integer array of host halo IDs, sorted in ascending order

    Returns
    --------
    idx_unique_array_vals : array
        Integer array storing the indices of the first appearance of
        each unique entry in sorted_array

    Notes
    ------
    By construction, the first element of `calculate_first_idx_unique_array_vals`
    will always be zero.

    Examples
    --------
    >>> sorted_array = np.array((0, 0, 1, 1, 4, 8, 8, 10))
    >>> result = calculate_first_idx_unique_array_vals(sorted_array)
    >>> assert np.all(result == (0, 2, 4, 5, 7))
    """
    return np.concatenate(([0], np.flatnonzero(np.diff(sorted_array)) + 1))


def calculate_last_idx_unique_array_vals(sorted_array):
    """ Given an integer array with possibly repeated entries in ascending order,
    return the indices of the last appearance of each unique value.

    Parameters
    ----------
    sorted_array : array
        Integer array of host halo IDs, sorted in ascending order

    Returns
    --------
    idx_unique_array_vals : array
        Integer array storing the indices of the last appearance of
        each unique entry in sorted_array

    Notes
    ------
    By construction, the first element of `calculate_first_idx_unique_array_vals`
    will always be len(sorted_array)-1.

    Examples
    --------
    >>> sorted_array = np.array((0, 0, 1, 1, 4, 8, 8, 10))
    >>> result = calculate_last_idx_unique_array_vals(sorted_array)
    >>> assert np.all(result == (1, 3, 4, 6, 7))
    """
    return np.append(np.flatnonzero(np.diff(sorted_array)), len(sorted_array)-1)


def sum_in_bins(arr, sorted_bin_numbers, testing_mode=False):
    """ Given an array of values ``arr`` and another equal-length array
    ``sorted_bin_numbers`` storing how these values have been binned into *Nbins*,
    calculate the sum of the values in each bin.

    Parameters
    -----------
    arr : array
        Array of length *Nvals* storing the quantity to be summed
        in the bins defined by ``sorted_bin_numbers``.

    sorted_bin_numbers : array
        Integer array of length *Nvals* storing the bin numbers
        of each entry of the input ``arr``,
        e.g., the result of np.digitize(arr, bins).
        The ``sorted_bin_numbers`` array may have repeated entries but must
        be in ascending order. That is, the subhalos whose property is
        stored in array ``arr`` will be presumed to be pre-grouped according to,
        for example, host halo mass, with lowest halo masses first,
        and higher halo masses at higher indices, in monotonic fashion.

    testing_mode : bool, optional
        Boolean specifying whether input arrays will be tested to see if they
        satisfy the assumptions required by the algorithm.
        Setting ``testing_mode`` to True is useful for unit-testing purposes,
        while setting it to False improves performance.
        Default is False.

    Returns
    --------
    binned_sum : array
        Array of length-*Nbins* storing the sum of ``arr`` values
        within the bins defined by ``sorted_bin_numbers``.

    Examples
    --------
    >>> Nvals = 5
    >>> arr = np.arange(Nvals)
    >>> sorted_bin_numbers = np.array((1, 2, 2, 6, 7))
    >>> result = sum_in_bins(arr, sorted_bin_numbers)
    """
    try:
        assert len(arr) == len(sorted_bin_numbers)
    except AssertionError:
        raise ValueError("Input ``arr`` and ``sorted_bin_numbers`` must have same length")

    if testing_mode is True:
        try:
            assert np.all(np.diff(sorted_bin_numbers) >= 0)
        except AssertionError:
            msg = ("Input ``sorted_bin_numbers`` array must be sorted in ascending order")
            raise ValueError(msg)

    last_idx = calculate_last_idx_unique_array_vals(sorted_bin_numbers)
    first_entry = arr[:last_idx[0]+1].sum()
    binned_sum = np.diff(np.cumsum(arr)[last_idx])
    binned_sum = np.concatenate(([first_entry], binned_sum))
    return binned_sum
