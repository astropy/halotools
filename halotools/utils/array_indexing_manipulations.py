"""
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

__all__ = ('calculate_first_idx_unique_array_vals',
    'calculate_last_idx_unique_array_vals', 'sum_in_bins',
    'random_indices_within_bin')


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


def random_indices_within_bin(binned_multiplicity, desired_binned_occupations,
        seed=None, min_required_entries_per_bin=1):
    """ Given two equal-length arrays, with ``desired_binned_occupations``
    defining the number of desired random draws per bin,
    and ``binned_multiplicity`` defining the number of indices in each bin
    that are available to be randomly drawn,
    return a set of indices such that
    only the appropriate indices will be drawn for each bin, and the total
    number of such random draws is in accord with the input ``desired_binned_occupations``.

    The ``random_indices_within_bin`` function is the kernel of the calculation in which
    satellites are assigned to host halos that do not have enough subhalos
    to serve as satellites. The algorithm implemented here enables, for example,
    the random selection of a subhalo that resides in a host of a nearby mass.

    Parameters
    -----------
    binned_multiplicity : array
        Array of length-*Nbins* storing how many total items
        reside in each bin.

        All entries of ``binned_multiplicity`` must be at least as large
        as ``min_required_entries_per_bin``, enforcing a user-specified requirement
        that in each bin, you must have "enough" entries to draw from.

    desired_binned_occupations : array
        Array of non-negative integers of length-*Nbins* storing
        the number of times to draw from each bin.

    seed : integer, optional
        Random number seed used when drawing random numbers with `numpy.random`.
        Useful when deterministic results are desired, such as during unit-testing.
        Default is None, producing stochastic results.

    min_required_entries_per_bin : int, optional
        Minimum requirement on the number of entries in each bin. Default is 1.

    Returns
    -------
    indices : array
        Integer array of length equal to desired_binned_occupations.sum()
        whose values can be used to index the appropriate entries of the subhalo table.

    Examples
    ---------
    >>> binned_multiplicity = np.array([1, 2, 2, 1, 3])
    >>> desired_binned_occupations = np.array([2, 1, 3, 0, 2])
    >>> idx = random_indices_within_bin(binned_multiplicity, desired_binned_occupations)

    The ``idx`` array has *desired_binned_occupations.sum()* total entries,
    with each entry storing the index of the subhalo table that will serve as a
    randomly selected satellite.
    """
    try:
        assert np.all(binned_multiplicity >= min_required_entries_per_bin)
    except AssertionError:
        msg = ("Input ``binned_multiplicity`` array must contain at least \n"
        "min_required_entries_per_bin = {0} entries. \nThis indicates that "
        "the host halo mass bins should be broader.\n".format(min_required_entries_per_bin))
        raise ValueError(msg)

    try:
        assert np.all(desired_binned_occupations >= 0)
    except AssertionError:
        msg = ("All entries of input ``desired_binned_occupations``\n"
            "must be non-negative integers.\n")
        raise ValueError(msg)

    num_draws = desired_binned_occupations.sum()
    if num_draws == 0:
        return np.array([], dtype=int)

    with NumpyRNGContext(seed):
        uniform_random = np.random.rand(num_draws)

    num_available_subs = np.repeat(binned_multiplicity.astype(int),
        desired_binned_occupations.astype(int))
    intra_bin_indices = np.floor(uniform_random*num_available_subs)

    first_bin_indices = np.concatenate(([0], np.cumsum(binned_multiplicity)[:-1]))
    repeated_first_bin_indices = np.repeat(first_bin_indices,
        desired_binned_occupations.astype(int))

    absolute_indices = intra_bin_indices + repeated_first_bin_indices
    return absolute_indices
