"""
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

__all__ = ('subhalo_indexing_array', )


def subhalo_indexing_array(subhalo_hostids, satellite_occupations, host_halo_ids,
        host_halo_bin_numbers, fill_remaining_satellites=True,
        seed=None, testing_mode=False):
    """ Function driving the selection of subhalos during HOD mock population.
    Given a catalog of subhalos, host halos and a desired number of satellites
    in each host, the `subhalo_indexing_array` function can be used to return the
    indices used to select subhalos to serve as satellites.

    In the situation in which a host halo does not have as many subhalos
    as the desired number of satellites, a subhalo within the same bin
    as that host (e.g., a subhalo with a similar host mass)
    will be randomly selected to serve as the satellite;
    we will refer to such satellites as *orphans.*
    Here, the input ``host_halo_bin_numbers``
    determines which host halos are grouped together into the same bin.

    The returned array is a length-*Nsats* array storing the indices of the
    ``subhalo_hostids`` that were selected. In case special treatment of the
    orphan satellites is desired, the ``subhalo_indexing_array`` function
    also returns a boolean array that can be used as a mask to identify the
    orphans.

    Parameters
    ----------
    subhalo_hostids : array
        Integer array of length *Nsubs* storing the id of the associated host halo.
        ``subhalo_hostids`` may have repeated values and must be in ascending order.

    satellite_occupations : array
        Integer array of length *Nhosts* storing the desired
        number of satellites in each host halo.

    host_halo_ids : array
        Integer array of length *Nhosts* storing each host halo's unique id,
        typically the ``halo_id`` column in a Halotools-catalog.

    host_halo_bin_numbers : array
        Integer array of length *Nhosts* storing the bin number of each host halo.

    fill_remaining_satellites : bool, optional
        To address cases of a host halo with fewer available subhalos
        than the desired number of satellites, the indices of randomly selected
        subhalos from the same host mass bin will be selected provided that
        ``fill_remaining_satellites`` is set to True.
        If ``fill_remaining_satellites`` is instead set to False, then the value
        -1 will be returned for all such entries. Default is True.

    seed : integer, optional
        Random number seed used when drawing random numbers with `numpy.random`.
        Useful when deterministic results are desired, such as during unit-testing.
        Default is None, producing stochastic results.

    testing_mode : bool, optional
        Boolean specifying whether input arrays will be tested to see if they
        satisfy the assumptions required by the algorithm.
        Setting ``testing_mode`` to True is useful for unit-testing purposes,
        while setting it to False improves performance.
        Default is False.

    Returns
    -------
    satellite_selection_indices : array
        Integer array storing the indices of the selected subhalos.
        Some indices have been multiplied by -1.
        This indicates that in one or more host halos, there were more desired
        satellites than the number of subhalos in that halo.
        In such a case, the index of a randomly selected subhalo in a similar
        host mass is selected, and this index is multiplied by -1.

    missing_subhalo_mask : array
        Boolean array that can be used to select the indices corresponding to
        satellites with no true subhalo in the associated host halo. This
        situation occurs whenever and entry of ``desired_occupations``
        exceeds the number of subhalos in that host halo,
        as described in the Notes.

    Examples
    --------
    We'll demonstrate basic usage here using a halo catalog taken from a
    `~halotools.sim_manager.FakeSim` object, which means we'll have to do a fair
    amount of work to arrange the memory layout into the required form.
    When ``subhalo_indexing_array`` is used as part of a Halotools model,
    this organization is typically accomplished in an automated fashion during a
    pre-processing phase of mock population.

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim()

    We are going to group our (sub)halos together by host halo mass. However,
    the `~halotools.sim_manager.FakeSim` does not come with a column for
    host halo mass, so we will need to calculate this ourselves using
    the `halotools.utils.crossmatch` function, using the algorithm described
    in the tutorial on :ref:`crossmatching_halo_catalogs`.

    >>> from halotools.utils import crossmatch
    >>> idxA, idxB = crossmatch(halocat.halo_table['halo_hostid'], halocat.halo_table['halo_id'])
    >>> halocat.halo_table['halo_mvir_host_halo'] = 0.
    >>> halocat.halo_table['halo_mvir_host_halo'][idxA] = halocat.halo_table['halo_mvir'][idxB]

    The `subhalo_indexing_array` algorithm requires that
    every entry of the input ``subhalo_hostids`` has a matching entry in
    the input ``host_halo_ids`` array. To address this, we will mask out
    those rare subhalos with no matching host halo
    (this situation occurs in <0.1% for typical Rockstar catalogs).

    >>> unmatched_mask = halocat.halo_table['halo_mvir_host_halo'] == 0. #  initialized column value
    >>> halos = halocat.halo_table[~unmatched_mask]

    Now we will sort the catalog by the ``sorting_keys`` list.

    >>> halos['negative_halo_vpeak'] = -halos['halo_vpeak']
    >>> sorting_keys = ['halo_mvir_host_halo', 'halo_hostid', 'halo_upid', 'negative_halo_vpeak']
    >>> halos.sort(sorting_keys)

    Our halo catalog is now sorted in ascending order of ``halo_mvir_host_halo``.
    Because the second entry of our ``sorting_keys`` is ``halo_hostid``, then within
    each bin of host halo mass, halos and subhalos with the same ``halo_hostid``
    will be grouped together. Since ``halo_upid`` is -1 for host halos and
    a positive long integer for subhalos, then choosing ``halo_upid`` as our third
    ``sorting_key`` entry, then within each host-sub system the host halo will
    appear first. Finally, the subhalos in each host will be arranged in
    *descending* order of ``halo_vpeak`` - this ensures that subhalos with particularly
    large ``halo_vpeak`` will be preferentially selected to serve as satellites.

    Now we separate host halos from subhalos (preserving the above memory layout),
    and define the arrays we'll use as inputs to the `subhalo_indexing_array` function.

    >>> host_halo_mask = halos['halo_upid'] == -1
    >>> hosts = halos[host_halo_mask]
    >>> subhalos = halos[~host_halo_mask]
    >>> mass_bins = np.logspace(9.9, 16.1, 5)
    >>> host_halo_bin_numbers = np.digitize(hosts['halo_mvir'].data, mass_bins)
    >>> subhalo_hostids = subhalos['halo_hostid'].data
    >>> satellite_occupations = np.random.randint(0, 5, len(hosts))
    >>> host_halo_ids = hosts['halo_id'].data

    With our arrays so defined, we call the ``subhalo_indexing_array`` function
    and demonstrate that it does indeed return a result that can serve as an indexing
    array providing us with the correct total number of satellites.

    >>> result = subhalo_indexing_array(subhalo_hostids, satellite_occupations, host_halo_ids, host_halo_bin_numbers, testing_mode=True)
    >>> satellite_selection_indices, missing_subhalo_mask = result
    >>> selected_subhalos = subhalos[satellite_selection_indices]
    >>> assert len(selected_subhalos) == satellite_occupations.sum()

    """
    idx_selected_subhalos, subhalo_occupations, subhalo_multiplicity = (
        calculate_selection_of_true_subhalos(
        subhalo_hostids, satellite_occupations, host_halo_ids, testing_mode=testing_mode))

    remaining_occupations = satellite_occupations - subhalo_occupations

    if fill_remaining_satellites is True:
        idx_remaining_subhalos = (
            calculate_selection_of_remaining_satellites(remaining_occupations,
                subhalo_occupations, subhalo_multiplicity, host_halo_bin_numbers,
                seed=seed, testing_mode=testing_mode)
            )
    else:
        idx_remaining_subhalos = np.zeros(remaining_occupations.sum()) - 1

    satellite_selection_indices, missing_subhalo_mask = array_weave(
        idx_selected_subhalos, idx_remaining_subhalos,
        subhalo_occupations, remaining_occupations, testing_mode=testing_mode)

    return satellite_selection_indices, missing_subhalo_mask


def calculate_selection_of_true_subhalos(subhalo_hostids, satellite_occupations, host_halo_ids,
        testing_mode=False):
    """
    """
    subhalo_multiplicity = calculate_subhalo_multiplicity(subhalo_hostids, host_halo_ids)

    subhalo_occupations = calculate_subhalo_occupations(satellite_occupations, subhalo_multiplicity)

    idx_selected_subhalos = indices_of_selected_subhalos(
        subhalo_hostids, subhalo_occupations, subhalo_multiplicity, testing_mode=testing_mode)

    return idx_selected_subhalos, subhalo_occupations, subhalo_multiplicity


def calculate_selection_of_remaining_satellites(remaining_occupations,
        subhalo_occupations, subhalo_multiplicity, host_halo_bin_numbers, seed=None,
        testing_mode=False):
    """
    Calculate the indices of subhalos that should be selected as satellites to
    address the remaining cases where a given host halo did not have as many
    subhalos as desired satellites. The strategy implemented here is to randomly
    select subhalos from the same host "mass" bin.

    Parameters
    -----------
    remaining_occupations : array
        Integer array of length-*Nhosts* storing the number of satellites that
        remain to be selected after using all the available subhalos in each host.

    subhalo_occupations : array
        Integer array of length-*Nhosts* storing the number of subhalos that have
        already been selected to serve as satellites.

    subhalo_multiplicity : array
        Length-*Nhosts* integer array storing the number of
        subhalos in each host halo.

    host_halo_bin_numbers : array
        Integer array of length *Nhosts* storing the bin numbers
        of each host halo, e.g., the result of np.digitize(host_halo_mass, mass_bins).
        The ``host_halo_bin_numbers`` array may have repeated entries but must
        be in ascending order.

    seed : integer, optional
        Random number seed used when drawing random numbers with `numpy.random`.
        Useful when deterministic results are desired, such as during unit-testing.
        Default is None, producing stochastic results.

    """

    binned_subhalo_multiplicity = sum_in_bins(subhalo_multiplicity, host_halo_bin_numbers)
    binned_remaining_occupations = sum_in_bins(remaining_occupations, host_halo_bin_numbers)
    remaining_indices = random_indices_within_bin(
        binned_subhalo_multiplicity, binned_remaining_occupations, seed)

    return remaining_indices


def calculate_subhalo_multiplicity(subhalo_hostids, host_halo_ids):
    """ Determine the number of subhalos in each host halo.

    Parameters
    ----------
    subhalo_hostids : array
        Length-*Nsubs* integer array storing
        the ID of the halo hosting each subhalo.
        The entries of ``subhalo_hostids`` may be repeated,
        but must be in ascending order.

    host_halo_ids : array
        Length-*Nhosts* integer array storing the ID
        of the host halos in which the subhalos (may) reside.
        All entries must be unique. Entries of ``host_halo_id``
        with no accompanying subhalos are permissible,
        and the ``host_halo_ids`` array can be sorted in any order.

    Returns
    ---------
    subhalo_multiplicity : array
        Length-*Nhosts* integer array storing the number of
        subhalos in each host halo.

    Examples
    --------
    >>> subhalo_hostids = np.array((1, 1, 2, 2, 2, 4, 5, 6, 6))
    >>> host_halo_ids = np.arange(7)
    >>> subhalo_multiplicity = calculate_subhalo_multiplicity(subhalo_hostids, host_halo_ids)
    >>> assert np.all(subhalo_multiplicity == (0, 2, 3, 0, 1, 1, 2))

    """
    unique_subhalo_hostid, unique_subhalo_multiplicity = (
        np.unique(subhalo_hostids, return_counts=True))
    host_halo_has_a_subhalo = np.in1d(host_halo_ids, unique_subhalo_hostid, assume_unique=True)

    subhalo_multiplicity = np.zeros_like(host_halo_ids)
    subhalo_multiplicity[host_halo_has_a_subhalo] = unique_subhalo_multiplicity
    return subhalo_multiplicity


def array_weave(val1, val2, mult1, mult2, testing_mode=False):
    """ Calculate the array that weaves together the values stored in the
    two arrays val1 and val2 according to the input mult1 and mult2.
    Additionally return a masking array that can be used to select
    either set of values from the returned woven array.

    Parameters
    ----------
    val1 : array
        Length-*num_vals1* array

    val2 : array
        Length-*num_vals2* array

    mult1 : array
        Length-*Nhosts* array describing the existing multiplicity of the
        ``val1`` entries.

    mult2 : array
        Length-*Nhosts* array describing the existing multiplicity of the
        ``val2`` entries.

    testing_mode : bool, optional
        Boolean specifying whether input arrays will be tested to see if they
        satisfy the assumptions required by the algorithm.
        Setting ``testing_mode`` to True is useful for unit-testing purposes,
        while setting it to False improves performance.
        Default is False.

    Returns
    ---------
    result : array
        Length-*num_vals1 + num_vals2* array consisting exclusively of
        ``val1`` and ``val2`` entries, woven together in a manner described in the
        Notes section below.

    val2_mask : array
        Boolean array that can be used to select the ``val2`` subset
        from the returned ``result``.
        The array ``~val2_mask`` selects the ``val1`` subset.

    Examples
    --------
    >>> val1 = np.array((1, 2, 2, 4))
    >>> val2 = np.array((-1, -1, -2, -3, -3))
    >>> mult1 = np.array((1, 2, 0, 1))
    >>> mult2 = np.array((2, 1, 2, 0))
    >>> result, val2_mask = array_weave(val1, val2, mult1, mult2)

    For each entry of the ``mult`` arrays, a sequence of the corresponding
    ``val1`` values appears first, followed by a sequence of ``val2`` entries.
    This alternation proceeds element-wise across the ``mult`` arrays,
    skipping ``val1`` and/or ``val2`` entries as needed.

    >>> assert np.all(result == (1, -1, -1, 2, 2, -2, -3, -3, 4))
    """
    try:
        assert len(mult1) == len(mult2)
    except AssertionError:
        msg = "Input ``mult1`` and ``mult2`` arrays must have equal length"
        raise AssertionError(msg)

    if testing_mode is True:
        total_val1_values = np.sum(mult1)
        try:
            assert total_val1_values == len(val1)
        except AssertionError:
            msg = "The sum of the ``mult1`` entries should equal the length of ``val1``"
            raise ValueError(msg)

        total_val2_values = np.sum(mult2)
        try:
            assert total_val2_values == len(val2)
        except AssertionError:
            msg = "The sum of the ``mult2`` entries should equal the length of ``val2``"
            raise ValueError(msg)

    mult = np.array([mult1, mult2]).ravel('F')
    tftf = np.tile([True, False], len(mult1))
    val1_mask = np.repeat(tftf, mult)

    result = np.empty(len(val1) + len(val2), int)
    result[val1_mask] = val1
    result[~val1_mask] = val2
    return result, ~val1_mask


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


def indices_of_selected_subhalos(subhalo_hostids, subhalo_occupations, subhalo_multiplicity,
            testing_mode=False):
    """
    Given a sorted array of integers ``subhalo_hostids`` whose entries store the
    ID of the host halo in which the subhalos reside,
    and given an integer array ``subhalo_occupations`` specifying how many subhalos
    should be selected from each host halo, and also given the number of
    subhalos in each host halo ``subhalo_multiplicity``,
    return the indices corresponding to the selected objects.

    Parameters
    -----------
    subhalo_hostids : array
        Sorted integer array of length-*Nsubs* storing the ID of the host halo
        in which each subhalo resides.
        It is permissible for there to be host halos represented
        in ``subhalo_occupations`` with no subhalos in ``subhalo_hostids``.
        However, every value of ``subhalo_hostids`` must correspond to the ID
        of some host halo represented by the ``subhalo_occupations`` array.
        This implies that there must be at least as many entries in
        ``subhalo_occupations`` as there are unique entries of ``subhalo_hostids``.

    subhalo_occupations : array
        Integer array of length-*Nhalos* storing the number of subhalos
        that should be selected from each host halo.
        No entry of ``subhalo_occupations`` may exceed the
        corresponding value of ``subhalo_multiplicity``.

    subhalo_multiplicity : array
        Integer array of length-*Nhalos* storing
        the number of subhalos in each host halo.
        The sum of the entries of``subhalo_multiplicity`` must equal *Nsub*.

    testing_mode : bool, optional
        Boolean specifying whether input arrays will be tested to see if they
        satisfy the assumptions required by the algorithm.
        Setting ``testing_mode`` to True is useful for unit-testing purposes,
        while setting it to False improves performance.
        Default is False.

    Returns
    -------
    index_array : array
        Integer array of length *Nsats = subhalo_occupations.sum()*
        storing the indices of the subhalos that should be selected.

    Examples
    ---------
    >>> subhalo_hostids = np.array((1, 1, 2, 2, 2, 3, 9, 9))
    >>> subhalo_multiplicity = np.array((0, 2, 3, 1, 0, 0, 2))
    >>> subhalo_occupations  = np.array((0, 2, 0, 1, 0, 0, 2))

    >>> index_array = indices_of_selected_subhalos(subhalo_hostids, subhalo_occupations, subhalo_multiplicity)
    >>> selected_subhalo_hostids = subhalo_hostids[index_array]
    """
    if testing_mode is True:

        Nhosts = len(subhalo_occupations)
        try:
            assert Nhosts == len(subhalo_multiplicity)
        except:
            msg = "Input ``subhalo_occupations`` and ``subhalo_multiplicity`` must have the same length"
            raise ValueError(msg)

        try:
            unique_subhalo_hostids_values = np.unique(subhalo_hostids)
            num_unique_subhalo_hostids = len(unique_subhalo_hostids_values)
            assert num_unique_subhalo_hostids <= Nhosts
        except AssertionError:
            msg = ("The input ``subhalo_hostids`` has {0} unique entries, \n"
            "but there are only {1} total entries in ``subhalo_occupations``.\n"
            "The host halo of each subhalo must be represented in the ``subhalo_occupations`` "
            "array, \n so this mismatch is not permissible.\n")
            raise ValueError(msg.format(num_unique_subhalo_hostids, Nhosts))

        try:
            assert np.all(subhalo_occupations <= subhalo_multiplicity)
        except AssertionError:
            msg = ("No entry of ``subhalo_occupations`` may "
                "exceed the corresponding entry of ``subhalo_multiplicity``\n")
            raise ValueError(msg)

        Nsubs = len(subhalo_hostids)
        try:
            total_subhalo_multiplicity = subhalo_multiplicity.sum()
            assert total_subhalo_multiplicity == Nsubs
        except AssertionError:
            msg = ("The sum of ``subhalo_multiplicity`` is {0}, \n"
                "which is inconsistent with the total number of "
                "entries of ``subhalo_hostids`` = {1}.")
            raise ValueError(msg.format(total_subhalo_multiplicity, Nsubs))

    clipped_subhalo_occupations = subhalo_occupations[subhalo_multiplicity>0]
    csum = clipped_subhalo_occupations.cumsum()
    num_subhalos_to_draw = csum[-1]

    idx_unique_subhalo_hostids = calculate_first_idx_unique_array_vals(subhalo_hostids)

    return (np.arange(num_subhalos_to_draw) +
        np.repeat(idx_unique_subhalo_hostids - csum + clipped_subhalo_occupations,
            clipped_subhalo_occupations))


def calculate_subhalo_occupations(satellite_occupations, subhalo_multiplicity):
    """ Given ``satellite_occupations``, the desired number of satellites
    in each host halo, as well as ``subhalo_multiplicity``,
    the number of subhalos available in each halo to serve as satellites,
    return the number of existing subhalos in each host that will serve as satellites.
    For cases where ``satellite_occupations`` exceeds ``subhalo_multiplicity``,
    all subhalos will be selected and the remaining satellites
    in that halo must be assigned by other means.

    Parameters
    ----------
    satellite_occupations : array
        Integer array of length-*Nhalos* storing the desired number of
        satellites in each halo.

    subhalo_multiplicity : array
        Integer array of length-*Nhalos* storing the number of subhalos
        residing in each halo

    Returns
    -------
    subhalo_occupations : array
        Integer array of length-*Nhalos* storing the number of subhalos
        in each halo that will be selected to serve as satellites.

    Examples
    ---------
    >>> satellite_occupations = np.array([1, 2, 3, 0])
    >>> subhalo_multiplicity = np.array([2, 1, 2, 3])
    >>> subhalo_occupations = calculate_subhalo_occupations(satellite_occupations, subhalo_multiplicity)

    >>> assert np.all(subhalo_occupations == (1, 1, 2, 0))
    """
    return np.where(satellite_occupations > subhalo_multiplicity,
        subhalo_multiplicity, satellite_occupations)


def random_indices_within_bin(binned_multiplicity, desired_binned_occupations,
        seed=None, min_required_subs_per_bin=1):
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
        Array of length-*Nbins* storing how many total subhalos reside in
        the collection of hosts in each bin.
        All entries must be at least as large as ``min_required_subs_per_bin``,
        enforcing a user-specified requirement that
        in each host halo mass bin, you must have "enough" subhalos to draw from.

    desired_binned_occupations : array
        Integer array of length-*Nbins* storing how many desired are satellites
        should be drawn for bin.

    seed : integer, optional
        Random number seed used when drawing random numbers with `numpy.random`.
        Useful when deterministic results are desired, such as during unit-testing.
        Default is None, producing stochastic results.

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
        assert np.all(binned_multiplicity >= min_required_subs_per_bin)
    except AssertionError:
        msg = ("Input ``binned_multiplicity`` array must contain at least \n"
        "min_required_subs_per_bin = {0} entries. \nThis indicates that "
        "the host halo mass bins should be broader.\n".format(min_required_subs_per_bin))
        raise ValueError(msg)

    num_draws = desired_binned_occupations.sum()
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