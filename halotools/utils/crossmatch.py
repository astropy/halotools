r""" Module containing the `~halotools.utils.crossmatch` function used to
calculate indices providing the correspondence between two data tables
sharing a common objectID.
"""
import numpy as np


__all__ = ("crossmatch", "compute_richness")


def crossmatch(x, y, skip_bounds_checking=False):
    r"""
    Finds where the elements of ``x`` appear in the array ``y``, including repeats.

    The elements in x may be repeated, but the elements in y must be unique.
    The arrays x and y may be only partially overlapping.

    The applications of this function envolve cross-matching
    two catalogs/data tables which share an objectID.
    For example, if you have a primary data table and a secondary data table containing
    supplementary information about (some of) the objects, the
    `~halotools.utils.crossmatch` function can be used to "value-add" the
    primary table with data from the second.

    For another example, suppose you have a single data table
    with an object ID column and also a column for a "host" ID column
    (e.g., ``halo_hostid`` in Halotools-provided catalogs),
    you can use the `~halotools.utils.crossmatch` function to create new columns
    storing properties of the associated host.

    See :ref:`crossmatching_halo_catalogs` and :ref:`crossmatching_galaxy_catalogs`
    for tutorials on common usages of this function with halo and galaxy catalogs.

    Parameters
    ----------
    x : integer array
        Array of integers with possibly repeated entries.

    y : integer array
        Array of unique integers.

    skip_bounds_checking : bool, optional
        The first step in the `crossmatch` function is to test that the input
        arrays satisfy the assumptions of the algorithm
        (namely that ``x`` and ``y`` store integers,
        and that all values in ``y`` are unique).
        If ``skip_bounds_checking`` is set to True,
        this testing is bypassed and the function evaluates faster.
        Default is False.

    Returns
    -------
    idx_x : integer array
        Integer array used to apply a mask to x
        such that x[idx_x] == y[idx_y]

    y_idx : integer array
        Integer array used to apply a mask to y
        such that x[idx_x] == y[idx_y]

    Notes
    -----
    The matching between ``x`` and ``y`` is done on the sorted arrays.  A consequence of
    this is that x[idx_x] and y[idx_y] will generally be a subset of ``x`` and ``y`` in
    sorted order.

    Examples
    --------
    Let's create some fake data to demonstrate basic usage of the function.
    First, let's suppose we have two tables of objects, ``table1`` and ``table2``.
    There are no repeated elements in any table, but these tables only partially overlap.
    The example below demonstrates how to transfer column data from ``table2``
    into ``table1`` for the subset of objects that appear in both tables.

    >>> num_table1 = int(1e6)
    >>> x = np.random.rand(num_table1)
    >>> objid = np.arange(num_table1)
    >>> from astropy.table import Table
    >>> table1 = Table({'x': x, 'objid': objid})

    >>> num_table2 = int(1e6)
    >>> objid = np.arange(5e5, num_table2+5e5)
    >>> y = np.random.rand(num_table2)
    >>> table2 = Table({'y': y, 'objid': objid})

    Note that ``table1`` and ``table2`` only partially overlap. In the code below,
    we will initialize a new ``y`` column for ``table1``, and for those rows
    with an ``objid`` that appears in both ``table1`` and ``table2``,
    we'll transfer the values of ``y`` from ``table2`` to ``table1``.

    >>> idx_table1, idx_table2 = crossmatch(table1['objid'].data, table2['objid'].data)
    >>> table1['y'] = np.zeros(len(table1), dtype = table2['y'].dtype)
    >>> table1['y'][idx_table1] = table2['y'][idx_table2]

    Now we'll consider a slightly more complicated example in which there
    are repeated entries in the input array ``x``. Suppose in this case that
    our data ``x`` comes with a natural grouping, for example into those
    galaxies that occupy a common halo. If we have a separate table ``y`` that
    stores attributes of the group, we may wish to broadcast some group property
    such as total group mass amongst all the group members.

    First create some new dummy data to demonstrate this application of
    the `crossmatch` function:

    >>> num_galaxies = int(1e6)
    >>> x = np.random.rand(num_galaxies)
    >>> objid = np.arange(num_galaxies)
    >>> num_groups = int(1e4)
    >>> groupid = np.random.randint(0, num_groups, num_galaxies)
    >>> galaxy_table = Table({'x': x, 'objid': objid, 'groupid': groupid})

    >>> groupmass = np.random.rand(num_groups)
    >>> groupid = np.arange(num_groups)
    >>> group_table = Table({'groupmass': groupmass, 'groupid': groupid})

    Now we use the `crossmatch` to paint the appropriate value of ``groupmass``
    onto each galaxy:

    >>> idx_galaxies, idx_groups = crossmatch(galaxy_table['groupid'].data, group_table['groupid'].data)
    >>> galaxy_table['groupmass'] = np.zeros(len(galaxy_table), dtype = group_table['groupmass'].dtype)
    >>> galaxy_table['groupmass'][idx_galaxies] = group_table['groupmass'][idx_groups]

    See the tutorials for additional demonstrations of alternative
    uses of the `crossmatch` function.

    See also
    ---------
    :ref:`crossmatching_halo_catalogs`

    :ref:`crossmatching_galaxy_catalogs`

    """
    # Ensure inputs are Numpy arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    # Require that the inputs meet the assumptions of the algorithm
    if skip_bounds_checking is True:
        pass
    else:
        try:
            assert len(set(y)) == len(y)
            assert np.all(np.array(y).astype(int) == y)
            assert np.shape(y) == (len(y),)
        except:
            msg = "Input array y must be a 1d sequence of unique integers"
            raise ValueError(msg)
        try:
            assert np.all(np.array(x).astype(int) == x)
            assert np.shape(x) == (len(x),)
        except:
            msg = "Input array x must be a 1d sequence of integers"
            raise ValueError(msg)

    # Internally, we will work with sorted arrays, and then undo the sorting at the end
    idx_x_sorted = np.argsort(x)
    idx_y_sorted = np.argsort(y)
    x_sorted = np.copy(x[idx_x_sorted])
    y_sorted = np.copy(y[idx_y_sorted])

    # x may have repeated entries, so find the unique values as well as their multiplicity
    unique_xvals, counts = np.unique(x_sorted, return_counts=True)

    # Determine which of the unique x values has a match in y
    unique_xval_has_match = np.in1d(unique_xvals, y_sorted, assume_unique=True)

    # Create a boolean array with True for each value in x with a match, otherwise False
    idx_x = np.repeat(unique_xval_has_match, counts)

    # For each unique value of x with a match in y, identify the index of the match
    matching_indices_in_y = np.searchsorted(
        y_sorted, unique_xvals[unique_xval_has_match]
    )

    # Repeat each matching index according to the multiplicity in x
    idx_y = np.repeat(matching_indices_in_y, counts[unique_xval_has_match])

    # Undo the original sorting and return the result
    return idx_x_sorted[idx_x], idx_y_sorted[idx_y]


def compute_richness(unique_halo_ids, halo_id_of_galaxies):
    r"""For every ID in unique_halo_ids,
    calculate the number of times the ID appears in halo_id_of_galaxies.

    Parameters
    ----------
    unique_halo_ids : ndarray
        Numpy array of shape (num_halos, ) storing unique integers

    halo_id_of_galaxies : ndarray
        Numpy integer array of shape (num_galaxies, ) storing the host ID of each galaxy

    Returns
    -------
    richness : ndarray
        Numpy integer array of shape (num_halos, ) storing richness of each host halo

    Examples
    --------
    >>> num_hosts = 100
    >>> num_sats = int(1e5)
    >>> unique_halo_ids = np.arange(5, num_hosts + 5)
    >>> halo_id_of_galaxies = np.random.randint(0, 5000, num_sats)
    >>> richness = compute_richness(unique_halo_ids, halo_id_of_galaxies)
    """
    unique_halo_ids = np.atleast_1d(unique_halo_ids).astype(int)
    halo_id_of_galaxies = np.atleast_1d(halo_id_of_galaxies).astype(int)
    richness_result = np.zeros_like(unique_halo_ids).astype(int)

    vals, counts = np.unique(halo_id_of_galaxies, return_counts=True)
    idxA, idxB = crossmatch(vals, unique_halo_ids)
    richness_result[idxB] = counts[idxA]
    return richness_result
