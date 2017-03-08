"""
This module contains private helper functions used throughout the
`~halotools.mock_observables.pair_counters` subpackage to perform
control flow on function arguments, bounds-checking and exception-handling.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from copy import copy

__author__ = ['Duncan Campbell', 'Andrew Hearin']

__all__ = ('_set_approximate_cell_sizes', '_cell1_parallelization_indices')


def _enclose_in_box(x1, y1, z1, x2, y2, z2, min_size=None):
    """
    Build box which encloses all points, shifting the points so that
    the "leftmost" point is (0,0,0).

    Parameters
    ----------
    x1,y1,z1 : array_like
        cartesian positions of points

    x2,y2,z2 : array_like
        cartesian positions of points

    min_size : array_like
        minimum lengths of a side of the box.  If the minimum box constructed around the
        points has a side i less than ``min_size[i]``, then the box is padded in order to
        obtain the minimum specified size.

    Returns
    -------
    x1, y1, z1, x2, y2, z2, Lbox
        shifted positions and box size.
    """
    xmin = np.min([np.min(x1), np.min(x2)])
    ymin = np.min([np.min(y1), np.min(y2)])
    zmin = np.min([np.min(z1), np.min(z2)])
    xmax = np.max([np.max(x1), np.max(x2)])
    ymax = np.max([np.max(y1), np.max(y2)])
    zmax = np.max([np.max(z1), np.max(z2)])

    xyzmin = np.min([xmin, ymin, zmin])
    xyzmax = np.max([xmax, ymax, zmax])-xyzmin

    x1 = x1 - xyzmin
    y1 = y1 - xyzmin
    z1 = z1 - xyzmin
    x2 = x2 - xyzmin
    y2 = y2 - xyzmin
    z2 = z2 - xyzmin

    Lbox = np.array([xyzmax, xyzmax, xyzmax])

    if min_size is not None:
        min_size = np.atleast_1d(min_size)
        if np.any(Lbox < min_size):
            Lbox[(Lbox < min_size)] = min_size[(Lbox < min_size)]

    return x1, y1, z1, x2, y2, z2, Lbox


def _enclose_in_square(x1, y1, x2, y2, min_size=None):
    """
    Build box which encloses all points, shifting the points so that
    the "leftmost" point is (0, 0).

    Parameters
    ----------
    x1, y1 : array_like
        cartesian positions of points

    x2, y2 : array_like
        cartesian positions of points

    min_size : array_like
        minimum lengths of a side of the box.  If the minimum box constructed around the
        points has a side i less than ``min_size[i]``, then the box is padded in order to
        obtain the minimum specified size.

    Returns
    -------
    x1, y1, x2, y2, Lbox
        shifted positions and box size.
    """
    xmin = np.min([np.min(x1), np.min(x2)])
    ymin = np.min([np.min(y1), np.min(y2)])
    xmax = np.max([np.max(x1), np.max(x2)])
    ymax = np.max([np.max(y1), np.max(y2)])

    xymin = np.min([xmin, ymin])
    xymax = np.max([xmax, ymax])-xymin

    x1 = x1 - xymin
    y1 = y1 - xymin
    x2 = x2 - xymin
    y2 = y2 - xymin

    Lbox = np.array([xymax, xymax])

    if min_size is not None:
        min_size = np.atleast_1d(min_size)
        if np.any(Lbox < min_size):
            Lbox[(Lbox < min_size)] = min_size[(Lbox < min_size)]

    return x1, y1, x2, y2, Lbox


def _set_approximate_cell_sizes(approx_cell1_size, approx_cell2_size, period):
    """
    process the approximate cell size parameters.
    If either is set to None, apply default settings.
    """

    #################################################
    # Set the approximate cell sizes of the trees
    if approx_cell1_size is None:
        approx_cell1_size = period/10.0
    else:
        approx_cell1_size = np.atleast_1d(approx_cell1_size)
        try:
            assert len(approx_cell1_size) == 3
            assert type(approx_cell1_size) is np.ndarray
            assert approx_cell1_size.ndim == 1
        except AssertionError:
            msg = ("Input ``approx_cell1_size`` must be a length-3 sequence")
            raise ValueError(msg)

    if approx_cell2_size is None:
        approx_cell2_size = copy(approx_cell1_size)
    else:
        approx_cell2_size = np.atleast_1d(approx_cell2_size)
        try:
            assert len(approx_cell2_size) == 3
            assert type(approx_cell2_size) is np.ndarray
            assert approx_cell2_size.ndim == 1
        except AssertionError:
            msg = ("Input ``approx_cell2_size`` must be a length-3 sequence")
            raise ValueError(msg)

    return approx_cell1_size, approx_cell2_size


def _set_approximate_2d_cell_sizes(approx_cell1_size, approx_cell2_size, period):
    """
    process the approximate cell size parameters.
    If either is set to None, apply default settings.
    """

    #################################################
    # Set the approximate cell sizes of the trees
    if approx_cell1_size is None:
        approx_cell1_size = period/10.0
    else:
        approx_cell1_size = np.atleast_1d(approx_cell1_size)
        try:
            assert len(approx_cell1_size) == 2
            assert type(approx_cell1_size) is np.ndarray
            assert approx_cell1_size.ndim == 1
        except AssertionError:
            msg = ("Input ``approx_cell1_size`` must be a length-3 sequence")
            raise ValueError(msg)

    if approx_cell2_size is None:
        approx_cell2_size = copy(approx_cell1_size)
    else:
        approx_cell2_size = np.atleast_1d(approx_cell2_size)
        try:
            assert len(approx_cell2_size) == 2
            assert type(approx_cell2_size) is np.ndarray
            assert approx_cell2_size.ndim == 1
        except AssertionError:
            msg = ("Input ``approx_cell2_size`` must be a length-3 sequence")
            raise ValueError(msg)

    return approx_cell1_size, approx_cell2_size


def _cell1_parallelization_indices(ncells, num_threads):
    """ Return a list of tuples that will be passed to multiprocessing.pool.map
    to count pairs in parallel. Each tuple has two entries storing the first and last
    cell_id that will be looped over in the outermost loop in the pair-counting engine.

    Parameters
    -----------
    ncells : int
        Total number of cells in the 3d mesh

    num_threads : int
        Number of cores requested to perform the pair-counting in parallel

    Returns
    -------
    num_threads : int
        Number of threads to use when counting pairs. Only differs from the
        input value for the case where the input num_threads > ncells

    list_of_tuples : list
        List of two-element tuples containing the first and last values of icell1
        that will be looped over in the outermost loop of the pair-counters.

    Notes
    ------
    Care is taken to avoid the problem of potentially having more threads available than cells.
    In the serial case, the returned list of tuples is a one-element list containing (0, ncells).
    If there are two cores available, cell1_tuples = [(0, ncells/2), (ncells/2, ncells)]

    """
    if num_threads == 1:
        return 1, [(0, ncells)]
    elif num_threads > ncells:
        return ncells, [(a, a+1) for a in np.arange(ncells)]
    else:
        list_with_possibly_empty_arrays = np.array_split(np.arange(ncells), num_threads)
        list_of_nonempty_arrays = [a for a in list_with_possibly_empty_arrays if len(a) > 0]
        list_of_tuples = [(x[0], x[0] + len(x)) for x in list_of_nonempty_arrays]
        return num_threads, list_of_tuples


def _enforce_maximum_search_length(search_length, period=None):
    """ The `~halotools.mock_observables.pair_counters.RectangularDoubleMesh`
    algorithm requires that the search length cannot exceed period/3 in any dimension.

    Parameters
    -----------
    search_length : float or len(period)-sequence
        Maximum search length over which pairs will be searched for.

    period : float or sequence, optional
        Periodicity of the simulation box. Default is None, in which case
        box will be assumed to be non-periodic.
        If a sequence is passed, the input ``search_length`` will be required
        to be less 1/3 of the smallest element of the sequence.
    """
    search_length = np.atleast_1d(search_length)
    if period is None:
        period = np.zeros_like(search_length) + np.inf
    period = np.atleast_1d(period)

    try:
        assert np.all(search_length < period/3.)
    except AssertionError:
        max_search_fraction = np.max(period - search_length)
        msg = ("The search algorithm used by the function you called \n"
            "does not permit you to look for pairs separated by values \n"
            "exceeding a fraction of Lbox/3. in any dimension.\n"
            "Your function call would require searching for pairs separated by a distance of {0:.2f}*Lbox.\n"
            "Either decrease your search length or use a larger simulation.")
        raise ValueError(msg.format(max_search_fraction))
