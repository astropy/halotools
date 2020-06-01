""" Module containing the `~halotools.mock_observables.pairwise_distance_3d` function
used to find pairs and their separation distance.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import multiprocessing
from functools import partial
from scipy.sparse import coo_matrix

from .pairwise_distance_3d import _get_r_max
from .rectangular_mesh import RectangularDoubleMesh
from .mesh_helpers import _set_approximate_cell_sizes, _enclose_in_box, _cell1_parallelization_indices
from .cpairs import pairwise_distance_xy_z_engine

from ...utils.array_utils import custom_len

__all__ = ('pairwise_distance_xy_z', )
__author__ = ('Andrew Hearin', 'Duncan Campbell')


def pairwise_distance_xy_z(data1, data2, rp_max, pi_max, period=None,
        verbose=False, num_threads=1,
        approx_cell1_size=None, approx_cell2_size=None):
    """
    Function returns pairs of points separated by
    a xy-projected distance smaller than or equal to the input ``rp_max`` and z distance ``pi_max``.

    Note that if data1 == data2 that the
    `~halotools.mock_observables.pairwise_distance_xy_z` function double-counts pairs.

    Parameters
    ----------
    data1 : array_like
        N1 by 3 numpy array of 3-dimensional positions.
        Values of each dimension should be between zero and the corresponding dimension
        of the input period.

    data2 : array_like
        N2 by 3 numpy array of 3-dimensional positions.
        Values of each dimension should be between zero and the corresponding dimension
        of the input period.

    rp_max : array_like
        radius of the cylinder to search for neighbors around galaxies in ``data1``.
        If a single float is given, ``rp_max`` is assumed to be the same for each galaxy in
        ``data1``. You may optionally pass in an array of length *Npts1*, in which case
        each point in ``data1`` will have its own individual neighbor-search projected radius.

        Length units assumed to be in Mpc/h, here and throughout Halotools.

    pi_max : array_like
        Half-length of cylinder to search for neighbors around galaxies in ``data1``.
        If a single float is given, ``pi_max`` is assumed to be the same for each galaxy in
        ``data1``. You may optionally pass in an array of length *Npts1*, in which case
        each point in ``data1`` will have its own individual neighbor-search cylinder half-length.

        Length units assumed to be in Mpc/h, here and throughout Halotools.

    period : array_like, optional
        Length-3 array defining the periodic boundary conditions.
        If only one number is specified, the enclosing volume is assumed to
        be a periodic cube (by far the most common case).
        If period is set to None, the default option,
        PBCs are set to infinity.

    verbose : Boolean, optional
        If True, print out information and progress.

    num_threads : int, optional
        Number of CPU cores to use in the pair counting.
        If ``num_threads`` is set to the string 'max', use all available cores.
        Default is 1 thread for a serial calculation that
        does not open a multiprocessing pool.

    approx_cell1_size : array_like, optional
        Length-3 array serving as a guess for the optimal manner by which
        the `~halotools.mock_observables.pair_counters.RectangularDoubleMesh`
        will apportion the ``data`` points into subvolumes of the simulation box.
        The optimum choice unavoidably depends on the specs of your machine.
        Default choice is to use 1/10 of the box size in each dimension,
        which will return reasonable result performance for most use-cases.
        Performance can vary sensitively with this parameter, so it is highly
        recommended that you experiment with this parameter when carrying out
        performance-critical calculations.

    approx_cell2_size : array_like, optional
        See comments for ``approx_cell1_size``.

    Returns
    -------
     distance : `~scipy.sparse.coo_matrix`
        sparse matrix in COO format containing distances
        between the ith entry in ``data1`` and jth in ``data2``.

    Examples
    --------
    For demonstration purposes we create randomly distributed sets of points within a
    periodic unit cube.

    >>> Npts1, Npts2, Lbox = 1000, 1000, 250.
    >>> period = [Lbox, Lbox, Lbox]
    >>> rp_max = 1.0
    >>> pi_max = 2.0

    >>> x1 = np.random.uniform(0, Lbox, Npts1)
    >>> y1 = np.random.uniform(0, Lbox, Npts1)
    >>> z1 = np.random.uniform(0, Lbox, Npts1)
    >>> x2 = np.random.uniform(0, Lbox, Npts2)
    >>> y2 = np.random.uniform(0, Lbox, Npts2)
    >>> z2 = np.random.uniform(0, Lbox, Npts2)

    We transform our *x, y, z* points into the array shape used by the pair-counter by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> data1 = np.vstack([x1, y1, z1]).T
    >>> data2 = np.vstack([x2, y2, z2]).T

    >>> perp_dist_matrix, para_dist_matrix = pairwise_distance_xy_z(data1, data2, rp_max, pi_max, period = period)

    """

    # Process the inputs with the helper function
    result = _pairwise_distance_xy_z_process_args(data1, data2, rp_max, pi_max, period,
            verbose, num_threads, approx_cell1_size, approx_cell2_size)
    x1in, y1in, z1in, x2in, y2in, z2in = result[0:6]
    rp_max, max_rp_max, pi_max, max_pi_max, period, num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[6:]
    xperiod, yperiod, zperiod = period

    search_xlength, search_ylength, search_zlength = max_rp_max, max_rp_max, max_pi_max

    # Compute the estimates for the cell sizes
    approx_cell1_size, approx_cell2_size = (
        _set_approximate_cell_sizes(approx_cell1_size, approx_cell2_size, period)
        )
    approx_x1cell_size, approx_y1cell_size, approx_z1cell_size = approx_cell1_size
    approx_x2cell_size, approx_y2cell_size, approx_z2cell_size = approx_cell2_size

    # Build the rectangular mesh
    double_mesh = RectangularDoubleMesh(x1in, y1in, z1in, x2in, y2in, z2in,
        approx_x1cell_size, approx_y1cell_size, approx_z1cell_size,
        approx_x2cell_size, approx_y2cell_size, approx_z2cell_size,
        search_xlength, search_ylength, search_zlength, xperiod, yperiod, zperiod, PBCs)

    # Create a function object that has a single argument, for parallelization purposes
    engine = partial(pairwise_distance_xy_z_engine,
        double_mesh, x1in, y1in, z1in, x2in, y2in, z2in, rp_max, pi_max)

    # Calculate the cell1 indices that will be looped over by the engine
    num_threads, cell1_tuples = _cell1_parallelization_indices(
        double_mesh.mesh1.ncells, num_threads)

    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        result = pool.map(engine, cell1_tuples)
        pool.close()
    else:
        result = [engine(cell1_tuples[0])]

    # unpack result
    d_perp = np.zeros((0,), dtype='float')
    d_para = np.zeros((0,), dtype='float')
    i_inds = np.zeros((0,), dtype='int')
    j_inds = np.zeros((0,), dtype='int')

    # unpack the results
    for i in range(len(result)):
        d_perp = np.append(d_perp, result[i][0])
        d_para = np.append(d_para, result[i][1])
        i_inds = np.append(i_inds, result[i][2])
        j_inds = np.append(j_inds, result[i][3])

    return (coo_matrix((d_perp, (i_inds, j_inds)), shape=(len(data1), len(data2))),
        coo_matrix((d_para, (i_inds, j_inds)), shape=(len(data1), len(data2))))


def _pairwise_distance_xy_z_process_args(data1, data2, rp_max, pi_max, period,
        verbose, num_threads, approx_cell1_size, approx_cell2_size):
    """
    helper function to process arguments for `~halotools.mock_observables.pairwise_distance_3d function.
    """
    if num_threads is not 1:
        if num_threads == 'max':
            num_threads = multiprocessing.cpu_count()
        if not isinstance(num_threads, int):
            msg = "Input ``num_threads`` argument must be an integer or the string 'max'"
            raise ValueError(msg)

    # Passively enforce that we are working with ndarrays
    x1 = data1[:, 0]
    y1 = data1[:, 1]
    z1 = data1[:, 2]
    x2 = data2[:, 0]
    y2 = data2[:, 1]
    z2 = data2[:, 2]

    rp_max = _get_r_max(data1, rp_max)
    pi_max = _get_r_max(data1, pi_max)
    max_rp_max = np.amax(rp_max)
    max_pi_max = np.amax(pi_max)

    # Set the boolean value for the PBCs variable
    if period is None:
        PBCs = False
        x1, y1, z1, x2, y2, z2, period = (
            _enclose_in_box(x1, y1, z1, x2, y2, z2,
                min_size=[max_rp_max*3.0, max_rp_max*3.0, max_pi_max*3.0]))
    else:
        PBCs = True
        period = np.atleast_1d(period).astype(float)
        if len(period) == 1:
            period = np.array([period[0]]*3)
        try:
            assert np.all(period < np.inf)
            assert np.all(period > 0)
        except AssertionError:
            msg = "Input ``period`` must be a bounded positive number in all dimensions"
            raise ValueError(msg)

    if approx_cell1_size is None:
        approx_cell1_size = [max_rp_max, max_rp_max, max_pi_max]
    elif custom_len(approx_cell1_size) == 1:
        approx_cell1_size = [approx_cell1_size, approx_cell1_size, approx_cell1_size]
    if approx_cell2_size is None:
        approx_cell2_size = [max_rp_max, max_rp_max, max_pi_max]
    elif custom_len(approx_cell2_size) == 1:
        approx_cell2_size = [approx_cell2_size, approx_cell2_size, approx_cell2_size]

    return (x1, y1, z1, x2, y2, z2,
        rp_max, max_rp_max, pi_max, max_pi_max, period, num_threads, PBCs,
        approx_cell1_size, approx_cell2_size)
