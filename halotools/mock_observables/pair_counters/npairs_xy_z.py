""" Module containing the `~halotools.mock_observables.npairs_xy_z` function
used to count pairs as a function of separation.
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import multiprocessing
from functools import partial

from .rectangular_mesh import RectangularDoubleMesh
from .mesh_helpers import (_set_approximate_cell_sizes, _enclose_in_box,
    _cell1_parallelization_indices)
from .cpairs import npairs_xy_z_engine
from ...utils.array_utils import array_is_monotonic, custom_len

__author__ = ('Andrew Hearin', 'Duncan Campbell')

__all__ = ('npairs_xy_z', )


def npairs_xy_z(sample1, sample2, rp_bins, pi_bins, period=None,
        num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    """
    Function counts the number of pairs of points with separation in the xy-plane
    less than the input ``rp_bins`` and separation in the z-dimension less than
    the input ``pi_bins``.

    Note that if sample1 == sample2 that the
    `~halotools.mock_observables.npairs_xy_z` function double-counts pairs.
    If your science application requires sample1==sample2 inputs and also pairs
    to not be double-counted, simply divide the final counts by 2.

    A common variation of pair-counting calculations is to count pairs with
    separations *between* two different distances *r1* and *r2*. You can retrieve
    this information from the `~halotools.mock_observables.npairs_xy_z`
    by taking `numpy.diff` of the returned array.

    Parameters
    ----------
    sample1 : array_like
        Numpy array of shape (Npts1, 3) containing 3-D positions of points.
        See the :ref:`mock_obs_pos_formatting` documentation page, or the
        Examples section below, for instructions on how to transform
        your coordinate position arrays into the
        format accepted by the ``sample1`` and ``sample2`` arguments.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    sample2 : array_like
        Numpy array of shape (Npts2, 3) containing 3-D positions of points.
        Should be identical to sample1 for cases of auto-sample pair counts.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    rp_bins : array_like
        array of boundaries defining the radial bins perpendicular to the LOS in which
        pairs are counted.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    pi_bins : array_like
        array of boundaries defining the p radial bins parallel to the LOS in which
        pairs are counted.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions
        in each dimension. If you instead provide a single scalar, Lbox,
        period is assumed to be the same in all Cartesian directions.
        If set to None (the default option), PBCs are set to infinity.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    num_threads : int, optional
        Number of threads to use in calculation, where parallelization is performed
        using the python ``multiprocessing`` module. Default is 1 for a purely serial
        calculation, in which case a multiprocessing Pool object will
        never be instantiated. A string 'max' may be used to indicate that
        the pair counters should use all available cores on the machine.

    approx_cell1_size : array_like, optional
        Length-3 array serving as a guess for the optimal manner by how points
        will be apportioned into subvolumes of the simulation box.
        The optimum choice unavoidably depends on the specs of your machine.
        Default choice is to use Lbox/10 in each dimension,
        which will return reasonable result performance for most use-cases.
        Performance can vary sensitively with this parameter, so it is highly
        recommended that you experiment with this parameter when carrying out
        performance-critical calculations.

    approx_cell2_size : array_like, optional
        Analogous to ``approx_cell1_size``, but for sample2.  See comments for
        ``approx_cell1_size`` for details.

    Returns
    -------
    num_pairs : array_like
        Numpy array of length len(rp_bins) storing the numbers of pairs in the input bins.

    Examples
    --------
    For demonstration purposes we create randomly distributed sets of points within a
    periodic unit cube.

    >>> Npts1, Npts2, Lbox = 1000, 1000, 250.
    >>> period = [Lbox, Lbox, Lbox]
    >>> rp_bins = np.logspace(-1, 1.5, 15)
    >>> pi_bins = [20, 40, 60]

    >>> x1 = np.random.uniform(0, Lbox, Npts1)
    >>> y1 = np.random.uniform(0, Lbox, Npts1)
    >>> z1 = np.random.uniform(0, Lbox, Npts1)
    >>> x2 = np.random.uniform(0, Lbox, Npts2)
    >>> y2 = np.random.uniform(0, Lbox, Npts2)
    >>> z2 = np.random.uniform(0, Lbox, Npts2)

    We transform our *x, y, z* points into the array shape used by the pair-counter by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> sample1 = np.vstack([x1, y1, z1]).T
    >>> sample2 = np.vstack([x2, y2, z2]).T

    >>> result = npairs_xy_z(sample1, sample2, rp_bins, pi_bins, period = period)

    """

    # Process the inputs with the helper function
    result = _npairs_xy_z_process_args(sample1, sample2, rp_bins, pi_bins, period,
            num_threads, approx_cell1_size, approx_cell2_size)
    x1in, y1in, z1in, x2in, y2in, z2in = result[0:6]
    rp_bins, pi_bins, period, num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[6:]
    xperiod, yperiod, zperiod = period

    rp_max = np.max(rp_bins)
    pi_max = np.max(pi_bins)
    search_xlength, search_ylength, search_zlength = rp_max, rp_max, pi_max

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

    # # Create a function object that has a single argument, for parallelization purposes
    engine = partial(npairs_xy_z_engine,
        double_mesh, x1in, y1in, z1in, x2in, y2in, z2in, rp_bins, pi_bins)

    # # Calculate the cell1 indices that will be looped over by the engine
    num_threads, cell1_tuples = _cell1_parallelization_indices(
        double_mesh.mesh1.ncells, num_threads)

    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        result = pool.map(engine, cell1_tuples)
        counts = np.sum(np.array(result), axis=0)
        pool.close()
    else:
        counts = engine(cell1_tuples[0])

    return np.array(counts)


def _npairs_xy_z_process_args(sample1, sample2, rp_bins, pi_bins, period,
        num_threads, approx_cell1_size, approx_cell2_size):
    """
    """
    if num_threads is not 1:
        if num_threads == 'max':
            num_threads = multiprocessing.cpu_count()
        if not isinstance(num_threads, int):
            msg = "Input ``num_threads`` argument must be an integer or the string 'max'"
            raise ValueError(msg)

    # Passively enforce that we are working with ndarrays
    x1 = sample1[:, 0]
    y1 = sample1[:, 1]
    z1 = sample1[:, 2]
    x2 = sample2[:, 0]
    y2 = sample2[:, 1]
    z2 = sample2[:, 2]

    rp_bins = np.atleast_1d(rp_bins).astype('f8')
    try:
        assert rp_bins.ndim == 1
        assert len(rp_bins) > 1
        if len(rp_bins) > 2:
            assert array_is_monotonic(rp_bins, strict=True) == 1
    except AssertionError:
        msg = ("Input ``rp_bins`` must be a monotonically increasing 1D array "
            "with at least two entries")
        raise ValueError(msg)
    rp_max = np.max(rp_bins)

    pi_bins = np.atleast_1d(pi_bins).astype('f8')
    try:
        assert pi_bins.ndim == 1
        assert len(pi_bins) > 1
        if len(pi_bins) > 2:
            assert array_is_monotonic(pi_bins, strict=True) == 1
    except AssertionError:
        msg = "Input ``pi_bins`` must be a monotonically increasing 1D array with at least two entries"
        raise ValueError(msg)
    pi_max = np.max(pi_bins)

    # Set the boolean value for the PBCs variable
    if period is None:
        PBCs = False
        x1, y1, z1, x2, y2, z2, period = (
            _enclose_in_box(x1, y1, z1, x2, y2, z2,
                min_size=[rp_max*3.0, rp_max*3.0, pi_max*3.0]))
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

    try:
        assert pi_max > 0.
        min_required_pi_max = period[2]/3.
        assert pi_max < min_required_pi_max
    except:
        msg = ("Input ``pi_max`` = %.2f must be a "
            "positive scalar less than period[2]/3 = %.2f" % (pi_max, min_required_pi_max))
        raise ValueError(msg)

    if approx_cell1_size is None:
        approx_cell1_size = [rp_max, rp_max, pi_max]
    elif custom_len(approx_cell1_size) == 1:
        approx_cell1_size = [approx_cell1_size, approx_cell1_size, approx_cell1_size]
    if approx_cell2_size is None:
        approx_cell2_size = [rp_max, rp_max, pi_max]
    elif custom_len(approx_cell2_size) == 1:
        approx_cell2_size = [approx_cell2_size, approx_cell2_size, approx_cell2_size]

    return (x1, y1, z1, x2, y2, z2,
        rp_bins, pi_bins, period, num_threads, PBCs,
        approx_cell1_size, approx_cell2_size)
