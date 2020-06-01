""" Module containing the `~halotools.mock_observables.weighted_npairs_xy` function
used to count pairs as a function of separation.
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import multiprocessing
from functools import partial

from .engines import weighted_npairs_xy_engine

from ..pair_counters.rectangular_mesh_2d import RectangularDoubleMesh2D
from ..pair_counters.mesh_helpers import _set_approximate_2d_cell_sizes
from ..pair_counters.mesh_helpers import _enclose_in_square, _cell1_parallelization_indices

from ...utils.array_utils import array_is_monotonic, custom_len


__author__ = ('Andrew Hearin', )

__all__ = ('weighted_npairs_xy', )


def weighted_npairs_xy(sample1, sample2, sample2_mass, rp_bins,
        period=None, num_threads=1,
        approx_cell1_size=None, approx_cell2_size=None):
    r"""
    Function counts the total mass of ``sample2`` enclosed within
    z-axis-aligned cylinders centered at ``sample1`` points.

    Parameters
    ----------
    sample1 : array_like
        Array of shape (Npts1, 2) containing XY positions of points.
        See the :ref:`mock_obs_pos_formatting` documentation page, or the
        Examples section below, for instructions on how to transform
        your coordinate position arrays into the
        format accepted by the ``sample1`` and ``sample2`` arguments.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    sample2 : array_like
        Array of shape (Npts2, 2) containing XY positions of points.

    sample2_mass : array_like
        Array of shape (Npts2, ) containing the masses of the ``sample2`` points.

        Note that you may get more numerically stable results
        if you are able to normalize your masses to order-unity values.

    rp_bins : array_like
        array of xy-distances defining the radial bins in which pairs are counted.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    period : array_like, optional
        Length-2 sequence defining the periodic boundary conditions
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
        Length-2 array serving as a guess for the optimal manner by how points
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
    enclosed_mass : array_like
        Numpy array of length len(rp_bins) storing the total mass enclosed
        within the input cylinders.

    Examples
    --------
    For demonstration purposes we create randomly distributed sets of points within a
    periodic unit cube.

    >>> Npts1, Npts2, Lbox = 1000, 1000, 450.
    >>> period = [Lbox, Lbox]
    >>> rp_bins = np.logspace(-1, 1.5, 15)

    >>> x1 = np.random.uniform(0, Lbox, Npts1)
    >>> y1 = np.random.uniform(0, Lbox, Npts1)
    >>> x2 = np.random.uniform(0, Lbox, Npts2)
    >>> y2 = np.random.uniform(0, Lbox, Npts2)
    >>> sample2_mass = np.random.uniform(0, 1, Npts2)

    We transform our *x, y* points into the array shape used by the pair-counter by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> sample1 = np.vstack([x1, y1]).T
    >>> sample2 = np.vstack([x2, y2]).T

    >>> result = weighted_npairs_xy(sample1, sample2, sample2_mass, rp_bins, period=period)

    """
    # Process the inputs with the helper function
    result = _weighted_npairs_xy_process_args(sample1, sample2, sample2_mass,
            rp_bins, period, num_threads, approx_cell1_size, approx_cell2_size)
    x1in, y1in, x2in, y2in, w2in = result[0:5]
    rp_bins, period, num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[5:]
    xperiod, yperiod = period

    rp_max = np.max(rp_bins)
    search_xlength, search_ylength = rp_max, rp_max

    # Compute the estimates for the cell sizes
    approx_cell1_size, approx_cell2_size = (
        _set_approximate_2d_cell_sizes(approx_cell1_size, approx_cell2_size, period)
        )
    approx_x1cell_size, approx_y1cell_size = approx_cell1_size
    approx_x2cell_size, approx_y2cell_size = approx_cell2_size

    # Build the rectangular mesh
    double_mesh = RectangularDoubleMesh2D(x1in, y1in, x2in, y2in,
        approx_x1cell_size, approx_y1cell_size,
        approx_x2cell_size, approx_y2cell_size,
        search_xlength, search_ylength, xperiod, yperiod, PBCs)

    # # Create a function object that has a single argument, for parallelization purposes
    counting_engine = partial(weighted_npairs_xy_engine,
        double_mesh, x1in, y1in, x2in, y2in, w2in, rp_bins)

    # # Calculate the cell1 indices that will be looped over by the engine
    num_threads, cell1_tuples = _cell1_parallelization_indices(
        double_mesh.mesh1.ncells, num_threads)

    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        result = pool.map(counting_engine, cell1_tuples)
        weighted_counts = np.sum(np.array(result), axis=0)
        pool.close()
    else:
        weighted_counts = counting_engine(cell1_tuples[0])

    return np.array(weighted_counts)


def _weighted_npairs_xy_process_args(sample1, sample2, w2, rp_bins, period,
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
    x2 = sample2[:, 0]
    y2 = sample2[:, 1]

    assert w2.shape[0] == sample2.shape[0]

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

    # Set the boolean value for the PBCs variable
    if period is None:
        PBCs = False
        x1, y1, x2, y2, period = (
            _enclose_in_square(x1, y1, x2, y2,
                min_size=[rp_max*3.0, rp_max*3.0]))
    else:
        PBCs = True
        period = np.atleast_1d(period).astype(float)
        if len(period) == 1:
            period = np.array([period[0]]*2)
        try:
            assert np.all(period < np.inf)
            assert np.all(period > 0)
        except AssertionError:
            msg = "Input ``period`` must be a bounded positive number in all dimensions"
            raise ValueError(msg)

    if approx_cell1_size is None:
        approx_cell1_size = [rp_max, rp_max]
    elif custom_len(approx_cell1_size) == 1:
        approx_cell1_size = [approx_cell1_size, approx_cell1_size]
    if approx_cell2_size is None:
        approx_cell2_size = [rp_max, rp_max]
    elif custom_len(approx_cell2_size) == 1:
        approx_cell2_size = [approx_cell2_size, approx_cell2_size]

    return (x1, y1, x2, y2, w2, rp_bins, period, num_threads, PBCs,
        approx_cell1_size, approx_cell2_size)
