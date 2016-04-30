""" Module containing the `~halotools.mock_observables.npairs_s_mu` function 
used to count pairs as a function of separation. 
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np 
import multiprocessing
from functools import partial 

__author__ = ('Andrew Hearin', 'Duncan Campbell')

from .rectangular_mesh import RectangularDoubleMesh
from .mesh_helpers import _set_approximate_cell_sizes, _cell1_parallelization_indices
from .cpairs import npairs_s_mu_engine
from .npairs_3d import _npairs_3d_process_args
from ...utils.array_utils import convert_to_ndarray, array_is_monotonic

__all__ = ('npairs_s_mu', )

def npairs_s_mu(data1, data2, s_bins, mu_bins, period = None, 
    verbose = False, num_threads = 1, approx_cell1_size = None, approx_cell2_size = None):
    """
    Function counts the number of pairs of points separated by less than
    radial separation, *s,* and :math:`\\mu\\equiv\\sin(\\theta_{\\rm los})`,
    where :math:`\\theta_{\\rm los}` is the line-of-sight angle
    between points and :math:`s^2 = r_{\\rm parallel}^2 + r_{\\rm perp}^2`.

    Note that if data1 == data2 that the
    `~halotools.mock_observables.npairs_s_mu` function double-counts pairs.
    If your science application requires data1==data2 inputs and also pairs
    to not be double-counted, simply divide the final counts by 2.

    A common variation of pair-counting calculations is to count pairs with
    separations *between* two different distances *r1* and *r2*. You can retrieve
    this information from the `~halotools.mock_observables.npairs_s_mu`
    by taking `numpy.diff` of the returned array.

    See Notes section for further clarification.

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

    s_bins : array_like
        numpy array of boundaries defining the radial bins in which pairs are counted.

    mu_bins : array_like
        numpy array of boundaries defining bins in :math:`\\sin(\\theta_{\\rm los})`
        in which the pairs are counted in.
        Note that using the sine is not common convention for
        calculating the two point correlation function (see notes).

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
        which will result in reasonable performance for most use-cases.
        Performance can vary sensitively with this parameter, so it is highly
        recommended that you experiment with it when carrying out
        performance-critical calculations.

    approx_cell2_size : array_like, optional
        See comments for ``approx_cell1_size``.

    Returns
    -------
    num_pairs : array of length len(rbins)
        number of pairs

    Notes
    ------
    The quantity :math:`\\mu` is defined as the :math:`\\sin(\\theta_{\\rm los})`
    and not the conventional :math:`\\cos(\\theta_{\\rm los})`. This is
    because the pair counter has been optimized under the assumption that its
    separation variable (in this case, :math:`\\mu`) *increases*
    as :math:`\\theta_{\\rm los})` increases.

    One final point of clarification concerning double-counting may be in order.
    Suppose data1==data2 and rbins[0]==0. Then the returned value for this bin
    will be len(data1), since each data1 point has distance 0 from itself.

    Returns
    -------
    N_pairs : array_like
        2-d array of length *Num_rp_bins x Num_pi_bins* storing the pair counts in each bin.

    Examples
    --------
    For demonstration purposes we create randomly distributed sets of points within a
    periodic unit cube.

    >>> Npts1, Npts2, Lbox = 1e3, 1e3, 200.
    >>> period = [Lbox, Lbox, Lbox]
    >>> s_bins = np.logspace(-1, 1.25, 15)
    >>> mu_bins = np.linspace(-0.5, 0.5)

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

    >>> result = npairs_s_mu(data1, data2, s_bins, mu_bins, period = period)
    """

    ### Process the inputs with the helper function
    result = _npairs_3d_process_args(data1, data2, s_bins, period,
            verbose, num_threads, approx_cell1_size, approx_cell2_size)
    x1in, y1in, z1in, x2in, y2in, z2in = result[0:6]
    s_bins, period, num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[6:]
    xperiod, yperiod, zperiod = period 

    rmax = np.max(s_bins)

    #process mu_bins parameter separately
    mu_bins = convert_to_ndarray(mu_bins)
    try:
        assert mu_bins.ndim == 1
        assert len(mu_bins) > 1
        if len(mu_bins) > 2:
            assert array_is_monotonic(mu_bins, strict = True) == 1
    except AssertionError:
        msg = ("\n Input `mu_bins` must be a monotonically increasing \n"
               "1D array with at least two entries")
        raise ValueError(msg)

    search_xlength, search_ylength, search_zlength = rmax, rmax, rmax 

    ### Compute the estimates for the cell sizes
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
    engine = partial(npairs_s_mu_engine, 
        double_mesh, data1[:,0], data1[:,1], data1[:,2], 
        data2[:,0], data2[:,1], data2[:,2], s_bins, mu_bins)

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

