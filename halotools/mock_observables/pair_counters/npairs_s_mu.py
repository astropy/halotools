""" Module containing the `~halotools.mock_observables.npairs_s_mu` function
used to count pairs as a function of separation.
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import multiprocessing
from functools import partial

from .rectangular_mesh import RectangularDoubleMesh
from .mesh_helpers import _set_approximate_cell_sizes, _cell1_parallelization_indices
from .cpairs import npairs_s_mu_engine
from .npairs_3d import _npairs_3d_process_args
from ...utils.array_utils import convert_to_ndarray, array_is_monotonic

__author__ = ('Andrew Hearin', 'Duncan Campbell')

__all__ = ('npairs_s_mu', )


def npairs_s_mu(sample1, sample2, s_bins, mu_bins, period=None,
        verbose=False, num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    """
    Function counts the number of pairs of points separated by less than
    radial separation, *s,* and :math:`\\mu\\equiv\\sin(\\theta_{\\rm los})`,
    where :math:`\\theta_{\\rm los}` is the line-of-sight angle
    between points and :math:`s^2 = r_{\\rm parallel}^2 + r_{\\rm perp}^2`.

    Note that if sample1 == sample2 that the
    `~halotools.mock_observables.npairs_s_mu` function double-counts pairs.
    If your science application requires sample1==sample2 inputs and also pairs
    to not be double-counted, simply divide the final counts by 2.

    A common variation of pair-counting calculations is to count pairs with
    separations *between* two different distances *r1* and *r2*. You can retrieve
    this information from the `~halotools.mock_observables.npairs_s_mu`
    by taking `numpy.diff` of the returned array.

    See Notes section for further clarification.

    Parameters
    ----------
    sample1 : array_like
        Npts1 x 3 numpy array containing 3-D positions of points.
        See the :ref:`mock_obs_pos_formatting` documentation page, or the
        Examples section below, for instructions on how to transform
        your coordinate position arrays into the
        format accepted by the ``sample1`` and ``sample2`` arguments.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    sample2 : array_like, optional
        Npts2 x 3 array containing 3-D positions of points.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    s_bins : array_like
        numpy array of :math:`s` boundaries defining the bins in which pairs are counted.

    mu_bins : array_like
        numpy array of :math:`\\cos(\\theta_{\\rm LOS})` boundaries defining the bins in
        which pairs are counted, and must be between [0,1].

        Note that using the sine is not common convention for
        calculating the two point correlation function (see notes).

    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions
        in each dimension. If you instead provide a single scalar, Lbox,
        period is assumed to be the same in all Cartesian directions.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    verbose : Boolean, optional
        If True, print out information and progress.

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
    Suppose sample1==sample2 and rbins[0]==0. Then the returned value for this bin
    will be len(sample1), since each sample1 point has distance 0 from itself.

    Returns
    -------
    N_pairs : array_like
        2-d array of length *Num_rp_bins x Num_pi_bins* storing the pair counts in each bin.

    Examples
    --------
    For demonstration purposes we create randomly distributed sets of points within a
    periodic unit cube.

    >>> Npts1, Npts2, Lbox = 1000, 1000, 200.
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

    >>> sample1 = np.vstack([x1, y1, z1]).T
    >>> sample2 = np.vstack([x2, y2, z2]).T

    >>> result = npairs_s_mu(sample1, sample2, s_bins, mu_bins, period = period)
    """

    ### Process the inputs with the helper function
    result = _npairs_3d_process_args(sample1, sample2, s_bins, period,
            verbose, num_threads, approx_cell1_size, approx_cell2_size)
    x1in, y1in, z1in, x2in, y2in, z2in = result[0:6]
    s_bins, period, num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[6:]
    xperiod, yperiod, zperiod = period

    rmax = np.max(s_bins)

    #process mu_bins parameter separately
    mu_bins = np.atleast_1d(mu_bins)
    try:
        assert mu_bins.ndim == 1
        assert len(mu_bins) > 1
        if len(mu_bins) > 2:
            assert array_is_monotonic(mu_bins, strict=True) == 1
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
        double_mesh, x1in, y1in, z1in, x2in, y2in, z2in, s_bins, mu_bins)

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
