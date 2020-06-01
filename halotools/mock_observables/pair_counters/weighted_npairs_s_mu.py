r""" Module containing the `~halotools.mock_observables.weighted_npairs_s_mu` function
used to count pairs as a function of separation.
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import multiprocessing
from functools import partial

from .rectangular_mesh import RectangularDoubleMesh
from .mesh_helpers import _set_approximate_cell_sizes, _cell1_parallelization_indices
from .cpairs import weighted_npairs_s_mu_engine
from .npairs_3d import _npairs_3d_process_args
from ...utils.array_utils import array_is_monotonic

__author__ = ('Andrew Hearin', 'Duncan Campbell')

__all__ = ('weighted_npairs_s_mu', )


def weighted_npairs_s_mu(sample1, sample2, weights1, weights2, s_bins, mu_bins, period=None,
        num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    r"""
    Function performs a *weighted* count of the number of pairs of points separated by less than
    distance r:math:`s`, given by ``s_bins`` along the line-of-sight (LOS), and
    angular distance, :math:`\mu\equiv\cos(\theta_{\rm los})`, given by ``mu_bins``,
    where :math:`\theta_{\rm los}` is the angle between :math:`\vec{s}` and
    the (LOS).

    The first two dimensions (x, y) define the plane for perpendicular distances.
    The third dimension (z) defines the LOS.  i.e. x,y positions are on
    the plane of the sky, and z is the radial distance coordinate.  This is the 'distant
    observer' approximation.

    A common variation of pair-counting calculations is to count pairs with
    separations *between* two different distances, e.g. [s1 ,s2] and [mu1, mu2].
    You can retrieve this information from `~halotools.mock_observables.weighted_npairs_s_mu`
    by taking `numpy.diff` of the returned array along each axis.

    See Notes section for further clarification.

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

    weights1 : array_like
        Numpy array of shape (Npts1, ) containing weights used to weight the pair counts.

    weights2 : array_like
        Numpy array of shape (Npts2, ) containing weights used to weight the pair counts.

    s_bins : array_like
        numpy array of shape (num_s_bin_edges, ) storing the :math:`s`
        boundaries defining the bins in which pairs are counted.

    mu_bins : array_like
        numpy array of shape (num_mu_bin_edges, ) storing the
        :math:`\cos(\theta_{\rm LOS})` boundaries defining the bins in
        which pairs are counted. All values must be between [0,1].

    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions
        in each dimension. If you instead provide a single scalar, Lbox,
        period is assumed to be the same in all Cartesian directions.
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
    num_pairs : array of shape (num_s_bin_edges, num_mu_bin_edges) storing the
        number of pairs separated by less than (s, mu)

    weighted_num_pairs : array of shape (num_s_bin_edges, num_mu_bin_edges) storing the
        weighted number of pairs separated by less than (s, mu). Each pair is
        weighted by `w1*w2`.

    Notes
    -----
    Let :math:`\vec{s}` be the radial vector connnecting two points.
    The magnitude, :math:`s`, is:

    .. math::
        s = \sqrt{r_{\parallel}^2+r_{\perp}^2},

    where :math:`r_{\parallel}` is the separation parallel to the LOS
    and :math:`r_{\perp}` is the separation perpednicular to the LOS.  :math:`\mu` is
    the cosine of the angle, :math:`\theta_{\rm LOS}`, between the LOS
    and :math:`\vec{s}`:

    .. math::
        \mu = \cos(\theta_{\rm LOS}) \equiv r_{\parallel}/s.

    Along the first dimension of ``num_pairs``, :math:`s` increases.
    Along the second dimension,  :math:`\mu` decreases,
    i.e. :math:`\theta_{\rm LOS}` increases.

    If sample1 == sample2 that the `~halotools.mock_observables.weighted_npairs_s_mu` function
    double-counts pairs. If your science application requires sample1==sample2 inputs
    and also pairs to not be double-counted, simply divide the final counts by 2.

    One final point of clarification concerning double-counting may be in order.
    Suppose sample1==sample2 and s_bins[0]==0. Then the returned value for this bin
    will be len(sample1), since each sample1 point has distance 0 from itself.

    Examples
    --------
    For demonstration purposes we create randomly distributed sets of points within a
    periodic unit cube.

    >>> Npts1, Npts2, Lbox = 1000, 1000, 200.
    >>> period = [Lbox, Lbox, Lbox]
    >>> s_bins = np.logspace(-1, 1.25, 15)
    >>> mu_bins = np.linspace(0, 1)

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
    >>> weights1 = np.random.rand(Npts1)
    >>> weights2 = np.random.rand(Npts2)

    >>> from halotools.mock_observables.pair_counters import weighted_npairs_s_mu
    >>> counts, weighted_counts = weighted_npairs_s_mu(sample1, sample2, weights1, weights2, s_bins, mu_bins, period=period)
    """

    # Process the inputs with the helper function
    result = _npairs_3d_process_args(sample1, sample2, s_bins, period,
            num_threads, approx_cell1_size, approx_cell2_size)
    x1in, y1in, z1in, x2in, y2in, z2in = result[0:6]
    s_bins, period, num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[6:]
    xperiod, yperiod, zperiod = period

    weights1 = np.atleast_1d(weights1)
    weights2 = np.atleast_1d(weights2)

    assert weights1.shape == x1in.shape, "``weights1`` should have shape ({0}, )".format(len(x1in))
    assert weights2.shape == x2in.shape, "``weights2`` should have shape ({0}, )".format(len(x2in))

    rmax = np.max(s_bins)

    # process mu_bins parameter separately
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
    # convert to mu=sin(theta_los) binning used by the cython engine.
    mu_bins_prime = np.sin(np.arccos(mu_bins))
    mu_bins_prime = np.sort(mu_bins_prime)
    # increasing mu_prime now corresponds to increasing theta_LOS

    search_xlength, search_ylength, search_zlength = rmax, rmax, rmax

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
    engine = partial(weighted_npairs_s_mu_engine,
        double_mesh, x1in, y1in, z1in, x2in, y2in, z2in, weights1, weights2, s_bins, mu_bins_prime)

    # Calculate the cell1 indices that will be looped over by the engine
    num_threads, cell1_tuples = _cell1_parallelization_indices(
        double_mesh.mesh1.ncells, num_threads)

    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        result = pool.map(engine, cell1_tuples)
        result = np.array(result)
        counts = result[:,0]
        weighted_counts = result[:,1]
        counts = np.sum(np.array(counts), axis=0)
        weighted_counts = np.sum(np.array(weighted_counts), axis=0)
        pool.close()
    else:
        result = engine(cell1_tuples[0])
        counts = result[0]
        weighted_counts = result[1]

    return np.array(counts), np.array(weighted_counts)
