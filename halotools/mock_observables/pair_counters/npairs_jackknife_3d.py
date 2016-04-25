""" Module containing the `~halotools.mock_observables.npairs_3d` function 
used to count pairs as a function of separation. 
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np 
import multiprocessing
from functools import partial 
from warnings import warn 

__author__ = ('Andrew Hearin', 'Duncan Campbell')

from .rectangular_mesh import RectangularDoubleMesh
from .mesh_helpers import _set_approximate_cell_sizes, _cell1_parallelization_indices
from .cpairs import npairs_jackknife_3d_engine
from .npairs_3d import _npairs_3d_process_args

from ...custom_exceptions import HalotoolsError

__all__ = ('npairs_jackknife_3d', )

def npairs_jackknife_3d(data1, data2, rbins, period=None, weights1=None, weights2=None,
    jtags1=None, jtags2=None, N_samples=0, verbose=False, num_threads=1,
    approx_cell1_size = None, approx_cell2_size = None):
    """
    Pair counter used to make jackknife error estimates of real-space pair counter
    `~halotools.mock_observables.pair_counters.npairs`.

    Parameters
    ----------
    data1 : array_like
        N1 by 3 numpy array of 3-dimensional positions.
        Values of each dimension should be between zero and the corresponding dimension
        of the input period.

    data2 : array_like
        N1 by 3 numpy array of 3-dimensional positions.
        Values of each dimension should be between zero and the corresponding dimension
        of the input period.

    rbins : array_like
        Boundaries defining the bins in which pairs are counted.

    period : array_like, optional
        Length-3 array defining the periodic boundary conditions.
        If only one number is specified, the enclosing volume is assumed to
        be a periodic cube (by far the most common case).
        If period is set to None, the default option,
        PBCs are set to infinity.

    weights1 : array_like, optional
        length N1 array containing weights used for weighted pair counts.

    weights2 : array_like, optional
        length N2 array containing weights used for weighted pair counts.

    jtags1 : array_like, optional
        length N1 array containing integer tags used to define jackknife sample
        membership. Tags are in the range [1, N_samples].
        The tag '0' is a reserved tag and should not be used.

    jtags2 : array_like, optional
        length N2 array containing integer tags used to define jackknife sample
        membership. Tags are in the range [1, N_samples].
        The tag '0' is a reserved tag and should not be used.

    N_samples : int, optional
        Total number of jackknife samples. All values of ``jtags1`` and ``jtags2``
        should be in the range [1, N_samples].

    verbose : Boolean, optional
        If True, print out information and progress.

    num_threads : int, optional
        Number of CPU cores to use in the pair counting.
        If ``num_threads`` is set to the string 'max', use all available cores.
        Default is 1 thread for a serial calculation that
        does not open a multiprocessing pool.

    approx_cell1_size : array_like, optional
        Length-3 array serving as a guess for the optimal manner by which
        the `~halotools.mock_observables.pair_counters.FlatRectanguloidDoubleTree`
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
    N_pairs : array_like
        Numpy array of shape (N_samples+1,len(rbins)).
        The sub-array N_pairs[0, :] stores numbers of pairs
        in the input bins for the entire sample.
        The sub-array N_pairs[i, :] stores numbers of pairs
        in the input bins for the :math:`i^{\\rm th}` jackknife sub-sample.

    Notes
    -----
    Jackknife weights are calculated using a weighting function.

    If both points are outside the sample, the weighting function returns 0.
    If both points are inside the sample, the weighting function returns (w1 * w2)
    If one point is inside, and the other is outside, the weighting function returns (w1 * w2)/2

    Examples
    --------
    For demonstration purposes we create randomly distributed sets of points within a
    periodic unit cube.

    >>> Npts1, Npts2, Lbox = 1e3, 1e3, 250.
    >>> period = [Lbox, Lbox, Lbox]
    >>> rbins = np.logspace(-1, 1.5, 15)

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

    Ordinarily, you would create ``jtags`` for the points by properly subdivide
    the points into spatial sub-volumes. For illustration purposes, we'll simply
    use randomly assigned sub-volumes as this has no impact on the calling signature:

    >>> N_samples = 10
    >>> jtags1 = np.random.random_integers(1, N_samples, Npts1)
    >>> jtags2 = np.random.random_integers(1, N_samples, Npts2)

    >>> result = npairs_jackknife_3d(data1, data2, rbins, period = period, jtags1=jtags1, jtags2=jtags2, N_samples = N_samples)

    """
    ### Process the inputs with the helper function
    result = _npairs_3d_process_args(data1, data2, rbins, period,
            verbose, num_threads, approx_cell1_size, approx_cell2_size)
    x1in, y1in, z1in, x2in, y2in, z2in = result[0:6]
    rbins, period, num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[6:]
    xperiod, yperiod, zperiod = period 

    rmax = np.max(rbins)
    search_xlength, search_ylength, search_zlength = rmax, rmax, rmax 

    # Process the input weights and jackknife-tags with the helper function
    weights1, weights2, jtags1, jtags2 = (
        _npairs_jackknife_3d_process_weights_jtags(data1, data2,
            weights1, weights2, jtags1, jtags2, N_samples))

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

    # Create a function object that has a single argument, for parallelization purposes
    engine = partial(npairs_jackknife_3d_engine, 
        double_mesh, data1[:,0], data1[:,1], data1[:,2], 
        data2[:,0], data2[:,1], data2[:,2], 
        weights1, weights2, jtags1, jtags2, N_samples, rbins)

    # Calculate the cell1 indices that will be looped over by the engine
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


def _npairs_jackknife_3d_process_weights_jtags(data1, data2, weights1, weights2, jtags1, jtags2, N_samples):
    """
    """

    #Process weights1 entry and check for consistency.
    if weights1 is None:
        weights1 = np.array([1.0]*np.shape(data1)[0], dtype=np.float64)
    else:
        weights1 = np.asarray(weights1).astype("float64")
        if np.shape(weights1)[0] != np.shape(data1)[0]:
            raise HalotoolsError("weights1 should have same len as data1")
    #Process weights2 entry and check for consistency.
    if weights2 is None:
        weights2 = np.array([1.0]*np.shape(data2)[0], dtype=np.float64)
    else:
        weights2 = np.asarray(weights2).astype("float64")
        if np.shape(weights2)[0] != np.shape(data2)[0]:
            raise HalotoolsError("weights2 should have same len as data2")
    
    #Process jtags_1 entry and check for consistency.
    if jtags1 is None:
        jtags1 = np.array([0]*np.shape(data1)[0], dtype=np.int)
    else:
        jtags1 = np.asarray(jtags1).astype("int")
        if np.shape(jtags1)[0] != np.shape(data1)[0]:
            raise HalotoolsError("jtags1 should have same len as data1")
    #Process jtags_2 entry and check for consistency.
    if jtags2 is None:
        jtags2 = np.array([0]*np.shape(data2)[0], dtype=np.int)
    else:
        jtags2 = np.asarray(jtags2).astype("int")
        if np.shape(jtags2)[0] != np.shape(data2)[0]:
            raise HalotoolsError("jtags2 should have same len as data2")
    
    #Check bounds of jackknife tags
    if np.min(jtags1) < 1: 
        raise HalotoolsError("jtags1 must be >= 1")
    if np.min(jtags2) < 1: 
        raise HalotoolsError("jtags2 must be >= 1")
    if np.max(jtags1) > N_samples: 
        raise HalotoolsError("jtags1 must be <= N_samples")
    if np.max(jtags2) > N_samples: 
        raise HalotoolsError("jtags2 must be <= N_samples")
    
    #throw warning if some tags do not exist
    if not np.array_equal(np.unique(jtags1),np.arange(1,N_samples+1)):
        warn("Warning: data1 does not contain points in every jackknife sample.")
    if not np.array_equal(np.unique(jtags1),np.arange(1,N_samples+1)):
        warn("Warning: data2 does not contain points in every jackknife sample.")
    
    if type(N_samples) is not int: 
        raise HalotoolsError("There must be an integer number of jackknife samples")
    if np.max(jtags1)>N_samples:
        raise HalotoolsError("There are more jackknife samples than indicated by N_samples")
    if np.max(jtags2)>N_samples:
        raise HalotoolsError("There are more jackknife samples than indicated by N_samples")

    return weights1, weights2, jtags1, jtags2






