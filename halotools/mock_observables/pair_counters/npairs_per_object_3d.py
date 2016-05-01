""" Module containing the `~halotools.mock_observables.npairs_3d` function 
used to count pairs as a function of separation. 
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np 
import multiprocessing
from functools import partial 

__author__ = ('Andrew Hearin', 'Duncan Campbell')

from .rectangular_mesh import RectangularDoubleMesh
from .mesh_helpers import _set_approximate_cell_sizes, _cell1_parallelization_indices
from .pair_counting_engines import npairs_per_object_3d_engine
from .npairs_3d import _npairs_3d_process_args

__all__ = ('npairs_per_object_3d', )

def npairs_per_object_3d(data1, data2, rbins, period = None,
    verbose = False, num_threads = 1,
    approx_cell1_size = None, approx_cell2_size = None):
    """
    Function counts the number of times the pair count between two samples exceeds a
    threshold value as a function of the 3d spatial separation *r*.

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

    rbins : array_like
        Boundaries defining the bins in which pairs are counted.

    n_thresh : int, optional
        positive integer number indicating the threshold pair count

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
    num_pairs : array_like
        Numpy array of length len(rbins) storing the numbers of pairs in the input bins.

    Examples
    --------
    For illustration purposes, we'll create some fake data and call the pair counter:

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

    >>> result = npairs_per_object_3d(data1, data2, rbins, period = period)
    """

    ### Process the inputs with the helper function
    result = _npairs_3d_process_args(data1, data2, rbins, period,
            verbose, num_threads, approx_cell1_size, approx_cell2_size)
    x1in, y1in, z1in, x2in, y2in, z2in = result[0:6]
    rbins, period, num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[6:]
    xperiod, yperiod, zperiod = period 

    rmax = np.max(rbins)
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

    # Create a function object that has a single argument, for parallelization purposes
    engine = partial(npairs_per_object_3d_engine, 
        double_mesh, data1[:,0], data1[:,1], data1[:,2], 
        data2[:,0], data2[:,1], data2[:,2], rbins)

    # Calculate the cell1 indices that will be looped over by the engine
    num_threads, cell1_tuples = _cell1_parallelization_indices(
        double_mesh.mesh1.ncells, num_threads)

    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        result = pool.map(engine, cell1_tuples)
        counts = np.vstack(result)
        pool.close()
    else:
        result = engine(cell1_tuples[0])
        counts = np.vstack(result)

    return np.array(counts)

