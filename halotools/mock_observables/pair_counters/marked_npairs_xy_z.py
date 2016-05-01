""" Module containing the `~halotools.mock_observables.npairs_3d` function 
used to count pairs as a function of separation. 
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np 
import multiprocessing
from functools import partial 

from .marked_npairs_3d import _marked_npairs_process_weights
from .npairs_xy_z import _npairs_xy_z_process_args
from .mesh_helpers import _set_approximate_cell_sizes, _cell1_parallelization_indices
from .rectangular_mesh import RectangularDoubleMesh

from .marked_pair_counting_engines import marked_npairs_xy_z_engine 

__author__ = ('Duncan Campbell', 'Andrew Hearin')


__all__ = ('marked_npairs_xy_z', )

def marked_npairs_xy_z(data1, data2, rp_bins, pi_bins, 
                  period=None, weights1 = None, weights2 = None,
                  weight_func_id = 0, verbose = False, num_threads = 1,
                  approx_cell1_size = None, approx_cell2_size = None):
    """
    Calculate the number of weighted pairs with seperations greater than 
    or equal to :math:`r_{\\perp}` and :math:`r_{\\parallel}`, :math:`W(>r_{\\perp},>r_{\\parallel})`.

    :math:`r_{\\perp}` and :math:`r_{\\parallel}` are defined wrt the z-direction.

    The weight given to each pair is determined by the weights for a pair,
    :math:`w_1`, :math:`w_2`, and a user-specified "weighting function", indicated
    by the ``wfunc`` parameter, :math:`f(w_1,w_2)`.

    Parameters
    ----------
    data1 : array_like
        *N1* by 3 array of 3-D positions.  If the ``period`` parameter is set, each
        component of the coordinates should be bounded between zero and the corresponding
        periodic boundary.

    data2 : array_like
        *N2* by 3 array of 3-D positions.  If the ``period`` parameter is set, each
        component of the coordinates should be bounded between zero and the corresponding
        periodic boundary.

    rp_bins : array_like
        numpy array of length Nrp_bins+1 defining the boundaries of bins of projected
        separation, :math:`r_{\\rm p}`, in which pairs are counted.

    pi_bins : array_like
        numpy array of length Npi_bins+1 defining the boundaries of bins of parallel
        separation, :math:`\\pi`, in which pairs are counted.

    period : array_like, optional
        Length-3 array defining axis-aligned periodic boundary conditions. If only
        one number, Lbox, is specified, the period is assumed to be np.array([Lbox]*3).

    weights1 : array_like, optional
        Either a 1-D array of length *N1*, or a 2-D array of length *N1* x *N_weights*,
        containing the weights used for the weighted pair counts. If this parameter is
        None, the weights are set to np.ones(*(N1,N_weights)*).

    weights2 : array_like, optional
        Either a 1-D array of length *N1*, or a 2-D array of length *N1* x *N_weights*,
        containing the weights used for the weighted pair counts. If this parameter is
        None, the weights are set to np.ones(*(N1,N_weights)*).

    wfunc : int, optional
        weighting function integer ID. Each weighting function requires a specific
        number of weights per point, *N_weights*.  See the Notes for a description of
        available weighting functions.

    verbose : Boolean, optional
        If True, print out information and progress.

    num_threads : int, optional
        number of 'threads' to use in the pair counting.  if set to 'max', use all
        available cores.  num_threads=0 is the default.

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
    wN_pairs : numpy.ndarray
        2-D array of shape *(Nrp_bins,Npi_bins)* containing the weighted number
        counts of pairs
    """

    ### Process the inputs with the helper function
    result = _npairs_xy_z_process_args(data1, data2, rp_bins, pi_bins, period,
            verbose, num_threads, approx_cell1_size, approx_cell2_size)
    x1in, y1in, z1in, x2in, y2in, z2in = result[0:6]
    rp_bins, pi_bins, period, num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[6:]
    xperiod, yperiod, zperiod = period 

    rp_max = np.max(rp_bins)
    pi_max = np.max(pi_bins)
    search_xlength, search_ylength, search_zlength = rp_max, rp_max, pi_max 

    # Process the input weights and with the helper function
    weights1, weights2 = _marked_npairs_process_weights(data1, data2,
            weights1, weights2, weight_func_id)

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
    engine = partial(marked_npairs_xy_z_engine, double_mesh, 
        x1in, y1in, z1in, x2in, y2in, z2in, 
        weights1, weights2, weight_func_id, rp_bins, pi_bins)

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






