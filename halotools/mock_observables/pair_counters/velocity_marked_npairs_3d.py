from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np 
import multiprocessing
from functools import partial 

from .npairs_3d import _npairs_3d_process_args
from .mesh_helpers import _set_approximate_cell_sizes, _cell1_parallelization_indices
from .rectangular_mesh import RectangularDoubleMesh

from .marked_cpairs import velocity_marked_npairs_3d_engine 

from ...utils.array_utils import convert_to_ndarray
from ...custom_exceptions import HalotoolsError 

__author__ = ('Duncan Campbell', 'Andrew Hearin')


__all__ = ('velocity_marked_npairs_3d', )

def velocity_marked_npairs_3d(data1, data2, rbins, period=None,
    weights1 = None, weights2 = None,
    weight_func_id = 0, verbose = False, num_threads = 1,
    approx_cell1_size = None, approx_cell2_size = None):
    """
    Calculate the number of velocity weighted pairs with seperations greater than or equal to r, :math:`W(>r)`.

    The weight given to each pair is determined by the weights for a pair,
    :math:`w_1`, :math:`w_2`, and a user-specified "velocity weighting function", indicated
    by the ``weight_func_id`` parameter, :math:`f(w_1,w_2)`.

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

    rbins : array_like
        numpy array of length Nrbins+1 defining the boundaries of bins in which
        pairs are counted.

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

    weight_func_id : int, optional
        velocity weighting function integer ID. Each weighting function requires a specific
        number of weights per point, *N_weights*.  See the Notes for a description of
        available weighting functions.

    verbose : Boolean, optional
        If True, print out information and progress.

    num_threads : int, optional
        number of 'threads' to use in the pair counting.  if set to 'max', use all
        available cores.  num_threads=0 is the default.

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
    w1N_pairs : numpy.array
        array of length *Nrbins* containing the weighted number counts of pairs
        The exact values depend on ``weight_func_id``
        (which weighting function was chosen).

    w2N_pairs : numpy.array
        array of length *Nrbins* containing the weighted number counts of pairs
        The exact values depend on ``weight_func_id``
        (which weighting function was chosen).

    w3N_pairs : numpy.array
        array of length *Nrbins* containing the weighted number counts of pairs
        The exact values depend on ``weight_func_id``
        (which weighting function was chosen).
    """

    result = _npairs_3d_process_args(data1, data2, rbins, period,
            verbose, num_threads, approx_cell1_size, approx_cell2_size)
    x1in, y1in, z1in, x2in, y2in, z2in = result[0:6]
    rbins, period, num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[6:]
    xperiod, yperiod, zperiod = period 

    rmax = np.max(rbins)
    search_xlength, search_ylength, search_zlength = rmax, rmax, rmax 

    # Process the input weights and with the helper function
    weights1, weights2 = (
        _velocity_marked_npairs_3d_process_weights(data1, data2,
            weights1, weights2, weight_func_id))

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
    engine = partial(velocity_marked_npairs_3d_engine, double_mesh, 
        x1in, y1in, z1in, x2in, y2in, z2in, 
        weights1, weights2, weight_func_id, rbins)

    # Calculate the cell1 indices that will be looped over by the engine
    num_threads, cell1_tuples = _cell1_parallelization_indices(
        double_mesh.mesh1.ncells, num_threads)

    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        result = np.array(pool.map(engine, cell1_tuples))
        counts1, counts2, counts3 = result[:,0], result[:,1], result[:,2]
        counts1 = np.sum(counts1,axis=0)
        counts2 = np.sum(counts2,axis=0)
        counts3 = np.sum(counts3,axis=0)
        pool.close()
    else:
        counts1, counts2, counts3  = np.array(engine(cell1_tuples[0]))

    return counts1, counts2, counts3


















def _velocity_marked_npairs_3d_process_weights(data1, data2, weights1, weights2, weight_func_id):
    """
    process weights and associated arguments for
    `~halotools.mock_observables.pair_counters.marked_double_tree_pairs.velocity_marked_npairs`
    """
    
    correct_num_weights = _func_signature_int_from_vel_weight_func_id(weight_func_id)
    npts_data1 = np.shape(data1)[0]
    npts_data2 = np.shape(data2)[0]
    correct_shape1 = (npts_data1, correct_num_weights)
    correct_shape2 = (npts_data2, correct_num_weights)
    
    ### Process the input weights1
    _converted_to_2d_from_1d = False
    # First convert weights1 into a 2-d ndarray
    if weights1 is None:
        weights1 = np.ones((npts_data1, 1), dtype = np.float64)
    else:
        weights1 = convert_to_ndarray(weights1)
        weights1 = weights1.astype("float64")
        if weights1.ndim == 1:
            _converted_to_2d_from_1d = True
            npts1 = len(weights1)
            weights1 = weights1.reshape((npts1, 1))
        elif weights1.ndim == 2:
            pass
        else:
            ndim1 = weights1.ndim
            msg = ("\n You must either pass in a 1-D or 2-D array \n"
                   "for the input `weights1`. Instead, an array of \n"
                   "dimension %i was received.")
            raise HalotoolsError(msg % ndim1)
    
    npts_weights1 = np.shape(weights1)[0]
    num_weights1 = np.shape(weights1)[1]
    # At this point, weights1 is guaranteed to be a 2-d ndarray
    ### now we check its shape
    if np.shape(weights1) != correct_shape1:
        if _converted_to_2d_from_1d is True:
            msg = ("\n You passed in a 1-D array for `weights1` that \n"
                   "does not have the correct length. The number of \n"
                   "points in `data1` = %i, while the number of points \n"
                   "in your input 1-D `weights1` array = %i")
            raise HalotoolsError(msg % (npts_data1, npts_weights1))
        else:
            msg = ("\n You passed in a 2-D array for `weights1` that \n"
                   "does not have a consistent shape with `data1`. \n"
                   "`data1` has length %i. The input value of `weight_func_id` = %i \n"
                   "For this value of `weight_func_id`, there should be %i weights \n"
                   "per point. The shape of your input `weights1` is (%i, %i)\n")
            raise HalotoolsError(msg % 
                (npts_data1, weight_func_id, correct_num_weights, npts_weights1, num_weights1))
    
    ### Process the input weights2
    _converted_to_2d_from_1d = False
    # Now convert weights2 into a 2-d ndarray
    if weights2 is None:
        weights2 = np.ones((npts_data2, 1), dtype = np.float64)
    else:
        weights2 = convert_to_ndarray(weights2)
        weights2 = weights2.astype("float64")
        if weights2.ndim == 1:
            _converted_to_2d_from_1d = True
            npts2 = len(weights2)
            weights2 = weights2.reshape((npts2, 1))
        elif weights2.ndim == 2:
            pass
        else:
            ndim2 = weights2.ndim
            msg = ("\n You must either pass in a 1-D or 2-D array \n"
                   "for the input `weights2`. Instead, an array of \n"
                   "dimension %i was received.")
            raise HalotoolsError(msg % ndim2)
    
    npts_weights2 = np.shape(weights2)[0]
    num_weights2 = np.shape(weights2)[1]
    # At this point, weights2 is guaranteed to be a 2-d ndarray
    ### now we check its shape
    if np.shape(weights2) != correct_shape2:
        if _converted_to_2d_from_1d is True:
            msg = ("\n You passed in a 1-D array for `weights2` that \n"
                   "does not have the correct length. The number of \n"
                   "points in `data2` = %i, while the number of points \n"
                   "in your input 1-D `weights2` array = %i")
            raise HalotoolsError(msg % (npts_data2, npts_weights2))
        else:
            msg = ("\n You passed in a 2-D array for `weights2` that \n"
                   "does not have a consistent shape with `data2`. \n"
                   "`data2` has length %i. The input value of `weight_func_id` = %i \n"
                   "For this value of `weight_func_id`, there should be %i weights \n"
                   "per point. The shape of your input `weights2` is (%i, %i)\n")
            raise HalotoolsError(msg % 
                (npts_data2, weight_func_id, correct_num_weights, npts_weights2, num_weights2))
    
    return weights1, weights2

def _func_signature_int_from_vel_weight_func_id(weight_func_id):
    """
    return the function signiture available velocity weighting functions.
    """
    if type(weight_func_id) != int:
        msg = "\n weight_func_id parameter must be an integer ID of a weighting function."
        raise HalotoolsError(msg)
    
    elif weight_func_id == 11:
        return 6
    elif weight_func_id == 12:
        return 6
    elif weight_func_id == 13:
        return 7
    elif weight_func_id == 14:
        return 1
    elif weight_func_id == 15:
        return 1
    elif weight_func_id == 16:
        return 2
    else:
        msg = ("The value ``weight_func_id`` = %i is not recognized")
        raise HalotoolsError(msg % weight_func_id)










