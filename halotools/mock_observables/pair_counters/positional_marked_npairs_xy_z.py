r""" Module containing the `~halotools.mock_observables.npairs_3d` function
used to count pairs as a function of separation.
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import multiprocessing
from functools import partial

from .npairs_xy_z import _npairs_xy_z_process_args
from .mesh_helpers import _set_approximate_cell_sizes, _cell1_parallelization_indices
from .rectangular_mesh import RectangularDoubleMesh

from .marked_cpairs import positional_marked_npairs_xy_z_engine

from ...custom_exceptions import HalotoolsError

__author__ = ('Duncan Campbell', 'Andrew Hearin')


__all__ = ('positional_marked_npairs_xy_z', )


def positional_marked_npairs_xy_z(sample1, sample2, rp_bins, pi_bins,
                  period=None, weights1=None, weights2=None,
                  weight_func_id=0, verbose=False, num_threads=1,
                  approx_cell1_size=None, approx_cell2_size=None):
    r"""
    Calculate the number of weighted pairs with separations greater than
    or equal to :math:`r_{\perp}` and :math:`r_{\parallel}`, :math:`W(>r_{\perp},>r_{\parallel})`.

    :math:`r_{\perp}` and :math:`r_{\parallel}` are defined wrt the z-direction.

    The weight given to each pair is determined by the weights for a pair,
    :math:`w_1`, :math:`w_2`, and a user-specified "weighting function", indicated
    by the ``wfunc`` parameter, :math:`f(w_1,w_2)`.

    Parameters
    ----------
    sample1 : array_like
        Numpy array of shape (Npts1, 3) containing 3-D positions of points.
        See the :ref:`mock_obs_pos_formatting` documentation page, or the
        Examples section below, for instructions on how to transform
        your coordinate position arrays into the
        format accepted by the ``sample1`` and ``sample2`` arguments.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    sample2 : array_like, optional
        Numpy array of shape (Npts2, 3) containing 3-D positions of points.
        Should be identical to sample1 for cases of auto-sample pair counts.

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
    wN_pairs : numpy.array
        Numpy array of shape (Nrp_bins, Nrpi_bins) containing the weighted number counts of pairs

    N_pairs : numpy.array
        Numpy array of shape (Nrp_bins, Nrpi_bins) containing the number counts of pairs

    Examples
    --------
    For demonstration purposes we create randomly distributed sets of points within a
    periodic unit cube, using random weights.

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

    We create a set of random weights:

    >>> weights1 = np.random.random((Npts1, 3))
    >>> weights2 = np.random.random((Npts2, 1))

    The weighted counts are calculated by:

    >>> weighted_counts, counts = positional_marked_npairs_xy_z(sample1, sample2, rp_bins, pi_bins, period=period, weights1=weights1, weights2=weights2, weight_func_id=1)

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

    # Process the input weights and with the helper function
    weights1, weights2 = _marked_npairs_process_weights(sample1, sample2,
            weights1, weights2, weight_func_id)

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
    engine = partial(positional_marked_npairs_xy_z_engine, double_mesh,
        x1in, y1in, z1in, x2in, y2in, z2in,
        weights1, weights2, weight_func_id, rp_bins, pi_bins)

    # Calculate the cell1 indices that will be looped over by the engine
    num_threads, cell1_tuples = _cell1_parallelization_indices(
        double_mesh.mesh1.ncells, num_threads)

    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        result = np.array(pool.map(engine, cell1_tuples))
        counts, marked_counts = result[:, 0, :, :], result[:, 1, :, :]
        marked_counts = np.sum(np.array(marked_counts), axis=0)
        counts = np.sum(np.array(counts), axis=0)
        pool.close()
    else:
        counts, marked_counts = engine(cell1_tuples[0])

    return np.array(marked_counts), np.array(counts)


def _marked_npairs_process_weights(sample1, sample2, weights1, weights2, weight_func_id):
    """
    """

    correct_num_weights1, correct_num_weights2 = _func_signature_int_from_wfunc(weight_func_id)
    npts_sample1 = np.shape(sample1)[0]
    npts_sample2 = np.shape(sample2)[0]
    correct_shape1 = (npts_sample1, correct_num_weights1)
    correct_shape2 = (npts_sample2, correct_num_weights2)

    # Process the input weights1
    _converted_to_2d_from_1d = False
    # First convert weights1 into a 2-d ndarray
    if weights1 is None:
        weights1 = np.ones(correct_shape1, dtype=np.float64)
    else:
        weights1 = np.atleast_1d(weights1)
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
    # now we check its shape
    if np.shape(weights1) != correct_shape1:
        if _converted_to_2d_from_1d is True:
            msg = ("\n You passed in a 1-D array for `weights1` that \n"
                   "does not have the correct length. The number of \n"
                   "points in `sample1` = %i, while the number of points \n"
                   "in your input 1-D `weights1` array = %i")
            raise HalotoolsError(msg % (npts_sample1, npts_weights1))
        else:
            msg = ("\n You passed in a 2-D array for `weights1` that \n"
                   "does not have a consistent shape with `sample1`. \n"
                   "`sample1` has length %i. The input value of `weight_func_id` = %i \n"
                   "For this value of `weight_func_id`, there should be %i weights \n"
                   "per point. The shape of your input `weights1` is (%i, %i)\n")
            raise HalotoolsError(msg %
                (npts_sample1, weight_func_id, correct_num_weights1, npts_weights1, num_weights1))

    # Process the input weights2
    _converted_to_2d_from_1d = False
    # Now convert weights2 into a 2-d ndarray
    if weights2 is None:
        weights2 = np.ones(correct_shape2, dtype=np.float64)
    else:
        weights2 = np.atleast_1d(weights2)
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
    # now we check its shape
    if np.shape(weights2) != correct_shape2:
        if _converted_to_2d_from_1d is True:
            msg = ("\n You passed in a 1-D array for `weights2` that \n"
                   "does not have the correct length. The number of \n"
                   "points in `sample2` = %i, while the number of points \n"
                   "in your input 1-D `weights2` array = %i")
            raise HalotoolsError(msg % (npts_sample2, npts_weights2))
        else:
            msg = ("\n You passed in a 2-D array for `weights2` that \n"
                   "does not have a consistent shape with `sample2`. \n"
                   "`sample2` has length %i. The input value of `weight_func_id` = %i \n"
                   "For this value of `weight_func_id`, there should be %i weights \n"
                   "per point. The shape of your input `weights2` is (%i, %i)\n")
            raise HalotoolsError(msg %
                (npts_sample2, weight_func_id, correct_num_weights2, npts_weights2, num_weights2))

    return  weights1, weights2


def _func_signature_int_from_wfunc(weight_func_id):
    """
    Return the function signature available weighting functions.
    """

    if type(weight_func_id) != int:
        msg = "\n weight_func_id parameter must be an integer ID of a weighting function."
        raise ValueError(msg)

    if weight_func_id == 1:
        return (3, 1)
    if weight_func_id == 2:
        return (3, 1)
    if weight_func_id == 3:
        return (3, 1)
    if weight_func_id == 4:
        return (3, 1)
    if weight_func_id == 5:
        return (3, 3)
    if weight_func_id == 6:
        return (3, 3)
    else:
        msg = ("The value ``weight_func_id`` = %i is not recognized")
        raise HalotoolsError(msg % weight_func_id)
