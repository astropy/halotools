r""" Module containing the `~halotools.mock_observables.npairs_3d` function
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

from .marked_cpairs import marked_npairs_xy_z_engine

__author__ = ('Duncan Campbell', 'Andrew Hearin')


__all__ = ('marked_npairs_xy_z', )


def marked_npairs_xy_z(sample1, sample2, rp_bins, pi_bins,
                  period=None, weights1=None, weights2=None,
                  weight_func_id=0, num_threads=1,
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
    wN_pairs : numpy.ndarray
        2-D array of shape *(Nrp_bins,Npi_bins)* containing the weighted number
        counts of pairs

    Notes
    -----
    See the docstring of the `~halotools.mock_observables.marked_tpcf` function
    for a description of the available marking functions that can be passed in
    via the ``wfunc`` optional argument.
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
