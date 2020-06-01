from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import multiprocessing
from functools import partial

from ..pair_counters.npairs_xy_z import _npairs_xy_z_process_args
from ..pair_counters.mesh_helpers import _set_approximate_cell_sizes, _cell1_parallelization_indices
from ..pair_counters.rectangular_mesh import RectangularDoubleMesh
from .velocity_marked_npairs_3d import _velocity_marked_npairs_3d_process_weights
from .engines import velocity_marked_npairs_xy_z_engine

__author__ = ('Duncan Campbell', 'Andrew Hearin')


__all__ = ('velocity_marked_npairs_xy_z', )


def velocity_marked_npairs_xy_z(sample1, sample2, rp_bins, pi_bins, period=None,
        weights1=None, weights2=None, weight_func_id=1, num_threads=1,
        approx_cell1_size=None, approx_cell2_size=None):
    r"""
    Calculate the number of velocity weighted pairs
    with separations greater than or equal to
    :math:`r_{\perp}` and :math:`r_{\parallel}`, :math:`W(>r_{\perp},>r_{\parallel})`.

    :math:`r_{\perp}` and :math:`r_{\parallel}` are defined wrt the z-direction.

    The weight given to each pair is determined by the weights for a pair,
    :math:`w_1`, :math:`w_2`, and a user-specified "velocity weighting function", indicated
    by the ``weight_func_id`` parameter, :math:`f(w_1,w_2)`.

    Parameters
    ----------
    sample1 : array_like
        Npts1 x 3 numpy array containing 3-D positions of points.
        See the :ref:`mock_obs_pos_formatting` documentation page, or the
        Examples section below, for instructions on how to transform
        your coordinate position arrays into the
        format accepted by the ``sample1`` and ``sample2`` arguments.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    sample2 : array_like
        Npts2 x 3 array containing 3-D positions of points.

    rp_bins : array_like
        array of boundaries defining the radial bins perpendicular to the LOS in which
        pairs are counted.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    pi_bins : array_like
        array of boundaries defining the bins parallel to the LOS in which
        pairs are counted.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions
        in each dimension. If you instead provide a single scalar, Lbox,
        period is assumed to be the same in all Cartesian directions.
        If set to None (the default option), PBCs are set to infinity.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    weights1 : array_like, optional
        Either a 1-D array of length *Npts1*, or a 2-D array of length *Npts1* x *N_weights*,
        containing the weights used for the weighted pair counts. If this parameter is
        None, the weights are set to np.ones(*(Npts1,N_weights)*).

    weights2 : array_like, optional
        Either a 1-D array of length *Npts2*, or a 2-D array of length *Npts2* x *N_weights*,
        containing the weights used for the weighted pair counts. If this parameter is
        None, the weights are set to np.ones(*(Npts2,N_weights)*).

    weight_func_id : int, optional
        velocity weighting function integer ID. Each weighting function requires a specific
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
    w1N_pairs : numpy.array
        2-D array of shape *(Nrp_bins,Npi_bins)* containing the weighted number counts
        of pairs. The exact values depend on ``weight_func_id``
        (which weighting function was chosen).

    w2N_pairs : numpy.array
        2-D array of shape *(Nrp_bins,Npi_bins)* containing the weighted number counts
        of pairs. The exact values depend on ``weight_func_id``
        (which weighting function was chosen).

    w3N_pairs : numpy.array
        2-D array of shape *(Nrp_bins,Npi_bins)* containing the weighted number counts
        of pairs. The exact values depend on ``weight_func_id``
        (which weighting function was chosen).

    Examples
    --------
    For demonstration purposes we will work with
    halos in the `~halotools.sim_manager.FakeSim`.

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim()

    >>> x = halocat.halo_table['halo_x']
    >>> y = halocat.halo_table['halo_y']
    >>> z = halocat.halo_table['halo_z']

    We transform our *x, y, z* points into the array shape used by the pair-counter by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> sample1 = np.vstack((x,y,z)).T

    We will do the same to get a random set of velocities.

    >>> vx = halocat.halo_table['halo_vx']
    >>> vy = halocat.halo_table['halo_vy']
    >>> vz = halocat.halo_table['halo_vz']
    >>> velocities = np.vstack((x,y,z,vx,vy,vz)).T

    >>> rp_bins = np.logspace(-2,-1,10)
    >>> pi_bins = np.linspace(0, 10, 5)
    >>> result = velocity_marked_npairs_xy_z(sample1, sample1, rp_bins, pi_bins, period=halocat.Lbox, weights1=velocities, weights2=velocities,)

    """
    result = _npairs_xy_z_process_args(sample1, sample2, rp_bins, pi_bins, period,
            num_threads, approx_cell1_size, approx_cell2_size)
    x1in, y1in, z1in, x2in, y2in, z2in = result[0:6]
    rp_bins, pi_bins, period, num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[6:]
    xperiod, yperiod, zperiod = period

    rp_max = np.max(rp_bins)
    pi_max = np.max(pi_bins)
    search_xlength, search_ylength, search_zlength = rp_max, rp_max, pi_max

    # Process the input weights and with the helper function
    weights1, weights2 = (
        _velocity_marked_npairs_3d_process_weights(sample1, sample2,
            weights1, weights2, weight_func_id))

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
    engine = partial(velocity_marked_npairs_xy_z_engine, double_mesh,
        x1in, y1in, z1in, x2in, y2in, z2in,
        weights1, weights2, weight_func_id, rp_bins, pi_bins)

    # Calculate the cell1 indices that will be looped over by the engine
    num_threads, cell1_tuples = _cell1_parallelization_indices(
        double_mesh.mesh1.ncells, num_threads)

    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        result = np.array(pool.map(engine, cell1_tuples))
        counts1, counts2, counts3 = result[:, 0], result[:, 1], result[:, 2]
        counts1 = np.sum(counts1, axis=0)
        counts2 = np.sum(counts2, axis=0)
        counts3 = np.sum(counts3, axis=0)
        pool.close()
    else:
        counts1, counts2, counts3 = np.array(engine(cell1_tuples[0]))

    return counts1, counts2, counts3
