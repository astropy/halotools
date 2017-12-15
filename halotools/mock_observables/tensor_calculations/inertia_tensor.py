"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import multiprocessing
from functools import partial

from .engines import inertia_tensor_per_object_3d_engine

from ..mock_observables_helpers import get_num_threads, get_period, enforce_sample_respects_pbcs
from ..pair_counters.mesh_helpers import (_set_approximate_cell_sizes,
    _cell1_parallelization_indices, _enclose_in_box, _enforce_maximum_search_length)
from ..pair_counters.rectangular_mesh import RectangularDoubleMesh


__all__ = ('inertia_tensor_per_object_3d', )


def inertia_tensor_per_object_3d(sample1, sample2, weights2, smoothing_scale,
            period=None, num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    r""" For each point in `sample1`, calculate the inertia tensor using all point masses
    in `sample2` within the input smoothing_scale.

    Parameters
    ----------
    sample1 : array_like
        Numpy array of shape (npts1, 3) storing 3-D positions of points.

        See the :ref:`mock_obs_pos_formatting` documentation page, or the
        Examples section below, for instructions on how to transform
        your coordinate position arrays into the
        format accepted by the ``sample1`` and ``sample2`` arguments.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

    sample2 : array_like
        Numpy array of shape (npts2, 3) storing 3-D positions of the point masses
        used to calculate the inertia tensor of every `sample1` point.

    weights2 : array_like
        Numpy array of shape (npts2,) storing the mass of each `sample2` point
        used to calculate the inertia tensor of every `sample1` point.

    smoothing_scale : float
        Three-dimensional distance from each `sample1` point defining
        which points in `sample2` are used to compute the inertia tensor

    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions
        in each dimension. If you instead provide a single scalar, Lbox,
        period is assumed to be the same in all Cartesian directions.
        Default is None, in which case no PBCs will be applied.

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
    inertia_tensors : ndarray
        Numpy array of shape (npts1, 3, 3) storing the inertia tensor for every
        object in `sample1`.

    Examples
    --------
    >>> npts1, npts2 = 50, 75
    >>> sample1 = np.random.random((npts1, 3))
    >>> sample2 = np.random.random((npts2, 3))
    >>> weights2 = np.random.random(npts2)
    >>> smoothing_scale = 0.1
    >>> tensors = inertia_tensor_per_object_3d(sample1, sample2, weights2, smoothing_scale)
    """
    num_threads = get_num_threads(num_threads, enforce_max_cores=False)
    period, PBCs = get_period(period)

    # At this point, period may still be set to None,
    # in which case we must remap our points inside the smallest enclosing cube
    # and set ``period`` equal to this cube size.
    if period is None:
        x1in, y1in, z1in, x2in, y2in, z2in, period = (
            _enclose_in_box(
                sample1[:, 0], sample1[:, 1], sample1[:, 2],
                sample2[:, 0], sample2[:, 1], sample2[:, 2],
                min_size=[smoothing_scale*3.0, smoothing_scale*3.0, smoothing_scale*3.0]))
    else:
        x1in = sample1[:, 0]
        y1in = sample1[:, 1]
        z1in = sample1[:, 2]
        x2in = sample2[:, 0]
        y2in = sample2[:, 1]
        z2in = sample2[:, 2]

    msg = "np.shape(weights2) = {0} should be ({1}, )"
    assert np.shape(weights2) == (sample2.shape[0], ), msg.format(np.shape(weights2), sample2.shape[0])

    xperiod, yperiod, zperiod = period
    _enforce_maximum_search_length(smoothing_scale, period)
    enforce_sample_respects_pbcs(x1in, y1in, z1in, period)
    enforce_sample_respects_pbcs(x2in, y2in, z2in, period)

    search_xlength = smoothing_scale
    search_ylength = smoothing_scale
    search_zlength = smoothing_scale

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
    engine = partial(inertia_tensor_per_object_3d_engine, double_mesh,
        x1in, y1in, z1in, x2in, y2in, z2in, weights2, smoothing_scale)

    # Calculate the cell1 indices that will be looped over by the engine
    num_threads, cell1_tuples = _cell1_parallelization_indices(
        double_mesh.mesh1.ncells, num_threads)

    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        result = np.array(pool.map(engine, cell1_tuples))
        tensors = np.sum(result, axis=0)
        pool.close()
    else:
        tensors = np.array(engine(cell1_tuples[0]))

    return tensors
