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


__all__ = ('inertia_tensor_per_object', 'inertia_tensors_principal_axes')


def inertia_tensor_per_object(sample1, sample2, weights2, smoothing_scale,
            num_threads=1, period=None, approx_cell1_size=None, approx_cell2_size=None):
    """
    Examples
    --------
    >>> npts1, npts2 = 50, 75
    >>> sample1 = np.random.random((npts1, 3))
    >>> sample2 = np.random.random((npts2, 3))
    >>> weights2 = np.random.random(npts2)
    >>> smoothing_scale = 0.1
    >>> tensors = inertia_tensor_per_object(sample1, sample2, weights2, smoothing_scale)
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

        # result = np.array(pool.map(engine, cell1_tuples))
        # marked_counts = np.sum(np.array(marked_counts), axis=0)
        # counts = np.sum(np.array(counts), axis=0)

        pool.close()
        raise NotImplementedError()

    else:
        tensor = engine(cell1_tuples[0])

    return tensor


def _principal_axes_from_matrices(matrices):
    evals, evecs = np.linalg.eigh(matrices)
    return evecs[:, :, 2], evals[:, 2]


def _sphericity_from_matrices(matrices):
    evals, __ = np.linalg.eigh(matrices)
    third_evals = evals[:, 0]
    first_evals = evals[:, 2]
    sphericity = third_evals/first_evals
    return sphericity


def _triaxility_from_matrices(matrices):
    evals, __ = np.linalg.eigh(matrices)
    third_evals = evals[:, 0]
    second_evals = evals[:, 1]
    first_evals = evals[:, 2]
    triaxility = (first_evals**2 - second_evals**2)/(first_evals**2 - third_evals**2)
    return triaxility


def inertia_tensors_principal_axes(sample1, sample2, weights, smoothing_scale):
    """
    """
    return _principal_axes_from_matrices(
            inertia_tensor_per_object(sample1, sample2, weights, smoothing_scale))

