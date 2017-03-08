""" Module containing the `~halotools.mock_observables.weighted_npairs_xy` function
used to count pairs as a function of separation.
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import multiprocessing
from functools import partial

from .engines import weighted_npairs_per_object_xy_engine
from .weighted_npairs_xy import _weighted_npairs_xy_process_args

from ..pair_counters.rectangular_mesh_2d import RectangularDoubleMesh2D
from ..pair_counters.mesh_helpers import _set_approximate_2d_cell_sizes
from ..pair_counters.mesh_helpers import _cell1_parallelization_indices


__author__ = ('Andrew Hearin', )

__all__ = ('weighted_npairs_per_object_xy', )


def weighted_npairs_per_object_xy(sample1, sample2, sample2_mass, rp_bins,
        period=None, verbose=False, num_threads=1,
        approx_cell1_size=None, approx_cell2_size=None):
    """
    """
    # Process the inputs with the helper function
    result = _weighted_npairs_xy_process_args(sample1, sample2, sample2_mass,
            rp_bins, period,
            verbose, num_threads, approx_cell1_size, approx_cell2_size)
    x1in, y1in, x2in, y2in, w2in = result[0:5]
    rp_bins, period, num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[5:]
    xperiod, yperiod = period

    rp_max = np.max(rp_bins)
    search_xlength, search_ylength = rp_max, rp_max

    # Compute the estimates for the cell sizes
    approx_cell1_size, approx_cell2_size = (
        _set_approximate_2d_cell_sizes(approx_cell1_size, approx_cell2_size, period)
        )
    approx_x1cell_size, approx_y1cell_size = approx_cell1_size
    approx_x2cell_size, approx_y2cell_size = approx_cell2_size

    # Build the rectangular mesh
    double_mesh = RectangularDoubleMesh2D(x1in, y1in, x2in, y2in,
        approx_x1cell_size, approx_y1cell_size,
        approx_x2cell_size, approx_y2cell_size,
        search_xlength, search_ylength, xperiod, yperiod, PBCs)

    # # Create a function object that has a single argument, for parallelization purposes
    counting_engine = partial(weighted_npairs_per_object_xy_engine,
        double_mesh, x1in, y1in, x2in, y2in, w2in, rp_bins)

    # # Calculate the cell1 indices that will be looped over by the engine
    num_threads, cell1_tuples = _cell1_parallelization_indices(
        double_mesh.mesh1.ncells, num_threads)

    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        result = pool.map(counting_engine, cell1_tuples)
        counts = np.sum(np.array(result), axis=0)
        pool.close()
    else:
        result = counting_engine(cell1_tuples[0])
        counts = np.vstack(result)

    return np.array(counts)
