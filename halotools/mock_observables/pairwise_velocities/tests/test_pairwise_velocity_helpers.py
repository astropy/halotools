"""
"""
from __future__ import (absolute_import, division, print_function)

import numpy as np
from astropy.tests.helper import pytest
from astropy.utils.misc import NumpyRNGContext

from ..pairwise_velocities_helpers import _pairwise_velocity_stats_process_args

fixed_seed = 43


def test1():
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((10, 3))
        velocities1 = np.random.random((10, 3))
        sample2 = np.random.random((10, 3))
        velocities2 = np.random.random((10, 3))

    period = None
    do_auto = False
    do_cross = False
    num_threads = 1
    max_sample_size = int(1e5)
    approx_cell1_size = 0.1
    approx_cell2_size = 0.1

    with pytest.raises(ValueError) as err:
        result = _pairwise_velocity_stats_process_args(sample1, velocities1, sample2, velocities2,
            period, do_auto, do_cross, num_threads, max_sample_size, approx_cell1_size, approx_cell2_size)
    substr = "Both ``do_auto`` and ``do_cross`` have been set to False"
    assert substr in err.value.args[0]

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((10, 3))
        velocities1 = np.random.random((10, 3))
        sample2 = np.random.random((10, 3))
        velocities2 = np.random.random((10, 3))

    do_auto = True

    result = _pairwise_velocity_stats_process_args(sample1, velocities1, None, None,
        period, do_auto, do_cross, num_threads, max_sample_size, approx_cell1_size, approx_cell2_size)
    sample1, velocities1, sample2, velocities2, period, do_auto,\
        do_cross, num_threads, _sample1_is_sample2, PBCs = result
    assert (do_auto is True) & (do_cross is False)

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((10, 3))
        velocities1 = np.random.random((10, 3))
        sample2 = np.random.random((10, 3))
        velocities2 = np.random.random((10, 3))

    result = _pairwise_velocity_stats_process_args(sample1, velocities1, sample2, velocities2,
        period, do_auto, do_cross, num_threads, max_sample_size, approx_cell1_size, approx_cell2_size)
    sample1, velocities1, sample2, velocities2, period, do_auto,\
        do_cross, num_threads, _sample1_is_sample2, PBCs = result

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((10, 3))
        velocities1 = np.random.random((10, 3))
        sample2 = np.random.random((10, 3))
        velocities2 = np.random.random((10, 3))

    max_sample_size = 5

    result = _pairwise_velocity_stats_process_args(sample1, velocities1, sample2, velocities2,
        period, do_auto, do_cross, num_threads, max_sample_size, approx_cell1_size, approx_cell2_size)
    sample1, velocities1, sample2, velocities2, period, do_auto,\
        do_cross, num_threads, _sample1_is_sample2, PBCs = result
    assert _sample1_is_sample2 is False
    assert len(sample1) == max_sample_size
    assert len(sample2) == max_sample_size

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((10, 3))
        velocities1 = np.random.random((10, 3))
        sample2 = np.random.random((10, 3))
        velocities2 = np.random.random((10, 3))

    result = _pairwise_velocity_stats_process_args(sample1, velocities1, None, None,
        period, do_auto, do_cross, num_threads, max_sample_size, approx_cell1_size, approx_cell2_size)
    sample1, velocities1, sample2, velocities2, period, do_auto,\
        do_cross, num_threads, _sample1_is_sample2, PBCs = result

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((10, 3))
        velocities1 = np.random.random((10, 3))
        sample2 = np.random.random((10, 3))
        velocities2 = np.random.random((10, 3))

    result = _pairwise_velocity_stats_process_args(sample1, velocities1, sample1, velocities1,
        period, do_auto, do_cross, num_threads, max_sample_size, approx_cell1_size, approx_cell2_size)
    sample1, velocities1, sample2, velocities2, period, do_auto,\
        do_cross, num_threads, _sample1_is_sample2, PBCs = result

    pass
