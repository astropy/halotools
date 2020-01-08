"""
"""
from __future__ import (absolute_import, division, print_function)

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext

from ..pairwise_velocities_helpers import (_pairwise_velocity_stats_process_args,
    _process_radial_bins, _process_rp_bins)

fixed_seed = 43


def test_pairwise_velocity_stats_process_args1():
    period = None
    do_auto = False
    do_cross = False
    num_threads = 1
    approx_cell1_size = 0.1
    approx_cell2_size = 0.1

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((10, 3))
        velocities1 = np.random.random((10, 3))
        sample2 = np.random.random((10, 3))
        velocities2 = np.random.random((10, 3))

    with pytest.raises(ValueError) as err:
        result = _pairwise_velocity_stats_process_args(sample1, velocities1, sample2, velocities2,
            period, do_auto, do_cross, num_threads, approx_cell1_size, approx_cell2_size,
            fixed_seed)
    substr = "Both ``do_auto`` and ``do_cross`` have been set to False"
    assert substr in err.value.args[0]


def test_pairwise_velocity_stats_process_args2():
    period = None
    do_auto = True
    do_cross = False
    num_threads = 1
    approx_cell1_size = 0.1
    approx_cell2_size = 0.1

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((10, 3))
        velocities1 = np.random.random((10, 3))
        sample2 = np.random.random((10, 3))
        velocities2 = np.random.random((10, 3))

    result = _pairwise_velocity_stats_process_args(sample1, velocities1, None, None,
        period, do_auto, do_cross, num_threads, approx_cell1_size, approx_cell2_size,
        fixed_seed)
    sample1, velocities1, sample2, velocities2, period, do_auto,\
        do_cross, num_threads, _sample1_is_sample2, PBCs = result
    assert (do_auto is True) & (do_cross is False)


def test_pairwise_velocity_stats_process_args3():
    period = None
    do_auto = True
    do_cross = False
    num_threads = 1
    approx_cell1_size = 0.1
    approx_cell2_size = 0.1

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((10, 3))
        velocities1 = np.random.random((10, 3))
        sample2 = np.random.random((10, 3))
        velocities2 = np.random.random((10, 3))

    result = _pairwise_velocity_stats_process_args(sample1, velocities1, sample2, velocities2,
        period, do_auto, do_cross, num_threads, approx_cell1_size, approx_cell2_size,
        fixed_seed)
    sample1, velocities1, sample2, velocities2, period, do_auto,\
        do_cross, num_threads, _sample1_is_sample2, PBCs = result


def test_pairwise_velocity_stats_process_args4():
    period = None
    do_auto = True
    do_cross = False
    num_threads = 1
    approx_cell1_size = 0.1
    approx_cell2_size = 0.1

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((10, 3))
        velocities1 = np.random.random((10, 3))
        sample2 = np.random.random((10, 3))
        velocities2 = np.random.random((10, 3))

    result = _pairwise_velocity_stats_process_args(sample1, velocities1, sample2, velocities2,
        period, do_auto, do_cross, num_threads, approx_cell1_size, approx_cell2_size,
        fixed_seed)
    sample1, velocities1, sample2, velocities2, period, do_auto,\
        do_cross, num_threads, _sample1_is_sample2, PBCs = result
    assert _sample1_is_sample2 is False


def test_pairwise_velocity_stats_process_args5():
    period = None
    do_auto = True
    do_cross = False
    num_threads = 1
    approx_cell1_size = 0.1
    approx_cell2_size = 0.1

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((10, 3))
        velocities1 = np.random.random((10, 3))
        sample2 = np.random.random((10, 3))
        velocities2 = np.random.random((10, 3))

    result = _pairwise_velocity_stats_process_args(sample1, velocities1, None, None,
        period, do_auto, do_cross, num_threads, approx_cell1_size, approx_cell2_size,
        fixed_seed)
    sample1, velocities1, sample2, velocities2, period, do_auto,\
        do_cross, num_threads, _sample1_is_sample2, PBCs = result


def test_pairwise_velocity_stats_process_args6():
    period = None
    do_auto = True
    do_cross = False
    num_threads = 1
    approx_cell1_size = 0.1
    approx_cell2_size = 0.1

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((10, 3))
        velocities1 = np.random.random((10, 3))

    with NumpyRNGContext(fixed_seed+1):
        sample2 = np.random.random((10, 3))
        velocities2 = np.random.random((10, 3))

    result = _pairwise_velocity_stats_process_args(sample1, velocities1, sample2, velocities2,
        period, do_auto, do_cross, num_threads, approx_cell1_size, approx_cell2_size,
        fixed_seed)
    sample1, velocities1, sample2, velocities2, period, do_auto,\
        do_cross, num_threads, _sample1_is_sample2, PBCs = result


def test_process_radial_bins1():
    input_rbins = np.linspace(0.1, 0.5, 5)
    period = None
    PBCs = False
    rbins = _process_radial_bins(input_rbins, period, PBCs)


def test_process_radial_bins2():
    input_rbins = np.arange(5)[::-1]
    period = None
    PBCs = False
    with pytest.raises(ValueError) as err:
        rbins = _process_radial_bins(input_rbins, period, PBCs)


def test_process_radial_bins3():
    input_rbins = np.linspace(0.1, 2, 5)
    period = 1
    PBCs = True
    with pytest.raises(ValueError) as err:
        rbins = _process_radial_bins(input_rbins, period, PBCs)


def test_process_rp_bins1():
    input_rpbins = np.linspace(0.1, 0.5, 5)
    period = None
    PBCs = False
    pi_max = 0.1
    rp_bins, pi_max = _process_rp_bins(input_rpbins, pi_max, period, PBCs)


def test_process_rp_bins2():
    input_rpbins = np.arange(5)[::-1]
    period = None
    PBCs = False
    pi_max = 0.1
    with pytest.raises(ValueError) as err:
        rp_bins, pi_max = _process_rp_bins(input_rpbins, pi_max, period, PBCs)
    substr = "Input `rp_bins` must be a monotonically increasing"
    assert substr in err.value.args[0]


def test_process_rp_bins3():
    input_rpbins = np.linspace(0.1, 0.5, 5)
    period = [1, 1, 1]
    PBCs = True
    pi_max = 0.1
    with pytest.raises(ValueError) as err:
        rp_bins, pi_max = _process_rp_bins(input_rpbins, pi_max, period, PBCs)
    substr = "The maximum length over which you search for pairs"
    assert substr in err.value.args[0]


def test_process_rp_bins4():
    input_rpbins = np.linspace(0.1, 0.2, 5)
    period = [1, 1, 1]
    PBCs = True
    pi_max = 0.5
    with pytest.raises(ValueError) as err:
        rp_bins, pi_max = _process_rp_bins(input_rpbins, pi_max, period, PBCs)
    substr = "The input ``pi_max`` = "
    assert substr in err.value.args[0]
