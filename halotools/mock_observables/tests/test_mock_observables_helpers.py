""" Module providing unit-testing for
`~halotools.mock_observables.mock_observables_helpers` functions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import multiprocessing
import pytest
import warnings

from ..mock_observables_helpers import (
    enforce_sample_respects_pbcs,
    get_num_threads,
    get_period,
)
from ..mock_observables_helpers import (
    enforce_sample_has_correct_shape,
    get_separation_bins_array,
)
from ..mock_observables_helpers import get_line_of_sight_bins_array

__all__ = (
    "test_enforce_sample_respects_pbcs",
    "test_get_num_threads",
    "test_get_period",
)

fixed_seed = 43


def test_enforce_sample_respects_pbcs():
    npts = 10
    x = np.linspace(0, 1, npts)
    y = np.linspace(0, 1, npts)
    z = np.linspace(0, 1, npts)

    enforce_sample_respects_pbcs(x, y, z, [1, 1, 1])

    with pytest.raises(ValueError) as err:
        enforce_sample_respects_pbcs(x, y, z, [0.9, 1, 1])
    substr = "You set xperiod = "
    assert substr in err.value.args[0]

    with pytest.raises(ValueError) as err:
        enforce_sample_respects_pbcs(x, y, z, [1, 0.9, 1])
    substr = "You set yperiod = "
    assert substr in err.value.args[0]

    with pytest.raises(ValueError) as err:
        enforce_sample_respects_pbcs(x, y, z, [1, 1, 0.9])
    substr = "You set zperiod = "
    assert substr in err.value.args[0]

    x = np.linspace(-1, 1, npts)
    with pytest.raises(ValueError) as err:
        enforce_sample_respects_pbcs(x, y, z, [1, 1, 1])
    substr = "your input data has negative values"
    assert substr in err.value.args[0]


def test_get_num_threads():

    input_num_threads = 1
    result = get_num_threads(input_num_threads, enforce_max_cores=False)
    assert result == 1

    input_num_threads = "max"
    result = get_num_threads(input_num_threads, enforce_max_cores=False)
    assert result == multiprocessing.cpu_count()

    max_cores = multiprocessing.cpu_count()

    input_num_threads = max_cores + 1
    with warnings.catch_warnings(record=True) as w:
        result = get_num_threads(input_num_threads, enforce_max_cores=False)
        assert "num_available_cores" in str(w[-1].message)
    assert result == input_num_threads

    input_num_threads = max_cores + 1
    with warnings.catch_warnings(record=True) as w:
        result = get_num_threads(input_num_threads, enforce_max_cores=True)
        assert "num_available_cores" in str(w[-1].message)
    assert result == max_cores

    input_num_threads = "$"
    with pytest.raises(ValueError) as err:
        result = get_num_threads(input_num_threads, enforce_max_cores=True)
    substr = "Input ``num_threads`` must be an integer"
    assert err.value.args[0] in substr


def test_get_period():
    period, PBCs = get_period(1)
    assert np.all(period == 1)
    assert PBCs is True

    period, PBCs = get_period([1, 1, 1])
    assert np.all(period == 1)
    assert PBCs is True

    with pytest.raises(ValueError) as err:
        period, PBCs = get_period([1, 1])
    substr = "Input ``period`` must be either a scalar or a 3-element sequence."
    assert substr in err.value.args[0]

    with pytest.raises(ValueError) as err:
        period, PBCs = get_period([1, 1, np.inf])
    substr = "All values must bounded positive numbers."
    assert substr in err.value.args[0]


def test_enforce_sample_has_correct_shape():

    npts = 100
    good_sample = np.zeros((npts, 3))
    _ = enforce_sample_has_correct_shape(good_sample)

    bad_sample = np.zeros((npts, 2))
    with pytest.raises(TypeError) as err:
        _ = enforce_sample_has_correct_shape(bad_sample)
    substr = "Input sample of points must be a Numpy ndarray of shape (Npts, 3)."
    assert substr in err.value.args[0]


def test_get_separation_bins_array():

    good_rbins = [1, 2]
    _ = get_separation_bins_array(good_rbins)

    good_rbins = np.linspace(1, 2, 10)
    _ = get_separation_bins_array(good_rbins)

    bad_rbins = [0, 1]
    with pytest.raises(TypeError) as err:
        _ = get_separation_bins_array(bad_rbins)
    substr = "Input separation bins must be a monotonically increasing "
    assert substr in err.value.args[0]

    bad_rbins = [1, 2, 2, 4]
    with pytest.raises(TypeError) as err:
        _ = get_separation_bins_array(bad_rbins)
    substr = "Input separation bins must be a monotonically increasing "
    assert substr in err.value.args[0]


def test_get_line_of_sight_bins_array():

    good_pi_bins = [1, 2]
    _ = get_line_of_sight_bins_array(good_pi_bins)

    good_pi_bins = np.linspace(1, 2, 10)
    _ = get_line_of_sight_bins_array(good_pi_bins)

    # Note that a zero-value in pi_bins is currently permissible
    good_pi_bins = [0, 1, 2]
    _ = get_line_of_sight_bins_array(good_pi_bins)

    bad_pi_bins = [1, 0.5, 0.1]
    with pytest.raises(TypeError) as err:
        _ = get_line_of_sight_bins_array(bad_pi_bins)
    substr = "Input separation bins must be a monotonically increasing "
    assert substr in err.value.args[0]

    bad_pi_bins = [1, 2, 2, 4]
    with pytest.raises(TypeError) as err:
        _ = get_line_of_sight_bins_array(bad_pi_bins)
    substr = "Input separation bins must be a monotonically increasing "
    assert substr in err.value.args[0]
