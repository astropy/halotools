"""
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np

from .. import array_indexing_manipulations as aim

__all__ = ('test_calculate_first_idx_unique_array_vals1',
    'test_calculate_first_idx_unique_array_vals2', 'test_calculate_last_idx_unique_array_vals1',
    'test_calculate_last_idx_unique_array_vals2', 'test_sum_in_bins1')


def test_calculate_first_idx_unique_array_vals1():
    arr = np.array([1, 1, 2, 2, 2, 3, 3])
    result = aim.calculate_first_idx_unique_array_vals(arr)
    correct_result = np.array([0, 2, 5])
    assert np.all(result == correct_result)


def test_calculate_first_idx_unique_array_vals2():
    arr = np.array([1, 2, 3])
    result = aim.calculate_first_idx_unique_array_vals(arr)
    correct_result = np.array([0, 1, 2])
    assert np.all(result == correct_result)


def test_calculate_last_idx_unique_array_vals1():
    arr = np.array([1, 1, 2, 2, 2, 3, 3])
    result = aim.calculate_last_idx_unique_array_vals(arr)
    correct_result = np.array([1, 4, 6])
    assert np.all(result == correct_result)


def test_calculate_last_idx_unique_array_vals2():
    arr = np.array([1, 2, 3])
    result = aim.calculate_last_idx_unique_array_vals(arr)
    correct_result = np.array([0, 1, 2])
    assert np.all(result == correct_result)


def test_sum_in_bins1():
    sorted_bin_numbers = np.array([1, 1, 2, 2, 2, 3, 3])
    values_in_bins = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3])
    result = aim.sum_in_bins(values_in_bins, sorted_bin_numbers)
    correct_result = np.array([0.2, 0.6, 0.6])
    assert np.allclose(result, correct_result)
