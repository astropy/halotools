"""
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext

from .. import subhalo_selection_kernel as ssk

__all__ = ('test_indices_of_selected_subhalos', )

fixed_seed = 43


def test_indices_of_selected_subhalos():
    """
    """
    objID = np.array([0, 0, 5, 5, 5, 7, 8, 8])
    multiplicity = np.array([2, 3, 1, 2])
    occupations = np.array([0, 2, 1, 2])
    result = ssk.indices_of_selected_subhalos(objID, occupations, multiplicity)
    correct_result = np.array([2, 3, 5, 6, 7])
    assert np.all(result == correct_result)


def test_full_index_selection():
    """ When testing_mode is set to True, all indices associated with
    ``remaining_occupations`` should just be equal to -1. This feature
    has no use for the end-user, but is useful for unit-testing purposes.
    """
    objID = np.array([4, 4, 6, 9, 9, 9, 10, 10, 15])
    hostIDs = np.array([3, 4, 5, 6, 9, 10, 12, 15, 16])
    host_halo_bins = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3])
    occupations = np.array([2, 1, 1, 0, 3, 3, 0, 2, 1])
    correct_result = np.array([-1, -1, 0, -1, 3, 4, 5, 6, 7, -1, 8, -1, -1])
    result, mask = ssk.subhalo_indexing_array(objID, occupations, hostIDs, host_halo_bins,
        testing_mode=True, fill_remaining_satellites=False)
    msg = "subhalo_indexing_array function is incorrect with testing_mode=True"
    assert np.all(result == correct_result), msg

    result2, mask = ssk.subhalo_indexing_array(objID, occupations, hostIDs, host_halo_bins,
        fill_remaining_satellites=True)
    assert np.all(result[result != -1] == result2[result != -1])

    try:
        selected_objects = objID[result2]
    except IndexError:
        msg = "Remaining subhalo selection returns incorrect indices"
        raise IndexError(msg)

    fake_satellite_mask = result == -1
    fake_satellite_objids = objID[result2[fake_satellite_mask]]
    assert set(fake_satellite_objids[0:2]) <= set((3, 4))
    assert set((fake_satellite_objids[2], )) <= set((5, 6))
    assert set((fake_satellite_objids[3], )) <= set((9, 10))
    assert set(fake_satellite_objids[4:]) <= set((12, 15, 16))


def test_random_indices_within_bin1():
    binned_multiplicity = np.array([1, 1, 1])
    desired_binned_occupations = np.array([2, 3, 4])

    result = ssk.random_indices_within_bin(
        binned_multiplicity, desired_binned_occupations, seed=43)
    correct_result = np.repeat(np.arange(len(binned_multiplicity)), desired_binned_occupations)
    assert np.all(result == correct_result)


def test_random_indices_within_bin2():
    binned_multiplicity = np.array([0, 1, 1])
    desired_binned_occupations = np.array([2, 3, 4])

    with pytest.raises(ValueError) as err:
        result = ssk.random_indices_within_bin(
            binned_multiplicity, desired_binned_occupations, seed=43)
    substr = "Input ``binned_multiplicity`` array must contain "
    assert substr in err.value.args[0]
    substr = "min_required_subs_per_bin = 1"
    assert substr in err.value.args[0]

    with pytest.raises(ValueError) as err:
        result = ssk.random_indices_within_bin(
            binned_multiplicity, desired_binned_occupations,
            seed=43, min_required_subs_per_bin=13)
    substr = "min_required_subs_per_bin = 13"
    assert substr in err.value.args[0]


def test_random_indices_within_bin3():
    """ The purpose of this test is to scour the set of possible inputs for
    edge cases that may not be covered by the hand-tailored tests.
    """

    num_bins = 10
    max_binned_multiplicity = 5
    seed_list = np.arange(100).astype(int)

    test_exists_with_more_subs_than_sats = False
    test_exists_with_more_sats_than_subs = False
    test_exists_with_more_sats_than_subs_alt_corner_case = False
    for seed in seed_list:
        with NumpyRNGContext(seed):
            binned_multiplicity = np.random.randint(1, max_binned_multiplicity, num_bins)
            desired_binned_occupations = np.random.randint(0, 2*max_binned_multiplicity, num_bins)

        result = ssk.random_indices_within_bin(
            binned_multiplicity, desired_binned_occupations, seed=43)

        if binned_multiplicity.sum() > desired_binned_occupations.sum():
            test_exists_with_more_subs_than_sats = True
            if np.any(desired_binned_occupations > binned_multiplicity):
                assert len(result) > len(set(result))
                test_exists_with_more_sats_than_subs_alt_corner_case = True
        if desired_binned_occupations.sum() > binned_multiplicity.sum():
            test_exists_with_more_sats_than_subs = True

        assert len(result) == desired_binned_occupations.sum()
        assert np.all(result >= 0)
        assert np.all(result <= binned_multiplicity.sum())

    # Verify that our chosen seed_list covered both of the following cases
    assert test_exists_with_more_subs_than_sats is True
    assert test_exists_with_more_sats_than_subs is True
    assert test_exists_with_more_sats_than_subs_alt_corner_case is True


def test_calculate_first_idx_unique_array_vals1():
    arr = np.array([1, 1, 2, 2, 2, 3, 3])
    result = ssk.calculate_first_idx_unique_array_vals(arr)
    correct_result = np.array([0, 2, 5])
    assert np.all(result == correct_result)


def test_calculate_first_idx_unique_array_vals2():
    arr = np.array([1, 2, 3])
    result = ssk.calculate_first_idx_unique_array_vals(arr)
    correct_result = np.array([0, 1, 2])
    assert np.all(result == correct_result)


def test_calculate_last_idx_unique_array_vals1():
    arr = np.array([1, 1, 2, 2, 2, 3, 3])
    result = ssk.calculate_last_idx_unique_array_vals(arr)
    correct_result = np.array([1, 4, 6])
    assert np.all(result == correct_result)


def test_calculate_last_idx_unique_array_vals2():
    arr = np.array([1, 2, 3])
    result = ssk.calculate_last_idx_unique_array_vals(arr)
    correct_result = np.array([0, 1, 2])
    assert np.all(result == correct_result)


def test_sum_in_bins1():
    sorted_bin_numbers = np.array([1, 1, 2, 2, 2, 3, 3])
    values_in_bins = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3])
    result = ssk.sum_in_bins(values_in_bins, sorted_bin_numbers)
    correct_result = np.array([0.2, 0.6, 0.6])
    assert np.allclose(result, correct_result)
