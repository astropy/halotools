""" Module provides testing for the array_indexing_manipulations functions.
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
import pytest

from .. import array_indexing_manipulations as aim

__all__ = ('test_calculate_first_idx_unique_array_vals1',
    'test_calculate_first_idx_unique_array_vals2', 'test_calculate_last_idx_unique_array_vals1',
    'test_calculate_last_idx_unique_array_vals2', 'test_sum_in_bins1')


def test_calculate_first_idx_unique_array_vals0():
    """ This function ensures that the appropriate exception
    will be raised when the function is executed with
    testing_mode set to True.
    """
    arr = np.arange(5)[::-1]
    __ = aim.calculate_first_idx_unique_array_vals(arr, testing_mode=False)
    with pytest.raises(ValueError) as err:
        __ = aim.calculate_first_idx_unique_array_vals(arr, testing_mode=True)
    substr = "Input ``sorted_array`` array must be sorted in ascending order"
    assert substr in err.value.args[0]


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


def test_calculate_first_idx_unique_array_vals3():
    """ This test uses random arrays to verify that the following hold:

    1. The length of the returned result equals the number of unique elements in ``arr``

    2. The entries of the returned result are unique

    3. All elements in arr[result[i]:result[i+1]] are the same

    4. arr[result[i]] != arr[result[i]-1]

    5. arr[result[i]] != arr[result[i+1]]

    6. arr[result[-1]:] == arr.max()
    """
    low, high = -1000, 1000
    npts = int(10*(high-low))
    seed_list = [1, 100, 500, 999]
    num_random_indices_to_test = 100
    for seed in seed_list:
        with NumpyRNGContext(seed):
            arr = np.sort(np.random.randint(low, high, npts))
            result = aim.calculate_first_idx_unique_array_vals(arr)
            assert result[0] == 0
            assert len(result) == len(set(arr))
            assert len(result) == len(set(result))

            # test the outer edge
            assert np.all(arr[result[-1]:] == arr.max())

            # test random indices and random arrays
            random_idx_to_test = np.random.choice(np.arange(1, len(result)-2),
                size=num_random_indices_to_test)
            for elt in random_idx_to_test:
                first = result[elt]
                last = result[elt+1]-1
                assert len(set(arr[first:last+1])) == 1
                assert arr[first] != arr[first-1]
                assert arr[first] != arr[last+1]
                assert arr[last] != arr[last+1]


def test_calculate_last_idx_unique_array_vals0():
    """ This function ensures that the appropriate exception
    will be raised when the function is executed with
    testing_mode set to True.
    """
    arr = np.arange(5)[::-1]
    __ = aim.calculate_last_idx_unique_array_vals(arr, testing_mode=False)

    with pytest.raises(ValueError) as err:
        __ = aim.calculate_last_idx_unique_array_vals(arr, testing_mode=True)
    substr = "Input ``sorted_array`` array must be sorted in ascending order"
    assert substr in err.value.args[0]


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


def test_calculate_last_idx_unique_array_vals3():
    """ This test uses random arrays to verify that the following hold:

    1. The length of the returned result equals the number of unique elements in ``arr``

    2. The entries of the returned result are unique

    3. All elements in arr[result[i]:result[i+1]] are the same

    4. arr[result[i]] != arr[result[i]-1]

    5. arr[result[i]] != arr[result[i+1]]

    6. arr[:result[0]+1] == arr.min()
    """
    low, high = -1000, 1000
    npts = int(10*(high-low))
    seed_list = [1, 100, 500, 999]
    num_random_indices_to_test = 100
    for seed in seed_list:
        with NumpyRNGContext(seed):
            arr = np.sort(np.random.randint(low, high, npts))
            result = aim.calculate_last_idx_unique_array_vals(arr)
            assert result[-1] == len(arr)-1
            assert len(result) == len(set(arr))
            assert len(result) == len(set(result))

            # test the outer edge
            assert np.all(arr[:result[0]+1] == arr.min())

            # test random indices and random arrays
            random_idx_to_test = np.random.choice(np.arange(1, len(result)-2),
                size=num_random_indices_to_test)
            for elt in random_idx_to_test:
                last = result[elt]
                first = result[elt-1]+1
                assert len(set(arr[first:last+1])) == 1
                assert arr[first] != arr[first-1]
                assert arr[first] != arr[last+1]
                assert arr[last] != arr[last+1]


def test_sum_in_bins1():
    sorted_bin_numbers = np.array([1, 1, 2, 2, 2, 3, 3])
    values_in_bins = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3])
    result = aim.sum_in_bins(values_in_bins, sorted_bin_numbers)
    correct_result = np.array([0.2, 0.6, 0.6])
    assert np.allclose(result, correct_result)


def test_sum_in_bins2():
    with pytest.raises(ValueError) as err:
        result = aim.sum_in_bins([1, 2], [1, 2, 3])
    substr = "Input ``arr`` and ``sorted_bin_numbers`` must have same length"
    assert substr in err.value.args[0]


def test_sum_in_bins3():
    with pytest.raises(ValueError) as err:
        result = aim.sum_in_bins([1, 2, 3], [1, 2, 1], testing_mode=True)
    substr = "Input ``sorted_bin_numbers`` array must be sorted in ascending order"
    assert substr in err.value.args[0]


def test_sum_in_bins4():
    """ This test uses random arrays to verify that the following hold:

    1. The length of the returned result equals the number of unique elements in ``sorted_bin_numbers``

    2. Randomly chosen entries are correct when calculated manually

    Test (2) is also a highly non-trivial integration test that the
    sum_in_bins, calculate_first_idx_unique_array_vals and
    calculate_last_idx_unique_array_vals work properly in concert with one another.
    """
    low, high = -1000, 1000
    npts = int(10*(high-low))
    seed_list = [1, 100, 500, 999]
    num_random_indices_to_test = 100
    for seed in seed_list:
        with NumpyRNGContext(seed):
            sorted_bin_numbers = np.sort(np.random.randint(low, high, npts))
            values_in_bins = np.random.rand(npts)
            result = aim.sum_in_bins(values_in_bins, sorted_bin_numbers)
            assert len(result) == len(set(sorted_bin_numbers))

            first_idx_array = aim.calculate_first_idx_unique_array_vals(sorted_bin_numbers)
            last_idx_array = aim.calculate_last_idx_unique_array_vals(sorted_bin_numbers)
            entries_to_test = np.random.choice(np.arange(len(first_idx_array)),
                num_random_indices_to_test)
            for i in entries_to_test:
                first, last = first_idx_array[i], last_idx_array[i]
                assert len(set(sorted_bin_numbers[first:last+1])) == 1
                correct_result = np.sum(values_in_bins[first:last+1])
                result_i = result[i]
                assert np.allclose(correct_result, result_i, rtol=0.0001)


def test_random_indices_within_bin_stochasticity():
    binned_multiplicity = np.ones(100)*5
    desired_binned_occupations = np.arange(100)
    result1 = aim.random_indices_within_bin(
        binned_multiplicity, desired_binned_occupations, seed=43)
    result2 = aim.random_indices_within_bin(
        binned_multiplicity, desired_binned_occupations, seed=43)
    result3 = aim.random_indices_within_bin(
        binned_multiplicity, desired_binned_occupations, seed=44)

    assert np.all(result1 == result2)
    assert not np.all(result1 == result3)


def test_random_indices_within_bin1():
    binned_multiplicity = np.array([1, 1, 1])
    desired_binned_occupations = np.array([2, 3, 4])

    result = aim.random_indices_within_bin(
        binned_multiplicity, desired_binned_occupations, seed=43)
    correct_result = np.repeat(np.arange(len(binned_multiplicity)), desired_binned_occupations)
    assert np.all(result == correct_result)


def test_random_indices_within_bin2():
    binned_multiplicity = np.array([0, 1, 1])
    desired_binned_occupations = np.array([2, 3, 4])

    with pytest.raises(ValueError) as err:
        result = aim.random_indices_within_bin(
            binned_multiplicity, desired_binned_occupations, seed=43)
    substr = "Input ``binned_multiplicity`` array must contain "
    assert substr in err.value.args[0]
    substr = "min_required_entries_per_bin = 1"
    assert substr in err.value.args[0]

    # Verify that this limit does not apply for entries with desired_binned_occupations = 0
    result = aim.random_indices_within_bin(
        binned_multiplicity, np.array([0, 3, 4]), seed=43)

    with pytest.raises(ValueError) as err:
        result = aim.random_indices_within_bin(
            binned_multiplicity, desired_binned_occupations,
            seed=43, min_required_entries_per_bin=13)
    substr = "min_required_entries_per_bin = 13"
    assert substr in err.value.args[0]


@pytest.mark.installation_test
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

        result = aim.random_indices_within_bin(
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


def test_calculate_entry_multiplicity():
    """
    """
    low, high = -1000, 1000
    unique_possible_hostids = np.arange(low, high)
    seed_list = [1, 100, 500, 999]
    for seed in seed_list:
        with NumpyRNGContext(seed):
            correct_multiplicity = np.random.randint(0, 5, len(unique_possible_hostids))
            sorted_repeated_hostids = np.repeat(unique_possible_hostids, correct_multiplicity)

            multiplicity = aim.calculate_entry_multiplicity(
                sorted_repeated_hostids, unique_possible_hostids)
            assert np.all(multiplicity == correct_multiplicity)


def test_calculate_entry_multiplicity2():
    """ Verify that the calculate_entry_multiplicity catches an unsorted input
    and raises the correct error message.
    """
    sorted_repeated_hostids = np.arange(10)[::-1]
    unique_possible_hostids = np.arange(10)

    # The following line should not raise an exception
    __ = aim.calculate_entry_multiplicity(sorted_repeated_hostids, unique_possible_hostids,
        testing_mode=False)

    with pytest.raises(ValueError) as err:
        __ = aim.calculate_entry_multiplicity(
            sorted_repeated_hostids, unique_possible_hostids, testing_mode=True)
    substr = "Input ``sorted_repeated_hostids`` array is not sorted in ascending order"
    assert substr in err.value.args[0]


def test_calculate_entry_multiplicity3():
    """ Verify that the calculate_entry_multiplicity catches a unique_possible_hostids
    array with repeated entries and raises the correct error message.
    """
    sorted_repeated_hostids = np.arange(10)
    unique_possible_hostids = np.append(np.arange(10), 0)

    with pytest.raises(ValueError) as err:
        __ = aim.calculate_entry_multiplicity(
            sorted_repeated_hostids, unique_possible_hostids, testing_mode=True)
    substr = "All entries of ``unique_possible_hostids`` must be unique"
    assert substr in err.value.args[0]


def test_calculate_entry_multiplicity4():
    """ Verify that the calculate_entry_multiplicity catches an entry of
    sorted_repeated_hostids that does not appear in unique_possible_hostids
    and raises the correct error message.
    """
    sorted_repeated_hostids = np.arange(11)
    unique_possible_hostids = np.arange(10)

    with pytest.raises(ValueError) as err:
        __ = aim.calculate_entry_multiplicity(
            sorted_repeated_hostids, unique_possible_hostids, testing_mode=True)
    substr = "must appear in unique_possible_hostids."
    assert substr in err.value.args[0]
