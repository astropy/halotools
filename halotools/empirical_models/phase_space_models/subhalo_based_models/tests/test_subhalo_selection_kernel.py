"""
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext

from .. import subhalo_selection_kernel as ssk

__all__ = ('test_calculate_satellite_selection_mask1', )

fixed_seed = 43
seed_array = np.arange(0, 10)


def test_calculate_satellite_selection_mask1():
    """ Create a hard-coded specific example for which the indices can be
    determined by hand, and explicitly enforce that the returned values agree.
    """
    objID = np.array([4, 4, 6, 9, 9, 9, 10, 10, 15])
    hostIDs = np.array([3, 4, 5, 6, 9, 10, 12, 15, 16])
    host_halo_bins = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3])
    occupations = np.array([2, 1, 1, 0, 3, 3, 0, 2, 1])
    correct_result = np.array([-1, -1, 0, -1, 3, 4, 5, 6, 7, -1, 8, -1, -1])
    result, mask = ssk.calculate_satellite_selection_mask(objID, occupations, hostIDs, host_halo_bins,
        testing_mode=True, fill_remaining_satellites=False)
    msg = "calculate_satellite_selection_mask function is incorrect with testing_mode=True"
    assert np.all(result == correct_result), msg
    assert np.all(result[mask] == -1)

    result2, mask2 = ssk.calculate_satellite_selection_mask(objID, occupations, hostIDs, host_halo_bins,
        fill_remaining_satellites=True)
    assert np.all(result[~mask] == result2[~mask])
    assert np.all(result[mask] != result2[mask])

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


def test_calculate_satellite_selection_mask2():
    """ Create a sequence of randomly selected inputs and verify that
    the calculate_satellite_selection_mask function returns sensible results
    """
    nhosts = 1000
    for seed in seed_array:
        with NumpyRNGContext(seed):
            host_halo_ids = np.sort(np.random.permutation(np.arange(0, int(1e5)))[0:nhosts])

            max_num_subs_per_halo = np.random.randint(3, 5)
            subhalo_multiplicity = np.random.randint(0, max_num_subs_per_halo, nhosts)
            subhalo_hostids = np.repeat(host_halo_ids, subhalo_multiplicity)

            max_num_sats_per_halo = np.random.randint(3, 5)
            satellite_occupations = np.random.randint(0, max_num_sats_per_halo, nhosts)

            subhalo_occupations = ssk.calculate_subhalo_occupations(
                satellite_occupations, subhalo_multiplicity)
            non_subhalo_occupations = satellite_occupations - subhalo_occupations

            nbins = np.random.randint(1, min(10, int(nhosts/10)))

            __ = np.random.randint(0, nbins, nhosts)
            __[0:nbins] = np.arange(nbins)
            host_halo_bin_numbers = np.sort(__)

            satellite_selection_indices, missing_subhalo_mask = ssk.calculate_satellite_selection_mask(
                subhalo_hostids, satellite_occupations, host_halo_ids, host_halo_bin_numbers,
                testing_mode=True, fill_remaining_satellites=False, seed=seed)

            # Verify that the correct number of indices are returned
            assert len(satellite_selection_indices) == satellite_occupations.sum()
            assert len(satellite_selection_indices[missing_subhalo_mask]) == non_subhalo_occupations.sum()

            # Verify that all masked entries are -1, and no others
            assert np.all(satellite_selection_indices[missing_subhalo_mask] == -1)
            assert not np.any(satellite_selection_indices[~missing_subhalo_mask] == -1)

            satellite_selection_indices2, missing_subhalo_mask2 = ssk.calculate_satellite_selection_mask(
                subhalo_hostids, satellite_occupations, host_halo_ids, host_halo_bin_numbers,
                testing_mode=True, fill_remaining_satellites=True, seed=None)

            # Verify that the returned array can successfully serve as a fancy indexing mask
            __ = subhalo_hostids[satellite_selection_indices2]

            #  Selection of true subhalos should be deterministic
            result1 = subhalo_hostids[satellite_selection_indices[~missing_subhalo_mask]]
            result2 = subhalo_hostids[satellite_selection_indices2[~missing_subhalo_mask2]]
            assert np.all(result1 == result2)


def test_array_weave1():
    nhosts1, nhosts2 = 5, 4
    mult1 = np.ones(nhosts1, dtype=int)
    mult2 = np.ones(nhosts2, dtype=int)
    uval1 = np.arange(len(mult1))
    uval2 = np.arange(len(mult2))
    val1 = np.repeat(uval1, mult1)
    val2 = np.repeat(uval2, mult2)

    with pytest.raises(ValueError) as err:
        __ = ssk.array_weave(val1, val2, mult1, mult2, testing_mode=True)
    substr = "Input ``mult1`` and ``mult2`` arrays must have equal length"
    assert substr in err.value.args[0]


def test_array_weave2():
    nhosts1, nhosts2 = 5, 5
    mult1 = np.ones(nhosts1, dtype=int)
    mult2 = np.ones(nhosts2, dtype=int)
    uval1 = np.arange(len(mult1))
    uval2 = np.arange(len(mult2))
    val1 = np.repeat(uval1, mult1)
    val2 = np.repeat(uval2, mult2)

    with pytest.raises(ValueError) as err:
        __ = ssk.array_weave(val1[1:], val2, mult1, mult2, testing_mode=True)
    substr = "The sum of the ``mult1`` entries should equal the length of ``val1``"
    assert substr in err.value.args[0]


def test_array_weave3():
    nhosts1, nhosts2 = 5, 5
    mult1 = np.ones(nhosts1, dtype=int)
    mult2 = np.ones(nhosts2, dtype=int)
    uval1 = np.arange(len(mult1))
    uval2 = np.arange(len(mult2))
    val1 = np.repeat(uval1, mult1)
    val2 = np.repeat(uval2, mult2)

    with pytest.raises(ValueError) as err:
        __ = ssk.array_weave(val1, val2[1:], mult1, mult2, testing_mode=True)
    substr = "The sum of the ``mult2`` entries should equal the length of ``val2``"
    assert substr in err.value.args[0]


def test_indices_of_selected_subhalos1():
    """
    """
    objID = np.array([0, 0, 5, 5, 5, 7, 8, 8])
    multiplicity = np.array([2, 3, 1, 2])
    occupations = np.array([0, 2, 1, 2])
    result = ssk.indices_of_selected_subhalos(objID, occupations, multiplicity)
    correct_result = np.array([2, 3, 5, 6, 7])
    assert np.all(result == correct_result)


def test_indices_of_selected_subhalos2():
    """
    """
    objID = np.array([0, 0, 5, 5, 5, 7, 8, 8])
    multiplicity = np.array([2, 3, 1, 2])
    occupations = np.array([0, 2, 1, 2])
    with pytest.raises(ValueError) as err:
        __ = ssk.indices_of_selected_subhalos(objID, occupations[1:], multiplicity,
            testing_mode=True)
    substr = "Input ``subhalo_occupations`` and ``subhalo_multiplicity`` must have the same length"
    assert substr in err.value.args[0]


def test_indices_of_selected_subhalos3():
    """
    """
    objID = np.arange(8)
    multiplicity = np.array([2, 3, 1, 2])
    occupations = np.array([0, 2, 1, 2])
    with pytest.raises(ValueError) as err:
        __ = ssk.indices_of_selected_subhalos(objID, occupations, multiplicity,
            testing_mode=True)
    substr = "The host halo of each subhalo must be represented"
    assert substr in err.value.args[0]


def test_indices_of_selected_subhalos4():
    """
    """
    objID = np.array([0, 0, 5, 5, 5, 7, 8, 8])
    multiplicity = np.array([2, 3, 1, 2])
    occupations = np.array([0, 1, 2, 2])
    with pytest.raises(ValueError) as err:
        __ = ssk.indices_of_selected_subhalos(objID, occupations, multiplicity,
            testing_mode=True)
    substr = "No entry of ``subhalo_occupations`` may exceed"
    assert substr in err.value.args[0]


def test_indices_of_selected_subhalos5():
    """
    """
    objID = np.array([0, 0, 5, 5, 5, 7, 8, 8])
    multiplicity = np.array([2, 2, 1, 2])
    occupations = np.array([0, 2, 1, 2])

    with pytest.raises(ValueError) as err:
        __ = ssk.indices_of_selected_subhalos(objID, occupations, multiplicity,
            testing_mode=True)
    substr = "The sum of ``subhalo_multiplicity`` is"
    assert substr in err.value.args[0]
    substr = "which is inconsistent with the total number of entries of ``subhalo_hostids``"
    assert substr in err.value.args[0]
