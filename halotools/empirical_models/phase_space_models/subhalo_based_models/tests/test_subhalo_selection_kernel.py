"""
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext

from .. import subhalo_selection_kernel as ssk

__all__ = ('test_subhalo_indexing_array1', )

fixed_seed = 43
seed_array = np.arange(0, 10)


def test_subhalo_indexing_array1():
    """ Create a hard-coded specific example for which the indices can be
    determined by hand, and explicitly enforce that the returned values agree.
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
    assert np.all(result[result == -1] != result2[result == -1])

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


def test_subhalo_indexing_array2():
    """
    """
    nhosts= 8
    nbins = int(nhosts/4)
    for seed in seed_array:
        with NumpyRNGContext(seed):
            host_halo_ids = np.random.permutation(np.arange(0, int(1e5)))[0:nhosts]
            subhalo_multiplicity = np.random.randint(0, 3, nhosts)
            subhalo_hostids = np.repeat(host_halo_ids, subhalo_multiplicity)
            satellite_occupations = np.random.randint(0, 3, nhosts)
            host_halo_bin_numbers = np.sort(np.random.randint(0, nbins, nhosts))

            satellite_selection_indices, missing_subhalo_mask = ssk.subhalo_indexing_array(
                subhalo_hostids, satellite_occupations, host_halo_ids, host_halo_bin_numbers,
                testing_mode=True, fill_remaining_satellites=False)

            assert len(satellite_selection_indices) == satellite_occupations.sum()

            nosub_satellite_occupations = satellite_occupations - subhalo_multiplicity
            correct_num_fake_satellites = nosub_satellite_occupations.sum()
            returned_num_fake_satellites = len(satellite_selection_indices[missing_subhalo_mask])
            assert returned_num_fake_satellites == correct_num_fake_satellites


            # subhalo_occupations = ssk.calculate_subhalo_occupations(
            #     satellite_occupations, subhalo_multiplicity)
            # correct_num_selected_subhalos = subhalo_occupations.sum()
            # num_selected_subhalos = len(satellite_selection_indices[satellite_selection_indices != -1])
            # assert num_selected_subhalos == correct_num_selected_subhalos
            # assert len(satellite_selection_indices[missing_subhalo_mask]) == satellite_occupations.sum() - subhalo_occupations.sum()


def test_indices_of_selected_subhalos():
    """
    """
    objID = np.array([0, 0, 5, 5, 5, 7, 8, 8])
    multiplicity = np.array([2, 3, 1, 2])
    occupations = np.array([0, 2, 1, 2])
    result = ssk.indices_of_selected_subhalos(objID, occupations, multiplicity)
    correct_result = np.array([2, 3, 5, 6, 7])
    assert np.all(result == correct_result)
