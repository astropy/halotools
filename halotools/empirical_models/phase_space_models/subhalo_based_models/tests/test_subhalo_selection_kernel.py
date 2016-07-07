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
