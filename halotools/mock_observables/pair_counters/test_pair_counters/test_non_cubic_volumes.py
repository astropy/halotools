"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from ..npairs_xy_z import npairs_xy_z
from ..pairs import xy_z_npairs as pure_python_brute_force_npairs_xy_z

__all__ = ('test_npairs_xy_z_brute_force_periodic_non_cubic',
        'test_npairs_xy_z_brute_force_non_periodic_non_cubic')

fixed_seed = 43


def test_npairs_xy_z_brute_force_periodic_non_cubic():
    """
    test npairs_xy_z with periodic boundary conditions.
    """
    npts1, npts2 = 100, 90
    period = [1, 2, 3]
    with NumpyRNGContext(fixed_seed):
        x1 = np.random.uniform(0, period[0], npts1)
        y1 = np.random.uniform(0, period[1], npts1)
        z1 = np.random.uniform(0, period[2], npts1)
        x2 = np.random.uniform(0, period[0], npts2)
        y2 = np.random.uniform(0, period[1], npts2)
        z2 = np.random.uniform(0, period[2], npts2)

    data1 = np.vstack((x1, y1, z1)).T
    data2 = np.vstack((x2, y2, z2)).T

    rp_bins = np.arange(0, 0.31, 0.1)
    pi_bins = np.arange(0, 0.31, 0.1)

    result = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=period)
    test_result = pure_python_brute_force_npairs_xy_z(data1, data2, rp_bins, pi_bins, period=period)

    assert np.shape(result) == (len(rp_bins), len(pi_bins))
    assert np.all(result == test_result)


def test_npairs_xy_z_brute_force_non_periodic_non_cubic():
    """
    test npairs_xy_z with periodic boundary conditions.
    """
    npts1, npts2 = 100, 90
    period = [1, 2, 3]
    with NumpyRNGContext(fixed_seed):
        x1 = np.random.uniform(0, period[0], npts1)
        y1 = np.random.uniform(0, period[1], npts1)
        z1 = np.random.uniform(0, period[2], npts1)
        x2 = np.random.uniform(0, period[0], npts2)
        y2 = np.random.uniform(0, period[1], npts2)
        z2 = np.random.uniform(0, period[2], npts2)

    data1 = np.vstack((x1, y1, z1)).T
    data2 = np.vstack((x2, y2, z2)).T

    rp_bins = np.arange(0, 0.31, 0.1)
    pi_bins = np.arange(0, 0.31, 0.1)

    result = npairs_xy_z(data1, data2, rp_bins, pi_bins)
    test_result = pure_python_brute_force_npairs_xy_z(data1, data2, rp_bins, pi_bins)

    assert np.shape(result) == (len(rp_bins), len(pi_bins))
    assert np.all(result == test_result)
