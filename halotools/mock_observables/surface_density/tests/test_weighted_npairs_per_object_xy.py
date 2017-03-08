"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.tests.helper import pytest
from astropy.utils.misc import NumpyRNGContext

from .pure_python_weighted_npairs_per_object_xy import pure_python_weighted_npairs_per_object_xy
from ..weighted_npairs_xy import weighted_npairs_xy
from ..weighted_npairs_per_object_xy import weighted_npairs_per_object_xy

from ...tests.cf_helpers import generate_locus_of_3d_points

__all__ = ('test_weighted_npairs_per_object_xy_brute_force_pbc', )

fixed_seed = 43


def test_weighted_npairs_per_object_xy_brute_force_pbc():
    """
    """
    npts1, npts2 = 500, 111
    with NumpyRNGContext(fixed_seed):
        data1 = np.random.random((npts1, 2))
        data2 = np.random.random((npts2, 2))
        w2 = np.random.rand(npts2)
    rp_bins = np.array((0.01, 0.1, 0.2, 0.3))
    xperiod, yperiod = 1, 1

    xarr1, yarr1 = data1[:, 0], data1[:, 1]
    xarr2, yarr2 = data2[:, 0], data2[:, 1]
    counts, python_weighted_counts = pure_python_weighted_npairs_per_object_xy(
        xarr1, yarr1, xarr2, yarr2, w2, rp_bins, xperiod, yperiod)

    cython_weighted_counts = weighted_npairs_per_object_xy(data1, data2, w2, rp_bins, period=1)
    assert np.allclose(cython_weighted_counts, python_weighted_counts)

    # Verify the PBC enforcement is non-trivial
    cython_weighted_counts = weighted_npairs_per_object_xy(data1, data2, w2, rp_bins)
    assert not np.allclose(cython_weighted_counts, python_weighted_counts)


def test_weighted_npairs_per_object_xy_brute_force_no_pbc():
    """
    """
    npts1, npts2 = 500, 111
    with NumpyRNGContext(fixed_seed):
        data1 = np.random.random((npts1, 2))
        data2 = np.random.random((npts2, 2))
        w2 = np.random.rand(npts2)
    rp_bins = np.array((0.01, 0.1, 0.2, 0.3))
    xperiod, yperiod = np.inf, np.inf

    xarr1, yarr1 = data1[:, 0], data1[:, 1]
    xarr2, yarr2 = data2[:, 0], data2[:, 1]
    counts, python_weighted_counts = pure_python_weighted_npairs_per_object_xy(
        xarr1, yarr1, xarr2, yarr2, w2, rp_bins, xperiod, yperiod)

    cython_weighted_counts = weighted_npairs_per_object_xy(data1, data2, w2, rp_bins)
    assert np.allclose(cython_weighted_counts, python_weighted_counts)

    # Verify the PBC enforcement is non-trivial
    cython_weighted_counts = weighted_npairs_per_object_xy(data1, data2, w2, rp_bins, period=1)
    assert not np.allclose(cython_weighted_counts, python_weighted_counts)




