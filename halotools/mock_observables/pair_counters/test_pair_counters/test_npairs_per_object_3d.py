""" Module providing testing for the `~halotools.mock_observables.counts_in_cells` function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from .pure_python_npairs_per_object_3d import pure_python_npairs_per_object_3d

from ..npairs_per_object_3d import npairs_per_object_3d

from ...tests.cf_helpers import generate_locus_of_3d_points, generate_3d_regular_mesh

__all__ = ('test1_npairs_per_object_3d', )

fixed_seed = 43


def test1_npairs_per_object_3d():
    """ For ``sample1`` a regular grid and ``sample2`` a tightly locus of points
    in the immediate vicinity of a grid node, verify that the returned counts
    are correct with scalar inputs for proj_search_radius and cylinder_half_length
    """
    period = 1
    # set(sample1) = 0.1, 0.3, 0.5, 0.7, 0.9, with mesh[0,:] = (0.1, 0.1, 0.1)
    sample1 = generate_3d_regular_mesh(5)

    npts2 = 100
    sample2 = generate_locus_of_3d_points(npts2, xc=0.101, yc=0.101, zc=0.101, seed=fixed_seed)

    rbins = [0.0001, 0.02]

    result = npairs_per_object_3d(sample1, sample2, rbins, period=period)
    assert np.shape(result) == (len(sample1), 2)
    assert np.all(result[0, :] == (0, npts2))
    assert np.all(result[1:, :] == 0)


def test2_npairs_per_object_3d():
    """ For ``sample1`` a regular grid and ``sample2`` a tightly locus of points
    in the immediate vicinity of a grid node, verify that the returned counts
    are correct with scalar inputs for proj_search_radius and cylinder_half_length
    """
    period = 1
    # set(sample1) = 0.1, 0.3, 0.5, 0.7, 0.9, with mesh[55,:] = (0.3,  0.5,  0.1)
    sample1 = generate_3d_regular_mesh(5)

    npts2 = 100
    idx_to_test = 55
    sample2 = generate_locus_of_3d_points(npts2, seed=fixed_seed,
        xc=sample1[idx_to_test, 0], yc=sample1[idx_to_test, 1], zc=sample1[idx_to_test, 2])

    rbins = [0.0001, 0.02]

    result = npairs_per_object_3d(sample1, sample2, rbins, period=period)
    assert np.all(result[idx_to_test, :] == (0, npts2))

    # Evidently, result[35, :] == (0, npts2), but the correct indexing should be 55


def test_npairs_per_object_3d_brute_force():
    """
    """
    npts1 = 100
    npts2 = 90
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))

    rbins = [0.05, 0.1, 0.25]
    brute_force_result = pure_python_npairs_per_object_3d(sample1, sample2, rbins)
    result = npairs_per_object_3d(sample1, sample2, rbins)
    assert brute_force_result.shape == result.shape
    assert np.all(result == brute_force_result)


def test_npairs_per_object_3d_brute_force2():
    """
    """
    npts1 = 100
    npts2 = 90
    with NumpyRNGContext(fixed_seed+1):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))

    rbins = [0.05, 0.1, 0.25]
    brute_force_result = pure_python_npairs_per_object_3d(sample1, sample2, rbins, period=1)
    result = npairs_per_object_3d(sample1, sample2, rbins, period=1)
    assert brute_force_result.shape == result.shape
    assert np.all(result == brute_force_result)


def test_npairs_per_object_3d_parallel():
    """ Regression test for GitHub Issue #634.
    """
    npts1 = 100
    npts2 = 90
    with NumpyRNGContext(fixed_seed+1):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))

    rbins = [0.05, 0.1, 0.25]
    serial_result = npairs_per_object_3d(sample1, sample2, rbins, period=1, num_threads=1)
    parallel_result = npairs_per_object_3d(sample1, sample2, rbins, period=1, num_threads=3)
    assert np.shape(serial_result) == np.shape(parallel_result)
    assert np.all(serial_result == parallel_result)
