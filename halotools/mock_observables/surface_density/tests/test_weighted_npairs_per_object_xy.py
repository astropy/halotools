"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from .pure_python_weighted_npairs_per_object_xy import pure_python_weighted_npairs_per_object_xy
from ..weighted_npairs_per_object_xy import weighted_npairs_per_object_xy

from ...tests.cf_helpers import generate_3d_regular_mesh
from ...tests.cf_helpers import generate_thin_cylindrical_shell_of_points


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


def test_regular_grid1():
    """ For ``sample1`` a regular grid and ``sample2`` a tightly locus of points
    in the immediate vicinity of a grid node, verify that the returned counts
    are correct with scalar inputs for proj_search_radius and cylinder_half_length
    """
    period = 1

    num_pts_per_dim = 5
    centers = generate_3d_regular_mesh(num_pts_per_dim)
    num_cyl = centers.shape[0]

    num_ptcl = 100
    particles = generate_thin_cylindrical_shell_of_points(num_ptcl, 0.01, 0.1,
            xc=0.101, yc=0.101, zc=0.101, seed=fixed_seed)
    masses = np.logspace(2, 5, particles.shape[0])
    rp_bins = np.array((0.005, 0.02))

    result = weighted_npairs_per_object_xy(centers, particles, masses, rp_bins, period=period)
    assert np.shape(result) == (num_cyl, 2)
    assert np.all(result[:, 0] == 0)

    mask = (centers[:, 0] == 0.1) & (centers[:, 1] == 0.1)
    assert np.allclose(result[:, 1][mask], masses.sum())
    assert np.allclose(result[:, 1][~mask], 0.)


def test_weighted_npairs_per_object_xy_parallel():
    """
    """
    npts1, npts2 = 500, 111
    with NumpyRNGContext(fixed_seed):
        data1 = np.random.random((npts1, 2))
        data2 = np.random.random((npts2, 2))
        w2 = np.random.rand(npts2)
    rp_bins = np.array((0.01, 0.1, 0.2, 0.3))

    serial_weighted_counts = weighted_npairs_per_object_xy(data1, data2, w2, rp_bins,
            period=1, num_threads=1)
    parallel_weighted_counts = weighted_npairs_per_object_xy(data1, data2, w2, rp_bins,
            period=1, num_threads=5)
    assert np.allclose(serial_weighted_counts, parallel_weighted_counts)

