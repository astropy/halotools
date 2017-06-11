"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext

from .pure_python_weighted_npairs_xy import pure_python_weighted_npairs_xy
from ..weighted_npairs_xy import weighted_npairs_xy

from ...tests.cf_helpers import generate_locus_of_3d_points

__all__ = ('test_weighted_npairs_xy_brute_force_pbc', )

fixed_seed = 43


def test_weighted_npairs_xy_brute_force_pbc():
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
    counts, python_weighted_counts = pure_python_weighted_npairs_xy(
        xarr1, yarr1, xarr2, yarr2, w2, rp_bins, xperiod, yperiod)

    cython_weighted_counts = weighted_npairs_xy(data1, data2, w2, rp_bins, period=1)
    assert np.allclose(cython_weighted_counts, python_weighted_counts)

    # Verify the PBC enforcement is non-trivial
    cython_weighted_counts = weighted_npairs_xy(data1, data2, w2, rp_bins)
    assert not np.allclose(cython_weighted_counts, python_weighted_counts)


def test_weighted_npairs_xy_brute_force_no_pbc():
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
    counts, python_weighted_counts = pure_python_weighted_npairs_xy(
        xarr1, yarr1, xarr2, yarr2, w2, rp_bins, xperiod, yperiod)

    cython_weighted_counts = weighted_npairs_xy(data1, data2, w2, rp_bins, period=None)
    assert np.allclose(cython_weighted_counts, python_weighted_counts)

    # Verify the PBC enforcement is non-trivial
    cython_weighted_counts = weighted_npairs_xy(data1, data2, w2, rp_bins, period=1)
    assert not np.allclose(cython_weighted_counts, python_weighted_counts)


def test_weighted_npairs_xy_tight_locus1():
    """ Verify that `halotools.mock_observables.weighted_npairs_xy` returns
    the correct weighted counts for two tight loci of points with constant weights.

    In this test, PBCs are irrelevant
    """
    npts1, npts2 = 100, 110
    data1 = generate_locus_of_3d_points(npts1, xc=0.05, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.1, zc=0.9, seed=fixed_seed)

    w = np.e
    weights2 = np.zeros(npts2) + w

    rp_bins = np.array((0.025, 0.1, 0.3))

    weighted_result = weighted_npairs_xy(data1, data2, weights2, rp_bins, period=1)
    assert np.allclose(weighted_result, [0, w*npts1*npts2, w*npts1*npts2])


def test_weighted_npairs_xy_tight_locus2():
    """ Verify that `halotools.mock_observables.weighted_npairs_xy` returns
    the correct weighted counts for two tight loci of points with constant weights.

    In this test, PBCs are irrelevant
    """
    npts1, npts2 = 100, 110
    data1 = generate_locus_of_3d_points(npts1, xc=0.05, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.95, yc=0.1, zc=0.9, seed=fixed_seed)

    w = np.e
    weights2 = np.zeros(npts2) + w

    rp_bins = np.array((0.025, 0.15, 0.3))

    weighted_result = weighted_npairs_xy(data1, data2, weights2, rp_bins, period=1)
    assert np.allclose(weighted_result, [0, w*npts1*npts2, w*npts1*npts2])


def test_weighted_npairs_xy_tight_locus3():
    """ Verify that `halotools.mock_observables.weighted_npairs_xy` returns
    the correct counts for two tight loci of points with variable weights.

    In this test, PBCs are important.
    """
    npts1, npts2 = 100, 300
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.05, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.95, zc=0.1, seed=fixed_seed)
    with NumpyRNGContext(fixed_seed):
        weights2 = np.random.rand(npts2)

    rp_bins = np.array((0.05, 0.15, 0.3))

    weighted_result = weighted_npairs_xy(data1, data2, weights2, rp_bins, period=1)
    correct_result = np.array((0, npts1*weights2.sum(), npts1*weights2.sum()))
    assert np.allclose(weighted_result, correct_result)


def test_parallel():
    """ Verify that `halotools.mock_observables.weighted_npairs_xy` returns
    identical counts whether it is run in serial or parallel.
    """
    Lbox = 15
    npts1, npts2 = 1500, 700
    with NumpyRNGContext(fixed_seed):
        data1 = np.random.random((npts1, 3))*Lbox
        data2 = np.random.random((npts2, 3))*Lbox
        weights2 = np.random.rand(data2.shape[0])

    rp_bins = np.linspace(0.05, 0.25, 15)

    serial_result = weighted_npairs_xy(data1, data2, weights2, rp_bins,
        period=Lbox, approx_cell1_size=0.1)
    parallel_result2 = weighted_npairs_xy(data1, data2, weights2, rp_bins,
        period=Lbox, approx_cell1_size=0.15, num_threads=2)
    parallel_result7 = weighted_npairs_xy(data1, data2, weights2, rp_bins,
        period=Lbox, approx_cell1_size=0.31, num_threads=7)
    assert np.allclose(serial_result, parallel_result2)
    assert np.allclose(serial_result, parallel_result7)


def test_sensible_num_threads():
    npts1, npts2 = 100, 100
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.1, zc=0.2, seed=fixed_seed)
    with NumpyRNGContext(fixed_seed):
        weights2 = np.random.rand(data2.shape[0])

    rp_bins = np.array((0.05, 0.15, 0.3))

    with pytest.raises(ValueError) as err:
        result = weighted_npairs_xy(data1, data2, weights2, rp_bins, period=1,
            num_threads="Cuba Gooding Jr.")
    substr = "Input ``num_threads`` argument must be an integer or the string 'max'"
    assert substr in err.value.args[0]


def test_sensible_rp_bins():
    npts1, npts2 = 100, 100
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.1, zc=0.2, seed=fixed_seed)
    with NumpyRNGContext(fixed_seed):
        weights2 = np.random.rand(data2.shape[0])

    rp_bins = 0.1

    with pytest.raises(ValueError) as err:
        result = weighted_npairs_xy(data1, data2, weights2, rp_bins, period=1)
    substr = "Input ``rp_bins`` must be a monotonically increasing 1D array with at least two entries"
    assert substr in err.value.args[0]


def test_sensible_period():
    npts1, npts2 = 100, 100
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.1, zc=0.2, seed=fixed_seed)
    with NumpyRNGContext(fixed_seed):
        weights2 = np.random.rand(data2.shape[0])

    rp_bins = np.array((0.05, 0.15, 0.3))

    with pytest.raises(ValueError) as err:
        result = weighted_npairs_xy(data1, data2, weights2, rp_bins, period=np.inf)
    substr = "Input ``period`` must be a bounded positive number in all dimensions"
    assert substr in err.value.args[0]
