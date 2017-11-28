"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext

from ..npairs_xy_z import npairs_xy_z
from ..pairs import xy_z_npairs as pure_python_brute_force_npairs_xy_z

from ...tests.cf_helpers import generate_locus_of_3d_points
from ...tests.cf_helpers import generate_3d_regular_mesh

__all__ = ('test_npairs_xy_z_tight_locus1', )

fixed_seed = 43


def test_npairs_xy_z_tight_locus1():
    """ Verify that `halotools.mock_observables.npairs_xy_z` returns
    the correct counts for two tight loci of points.

    In this test, PBCs are irrelevant
    """
    npts1, npts2 = 100, 90
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.2, zc=0.105, seed=fixed_seed)

    rp_bins = np.array((0.05, 0.15, 0.3))
    pi_bins = np.array([0, 0.15])

    result = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1)
    assert np.all(result[:, 0] == [0, 0, 0])
    assert np.all(result[:, 1] == [0, npts1*npts2, npts1*npts2])


def test_rectangular_mesh_pairs_tight_locus2():
    """ Verify that `halotools.mock_observables.npairs_projected` returns
    the correct counts for two tight loci of points.

    In this test, PBCs are important.
    """
    npts1, npts2 = 100, 300
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.05, zc=0.05, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.95, zc=0.95, seed=fixed_seed)

    rp_bins = np.array((0.05, 0.25, 0.3))
    pi_bins = np.array([0.05, 0.15])

    result = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1)
    assert np.all(result[:, 0] == [0, 0, 0])
    assert np.all(result[:, 1] == [0, npts1*npts2, npts1*npts2])

    result = npairs_xy_z(data1, data2, rp_bins, pi_bins)
    assert np.all(result[:, 0] == [0, 0, 0])
    assert np.all(result[:, 1] == [0, 0, 0])


def test_npairs_xy_z_tight_locus_cell1_sizes():
    """ Verify that the pair counters return the correct results
    when operating on a tight locus of points, regardless of how the
    cell sizes are set.

    For this test, PBCs have no impact.
    """
    npts1, npts2 = 100, 90
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.2, zc=0.1, seed=fixed_seed)

    rp_bins = np.array((0.05, 0.15, 0.3))
    pi_bins = np.array([0, 0.3])

    result1 = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1,
        approx_cell1_size=0.1)
    assert np.all(result1[:, 1] == [0, npts1*npts2, npts1*npts2])

    result2 = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1,
        approx_cell1_size=[0.1, 0.1, 0.1])
    assert np.all(result2[:, 1] == [0, npts1*npts2, npts1*npts2])

    result3 = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1,
        approx_cell1_size=[0.3, 0.3, 0.3])
    assert np.all(result3[:, 1] == [0, npts1*npts2, npts1*npts2])

    result4 = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1,
        approx_cell1_size=[0.1, 0.3, 0.3])
    assert np.all(result4[:, 1] == [0, npts1*npts2, npts1*npts2])

    result5 = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1,
        approx_cell1_size=[0.1, 0.2, 0.3])
    assert np.all(result5[:, 1] == [0, npts1*npts2, npts1*npts2])


def test_npairs_xy_z_tight_locus_cell2_sizes():
    """ Verify that the pair counters return the correct results
    when operating on a tight locus of points.

    For this test, PBCs have no impact.
    """
    npts1, npts2 = 100, 90
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.2, zc=0.1, seed=fixed_seed)

    rp_bins = np.array((0.05, 0.15, 0.3))
    pi_bins = np.array([0, 0.3])

    result1 = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1,
        approx_cell2_size=0.1)
    assert np.all(result1[:, 1] == [0, npts1*npts2, npts1*npts2])

    result2 = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1,
        approx_cell2_size=[0.1, 0.1, 0.1])
    assert np.all(result2[:, 1] == [0, npts1*npts2, npts1*npts2])

    result3 = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1,
        approx_cell2_size=[0.3, 0.3, 0.3])
    assert np.all(result3[:, 1] == [0, npts1*npts2, npts1*npts2])

    result4 = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1,
        approx_cell2_size=[0.1, 0.3, 0.3])
    assert np.all(result4[:, 1] == [0, npts1*npts2, npts1*npts2])

    result5 = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1,
        approx_cell2_size=[0.1, 0.2, 0.3])
    assert np.all(result5[:, 1] == [0, npts1*npts2, npts1*npts2])


def test_npairs_xy_z_tight_locus_cell1_cell2_sizes():
    """ Verify that the pair counters return the correct results
    when operating on a tight locus of points.

    For this test, PBCs have no impact.
    """
    npts1, npts2 = 100, 90
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.2, zc=0.1, seed=fixed_seed)

    rp_bins = np.array((0.05, 0.15, 0.3))
    pi_bins = np.array([0, 0.3])

    result1 = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1,
        approx_cell1_size=0.1, approx_cell2_size=0.1)
    assert np.all(result1[:, 1] == [0, npts1*npts2, npts1*npts2])

    result1 = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1,
        approx_cell1_size=0.2, approx_cell2_size=0.1)
    assert np.all(result1[:, 1] == [0, npts1*npts2, npts1*npts2])

    result1 = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1,
        approx_cell1_size=0.1, approx_cell2_size=0.2)
    assert np.all(result1[:, 1] == [0, npts1*npts2, npts1*npts2])

    result2 = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1,
        approx_cell1_size=[0.1, 0.1, 0.1], approx_cell2_size=[0.1, 0.1, 0.1])
    assert np.all(result2[:, 1] == [0, npts1*npts2, npts1*npts2])

    result2 = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1,
        approx_cell1_size=[0.1, 0.1, 0.1], approx_cell2_size=[0.2, 0.2, 0.2])
    assert np.all(result2[:, 1] == [0, npts1*npts2, npts1*npts2])

    result2 = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1,
        approx_cell1_size=[0.2, 0.2, 0.2], approx_cell2_size=[0.1, 0.1, 0.1])
    assert np.all(result2[:, 1] == [0, npts1*npts2, npts1*npts2])

    result3 = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1,
        approx_cell2_size=[0.3, 0.3, 0.3])
    assert np.all(result3[:, 1] == [0, npts1*npts2, npts1*npts2])

    result3 = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1,
        approx_cell1_size=[0.1, 0.2, 0.3], approx_cell2_size=[0.23, 0.32, 0.11])
    assert np.all(result3[:, 1] == [0, npts1*npts2, npts1*npts2])


def test_npairs_xy_z_mesh1():
    """ Verify that `halotools.mock_observables.npairs_projected` returns
    the correct counts for two regularly spaced grids of points in the limit
    where pi_max << grid_spacing
    """
    npts_per_dim = 10
    data1 = generate_3d_regular_mesh(npts_per_dim)
    data2 = generate_3d_regular_mesh(npts_per_dim)
    grid_spacing = 1./npts_per_dim
    Lbox = 1.
    r1 = grid_spacing/100.
    epsilon = 0.0001
    r2 = grid_spacing + epsilon
    r3 = grid_spacing*np.sqrt(2) + epsilon
    r4 = grid_spacing*np.sqrt(3) + epsilon
    rp_bins = np.array([r1, r2, r3, r4])
    pi_max = 1.1*grid_spacing
    pi_bins = np.array([0, pi_max])
    result = npairs_xy_z(data1, data2, rp_bins, pi_bins,
        period=Lbox, approx_cell1_size=0.1)
    assert np.all(result[:, 0] ==
        [npts_per_dim**3, 5*npts_per_dim**3, 9*npts_per_dim**3, 9*npts_per_dim**3])
    assert np.all(result[:, 1] ==
        [3*npts_per_dim**3, 15*npts_per_dim**3, 27*npts_per_dim**3, 27*npts_per_dim**3])


def test_parallel():
    """ Verify that `halotools.mock_observables.npairs_projected` returns
    identical counts whether it is run in serial or parallel.
    """
    npts_per_dim = 10
    data1 = generate_3d_regular_mesh(npts_per_dim)
    data2 = generate_3d_regular_mesh(npts_per_dim)
    grid_spacing = 1./npts_per_dim
    Lbox = 1.
    r1 = grid_spacing/100.
    epsilon = 0.0001
    r2 = grid_spacing + epsilon
    r3 = grid_spacing*np.sqrt(2) + epsilon
    r4 = grid_spacing*np.sqrt(3) + epsilon
    rp_bins = np.array([r1, r2, r3, r4])
    pi_max = 0.1
    pi_bins = np.array([0, pi_max])

    serial_result = npairs_xy_z(data1, data2, rp_bins, pi_bins,
        period=Lbox, approx_cell1_size=0.1)
    parallel_result2 = npairs_xy_z(data1, data2, rp_bins, pi_bins,
        period=Lbox, approx_cell1_size=0.1, num_threads=2)
    parallel_result7 = npairs_xy_z(data1, data2, rp_bins, pi_bins,
        period=Lbox, approx_cell1_size=0.1, num_threads=7)
    assert np.all(serial_result == parallel_result2)
    assert np.all(serial_result == parallel_result7)


def test_npairs_xy_z_brute_force_periodic():
    """
    test npairs_xy_z with periodic boundary conditions.
    """
    npts1, npts2 = 100, 90
    with NumpyRNGContext(fixed_seed):
        data1 = np.random.random((npts1, 3))
        data2 = np.random.random((npts2, 3))

    rp_bins = np.arange(0, 0.31, 0.1)
    pi_bins = np.arange(0, 0.31, 0.1)

    result = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1)
    test_result = pure_python_brute_force_npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1)

    assert np.shape(result) == (len(rp_bins), len(pi_bins))
    assert np.all(result == test_result)


def test_npairs_xy_z_brute_force_non_periodic():
    """
    test npairs_xy_z with periodic boundary conditions.
    """
    npts1, npts2 = 100, 90
    with NumpyRNGContext(fixed_seed):
        data1 = np.random.random((npts1, 3))
        data2 = np.random.random((npts2, 3))

    rp_bins = np.arange(0, 0.31, 0.1)
    pi_bins = np.arange(0, 0.31, 0.1)

    result = npairs_xy_z(data1, data2, rp_bins, pi_bins)
    test_result = pure_python_brute_force_npairs_xy_z(data1, data2, rp_bins, pi_bins)

    assert np.shape(result) == (len(rp_bins), len(pi_bins))
    assert np.all(result == test_result)


def test_sensible_num_threads():
    npts1, npts2 = 100, 100
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.1, zc=0.2, seed=fixed_seed)

    rp_bins = np.array((0.05, 0.15, 0.3))
    pi_max = 0.1
    pi_bins = np.array([0, pi_max])

    with pytest.raises(ValueError) as err:
        result = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1,
            num_threads="Cuba Gooding Jr.")
    substr = "Input ``num_threads`` argument must be an integer or the string 'max'"
    assert substr in err.value.args[0]


def test_sensible_rp_bins():
    npts1, npts2 = 100, 100
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.1, zc=0.2, seed=fixed_seed)

    rp_bins = 0.1
    pi_max = 0.1
    pi_bins = np.array([0, pi_max])

    with pytest.raises(ValueError) as err:
        result = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1)
    substr = "Input ``rp_bins`` must be a monotonically increasing 1D array with at least two entries"
    assert substr in err.value.args[0]


def test_sensible_period():
    npts1, npts2 = 100, 100
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.1, zc=0.2, seed=fixed_seed)
    rp_bins = np.array((0.05, 0.15, 0.3))
    pi_max = 0.1
    pi_bins = np.array([0, pi_max])

    with pytest.raises(ValueError) as err:
        result = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=np.inf)
    substr = "Input ``period`` must be a bounded positive number in all dimensions"
    assert substr in err.value.args[0]


def test_pure_python_npairs_xy_z_argument_handling1():
    """
    """
    npts = 10
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 3))
        sample2 = np.random.random((npts, 2))
    rp_bins = np.linspace(0.01, 0.1, 5)
    pi_bins = np.linspace(0.01, 0.1, 5)

    with pytest.raises(ValueError) as err:
        __ = pure_python_brute_force_npairs_xy_z(sample1, sample2, rp_bins, pi_bins, period=None)
    substr = "sample1 and sample2 inputs do not have the same dimension"
    assert substr in err.value.args[0]


def test_pure_python_npairs_3d_argument_handling3():
    """
    """
    npts = 10
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 2))
        sample2 = np.random.random((npts, 2))
    rp_bins = np.linspace(0.01, 0.1, 5)
    pi_bins = np.linspace(0.01, 0.1, 5)

    with pytest.raises(ValueError) as err:
        __ = pure_python_brute_force_npairs_xy_z(sample1, sample2, rp_bins, pi_bins, period=[1, 1, 1])
    substr = "period should have len == dimension of points"
    assert substr in err.value.args[0]

