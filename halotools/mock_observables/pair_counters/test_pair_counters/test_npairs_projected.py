"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext

from ..npairs_projected import npairs_projected
from ..pairs import xy_z_npairs as pure_python_brute_force_npairs_projected

from ...tests.cf_helpers import generate_locus_of_3d_points
from ...tests.cf_helpers import generate_3d_regular_mesh

__all__ = ('test_rectangular_mesh_pairs_tight_locus_xy1', )

fixed_seed = 43


def test_rectangular_mesh_pairs_tight_locus_xy1():
    """ Verify that `halotools.mock_observables.npairs_projected` returns
    the correct counts for two tight loci of points.

    In this test, PBCs are irrelevant
    """
    npts1, npts2 = 100, 90
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.2, zc=0.1, seed=fixed_seed)

    rp_bins = np.array((0.05, 0.15, 0.3))
    pi_max = np.max(rp_bins)

    result = npairs_projected(data1, data2, rp_bins, pi_max, period=1)
    assert np.all(result == [0, npts1*npts2, npts1*npts2])


def test_rectangular_mesh_pairs_tight_locus_z1():
    """ Verify that `halotools.mock_observables.npairs_projected` returns
    the correct counts for two tight loci of points.

    In this test, PBCs are irrelevant
    """
    npts1, npts2 = 100, 110
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.1, zc=0.2, seed=fixed_seed)

    rp_bins = np.array((0.05, 0.15, 0.3))

    pi_max = 0.05
    result = npairs_projected(data1, data2, rp_bins, pi_max, period=1)
    assert np.all(result == [0, 0, 0])

    pi_max = 0.15
    result = npairs_projected(data1, data2, rp_bins, pi_max, period=1)
    assert np.all(result == [npts1*npts2, npts1*npts2, npts1*npts2])


def test_rectangular_mesh_pairs_tight_locus_xy2():
    """ Verify that `halotools.mock_observables.npairs_projected` returns
    the correct counts for two tight loci of points.

    In this test, PBCs are important.
    """
    npts1, npts2 = 100, 300
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.05, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.95, zc=0.1, seed=fixed_seed)

    rp_bins = np.array((0.05, 0.15, 0.3))
    pi_max = 0.1

    result = npairs_projected(data1, data2, rp_bins, pi_max, period=1)
    assert np.all(result == [0, npts1*npts2, npts1*npts2])

    result = npairs_projected(data1, data2, rp_bins, pi_max)
    assert np.all(result == [0, 0, 0])


def test_rectangular_mesh_pairs_tight_locus_z2():
    """ Verify that `halotools.mock_observables.npairs_projected` returns
    the correct counts for two tight loci of points.

    In this test, PBCs are important.
    """
    npts1, npts2 = 100, 101
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.05, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.2, zc=0.95, seed=fixed_seed)

    rp_bins = np.array((0.05, 0.15, 0.3))

    pi_max = 0.2
    result = npairs_projected(data1, data2, rp_bins, pi_max, period=1)
    assert np.all(result == [0, npts1*npts2, npts1*npts2])

    pi_max = 0.05
    result = npairs_projected(data1, data2, rp_bins, pi_max, period=1)
    assert np.all(result == [0, 0, 0])

    pi_max = 0.2
    result = npairs_projected(data1, data2, rp_bins, pi_max)
    assert np.all(result == [0, 0, 0])

    pi_max = 0.05
    result = npairs_projected(data1, data2, rp_bins, pi_max)
    assert np.all(result == [0, 0, 0])


def test_rectangular_mesh_pairs_tight_locus_cell1_sizes():
    """ Verify that the pair counters return the correct results
    when operating on a tight locus of points, regardless of how the
    cell sizes are set.

    For this test, PBCs have no impact.
    """
    npts1, npts2 = 100, 90
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.2, zc=0.1, seed=fixed_seed)

    rp_bins = np.array((0.05, 0.15, 0.3))
    pi_max = np.max(rp_bins)

    result1 = npairs_projected(data1, data2, rp_bins, pi_max, period=1,
        approx_cell1_size=0.1)
    assert np.all(result1 == [0, npts1*npts2, npts1*npts2])

    result2 = npairs_projected(data1, data2, rp_bins, pi_max, period=1,
        approx_cell1_size=[0.1, 0.1, 0.1])
    assert np.all(result2 == [0, npts1*npts2, npts1*npts2])

    result3 = npairs_projected(data1, data2, rp_bins, pi_max, period=1,
        approx_cell1_size=[0.3, 0.3, 0.3])
    assert np.all(result3 == [0, npts1*npts2, npts1*npts2])

    result4 = npairs_projected(data1, data2, rp_bins, pi_max, period=1,
        approx_cell1_size=[0.1, 0.3, 0.3])
    assert np.all(result4 == [0, npts1*npts2, npts1*npts2])

    result5 = npairs_projected(data1, data2, rp_bins, pi_max, period=1,
        approx_cell1_size=[0.1, 0.2, 0.3])
    assert np.all(result5 == [0, npts1*npts2, npts1*npts2])


def test_rectangular_mesh_pairs_tight_locus_cell2_sizes():
    """ Verify that the pair counters return the correct results
    when operating on a tight locus of points.

    For this test, PBCs have no impact.
    """
    npts1, npts2 = 100, 90
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.2, zc=0.1, seed=fixed_seed)

    rp_bins = np.array((0.05, 0.15, 0.3))
    pi_max = np.max(rp_bins)

    result1 = npairs_projected(data1, data2, rp_bins, pi_max, period=1,
        approx_cell2_size=0.1)
    assert np.all(result1 == [0, npts1*npts2, npts1*npts2])

    result2 = npairs_projected(data1, data2, rp_bins, pi_max, period=1,
        approx_cell2_size=[0.1, 0.1, 0.1])
    assert np.all(result2 == [0, npts1*npts2, npts1*npts2])

    result3 = npairs_projected(data1, data2, rp_bins, pi_max, period=1,
        approx_cell2_size=[0.3, 0.3, 0.3])
    assert np.all(result3 == [0, npts1*npts2, npts1*npts2])

    result4 = npairs_projected(data1, data2, rp_bins, pi_max, period=1,
        approx_cell2_size=[0.1, 0.3, 0.3])
    assert np.all(result4 == [0, npts1*npts2, npts1*npts2])

    result5 = npairs_projected(data1, data2, rp_bins, pi_max, period=1,
        approx_cell2_size=[0.1, 0.2, 0.3])
    assert np.all(result5 == [0, npts1*npts2, npts1*npts2])


def test_rectangular_mesh_pairs_tight_locus_cell1_cell2_sizes():
    """ Verify that the pair counters return the correct results
    when operating on a tight locus of points.

    For this test, PBCs have no impact.
    """
    npts1, npts2 = 100, 90
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.2, zc=0.1, seed=fixed_seed)

    rp_bins = np.array((0.05, 0.15, 0.3))
    pi_max = np.max(rp_bins)

    result1 = npairs_projected(data1, data2, rp_bins, pi_max, period=1,
        approx_cell1_size=0.1, approx_cell2_size=0.1)
    assert np.all(result1 == [0, npts1*npts2, npts1*npts2])

    result1 = npairs_projected(data1, data2, rp_bins, pi_max, period=1,
        approx_cell1_size=0.2, approx_cell2_size=0.1)
    assert np.all(result1 == [0, npts1*npts2, npts1*npts2])

    result1 = npairs_projected(data1, data2, rp_bins, pi_max, period=1,
        approx_cell1_size=0.1, approx_cell2_size=0.2)
    assert np.all(result1 == [0, npts1*npts2, npts1*npts2])

    result2 = npairs_projected(data1, data2, rp_bins, pi_max, period=1,
        approx_cell1_size=[0.1, 0.1, 0.1], approx_cell2_size=[0.1, 0.1, 0.1])
    assert np.all(result2 == [0, npts1*npts2, npts1*npts2])

    result2 = npairs_projected(data1, data2, rp_bins, pi_max, period=1,
        approx_cell1_size=[0.1, 0.1, 0.1], approx_cell2_size=[0.2, 0.2, 0.2])
    assert np.all(result2 == [0, npts1*npts2, npts1*npts2])

    result2 = npairs_projected(data1, data2, rp_bins, pi_max, period=1,
        approx_cell1_size=[0.2, 0.2, 0.2], approx_cell2_size=[0.1, 0.1, 0.1])
    assert np.all(result2 == [0, npts1*npts2, npts1*npts2])

    result3 = npairs_projected(data1, data2, rp_bins, pi_max, period=1,
        approx_cell2_size=[0.3, 0.3, 0.3])
    assert np.all(result3 == [0, npts1*npts2, npts1*npts2])

    result3 = npairs_projected(data1, data2, rp_bins, pi_max, period=1,
        approx_cell1_size=[0.1, 0.2, 0.3], approx_cell2_size=[0.23, 0.32, 0.11])
    assert np.all(result3 == [0, npts1*npts2, npts1*npts2])


def test_rectangular_mesh_pairs1():
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
    pi_max = grid_spacing/100.
    result = npairs_projected(data1, data2, rp_bins, pi_max,
        period=Lbox, approx_cell1_size=0.1)
    assert np.all(result ==
        [npts_per_dim**3, 5*npts_per_dim**3, 9*npts_per_dim**3, 9*npts_per_dim**3])


def test_rectangular_mesh_pairs2():
    """ Verify that `halotools.mock_observables.npairs_projected` returns
    the correct counts for two regularly spaced grids of points in the limit
    where grid_spacing < pi_max < 2*grid_spacing exceeds the spacing of the grid
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
    result = npairs_projected(data1, data2, rp_bins, pi_max,
        period=Lbox, approx_cell1_size=0.1)
    assert np.all(result ==
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

    serial_result = npairs_projected(data1, data2, rp_bins, pi_max,
        period=Lbox, approx_cell1_size=0.1)
    parallel_result2 = npairs_projected(data1, data2, rp_bins, pi_max,
        period=Lbox, approx_cell1_size=0.1, num_threads=2)
    parallel_result7 = npairs_projected(data1, data2, rp_bins, pi_max,
        period=Lbox, approx_cell1_size=0.1, num_threads=7)
    assert np.all(serial_result == parallel_result2)
    assert np.all(serial_result == parallel_result7)


def test_npairs_projected_brute_force_periodic():
    """
    Function tests npairs with periodic boundary conditions.
    """
    Npts = 1000
    with NumpyRNGContext(fixed_seed):
        random_sample = np.random.random((Npts, 3))
    period = np.array([1.0, 1.0, 1.0])
    rp_bins = np.array([0.001, 0.1, 0.2, 0.3])
    pi_max = 0.1
    pi_bins = [0, pi_max]

    result = npairs_projected(random_sample, random_sample, rp_bins, pi_max,
        period=period)

    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result) == (len(rp_bins),), msg

    test_result = pure_python_brute_force_npairs_projected(
        random_sample, random_sample, rp_bins, pi_bins, period=period)

    msg = "The double tree's result(s) are not equivalent to simple pair counter's."
    assert np.all(test_result[:, 1] == result), msg


def test_npairs_projected_brute_force_non_periodic():
    """
    Function tests npairs with periodic boundary conditions.
    """
    Npts = 1000
    with NumpyRNGContext(fixed_seed):
        random_sample = np.random.random((Npts, 3))
    rp_bins = np.array([0.001, 0.1, 0.2, 0.3])
    pi_max = 0.1
    pi_bins = [0, pi_max]

    result = npairs_projected(random_sample, random_sample, rp_bins, pi_max,
        period=None)

    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result) == (len(rp_bins),), msg

    test_result = pure_python_brute_force_npairs_projected(
        random_sample, random_sample, rp_bins, pi_bins, period=None)

    msg = "The double tree's result(s) are not equivalent to simple pair counter's."
    assert np.all(test_result[:, 1] == result), msg


def test_sensible_num_threads():
    npts1, npts2 = 100, 100
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.1, zc=0.2, seed=fixed_seed)

    rp_bins = np.array((0.05, 0.15, 0.3))
    pi_max = 0.1

    with pytest.raises(ValueError) as err:
        result = npairs_projected(data1, data2, rp_bins, pi_max, period=1,
            num_threads="Cuba Gooding Jr.")
    substr = "Input ``num_threads`` argument must be an integer or the string 'max'"
    assert substr in err.value.args[0]


def test_sensible_rp_bins():
    npts1, npts2 = 100, 100
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.1, zc=0.2, seed=fixed_seed)

    rp_bins = 0.1
    pi_max = 0.1

    with pytest.raises(ValueError) as err:
        result = npairs_projected(data1, data2, rp_bins, pi_max, period=1)
    substr = "Input ``rp_bins`` must be a monotonically increasing 1D array with at least two entries"
    assert substr in err.value.args[0]


def test_sensible_period():
    npts1, npts2 = 100, 100
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.1, zc=0.2, seed=fixed_seed)
    rp_bins = np.array((0.05, 0.15, 0.3))
    pi_max = 0.1

    with pytest.raises(ValueError) as err:
        result = npairs_projected(data1, data2, rp_bins, pi_max, period=np.inf)
    substr = "Input ``period`` must be a bounded positive number in all dimensions"
    assert substr in err.value.args[0]
