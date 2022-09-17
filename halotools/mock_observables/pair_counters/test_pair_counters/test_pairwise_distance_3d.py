"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from astropy.utils.misc import NumpyRNGContext
import pytest

from .pure_python_distance_matrix import (
    pure_python_distance_matrix_3d,
    pure_python_distance_matrix_xy_z,
)

from ..pairwise_distance_3d import pairwise_distance_3d, _get_r_max
from ..pairwise_distance_xy_z import pairwise_distance_xy_z

from ...tests.cf_helpers import generate_locus_of_3d_points
from ...tests.cf_helpers import generate_3d_regular_mesh

fixed_seed = 43


def test_pairwise_distance_3d_periodic_mesh_grid_1():
    """
    test that each point is paired with its neighbor on a regular grid
    """
    # regular grid
    Npts_per_dim = 10
    mesh_sample = generate_3d_regular_mesh(Npts_per_dim)
    period = 1.0

    # test on a uniform grid
    rmax = 0.10001
    m = pairwise_distance_3d(mesh_sample, mesh_sample, rmax, period=period)

    # each point has 7 connections including 1 self connection
    # N = (10^3)*7
    assert m.getnnz() == 7000

    # diagonal self matches should have distance 0
    assert np.all(m.diagonal() == 0.0)

    # off diagonal should all be 0.1
    i, j = m.nonzero()
    assert np.allclose(np.asarray(m.todense()[i, j]), 0.1)


def test_pairwise_distance_3d_nonperiodic_mesh_grid_1():
    """
    test that each point is paired with its neighbor on a regular grid
    """
    # regular grid
    Npts_per_dim = 10
    mesh_sample = generate_3d_regular_mesh(Npts_per_dim)
    period = 1.0

    # test on a uniform grid
    rmax = 0.10001
    m = pairwise_distance_3d(mesh_sample, mesh_sample, rmax, period=None)

    # each point has 7 connections including 1 self connection
    # N = (10^3)*7
    # points on the 6 faces have fewer connections N - (10*10*6) = 6400
    # overlapping accounts for edges (2 less) and corners (3 less)
    assert m.getnnz() == 6400

    # diagonal self matches should have distance 0
    assert np.all(m.diagonal() == 0.0)

    # off diagonal should all be 0.1
    i, j = m.nonzero()
    assert np.allclose(np.asarray(m.todense()[i, j]), 0.1)


def test_pairwise_distance_3d_nonperiodic_random_1():
    """
    test that each point is paired with every point when rmax is large enough
    """
    # random points
    Npts = 10
    with NumpyRNGContext(fixed_seed):
        random_sample = np.random.random((Npts, 3))
    period = 1.0

    rmax = 10.0
    m = pairwise_distance_3d(random_sample, random_sample, rmax, period=None)

    # each point is paired with every other point
    assert m.getnnz() == Npts**2


def test_pairwise_distance_3d_periodic_tight_locus1():
    """
    test that each point in two loci of points are paired across PBCs when rmax
    is large enough
    """
    # tigh locus
    Npts1 = 10
    Npts2 = 10
    data1 = generate_locus_of_3d_points(
        Npts1, xc=0.05, yc=0.05, zc=0.05, seed=fixed_seed
    )
    data2 = generate_locus_of_3d_points(
        Npts2, xc=0.95, yc=0.95, zc=0.95, seed=fixed_seed
    )
    period = 1.0

    # should be no connections
    rmax = 0.01
    m = pairwise_distance_3d(data1, data2, rmax, period=period)

    # each point has 0 connections including 1 self connection
    assert m.getnnz() == 0

    # should be 10*10 connections
    rmax = 0.3
    m = pairwise_distance_3d(data1, data2, rmax, period=period)

    # each point has 10 connections
    assert m.getnnz() == Npts1 * Npts2


def test_pairwise_distance_3d_nonperiodic_tight_locus1():
    """
    test that each point in two loci of points are NOT paired across PBCs
    """
    # tight locus
    Npts1 = 10
    Npts2 = 10
    data1 = generate_locus_of_3d_points(
        Npts1, xc=0.05, yc=0.05, zc=0.05, seed=fixed_seed
    )
    data2 = generate_locus_of_3d_points(
        Npts2, xc=0.95, yc=0.95, zc=0.95, seed=fixed_seed
    )
    period = 1.0

    # should be no connections
    rmax = 0.01
    m = pairwise_distance_3d(data1, data2, rmax, period=None)

    # each point has 0 connections including 1 self connection
    assert m.getnnz() == 0

    # should be 0 connections
    rmax = 0.3
    m = pairwise_distance_3d(data1, data2, rmax, period=None)

    # each point has 0 connections
    assert m.getnnz() == 0


def test_pairwise_distance_3d_nonperiodic_tight_locus2():
    """
    test that pairs of points in two loci are paired when rmax is large enough
    """
    # tight locus
    Npts1 = 10
    Npts2 = 10
    data1 = generate_locus_of_3d_points(Npts1, xc=0.5, yc=0.5, zc=0.5, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(Npts2, xc=0.6, yc=0.6, zc=0.6, seed=fixed_seed)
    period = 1.0

    # should be no connections
    rmax = 0.01
    m = pairwise_distance_3d(data1, data2, rmax, period=None)

    # each point has 0 connections including 1 self connection
    assert m.getnnz() == 0

    # should 10
    rmax = 0.2
    m = pairwise_distance_3d(data1, data2, rmax, period=None)

    # each point has 0 connections including 1 self connection
    assert m.getnnz() == Npts1 * Npts2


@pytest.mark.installation_test
def test_3d_brute_force_elementwise_comparison():
    Npts1, Npts2 = int(1e2), int(1e2)

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))

    r_max = 0.3

    sparse_matrix = pairwise_distance_3d(sample1, sample2, r_max, period=1)
    dense_matrix = sparse_matrix.tocsc()

    pure_python_dense_matrix = pure_python_distance_matrix_3d(
        sample1, sample2, r_max, Lbox=1
    )

    for i in range(pure_python_dense_matrix.shape[0]):
        for j in range(pure_python_dense_matrix.shape[1]):
            brute_force_element = pure_python_dense_matrix[i, j]
            sparse_matrix_element = dense_matrix[i, j]
            assert np.allclose(brute_force_element, sparse_matrix_element, rtol=0.001)


def test_xy_z_brute_force_elementwise_comparison():
    Npts1, Npts2 = int(1e2), int(1e2)

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))

    rp_max, pi_max = 0.2, 0.2

    sparse_matrix_xy, sparse_matrix_z = pairwise_distance_xy_z(
        sample1, sample2, rp_max, pi_max, period=1
    )
    dense_matrix_xy = sparse_matrix_xy.tocsc()
    dense_matrix_z = sparse_matrix_z.tocsc()

    (
        pure_python_dense_matrix_xy,
        pure_python_dense_matrix_z,
    ) = pure_python_distance_matrix_xy_z(sample1, sample2, rp_max, pi_max, Lbox=1)

    for i in range(pure_python_dense_matrix_xy.shape[0]):
        for j in range(pure_python_dense_matrix_xy.shape[1]):

            brute_force_element = pure_python_dense_matrix_xy[i, j]
            sparse_matrix_element = dense_matrix_xy[i, j]
            assert np.allclose(brute_force_element, sparse_matrix_element, rtol=0.001)

            brute_force_element = pure_python_dense_matrix_z[i, j]
            sparse_matrix_element = dense_matrix_z[i, j]
            assert np.allclose(brute_force_element, sparse_matrix_element, rtol=0.001)


def test_get_rmax1():
    """ """
    sample1 = np.zeros((100, 3))

    with pytest.raises(ValueError) as err:
        __ = _get_r_max(sample1, [0.1, 0.2])
    substr = "Input ``r_max`` must be the same length as ``sample1``."
    assert substr in err.value.args[0]


def test_get_rmax2():
    """ """
    sample1 = np.zeros((100, 3))

    with pytest.raises(ValueError) as err:
        __ = _get_r_max(sample1, np.inf)
    substr = "Input ``r_max`` must be an array of bounded positive numbers."
    assert substr in err.value.args[0]


@pytest.mark.installation_test
def test_parallel_serial_consistency():
    Npts1, Npts2 = int(50), int(51)

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))

    r_max = 0.3

    sparse_matrix_serial = pairwise_distance_3d(
        sample1, sample2, r_max, period=1, num_threads=1
    )
    sparse_matrix_parallel = pairwise_distance_3d(
        sample1, sample2, r_max, period=1, num_threads=3
    )
    sparse_matrix_parallel_max = pairwise_distance_3d(
        sample1, sample2, r_max, period=1, num_threads="max"
    )
    dense_matrix_serial = sparse_matrix_serial.tocsc()
    dense_matrix_parallel = sparse_matrix_parallel.tocsc()
    dense_matrix_parallel_max = sparse_matrix_parallel_max.tocsc()

    pure_python_dense_matrix = pure_python_distance_matrix_3d(
        sample1, sample2, r_max, Lbox=1
    )
    for i in range(pure_python_dense_matrix.shape[0]):
        for j in range(pure_python_dense_matrix.shape[1]):

            matrix_element_serial = dense_matrix_serial[i, j]
            matrix_element_parallel = dense_matrix_parallel[i, j]
            matrix_element_parallel_max = dense_matrix_parallel_max[i, j]
            assert np.allclose(
                matrix_element_serial, matrix_element_parallel, rtol=0.001
            )
            assert np.allclose(
                matrix_element_serial, matrix_element_parallel_max, rtol=0.001
            )


def test_args1():
    Npts1, Npts2 = int(50), int(51)

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))

    r_max = 0.3

    with pytest.raises(ValueError) as err:
        __ = pairwise_distance_3d(sample1, sample2, r_max, period=1, num_threads="a")
    substr = "Input ``num_threads`` argument must be an integer or the string 'max'"
    assert substr in err.value.args[0]


def test_args2():
    Npts1, Npts2 = int(50), int(51)

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))

    r_max = 0.3

    with pytest.raises(ValueError) as err:
        __ = pairwise_distance_3d(sample1, sample2, r_max, period=-11)
    substr = "Input ``period`` must be a bounded positive number in all dimensions"
    assert substr in err.value.args[0]


def test_args3():
    Npts1, Npts2 = int(50), int(51)

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))

    r_max = 0.3

    sparse_matrix = pairwise_distance_3d(
        sample1, sample2, r_max, approx_cell1_size=1, approx_cell2_size=1
    )
