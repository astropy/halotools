"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext
from astropy.config.paths import _find_home

from ..npairs_3d import npairs_3d
from ..pairs import npairs as pure_python_brute_force_npairs_3d

from ...tests.cf_helpers import generate_locus_of_3d_points
from ...tests.cf_helpers import generate_3d_regular_mesh

__all__ = ('test_rectangular_mesh_pairs_tight_locus1', )

fixed_seed = 43

# Determine whether the machine is mine
# This will be used to select tests whose
# returned values depend on the configuration
# of my personal cache directory files
aph_home = '/Users/aphearin'
detected_home = _find_home()
if aph_home == detected_home:
    APH_MACHINE = True
else:
    APH_MACHINE = False


@pytest.mark.installation_test
def test_rectangular_mesh_pairs_tight_locus1():
    """ Verify that `halotools.mock_observables.npairs_3d` returns
    the correct counts for two tight loci of points.

    In this test, PBCs are irrelevant
    """
    npts1, npts2 = 100, 100
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.1, zc=0.2, seed=fixed_seed)

    rbins = np.array((0.05, 0.15, 0.3))
    result = npairs_3d(data1, data2, rbins, period=1)
    assert np.all(result == [0, npts1*npts2, npts1*npts2])


def test_rectangular_mesh_pairs_tight_locus2():
    """ Verify that `halotools.mock_observables.npairs_3d` returns
    the correct counts for two tight loci of points.

    In this test, PBCs are important.
    """
    npts1, npts2 = 100, 100
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.05, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.1, zc=0.95, seed=fixed_seed)

    rbins = np.array((0.05, 0.15, 0.3))
    result = npairs_3d(data1, data2, rbins, period=1)
    assert np.all(result == [0, npts1*npts2, npts1*npts2])


def test_rectangular_mesh_pairs_tight_locus3():
    """ Verify that the pair counters return the correct results
    when operating on a tight locus of points.

    For this test, PBCs have no impact.
    """
    npts1, npts2 = 100, 200
    points1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    points2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.1, zc=0.25, seed=fixed_seed)
    rbins = np.array([0.1, 0.2, 0.3])
    correct_result = np.array([0, npts1*npts2, npts1*npts2])

    counts = npairs_3d(points1, points2, rbins, num_threads='max')
    assert np.all(counts == correct_result)


def test_rectangular_mesh_pairs_tight_locus4():
    """ Verify that the pair counters return the correct results
    when operating on a tight locus of points.

    For this test, PBCs have no impact.
    """
    npts1, npts2 = 100, 200
    points1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    points2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.1, zc=0.25, seed=fixed_seed)
    rbins = np.array([0.1, 0.2, 0.3])
    correct_result = np.array([0, npts1*npts2, npts1*npts2])

    counts = npairs_3d(points1, points2, rbins, num_threads=1)
    assert np.all(counts == correct_result)


def test_rectangular_mesh_pairs_tight_locus5():
    """ Verify that the pair counters return the correct results
    when operating on a tight locus of points.

    For this test, PBCs have no impact.
    """
    npts1, npts2 = 100, 200
    points1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    points2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.1, zc=0.25, seed=fixed_seed)
    rbins = np.array([0.1, 0.2, 0.3])
    correct_result = np.array([0, npts1*npts2, npts1*npts2])

    counts = npairs_3d(points1, points2, rbins, period=1.)
    assert np.all(counts == correct_result)


def test_rectangular_mesh_pairs_tight_locus6():
    """ Verify that the pair counters return the correct results
    when operating on a tight locus of points.

    For this test, PBCs have no impact.
    """
    npts1, npts2 = 100, 200
    points1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    points2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.1, zc=0.25, seed=fixed_seed)
    rbins = np.array([0.1, 0.2, 0.3])
    correct_result = np.array([0, npts1*npts2, npts1*npts2])

    counts = npairs_3d(points1, points2, rbins, approx_cell1_size=[0.1, 0.1, 0.1])
    assert np.all(counts == correct_result)


def test_rectangular_mesh_pairs_tight_locus7():
    """ Verify that the pair counters return the correct results
    when operating on a tight locus of points.

    For this test, PBCs have no impact.
    """
    npts1, npts2 = 100, 200
    points1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    points2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.1, zc=0.25, seed=fixed_seed)
    rbins = np.array([0.1, 0.2, 0.3])
    correct_result = np.array([0, npts1*npts2, npts1*npts2])

    counts = npairs_3d(points1, points2, rbins,
        approx_cell1_size=[0.1, 0.1, 0.1],
        approx_cell2_size=[0.1, 0.1, 0.1])
    assert np.all(counts == correct_result)


def test_rectangular_mesh_pairs_tight_locus8():
    """ Verify that the pair counters return the correct results
    when operating on a tight locus of points.

    For this test, PBCs have no impact.
    """
    npts1, npts2 = 100, 200
    points1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    points2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.1, zc=0.25, seed=fixed_seed)
    rbins = np.array([0.1, 0.2, 0.3])
    correct_result = np.array([0, npts1*npts2, npts1*npts2])

    counts = npairs_3d(points1, points2, rbins,
        approx_cell1_size=0.1, approx_cell2_size=0.1, period=1)
    assert np.all(counts == correct_result)


def test_rectangular_mesh_pairs_tight_locus9():
    """ Verify that the pair counters return the correct results
    when operating on a tight locus of points.

    For this test, PBCs have no impact.
    """
    npts1, npts2 = 100, 200
    points1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    points2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.1, zc=0.25, seed=fixed_seed)
    rbins = np.array([0.1, 0.2, 0.3])
    correct_result = np.array([0, npts1*npts2, npts1*npts2])

    counts = npairs_3d(points1, points2, rbins,
        approx_cell1_size=[0.2, 0.2, 0.2],
        approx_cell2_size=[0.15, 0.15, 0.15], period=1)
    assert np.all(counts == correct_result)


def test_rectangular_mesh_pairs():
    """ Verify that `halotools.mock_observables.npairs_3d` returns
    the correct counts for two regularly spaced grids of points.
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
    rbins = np.array([r1, r2, r3, r4])
    result = npairs_3d(data1, data2, rbins, period=Lbox, approx_cell1_size=0.1)
    assert np.all(result ==
        [npts_per_dim**3, 7*npts_per_dim**3, 19*npts_per_dim**3, 27*npts_per_dim**3])


@pytest.mark.skipif('not APH_MACHINE')
def test_parallel():
    """ Verify that `halotools.mock_observables.npairs_3d` returns
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
    rbins = np.array([r1, r2, r3, r4])
    serial_result = npairs_3d(data1, data2, rbins, period=Lbox, approx_cell1_size=0.1)
    parallel_result2 = npairs_3d(data1, data2, rbins, period=Lbox,
        approx_cell1_size=0.1, num_threads=2)
    parallel_result7 = npairs_3d(data1, data2, rbins, period=Lbox,
        approx_cell1_size=0.1, num_threads=3)
    assert np.all(serial_result == parallel_result2)
    assert np.all(serial_result == parallel_result7)


@pytest.mark.installation_test
def test_npairs_brute_force_periodic():
    """
    Function tests npairs with periodic boundary conditions.
    """
    Npts = 1000
    with NumpyRNGContext(fixed_seed):
        random_sample = np.random.random((Npts, 3))
    period = np.array([1.0, 1.0, 1.0])
    rbins = np.array([0.001, 0.1, 0.2, 0.3])

    result = npairs_3d(random_sample, random_sample, rbins, period=period)

    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result) == (len(rbins),), msg

    test_result = pure_python_brute_force_npairs_3d(
        random_sample, random_sample, rbins, period=period)

    msg = "The double tree's result(s) are not equivalent to simple pair counter's."
    assert np.all(test_result == result), msg


def test_npairs_brute_force_nonperiodic():
    """
    test npairs without periodic boundary conditions.
    """

    Npts = 1000
    with NumpyRNGContext(fixed_seed):
        random_sample = np.random.random((Npts, 3))
    rbins = np.array([0.001, 0.1, 0.2, 0.3])

    result = npairs_3d(random_sample, random_sample, rbins, period=None)

    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result) == (len(rbins),), msg

    test_result = pure_python_brute_force_npairs_3d(
        random_sample, random_sample, rbins, period=None)

    msg = "The double tree's result(s) are not equivalent to simple pair counter's."
    assert np.all(test_result == result), msg


def test_sensible_num_threads():
    npts1, npts2 = 100, 100
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.1, zc=0.2, seed=fixed_seed)

    rbins = np.array((0.05, 0.15, 0.3))
    with pytest.raises(ValueError) as err:
        result = npairs_3d(data1, data2, rbins, period=1,
            num_threads="Cuba Gooding Jr.")
    substr = "Input ``num_threads`` argument must be an integer or the string 'max'"
    assert substr in err.value.args[0]


def test_sensible_rbins():
    npts1, npts2 = 100, 100
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.1, zc=0.2, seed=fixed_seed)

    rbins = 0.1
    with pytest.raises(ValueError) as err:
        result = npairs_3d(data1, data2, rbins, period=1)
    substr = "Input ``rbins`` must be a monotonically increasing 1D array with at least two entries"
    assert substr in err.value.args[0]


def test_sensible_period():
    npts1, npts2 = 100, 100
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.1, zc=0.2, seed=fixed_seed)
    rbins = np.array((0.05, 0.15, 0.3))

    with pytest.raises(ValueError) as err:
        result = npairs_3d(data1, data2, rbins, period=np.inf)
    substr = "Input ``period`` must be a bounded positive number in all dimensions"
    assert substr in err.value.args[0]


def test_pure_python_npairs_3d_argument_handling1():
    """
    """
    npts = 10
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 3))
        sample2 = np.random.random((npts, 2))
    rbins = np.linspace(0.01, 0.1, 5)

    with pytest.raises(ValueError) as err:
        __ = pure_python_brute_force_npairs_3d(sample1, sample2, rbins, period=None)
    substr = "sample1 and sample2 inputs do not have the same dimension"
    assert substr in err.value.args[0]


def test_pure_python_npairs_3d_argument_handling2():
    """
    """
    npts = 10
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 3))
        sample2 = np.random.random((npts, 3))
    rbins = np.linspace(0.01, 0.1, 5)

    __ = pure_python_brute_force_npairs_3d(sample1, sample2, rbins, period=1)


def test_pure_python_npairs_3d_argument_handling3():
    """
    """
    npts = 10
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 2))
        sample2 = np.random.random((npts, 2))
    rbins = np.linspace(0.01, 0.1, 5)

    with pytest.raises(ValueError) as err:
        __ = pure_python_brute_force_npairs_3d(sample1, sample2, rbins, period=[1, 1, 1])
    substr = "period should have len == dimension of points"
    assert substr in err.value.args[0]
