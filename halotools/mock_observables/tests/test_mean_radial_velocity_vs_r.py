""" Module providing testing of `halotools.mock_observables.mean_radial_velocity_vs_r`
"""
from __future__ import (absolute_import, division, print_function)
import numpy as np 
from astropy.tests.helper import pytest
from astropy.utils.misc import NumpyRNGContext

from ..pairwise_velocity_stats import mean_radial_velocity_vs_r
from .cf_helpers import generate_locus_of_3d_points

__all__ = ('test_mean_radial_velocity_vs_r_correctness1', 
    'test_mean_radial_velocity_vs_r_correctness2', 'test_mean_radial_velocity_vs_r_correctness3', 
    'test_mean_radial_velocity_vs_r_correctness4', 'test_mean_radial_velocity_vs_r_correctness5', 
    'test_mean_radial_velocity_vs_r_parallel1', 'test_mean_radial_velocity_vs_r_parallel2', 
    'test_mean_radial_velocity_vs_r_parallel3', 'test_mean_radial_velocity_vs_r_auto_consistency', 
    'test_mean_radial_velocity_vs_r_cross_consistency')

fixed_seed = 43

@pytest.mark.slow
def test_mean_radial_velocity_vs_r_correctness1():
    """ This function tests that the 
    `~halotools.mock_observables.mean_radial_velocity_vs_r` function returns correct 
    results for a controlled distribution of points whose mean radial velocity 
    is analytically calculable. 

    For this test, the configuration is two tight localizations of points, 
    the first at (0.5, 0.5, 0.1), the second at (0.5, 0.5, 0.25). 
    The first set of points is moving at +50 in the z-direction; 
    the second set of points is at rest. 

    PBCs are set to infinity in this test. 

    So in this configuration, the two sets of points are moving towards each other, 
    and so the radial component of the relative velocity 
    should be -50 for cross-correlations in the radial separation bin containing the 
    pair of points. For any separation bin containing only 
    one set or the other, the auto-correlations should be 0 because each set of 
    points moves coherently. 

    The tests will be run with the two point configurations passed in as 
    separate ``sample1`` and ``sample2`` distributions, as well as bundled 
    together into the same distribution.

    """
    correct_relative_velocity = -50

    npts = 100

    xc1, yc1, zc1 = 0.5, 0.5, 0.1
    xc2, yc2, zc2 = 0.5, 0.5, 0.25

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)
    
    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities2 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,2] = 50.

    rbins = np.array([0, 0.1, 0.3])

    s1s1, s1s2, s2s2 = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s2s2[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)
    assert np.allclose(s1s2[1], correct_relative_velocity, rtol=0.01)
    assert np.allclose(s2s2[1], 0, rtol=0.01)

    # Now bundle sample2 and sample1 together and only pass in the concatenated sample
    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], correct_relative_velocity, rtol=0.01)


@pytest.mark.slow
def test_mean_radial_velocity_vs_r_correctness2():
    """ This function tests that the 
    `~halotools.mock_observables.mean_radial_velocity_vs_r` function returns correct 
    results for a controlled distribution of points whose mean radial velocity 
    is analytically calculable. 

    For this test, the configuration is two tight localizations of points, 
    the first at (0.5, 0.5, 0.05), the second at (0.5, 0.5, 0.95). 
    The first set of points is moving at +50 in the z-direction; 
    the second set of points is at rest. 

    So in this configuration, when PBCs are operative 
    the two sets of points are moving away from each other, 
    and so the radial component of the relative velocity 
    should be +50 for cross-correlations in the radial separation bin containing the 
    pair of points. For any separation bin containing only 
    one set or the other, the auto-correlations should be 0 because each set of 
    points moves coherently. 
    When PBCs are turned off, the function should return zero as the points 
    are too distant to find pairs. 

    These tests will be applied with and without periodic boundary conditions. 
    The tests will also be run with the two point configurations passed in as 
    separate ``sample1`` and ``sample2`` distributions, as well as bundled 
    together into the same distribution.

    """
    correct_relative_velocity = +50

    npts = 100

    xc1, yc1, zc1 = 0.5, 0.5, 0.05
    xc2, yc2, zc2 = 0.5, 0.5, 0.9

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)
    
    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities2 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,2] = 50.

    rbins = np.array([0, 0.1, 0.3])

    # First run the calculation with PBCs set to unity
    s1s1, s1s2, s2s2 = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2, period=1)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s2s2[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)
    assert np.allclose(s1s2[1], correct_relative_velocity, rtol=0.01)
    assert np.allclose(s2s2[1], 0, rtol=0.01)

    # Now set PBCs to infinity and verify that we instead get zeros
    s1s1, s1s2, s2s2 = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s2s2[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)
    assert np.allclose(s1s2[1], 0, rtol=0.01)
    assert np.allclose(s2s2[1], 0, rtol=0.01)

    # Bundle sample2 and sample1 together and only pass in the concatenated sample
    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    # Now repeat the above tests, with and without PBCs
    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins, period=1)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], correct_relative_velocity, rtol=0.01)

    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)

def test_mean_radial_velocity_vs_r_correctness3():
    """ This function tests that the 
    `~halotools.mock_observables.mean_radial_velocity_vs_r` function returns correct 
    results for a controlled distribution of points whose mean radial velocity 
    is analytically calculable. 

    For this test, the configuration is two tight localizations of points, 
    the first at (0.95, 0.5, 0.5), the second at (0.05, 0.5, 0.5). 
    The first set of points is moving at +50 in the x-direction; 
    the second set of points is moving at +25 in the x-direction.

    So in this configuration, when PBCs are operative 
    the two sets of points are moving towards each other, 
    as the first set of points is "gaining ground" on the second set in the x-direction. 
    So the radial component of the relative velocity 
    should be -25 for cross-correlations in the radial separation bin containing the 
    pair of points. For any separation bin containing only 
    one set or the other, the auto-correlations should be 0 because each set of 
    points moves coherently. 
    When PBCs are turned off, the function should return zero as the points 
    are too distant to find pairs. 

    These tests will be applied with and without periodic boundary conditions. 
    The tests will also be run with the two point configurations passed in as 
    separate ``sample1`` and ``sample2`` distributions, as well as bundled 
    together into the same distribution.

    """
    correct_relative_velocity = -25

    npts = 100

    xc1, yc1, zc1 = 0.95, 0.5, 0.5
    xc2, yc2, zc2 = 0.05, 0.5, 0.5

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)
    
    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities2 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,0] = 50.
    velocities2[:,0] = 25.

    rbins = np.array([0, 0.05, 0.3])

    s1s1, s1s2, s2s2 = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2, period=1)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s2s2[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)
    assert np.allclose(s1s2[1], correct_relative_velocity, rtol=0.01)
    assert np.allclose(s2s2[1], 0, rtol=0.01)

    # repeat the test but with PBCs set to infinity
    s1s1, s1s2, s2s2 = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s2s2[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)
    assert np.allclose(s1s2[1], 0, rtol=0.01)
    assert np.allclose(s2s2[1], 0, rtol=0.01)

    # Now bundle sample2 and sample1 together and only pass in the concatenated sample
    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    # Repeat the above tests
    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins, period=1)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], correct_relative_velocity, rtol=0.01)

    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)


def test_mean_radial_velocity_vs_r_correctness4():
    """ This function tests that the 
    `~halotools.mock_observables.mean_radial_velocity_vs_r` function returns correct 
    results for a controlled distribution of points whose mean radial velocity 
    is analytically calculable. 

    For this test, the configuration is two tight localizations of points, 
    the first at (0.5, 0.95, 0.5), the second at (0.5, 0.05, 0.5). 
    The first set of points is moving at -50 in the y-direction; 
    the second set of points is moving at +25 in the y-direction.

    So in this configuration, when PBCs are operative 
    the two sets of points are each moving away from each other, 
    so the radial component of the relative velocity 
    should be +75 for cross-correlations in the radial separation bin containing the 
    pair of points. For any separation bin containing only 
    one set or the other, the auto-correlations should be 0 because each set of 
    points moves coherently. 
    When PBCs are turned off, the function should return zero as the points 
    are too distant to find pairs. 

    These tests will be applied with and without periodic boundary conditions. 
    The tests will also be run with the two point configurations passed in as 
    separate ``sample1`` and ``sample2`` distributions, as well as bundled 
    together into the same distribution.

    """
    correct_relative_velocity = +75 

    npts = 100

    xc1, yc1, zc1 = 0.5, 0.95, 0.5
    xc2, yc2, zc2 = 0.5, 0.05, 0.5

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)
    
    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,1] = -50.
    velocities2 = np.zeros(npts*3).reshape(npts, 3)
    velocities2[:,1] = 25.

    rbins = np.array([0, 0.05, 0.3])

    s1s1, s1s2, s2s2 = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2, period=1)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s2s2[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)
    assert np.allclose(s1s2[1], correct_relative_velocity, rtol=0.01)
    assert np.allclose(s2s2[1], 0, rtol=0.01)

    # repeat the test but with PBCs set to infinity
    s1s1, s1s2, s2s2 = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s2s2[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)
    assert np.allclose(s1s2[1], 0, rtol=0.01)
    assert np.allclose(s2s2[1], 0, rtol=0.01)

    # Now bundle sample2 and sample1 together and only pass in the concatenated sample
    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    # Repeat the above tests
    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins, period=1)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], correct_relative_velocity, rtol=0.01)

    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)

def test_mean_radial_velocity_vs_r_correctness5():
    """ This function tests that the 
    `~halotools.mock_observables.mean_radial_velocity_vs_r` function returns correct 
    results for a controlled distribution of points whose mean radial velocity 
    is analytically calculable. 

    For this test, the configuration is two tight localizations of points, 
    the first at (0.05, 0.05, 0.05), the second at (0.95, 0.95, 0.95). 
    The first set of points is moving at (+50, +50, +50); 
    the second set of points is moving at (-50, -50, -50).

    So in this configuration, when PBCs are operative 
    the two sets of points are each moving towards each other, 
    so the radial component of the relative velocity 
    should be +100*sqrt(3) for cross-correlations in the radial separation bin containing the 
    pair of points. For any separation bin containing only 
    one set or the other, the auto-correlations should be 0 because each set of 
    points moves coherently. 
    When PBCs are turned off, the function should return zero as the points 
    are too distant to find pairs. 

    These tests will be applied with and without periodic boundary conditions. 
    The tests will also be run with the two point configurations passed in as 
    separate ``sample1`` and ``sample2`` distributions, as well as bundled 
    together into the same distribution.

    """
    correct_relative_velocity = np.sqrt(3)*100.

    npts = 91
    xc1, yc1, zc1 = 0.05, 0.05, 0.05
    xc2, yc2, zc2 = 0.95, 0.95, 0.95

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)

    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities2 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,:] = 50.
    velocities2[:,:] = -50.

    rbins = np.array([0, 0.1, 0.3])

    s1s1, s1s2, s2s2 = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2, period=1)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s2s2[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)
    assert np.allclose(s1s2[1], correct_relative_velocity, rtol=0.01)
    assert np.allclose(s2s2[1], 0, rtol=0.01)

    # repeat the test but with PBCs set to infinity
    s1s1, s1s2, s2s2 = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s2s2[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)
    assert np.allclose(s1s2[1], 0, rtol=0.01)
    assert np.allclose(s2s2[1], 0, rtol=0.01)

    # Now bundle sample2 and sample1 together and only pass in the concatenated sample
    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    # Repeat the above tests
    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins, period=1)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], correct_relative_velocity, rtol=0.01)

    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)

@pytest.mark.slow
def test_mean_radial_velocity_vs_r_parallel1():
    """ 
    Verify that the `~halotools.mock_observables.mean_radial_velocity_vs_r` function 
    returns identical results for two tight loci of points whether the function 
    runs in parallel or serial. 
    """

    npts = 91
    xc1, yc1, zc1 = 0.5, 0.05, 0.05
    xc2, yc2, zc2 = 0.45, 0.05, 0.1

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)

    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities2 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,:] = 50.
    velocities2[:,:] = 0.

    rbins = np.array([0, 0.1, 0.3])

    s1s1_parallel, s1s2_parallel, s2s2_parallel = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2, num_threads = 3, period=1)

    s1s1_serial, s1s2_serial, s2s2_serial = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2, num_threads = 1, period=1)

    assert np.all(s1s1_serial == s1s1_parallel)
    assert np.all(s1s2_serial == s1s2_parallel)
    assert np.all(s2s2_serial == s2s2_parallel)

@pytest.mark.slow
def test_mean_radial_velocity_vs_r_parallel2():
    """ 
    Verify that the `~halotools.mock_observables.mean_radial_velocity_vs_r` function 
    returns identical results for two random distributions of points whether the function 
    runs in parallel or serial, with PBCs operative. 
    """
    npts = 101

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 3))
        velocities1 = np.random.normal(loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
        sample2 = np.random.random((npts, 3))
        velocities2 = np.random.normal(loc = 0, scale = 100, size=npts*3).reshape((npts, 3))

    rbins = np.array([0, 0.1, 0.3])

    s1s1_parallel, s1s2_parallel, s2s2_parallel = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2, num_threads = 2, period=1)

    s1s1_serial, s1s2_serial, s2s2_serial = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2, num_threads = 1, period=1)

    assert np.allclose(s1s1_serial, s1s1_parallel, rtol = 0.001)
    assert np.allclose(s1s2_serial, s1s2_parallel, rtol = 0.001)
    assert np.allclose(s2s2_serial, s2s2_parallel, rtol = 0.001)

@pytest.mark.slow
def test_mean_radial_velocity_vs_r_parallel3():
    """ 
    Verify that the `~halotools.mock_observables.mean_radial_velocity_vs_r` function 
    returns identical results for two random distributions of points whether the function 
    runs in parallel or serial, with PBCs turned off. 
    """
    npts = 101

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 3))
        velocities1 = np.random.normal(loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
        sample2 = np.random.random((npts, 3))
        velocities2 = np.random.normal(loc = 0, scale = 100, size=npts*3).reshape((npts, 3))

    rbins = np.array([0, 0.1, 0.3])

    s1s1_parallel, s1s2_parallel, s2s2_parallel = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2, num_threads = 2)

    s1s1_serial, s1s2_serial, s2s2_serial = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2, num_threads = 1)

    assert np.allclose(s1s1_serial, s1s1_parallel, rtol = 0.001)
    assert np.allclose(s1s2_serial, s1s2_parallel, rtol = 0.001)
    assert np.allclose(s2s2_serial, s2s2_parallel, rtol = 0.001)


@pytest.mark.slow
def test_mean_radial_velocity_vs_r_auto_consistency():
    """ Verify that the `~halotools.mock_observables.mean_radial_velocity_vs_r` function  
    returns self-consistent auto-correlation results 
    regardless of whether we ask for cross-correlations. 
    """
    npts = 101

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 3))
        velocities1 = np.random.normal(loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
        sample2 = np.random.random((npts, 3))
        velocities2 = np.random.normal(loc = 0, scale = 100, size=npts*3).reshape((npts, 3))

    rbins = np.linspace(0, 0.3, 10)
    s1s1a, s1s2a, s2s2a = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2)
    s1s1b, s2s2b = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2, 
        do_cross = False)

    assert np.allclose(s1s1a,s1s1b, rtol=0.001)
    assert np.allclose(s2s2a,s2s2b, rtol=0.001)


@pytest.mark.slow
def test_mean_radial_velocity_vs_r_cross_consistency():
    """ Verify that the `~halotools.mock_observables.mean_radial_velocity_vs_r` function  
    returns self-consistent cross-correlation results 
    regardless of whether we ask for auto-correlations. 
    """
    npts = 101

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 3))
        velocities1 = np.random.normal(loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
        sample2 = np.random.random((npts, 3))
        velocities2 = np.random.normal(loc = 0, scale = 100, size=npts*3).reshape((npts, 3))

    rbins = np.linspace(0, 0.3, 10)
    s1s1a, s1s2a, s2s2a = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2)
    s1s2b = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2, 
        do_auto = False)

    assert np.allclose(s1s2a,s1s2b, rtol=0.001)














