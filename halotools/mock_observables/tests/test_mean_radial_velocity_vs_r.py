#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import numpy as np 
from astropy.tests.helper import pytest

from ..pairwise_velocity_stats import mean_radial_velocity_vs_r
from .cf_helpers import generate_locus_of_3d_points

__all__ = ('test_mean_radial_velocity_vs_r_correctness1a', 
    'test_mean_radial_velocity_vs_r_correctness1b')


@pytest.mark.slow
def test_mean_radial_velocity_vs_r_correctness1a():
    """ Create two tight localizations of points, 
    one at (0.5, 0.5, 0.1), the other at (0.5, 0.5, 0.2). 
    The first set of points is moving at -50 in the z-direction, 
    the second set of points is at rest. 

    In this example PBCs are irrelevant and we pass in a sample2
    """
    np.random.seed(43)

    npts = 200
    sample1 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.5, zc=0.1, epsilon = 0.0001)
    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,2] = 50.

    sample2 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.5, zc=0.25, epsilon = 0.0001)
    velocities2 = np.zeros(npts*3).reshape(npts, 3)

    rbins = np.array([0, 0.1, 0.3])
    s1s1, s1s2, s2s2 = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2)

    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s2s2[0], 0, rtol=0.01)

    assert np.allclose(s1s1[1], 0, rtol=0.01)
    assert np.allclose(s1s2[1], -50, rtol=0.01)
    assert np.allclose(s2s2[1], 0, rtol=0.01)

@pytest.mark.slow
def test_mean_radial_velocity_vs_r_correctness1b():
    """ Create two tight localizations of points, 
    one at (0.5, 0.5, 0.1), the other at (0.5, 0.5, 0.2). 
    The first set of points is moving at -50 in the z-direction, 
    the second set of points is at rest. These two sets of points 
    are concatenated together into a single sample1.

    In this example PBCs are irrelevant and we only pass in a sample1. 
    """
    np.random.seed(43)

    npts = 200
    sample1 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.5, zc=0.1, epsilon = 0.0001)
    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,2] = 50.

    sample2 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.5, zc=0.25, epsilon = 0.0001)
    velocities2 = np.zeros(npts*3).reshape(npts, 3)

    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    rbins = np.array([0, 0.1, 0.3])
    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins)

    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], -50, rtol=0.01)


@pytest.mark.slow
def test_mean_radial_velocity_vs_r_correctness2a():
    """ Create two tight localizations of points, 
    one at (0.5, 0.5, 0.1), the other at (0.5, 0.5, 0.2). 
    The first set of points is moving at -50 in the z-direction, 
    the second set of points is at rest. 

    In this example PBCs are important and we pass in a sample2
    """
    np.random.seed(43)

    npts = 200
    sample1 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.5, zc=0.1, epsilon = 0.0001)
    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,2] = -50.

    sample2 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.5, zc=0.95, epsilon = 0.0001)
    velocities2 = np.zeros(npts*3).reshape(npts, 3)

    rbins = np.array([0, 0.1, 0.3])
    s1s1, s1s2, s2s2 = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2, period=1)

    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s2s2[0], 0, rtol=0.01)

    assert np.allclose(s1s1[1], 0, rtol=0.01)
    assert np.allclose(s1s2[1], -50, rtol=0.01)
    assert np.allclose(s2s2[1], 0, rtol=0.01)

@pytest.mark.slow
def test_mean_radial_velocity_vs_r_correctness2b():
    """ Create two tight localizations of points, 
    one at (0.5, 0.5, 0.1), the other at (0.5, 0.5, 0.2). 
    The first set of points is moving at -50 in the z-direction, 
    the second set of points is at rest. These two sets of points 
    are concatenated together into a single sample1.

    In this example PBCs are important and we only pass in a sample1. 
    """
    np.random.seed(43)

    npts = 200
    sample1 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.5, zc=0.1, epsilon = 0.0001)
    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,2] = -50.

    sample2 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.5, zc=0.95, epsilon = 0.0001)
    velocities2 = np.zeros(npts*3).reshape(npts, 3)

    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    rbins = np.array([0, 0.1, 0.3])
    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins, period=1)

    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], -50, rtol=0.01)


@pytest.mark.slow
def test_mean_radial_velocity_vs_r_correctness3a():
    """ Create two tight localizations of points, 
    one at (0.5, 0.5, 0.1), the other at (0.5, 0.5, 0.2). 
    The first set of points is moving at -50 in the z-direction, 
    the second set of points is at rest. 

    In this example PBCs are important and we pass in a sample2

    This is the same exact test as test_mean_radial_velocity_vs_r_correctness2a, 
    only here we verify the opposite sign flip in the velocities.
    """
    np.random.seed(43)

    npts = 200
    sample1 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.5, zc=0.1, epsilon = 0.0001)
    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,2] = 50.

    sample2 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.5, zc=0.95, epsilon = 0.0001)
    velocities2 = np.zeros(npts*3).reshape(npts, 3)

    rbins = np.array([0, 0.1, 0.3])
    s1s1, s1s2, s2s2 = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2, period=1)

    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s2s2[0], 0, rtol=0.01)

    assert np.allclose(s1s1[1], 0, rtol=0.01)
    assert np.allclose(s1s2[1], 50, rtol=0.01)
    assert np.allclose(s2s2[1], 0, rtol=0.01)

@pytest.mark.slow
def test_mean_radial_velocity_vs_r_correctness3b():
    """ Create two tight localizations of points, 
    one at (0.5, 0.5, 0.1), the other at (0.5, 0.5, 0.2). 
    The first set of points is moving at -50 in the z-direction, 
    the second set of points is at rest. These two sets of points 
    are concatenated together into a single sample1.

    In this example PBCs are important and we only pass in a sample1. 

    This is the same exact test as test_mean_radial_velocity_vs_r_correctness2b, 
    only here we verify the opposite sign flip in the velocities.
    """
    np.random.seed(43)

    npts = 200
    sample1 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.5, zc=0.1, epsilon = 0.0001)
    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,2] = 50.

    sample2 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.5, zc=0.95, epsilon = 0.0001)
    velocities2 = np.zeros(npts*3).reshape(npts, 3)

    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    rbins = np.array([0, 0.1, 0.3])
    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins, period=1)

    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 50, rtol=0.01)

@pytest.mark.slow
def test_mean_radial_velocity_vs_r_parallel1():
    """ 
    Verify that the parallel and serial results are identical 
    for two tight localizations of points with PBCs operative. 
    """
    np.random.seed(43)

    npts = 200
    sample1 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.5, zc=0.1, epsilon = 0.0001)
    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,2] = 50.

    sample2 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.5, zc=0.25, epsilon = 0.0001)
    velocities2 = np.zeros(npts*3).reshape(npts, 3)

    rbins = np.array([0, 0.1, 0.3])

    s1s1_parallel, s1s2_parallel, s2s2_parallel = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2, num_threads = 2, period=1)

    s1s1_serial, s1s2_serial, s2s2_serial = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2, num_threads = 1, period=1)

    assert np.all(s1s1_serial == s1s1_parallel)
    assert np.all(s1s2_serial == s1s2_parallel)
    assert np.all(s2s2_serial == s2s2_parallel)

@pytest.mark.slow
def test_mean_radial_velocity_vs_r_parallel2():
    """ 
    Verify that the parallel and serial results are identical 
    for random points and velocities, with PBCs operative. 
    """
    np.random.seed(43)

    npts = 200
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
    Verify that the parallel and serial results are identical 
    for random points and velocities, with PBCs turned off. 
    """
    np.random.seed(43)

    npts = 200
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
    """ Verify that we get self-consistent auto-correlation results 
    regardless of whether we ask for cross-correlations. 
    """
    np.random.seed(43)

    npts = 200
    sample1 = np.random.rand(npts, 3)
    velocities1 = np.random.normal(
        loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
    sample2 = np.random.rand(npts, 3)
    velocities2 = np.random.normal(
        loc = 0, scale = 100, size=npts*3).reshape((npts, 3))

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
    """ Verify that we get self-consistent cross-correlation results 
    regardless of whether we ask for auto-correlations. 
    """
    np.random.seed(43)

    npts = 200
    sample1 = np.random.rand(npts, 3)
    velocities1 = np.random.normal(
        loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
    sample2 = np.random.rand(npts, 3)
    velocities2 = np.random.normal(
        loc = 0, scale = 100, size=npts*3).reshape((npts, 3))

    rbins = np.linspace(0, 0.3, 10)
    s1s1a, s1s2a, s2s2a = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2)
    s1s2b = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2, 
        do_auto = False)

    assert np.allclose(s1s2a,s1s2b, rtol=0.001)













