#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import numpy as np 
from astropy.tests.helper import pytest

from ..pairwise_velocity_stats import radial_pvd_vs_r

from .cf_helpers import generate_locus_of_3d_points

__all__ = ('test_radial_pvd_vs_r1', 'test_radial_pvd_vs_r_auto_consistency', 
    'test_radial_pvd_vs_r_cross_consistency')

@pytest.mark.slow
def test_radial_pvd_vs_r_correctness1a():
    """ Create two tight localizations of points, 
    one at (0.5, 0.5, 0.1), the other at (0.5, 0.5, 0.25). 
    The first set of points is moving with uniform random velocities 
    between 0 and 1, the second set of points is at rest. 

    In this example PBCs are irrelevant and we pass in a sample2
    """
    npts = 100
    sample1 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.5, zc=0.1, epsilon = 0.0001)
    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,2] = np.random.uniform(0, 1, npts)

    sample2 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.5, zc=0.25, epsilon = 0.0001)
    velocities2 = np.zeros(npts*3).reshape(npts, 3)

    rbins = np.array([0.001, 0.1, 0.3])
    s1s1, s1s2, s2s2 = radial_pvd_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2)

    correct_cross_pvd = np.std(np.repeat(velocities1[:,2], npts))

    assert np.allclose(s1s2[0], 0, rtol=0.1)
    assert np.allclose(s1s2[1], correct_cross_pvd, rtol=0.001)

@pytest.mark.slow
def test_radial_pvd_vs_r_correctness1b():
    """ Create two tight localizations of points, 
    one at (0.5, 0.5, 0.1), the other at (0.5, 0.5, 0.25). 
    The first set of points is moving with uniform random velocities 
    between 0 and 1, the second set of points is at rest. 

    In this example PBCs are important and we pass in a sample2
    """
    npts = 100
    sample1 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.5, zc=0.1, epsilon = 0.0001)
    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,2] = np.random.uniform(0, 1, npts)

    sample2 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.5, zc=0.95, epsilon = 0.0001)
    velocities2 = np.zeros(npts*3).reshape(npts, 3)

    rbins = np.array([0.001, 0.1, 0.3])
    s1s1, s1s2, s2s2 = radial_pvd_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2, period=1)

    correct_cross_pvd = np.std(np.repeat(velocities1[:,2], npts))

    assert np.allclose(s1s2[0], 0, rtol=0.1)
    assert np.allclose(s1s2[1], correct_cross_pvd, rtol=0.001)

@pytest.mark.slow
def test_radial_pvd_vs_r_correctness2a():
    """ Create two tight localizations of points, 
    one at (0.5, 0.5, 0.1), the other at (0.5, 0.5, 0.25). 
    The first set of points is moving with uniform random velocities 
    between 0 and 1, the second set of points is at rest. 

    In this example PBCs are irrelevant and we only pass in sample1
    """
    npts = 100
    sample1 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.5, zc=0.1, epsilon = 0.0001)
    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,2] = np.random.uniform(0, 1, npts)

    sample2 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.5, zc=0.25, epsilon = 0.0001)
    velocities2 = np.zeros(npts*3).reshape(npts, 3)

    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    rbins = np.array([0.001, 0.1, 0.3])
    s1s1 = radial_pvd_vs_r(sample, velocities, rbins)

    correct_cross_pvd = np.std(np.repeat(velocities1[:,2], npts))

    assert np.allclose(s1s1[1], correct_cross_pvd, rtol=0.001)

@pytest.mark.slow
def test_radial_pvd_vs_r_correctness2b():
    """ Create two tight localizations of points, 
    one at (0.5, 0.5, 0.1), the other at (0.5, 0.5, 0.25). 
    The first set of points is moving with uniform random velocities 
    between 0 and 1, the second set of points is at rest. 

    In this example PBCs are important and we only pass in sample1
    """
    npts = 100
    sample1 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.5, zc=0.1, epsilon = 0.0001)
    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,2] = np.random.uniform(0, 1, npts)

    sample2 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.5, zc=0.95, epsilon = 0.0001)
    velocities2 = np.zeros(npts*3).reshape(npts, 3)

    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    rbins = np.array([0.001, 0.1, 0.3])
    s1s1 = radial_pvd_vs_r(sample, velocities, rbins, period=1)

    correct_cross_pvd = np.std(np.repeat(velocities1[:,2], npts))

    assert np.allclose(s1s1[1], correct_cross_pvd, rtol=0.001)


@pytest.mark.slow
def test_radial_pvd_vs_r1():
    """ Verify that different choices for the cell size has no 
    impact on the results. 
    """
    np.random.seed(43)

    npts = 200
    sample1 = np.random.rand(npts, 3)
    velocities1 = np.random.normal(
        loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
    rbins = np.linspace(0, 0.3, 10)
    result1 = radial_pvd_vs_r(sample1, velocities1, rbins)
    result2 = radial_pvd_vs_r(sample1, velocities1, rbins, 
        approx_cell1_size = [0.2, 0.2, 0.2])
    assert np.allclose(result1, result2, rtol=0.0001)

@pytest.mark.slow
def test_radial_pvd_vs_r_auto_consistency():
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
    s1s1a, s1s2a, s2s2a = radial_pvd_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2)
    s1s1b, s2s2b = radial_pvd_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2, 
        do_cross = False)

    assert np.allclose(s1s1a,s1s1b, rtol=0.001)
    assert np.allclose(s2s2a,s2s2b, rtol=0.001)

@pytest.mark.slow
def test_radial_pvd_vs_r_cross_consistency():
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
    s1s1a, s1s2a, s2s2a = radial_pvd_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2)
    s1s2b = radial_pvd_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2, 
        do_auto = False)

    assert np.allclose(s1s2a,s1s2b, rtol=0.001)

