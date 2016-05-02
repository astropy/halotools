#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import numpy as np 
from astropy.tests.helper import pytest

from ..pairwise_velocity_stats import mean_los_velocity_vs_rp
from .cf_helpers import generate_locus_of_3d_points

__all__ = ('test_mean_los_velocity_vs_rp_correctness1a', )

@pytest.mark.slow
def test_mean_los_velocity_vs_rp_correctness1a():
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
        xc=0.5, yc=0.35, zc=0.25, epsilon = 0.0001)
    velocities2 = np.zeros(npts*3).reshape(npts, 3)

    rp_bins, pi_max = np.array([0, 0.1, 0.3]), 0.2
    s1s1, s1s2, s2s2 = mean_los_velocity_vs_rp(sample1, velocities1, rp_bins, pi_max, 
        sample2 = sample2, velocities2 = velocities2)

    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s2s2[0], 0, rtol=0.01)

    assert np.allclose(s1s1[1], 0, rtol=0.01)
    assert np.allclose(s1s2[1], -50, rtol=0.01)
    assert np.allclose(s2s2[1], 0, rtol=0.01)


@pytest.mark.slow
def test_mean_los_velocity_vs_rp_correctness1b():
    """ Create two tight localizations of points, 
    one at (0.5, 0.5, 0.1), the other at (0.5, 0.5, 0.2). 
    The first set of points is moving at 50 in the z-direction, 
    the second set of points is at rest. These two sets of points 
    are concatenated together into a single sample1.

    This test is identical to test_mean_los_velocity_vs_rp_correctness1a, 
    except here we concatenate sample1 and sample2 and only pass in a sample1. 
    """
    np.random.seed(43)

    npts = 200
    sample1 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.5, zc=0.1, epsilon = 0.0001)
    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,2] = 50.

    sample2 = generate_locus_of_3d_points(npts, 
        xc=0.5, yc=0.35, zc=0.25, epsilon = 0.0001)
    velocities2 = np.zeros(npts*3).reshape(npts, 3)

    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    rp_bins, pi_max = np.array([0, 0.1, 0.3]), 0.2
    s1s1 = mean_los_velocity_vs_rp(sample, velocities, rp_bins, pi_max)

    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], -50, rtol=0.01)

@pytest.mark.slow
def test_mean_los_velocity_vs_rp_correctness2a():
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
        xc=0.5, yc=0.35, zc=0.95, epsilon = 0.0001)
    velocities2 = np.zeros(npts*3).reshape(npts, 3)

    rp_bins, pi_max = np.array([0, 0.1, 0.3]), 0.2
    s1s1, s1s2, s2s2 = mean_los_velocity_vs_rp(sample1, velocities1, rp_bins, pi_max, 
        sample2 = sample2, velocities2 = velocities2, period=1)

    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s2s2[0], 0, rtol=0.01)

    assert np.allclose(s1s1[1], 0, rtol=0.01)
    assert np.allclose(s1s2[1], 50, rtol=0.01)
    assert np.allclose(s2s2[1], 0, rtol=0.01)




