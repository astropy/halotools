#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import numpy as np 
from astropy.tests.helper import pytest

from ..pairwise_velocity_stats import mean_los_velocity_vs_rp
from .cf_helpers import generate_locus_of_3d_points

__all__ = ('test_mean_los_velocity_vs_rp_correctness1', 'test_mean_los_velocity_vs_rp_correctness2', 
    'test_mean_los_velocity_vs_rp_correctness3', 'test_mean_los_velocity_vs_rp_correctness4')

fixed_seed = 43

@pytest.mark.slow
def test_mean_los_velocity_vs_rp_correctness1():
    """ Create two tight localizations of points, 
    the first at (1, 0, 0.1), the second at (1, 0.2, 0.25). 
    The first set of points is moving at +50 in the z-direction, 
    towards the second set of points, which is at rest. 
    So the first set of points is moving towards the second set. 

    PBCs are set to infinity. 

    Verify that the `~halotools.mock_observables.mean_los_velocity_vs_rp` function 
    correctly identifies the relative z-velocity to be -50. 
    """
    npts = 100

    xc1, yc1, zc1 = 1, 0, 0.1
    xc2, yc2, zc2 = 1, 0.2, 0.25

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)
    
    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,2] = 50.
    velocities2 = np.zeros(npts*3).reshape(npts, 3)

    rp_bins, pi_max = np.array([0, 0.1, 0.15, 0.21, 0.25]), 0.2

    s1s1, s1s2, s2s2 = mean_los_velocity_vs_rp(sample1, velocities1, rp_bins, pi_max, 
        sample2 = sample2, velocities2 = velocities2)
    assert np.allclose(s1s1[0:2], 0, rtol=0.01)
    assert np.allclose(s1s2[0:2], 0, rtol=0.01)
    assert np.allclose(s2s2[0:2], 0, rtol=0.01)
    assert np.allclose(s1s1[2], 0, rtol=0.01)
    assert np.allclose(s1s2[2], -50, rtol=0.01)
    assert np.allclose(s2s2[2], 0, rtol=0.01)
    assert np.allclose(s1s1[3], 0, rtol=0.01)
    assert np.allclose(s1s2[3], 0, rtol=0.01)
    assert np.allclose(s2s2[3], 0, rtol=0.01)

    # Now bundle sample2 and sample1 together and only pass in the concatenated sample
    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    s1s1 = mean_los_velocity_vs_rp(sample, velocities, rp_bins, pi_max)
    assert np.allclose(s1s1[0:2], 0, rtol=0.01)
    assert np.allclose(s1s1[2], -50, rtol=0.01)
    assert np.allclose(s1s1[3], 0, rtol=0.01)

@pytest.mark.slow
def test_mean_los_velocity_vs_rp_correctness2():
    """ Create two tight localizations of points, 
    one at (0.5, 0.5, 0.1), the other at (0.5, 0.35, 0.2). 
    The first set of points is moving at -50 in the z-direction, 
    the second set of points is at rest. 
    So the first set of points is moving away from the second set. 

    PBCs are set to infinity. 

    Verify that the `~halotools.mock_observables.mean_los_velocity_vs_rp` function 
    correctly identifies the relative z-velocity to be +50. 
    """
    npts = 100

    xc1, yc1, zc1 = 0.5, 0.5, 0.1
    xc2, yc2, zc2 = 0.5, 0.35, 0.25

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)
    
    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities2 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,2] = -50.

    rp_bins, pi_max = np.array([0, 0.1, 0.3]), 0.2
    s1s1, s1s2, s2s2 = mean_los_velocity_vs_rp(sample1, velocities1, rp_bins, pi_max, 
        sample2 = sample2, velocities2 = velocities2)

    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s2s2[0], 0, rtol=0.01)

    assert np.allclose(s1s1[1], 0, rtol=0.01)
    assert np.allclose(s1s2[1], +50, rtol=0.01)
    assert np.allclose(s2s2[1], 0, rtol=0.01)


@pytest.mark.slow
def test_mean_los_velocity_vs_rp_correctness3():
    """ Create two tight localizations of points, 
    one at (0.5, 0.5, 0.25), the other at (0.5, 0.35, 0.1). 
    The first set of points is moving at 50 in the z-direction, 
    the second set of points is at rest. So the first set of points
    is moving away from the second set. 

    PBCs are set to infinity. 

    Verify that the `~halotools.mock_observables.mean_los_velocity_vs_rp` function 
    correctly identifies the relative z-velocity to be +50. 

    """
    npts = 100

    xc1, yc1, zc1 = 0.5, 0.5, 0.25
    xc2, yc2, zc2 = 0.5, 0.35, 0.1

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)
    
    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities2 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,2] = +50.
    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    rp_bins, pi_max = np.array([0, 0.1, 0.3]), 0.2
    s1s1 = mean_los_velocity_vs_rp(sample, velocities, rp_bins, pi_max)

    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 50, rtol=0.01)

@pytest.mark.slow
def test_mean_los_velocity_vs_rp_correctness4():
    """ Create two tight localizations of points, 
    one at (0.5, 0.5, 0.1), the other at (0.5, 0.35, 0.95). 
    The first set of points is moving at -50 in the z-direction, 
    towards the second set of points, which is at rest. 
    So the first set of points is getting closer to the second set. 

    In this example PBCs are important and we pass in a sample2
    """
    npts = 100

    xc1, yc1, zc1 = 0.5, 0.5, 0.1
    xc2, yc2, zc2 = 0.5, 0.35, 0.95

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)
    
    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities2 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,2] = -50.

    rp_bins, pi_max = np.array([0, 0.1, 0.3]), 0.2
    s1s1, s1s2, s2s2 = mean_los_velocity_vs_rp(sample1, velocities1, rp_bins, pi_max, 
        sample2 = sample2, velocities2 = velocities2, period=1)

    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s2s2[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)
    assert np.allclose(s1s2[1], -50, rtol=0.01)
    assert np.allclose(s2s2[1], 0, rtol=0.01)




