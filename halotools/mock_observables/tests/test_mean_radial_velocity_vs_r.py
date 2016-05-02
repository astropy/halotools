""" Module providing testing of `halotools.mock_observables.mean_radial_velocity_vs_r`
"""
from __future__ import (absolute_import, division, print_function)
import numpy as np 
from astropy.tests.helper import pytest
from astropy.utils.misc import NumpyRNGContext

from ..pairwise_velocity_stats import mean_radial_velocity_vs_r
from .cf_helpers import generate_locus_of_3d_points

__all__ = ('test_mean_radial_velocity_vs_r_correctness1', )

fixed_seed = 43

@pytest.mark.slow
def test_mean_radial_velocity_vs_r_correctness1():
    """ Create two tight localizations of points, 
    the first at (0.5, 0.5, 0.1), the second at (0.5, 0.5, 0.25). 
    The first set of points is moving at +50 in the z-direction, 
    towards the second set of points, which is at rest. 

    PBCs are set to infinity. 

    Verify that the `~halotools.mock_observables.mean_radial_velocity_vs_r` function 
    correctly identifies the radial component of the relative velocity to be -50. 
    """
    npts = 100

    xc1, yc1, zc1 = 0.5, 0.5, 0.1
    xc2, yc2, zc2 = 0.5, 0.5, 0.25

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)
    
    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,2] = 50.
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

    # Now bundle sample2 and sample1 together and only pass in the concatenated sample
    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins)

    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], -50, rtol=0.01)


@pytest.mark.slow
def test_mean_radial_velocity_vs_r_correctness2():
    """ Create two tight localizations of points, 
    the first at (0.5, 0.5, 0.05), the second at (0.5, 0.5, 0.95). 
    The first set of points is moving at +50 in the z-direction, 
    away from the second set of points, which is at rest. 

    Verify that the `~halotools.mock_observables.mean_radial_velocity_vs_r` function 
    correctly identifies the radial component of the relative velocity to be -50. 
    """
    npts = 100

    xc1, yc1, zc1 = 0.5, 0.5, 0.05
    xc2, yc2, zc2 = 0.5, 0.5, 0.9

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)
    
    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,2] = 50.
    velocities2 = np.zeros(npts*3).reshape(npts, 3)

    rbins = np.array([0, 0.1, 0.3])

    # First run the calculation with PBCs set to unity
    s1s1, s1s2, s2s2 = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2, period=1)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s2s2[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)
    assert np.allclose(s1s2[1], 50, rtol=0.01)
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
    assert np.allclose(s1s1[1], 50, rtol=0.01)

    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)

def test_mean_radial_velocity_vs_r_correctness3():
    """ Create two tight localizations of points, 
    the first at (0.95, 0.5, 0.5), the second at (0.05, 0.5, 0.5). 
    The first set of points is moving at +50 in the x-direction, 
    towards the second set of points, which is moving at +25 in the x-direction. 
    So the first set of points is "gaining ground" on the second set in the x-direction. 

    Verify that the `~halotools.mock_observables.mean_radial_velocity_vs_r` function 
    correctly identifies the radial component of the relative velocity to be -25. 
    """
    npts = 100

    xc1, yc1, zc1 = 0.95, 0.5, 0.5
    xc2, yc2, zc2 = 0.05, 0.5, 0.5

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)
    
    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,0] = 50.
    velocities2 = np.zeros(npts*3).reshape(npts, 3)
    velocities2[:,0] = 25.

    rbins = np.array([0, 0.05, 0.3])

    s1s1, s1s2, s2s2 = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
        sample2 = sample2, velocities2 = velocities2, period=1)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s2s2[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)
    assert np.allclose(s1s2[1], -25, rtol=0.01)
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
    assert np.allclose(s1s1[1], -25, rtol=0.01)

    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)


def test_mean_radial_velocity_vs_r_correctness4():
    """ Create two tight localizations of points, 
    the first at (0.5, 0.95, 0.5), the second at (0.5, 0.05, 0.5). 
    The first set of points is moving at -50 in the y-direction, 
    towards the second set of points, which is moving at +25 in the y-direction. 
    So these sets of points are mutually moving away from each other in the y-direction. 

    Verify that the `~halotools.mock_observables.mean_radial_velocity_vs_r` function 
    correctly identifies the radial component of the relative velocity to be +75. 
    """
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
    assert np.allclose(s1s2[1], 75, rtol=0.01)
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
    assert np.allclose(s1s1[1], 75, rtol=0.01)

    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)


@pytest.mark.slow
def test_mean_radial_velocity_vs_r_parallel1():
    """ 
    Verify that the parallel and serial results are identical 
    for two tight localizations of points with PBCs operative. 
    """

    npts = 91
    xc1, yc1, zc1 = 0.5, 0.5, 0.1
    xc2, yc2, zc2 = 0.5, 0.5, 0.25

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)

    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:,2] = 50.
    velocities2 = np.zeros(npts*3).reshape(npts, 3)

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
    Verify that the parallel and serial results are identical 
    for random points and velocities, with PBCs operative. 
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
    Verify that the parallel and serial results are identical 
    for random points and velocities, with PBCs turned off. 
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
    """ Verify that we get self-consistent auto-correlation results 
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
    """ Verify that we get self-consistent cross-correlation results 
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














