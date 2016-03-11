#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import numpy as np 
from astropy.tests.helper import pytest
from ...custom_exceptions import HalotoolsError

from ..pairwise_velocity_stats import *
from .cf_helpers import generate_locus_of_3d_points

__all__ = ('test_mean_radial_velocity_vs_r_auto_consistency', 
	'test_mean_radial_velocity_vs_r_cross_consistency', 
	'test_radial_pvd_vs_r1', 'test_radial_pvd_vs_r_auto_consistency', 
	'test_radial_pvd_vs_r_cross_consistency')

@pytest.mark.slow
def test_mean_radial_velocity_vs_r_auto_consistency():
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

@pytest.mark.slow
def test_radial_pvd_vs_r1():
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

@pytest.mark.slow
def test_mean_los_velocity_vs_rp_auto_consistency():
	np.random.seed(43)

	npts = 200
	sample1 = np.random.rand(npts, 3)
	velocities1 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
	sample2 = np.random.rand(npts, 3)
	velocities2 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))

	rbins = np.linspace(0, 0.3, 10)
	pi_max = 0.2
	s1s1a, s1s2a, s2s2a = mean_los_velocity_vs_rp(sample1, velocities1, rbins, pi_max, 
		sample2 = sample2, velocities2 = velocities2)
	s1s1b, s2s2b = mean_los_velocity_vs_rp(sample1, velocities1, rbins, pi_max, 
		sample2 = sample2, velocities2 = velocities2, 
		do_cross = False)

	assert np.allclose(s1s1a,s1s1b, rtol=0.001)
	assert np.allclose(s2s2a,s2s2b, rtol=0.001)


@pytest.mark.slow
def test_mean_los_velocity_vs_rp_cross_consistency():
	np.random.seed(43)

	npts = 200
	sample1 = np.random.rand(npts, 3)
	velocities1 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
	sample2 = np.random.rand(npts, 3)
	velocities2 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))

	rbins = np.linspace(0, 0.3, 10)
	pi_max = 0.3
	s1s1a, s1s2a, s2s2a = mean_los_velocity_vs_rp(sample1, velocities1, rbins, pi_max, 
		sample2 = sample2, velocities2 = velocities2)
	s1s2b = mean_los_velocity_vs_rp(sample1, velocities1, rbins, pi_max, 
		sample2 = sample2, velocities2 = velocities2, 
		do_auto = False)

	assert np.allclose(s1s2a,s1s2b, rtol=0.001)


@pytest.mark.slow
def test_los_pvd_vs_rp_auto_consistency():
	np.random.seed(43)

	npts = 200
	sample1 = np.random.rand(npts, 3)
	velocities1 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
	sample2 = np.random.rand(npts, 3)
	velocities2 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))

	rbins = np.linspace(0, 0.3, 10)
	pi_max = 0.2
	s1s1a, s1s2a, s2s2a = los_pvd_vs_rp(sample1, velocities1, rbins, pi_max, 
		sample2 = sample2, velocities2 = velocities2)
	s1s1b, s2s2b = los_pvd_vs_rp(sample1, velocities1, rbins, pi_max, 
		sample2 = sample2, velocities2 = velocities2, 
		do_cross = False)

	assert np.allclose(s1s1a,s1s1b, rtol=0.001)
	assert np.allclose(s2s2a,s2s2b, rtol=0.001)

@pytest.mark.slow
def test_los_pvd_vs_rp_cross_consistency():
	np.random.seed(43)

	npts = 200
	sample1 = np.random.rand(npts, 3)
	velocities1 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
	sample2 = np.random.rand(npts, 3)
	velocities2 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))

	rbins = np.linspace(0, 0.3, 10)
	pi_max = 0.3
	s1s1a, s1s2a, s2s2a = los_pvd_vs_rp(sample1, velocities1, rbins, pi_max, 
		sample2 = sample2, velocities2 = velocities2)
	s1s2b = los_pvd_vs_rp(sample1, velocities1, rbins, pi_max, 
		sample2 = sample2, velocities2 = velocities2, 
		do_auto = False)

	assert np.allclose(s1s2a,s1s2b, rtol=0.001)

@pytest.mark.slow
def test_mean_radial_velocity_vs_r_correctness():
	""" Create two tight localizations of points, 
	one at (0.5, 0.5, 0.1), the other at (0.5, 0.5, 0.2). 
	The first set of points is moving at -50 in the z-direction, 
	the second set of points is at rest. 
	Verify that mean_radial_velocity_vs_r returns -50 
	for the outer bin. 
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
	assert np.allclose(s2s2[0], 0, rtol=0.01)
	assert np.allclose(s1s2[1], -50, rtol=0.01)

@pytest.mark.slow
def test_mean_radial_velocity_vs_r_correctness_pbc():
	""" Create two tight localizations of points, 
	one at (0.5, 0.5, 0.1), the other at (0.5, 0.5, 0.2). 
	The first set of points is moving at -50 in the z-direction, 
	the second set of points is at rest. 
	Verify that mean_radial_velocity_vs_r returns -50 
	for the outer bin. 
	Same test as test_mean_radial_velocity_vs_r_correctness, 
	only here we apply PBC, which should not matter in this case. 
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
		sample2 = sample2, velocities2 = velocities2, period=1.)

	assert np.allclose(s1s1[0], 0, rtol=0.01)
	assert np.allclose(s2s2[0], 0, rtol=0.01)
	assert np.allclose(s1s2[1], -50, rtol=0.01)

@pytest.mark.slow
def test_mean_radial_velocity_vs_r_correctness_pbc2():
	""" Create two tight localizations of points, 
	one at (0.5, 0.5, 0.05), the other at (0.5, 0.5, 0.9). 
	The first set of points is moving at -50 in the z-direction, 
	the second set of points is at rest. 
	Verify that mean_radial_velocity_vs_r returns -50 
	for the outer bin. 

	Same test as test_mean_radial_velocity_vs_r_correctness_pbc, 
	only here the PBC should matter. 
	"""
	np.random.seed(43)

	npts = 200
	sample1 = generate_locus_of_3d_points(npts, 
		xc=0.5, yc=0.5, zc=0.05, epsilon = 0.0001)
	velocities1 = np.zeros(npts*3).reshape(npts, 3)
	velocities1[:,2] = -50.

	sample2 = generate_locus_of_3d_points(npts, 
		xc=0.5, yc=0.5, zc=0.9, epsilon = 0.0001)
	velocities2 = np.zeros(npts*3).reshape(npts, 3)

	rbins = np.array([0, 0.1, 0.3])
	s1s1, s1s2, s2s2 = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
		sample2 = sample2, velocities2 = velocities2, period=1.)

	assert np.allclose(s1s1[0], 0, rtol=0.01)
	assert np.allclose(s2s2[0], 0, rtol=0.01)
	assert np.allclose(s1s2[1], -50, rtol=0.01)

