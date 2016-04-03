#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import numpy as np 
from astropy.tests.helper import pytest
from ...custom_exceptions import HalotoolsError

from ..large_scale_density import *
from .cf_helpers import generate_locus_of_3d_points

__all__ = ('test_large_scale_density_spherical_volume1', )

def test_large_scale_density_spherical_volume_exception_handling():
	"""
	"""
	npts1, npts2 = 100, 200
	sample = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1)
	tracers = generate_locus_of_3d_points(npts2, xc=0.15, yc=0.1, zc=0.1)
	radius = 0.1

	with pytest.raises(HalotoolsError) as err:
		result = large_scale_density_spherical_volume(
			sample, tracers, radius)
	substr = "If period is None, you must pass in ``sample_volume``."
	assert substr in err.value.args[0]

	with pytest.raises(HalotoolsError) as err:
		result = large_scale_density_spherical_volume(
			sample, tracers, radius, period=[1,1])
	substr = "Input ``period`` must either be a float or length-3 sequence"
	assert substr in err.value.args[0]

	with pytest.raises(HalotoolsError) as err:
		result = large_scale_density_spherical_volume(
			sample, tracers, radius, period=1, sample_volume=0.4)
	substr = "If period is not None, do not pass in sample_volume"
	assert substr in err.value.args[0]

def test_large_scale_density_spherical_volume1():
	"""
	"""
	npts1, npts2 = 100, 200
	sample = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1)
	tracers = generate_locus_of_3d_points(npts2, xc=0.15, yc=0.1, zc=0.1)
	radius = 0.1
	result = large_scale_density_spherical_volume(
		sample, tracers, radius, period=1)

	environment_volume = (4/3.)*np.pi*radius**3
	correct_answer = 200/environment_volume
	assert np.allclose(result, correct_answer, rtol=0.001)


def test_large_scale_density_spherical_volume2():
	"""
	"""
	npts1, npts2 = 100, 200
	sample = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1)
	tracers = generate_locus_of_3d_points(npts2, xc=0.95, yc=0.1, zc=0.1)
	radius = 0.2
	result = large_scale_density_spherical_volume(
		sample, tracers, radius, period=[1,1,1], norm_by_mean_density=True)
	mean_density = float(npts2)

	environment_volume = (4/3.)*np.pi*radius**3
	correct_answer = 200/environment_volume/mean_density
	assert np.allclose(result, correct_answer, rtol=0.001)

def test_large_scale_density_spherical_annulus_exception_handling():
	"""
	"""
	npts1, npts2 = 100, 200
	sample = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1)
	tracers = generate_locus_of_3d_points(npts2, xc=0.15, yc=0.1, zc=0.1)
	inner_radius, outer_radius = 0.1, 0.2

	with pytest.raises(HalotoolsError) as err:
		result = large_scale_density_spherical_annulus(
			sample, tracers, inner_radius, outer_radius)
	substr = "If period is None, you must pass in ``sample_volume``."
	assert substr in err.value.args[0]

	with pytest.raises(HalotoolsError) as err:
		result = large_scale_density_spherical_annulus(
			sample, tracers, inner_radius, outer_radius, period=[1,1])
	substr = "Input ``period`` must either be a float or length-3 sequence"
	assert substr in err.value.args[0]

	with pytest.raises(HalotoolsError) as err:
		result = large_scale_density_spherical_annulus(
			sample, tracers, inner_radius, outer_radius, period=1, sample_volume=0.4)
	substr = "If period is not None, do not pass in sample_volume"
	assert substr in err.value.args[0]

	with pytest.raises(HalotoolsError) as err:
		result = large_scale_density_spherical_annulus(
			sample, tracers, 0.5, outer_radius, period=1, sample_volume=0.4)
	substr = "Input ``outer_radius`` must be larger than input ``inner_radius``"
	assert substr in err.value.args[0]

def test_large_scale_density_spherical_annulus1():
	"""
	"""
	npts1, npts2 = 100, 200
	sample = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1)
	tracers = generate_locus_of_3d_points(npts2, xc=0.15, yc=0.1, zc=0.1)
	inner_radius, outer_radius = 0.04, 0.1
	result = large_scale_density_spherical_annulus(
		sample, tracers, inner_radius, outer_radius, period=1)

	environment_volume = (4/3.)*np.pi*(outer_radius**3 - inner_radius**3)
	correct_answer = 200/environment_volume
	assert np.allclose(result, correct_answer, rtol=0.001)

def test_large_scale_density_spherical_annulus2():
	"""
	"""
	npts1, npts2 = 100, 200
	sample = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1)
	tracers = generate_locus_of_3d_points(npts2, xc=0.95, yc=0.1, zc=0.1)
	inner_radius, outer_radius = 0.1, 0.2
	result = large_scale_density_spherical_annulus(
		sample, tracers, inner_radius, outer_radius, 
		period=[1,1,1], norm_by_mean_density=True)

	environment_volume = (4/3.)*np.pi*(outer_radius**3 - inner_radius**3)
	mean_density = float(npts2)
	correct_answer = 200/environment_volume/mean_density
	assert np.allclose(result, correct_answer, rtol=0.001)











