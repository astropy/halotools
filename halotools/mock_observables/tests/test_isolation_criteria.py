#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import numpy as np 
from astropy.tests.helper import pytest
from ...custom_exceptions import HalotoolsError

from ..isolation_criteria import *
from .cf_helpers import generate_locus_of_3d_points

__all__ = ['test_spherical_isolation_criteria1']

sample1 = generate_locus_of_3d_points(100, xc=0.1, yc=0.1, zc=0.1)

def test_spherical_isolation_criteria1():
	sample2 = generate_locus_of_3d_points(100, xc=0.5)
	r_max = 0.3
	iso = spherical_isolation(sample1, sample2, r_max)
	assert np.all(iso == True)

def test_spherical_isolation_criteria2():
	sample2 = generate_locus_of_3d_points(100, xc=0.11)
	r_max = 0.3
	iso = spherical_isolation(sample1, sample2, r_max)
	assert np.all(iso == False)

def test_spherical_isolation_criteria3():
	sample2 = generate_locus_of_3d_points(100, xc=0.95)
	r_max = 0.3
	iso = spherical_isolation(sample1, sample2, r_max, period=1)
	assert np.all(iso == False)
	iso2 = spherical_isolation(sample1, sample2, r_max)
	assert np.all(iso2 == True)

def test_cylindrical_isolation1():
	""" For two tight localizations of points right on top of each other, 
	all points in sample1 should not be isolated. 
	"""
	sample2 = generate_locus_of_3d_points(100)
	pi_max = 0.1
	rp_max = 0.1
	iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max)
	assert np.all(iso == False)

def test_cylindrical_isolation2():
	""" For two tight localizations of distant points, 
	all points in sample1 should be isolated unless PBCs are turned on
	"""
	sample1 = generate_locus_of_3d_points(100, xc=0.05, yc=0.05, zc=0.05)
	sample2 = generate_locus_of_3d_points(100, xc=0.95, yc=0.95, zc=0.95)
	pi_max = 0.2
	rp_max = 0.2
	iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max)
	assert np.all(iso == True)
	iso2 = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=1.)
	assert np.all(iso2 == False)

def test_cylindrical_isolation3():
	""" For two tight localizations of distant points, 
	verify independently correct behavior for pi_max and rp_max
	"""
	sample1 = generate_locus_of_3d_points(10, xc=0.05, yc=0.05, zc=0.05)
	sample2 = generate_locus_of_3d_points(10, xc=0.95, yc=0.95, zc=0.95)

	rp_max, pi_max = 0.2, 0.2
	iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max)
	assert np.all(iso == True)
	iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=[1, 1, 1])
	assert np.all(iso == False)

	rp_max, pi_max = 0.2, 0.2
	iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=[1000, 1000, 1])
	assert np.all(iso == True)
	iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=[1, 1, 1000])
	assert np.all(iso == True)

	rp_max, pi_max = 0.05, 0.2
	iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=1)
	assert np.all(iso == True)
	rp_max, pi_max = 0.2, 0.05
	iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=1)
	assert np.all(iso == True)

def test_conditional_spherical_isolation_cond_func1():
	sample1 = generate_locus_of_3d_points(10, xc=0.05, yc=0.05, zc=0.05)
	sample2 = generate_locus_of_3d_points(10, xc=0.95, yc=0.95, zc=0.95)
	r_max = 0.2
	marks1 = np.ones(len(sample1))
	marks2 = np.zeros(len(sample2))
	
	iso = spherical_isolation(sample1, sample2, r_max, period=1)
	assert np.all(iso == False)

	cond_func = 1
	marked_iso1a = conditional_spherical_isolation(sample1, sample2, r_max, 
		marks1, marks2, cond_func, period=1)
	marked_iso1b = conditional_spherical_isolation(sample1, sample2, r_max, 
		marks2, marks1, cond_func, period=1)
	assert np.all(marked_iso1a == False)
	assert np.all(marked_iso1b == True)

def test_conditional_spherical_isolation_cond_func2():
	sample1 = generate_locus_of_3d_points(10, xc=0.05, yc=0.05, zc=0.05)
	sample2 = generate_locus_of_3d_points(10, xc=0.95, yc=0.95, zc=0.95)
	r_max = 0.2
	marks1 = np.ones(len(sample1))
	marks2 = np.zeros(len(sample2))
	
	iso = spherical_isolation(sample1, sample2, r_max, period=1.0)
	assert np.all(iso == False)

	cond_func = 2
	marked_iso2a = conditional_spherical_isolation(sample1, sample2, r_max, 
		marks1, marks2, cond_func, period=1)
	marked_iso2b = conditional_spherical_isolation(sample1, sample2, r_max, 
		marks2, marks1, cond_func, period=1)
	assert np.all(marked_iso2a == True)
	assert np.all(marked_iso2b == False)

def test_conditional_spherical_isolation_cond_func3():
	sample1 = generate_locus_of_3d_points(10, xc=0.05, yc=0.05, zc=0.05)
	sample2 = generate_locus_of_3d_points(10, xc=0.95, yc=0.95, zc=0.95)
	r_max = 0.2
	marks1 = np.ones(len(sample1))
	marks2 = np.zeros(len(sample2))
	
	iso = spherical_isolation(sample1, sample2, r_max, period=1)
	assert np.all(iso == False)

	cond_func = 3
	marked_iso3 = conditional_spherical_isolation(sample1, sample2, r_max, 
		marks1, marks2, cond_func, period=1)
	assert np.all(marked_iso3 == True)

def test_conditional_spherical_isolation_cond_func4():
	sample1 = generate_locus_of_3d_points(10, xc=0.05, yc=0.05, zc=0.05)
	sample2 = generate_locus_of_3d_points(10, xc=0.95, yc=0.95, zc=0.95)
	r_max = 0.2
	marks1 = np.ones(len(sample1))
	marks2 = np.zeros(len(sample2))
	
	iso = spherical_isolation(sample1, sample2, r_max, period=1)
	assert np.all(iso == False)

	cond_func = 4
	marked_iso4 = conditional_spherical_isolation(sample1, sample2, r_max, 
		marks1, marks1, cond_func, period=1)
	assert np.all(marked_iso4 == True)

def test_conditional_spherical_isolation_cond_func5():
	sample1 = generate_locus_of_3d_points(10, xc=0.05, yc=0.05, zc=0.05)
	sample2 = generate_locus_of_3d_points(10, xc=0.95, yc=0.95, zc=0.95)
	r_max = 0.25

	# First verify that the two points are not isolated when ignoring the marks
	iso = spherical_isolation(sample1, sample2, r_max, period=1)
	assert np.all(iso == False)

	# The first sample1 mark is 1, the second sample1 mark is also 1
	# All sample2 marks are 0
	# Thus w_1[0] > (w_2[0]+w_1[1]) NEVER holds, 
	# and so the marked isolation should always be True
	marks1a, marks1b = np.ones(len(sample1)), np.ones(len(sample1)) 
	marks1 = np.vstack([marks1a, marks1b]).T 
	marks2a, marks2b = np.zeros(len(sample2)), np.zeros(len(sample2)) 
	marks2 = np.vstack([marks2a, marks2b]).T 
	
	cond_func = 5
	marked_iso5 = conditional_spherical_isolation(sample1, sample2, r_max, 
		marks1, marks2, cond_func, period=1)
	assert np.all(marked_iso5 == True)

	# The first sample1 mark is 1, the second sample1 mark is 0
	# All sample2 marks are 0
	# Thus w_1[0] > (w_2[0]+w_1[1]) ALWAYS holds, 
	# and so the marked isolation should be equivalent to the unmarked isolation 
	marks1a, marks1b = np.ones(len(sample1)), np.zeros(len(sample1)) 
	marks1 = np.vstack([marks1a, marks1b]).T 
	marks2a, marks2b = np.zeros(len(sample2)), np.zeros(len(sample2)) 
	marks2 = np.vstack([marks2a, marks2b]).T 
	
	cond_func = 5
	marked_iso5 = conditional_spherical_isolation(sample1, sample2, r_max, 
		marks1, marks2, cond_func, period=1)
	assert np.all(marked_iso5 == False)

def test_conditional_cylindrical_isolation_cond_func1():

	sample1 = generate_locus_of_3d_points(10, xc=0.05, yc=0.05, zc=0.05)
	sample2 = generate_locus_of_3d_points(10, xc=0.95, yc=0.95, zc=0.95)

	rp_max, pi_max = 0.2, 0.2
	iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=[1, 1, 1])
	assert np.all(iso == False)

	marks1 = np.ones(len(sample1))
	marks2 = np.zeros(len(sample2))
	
	cond_func = 1
	marked_iso1a = conditional_cylindrical_isolation(sample1, sample2, 
		rp_max, pi_max, marks1, marks2, cond_func, period=1)
	marked_iso1b = conditional_cylindrical_isolation(sample1, sample2, 
		rp_max, pi_max, marks2, marks1, cond_func, period=1)
	assert np.all(marked_iso1a == False)
	assert np.all(marked_iso1b == True)


def test_conditional_cylindrical_isolation_cond_func2():

	sample1 = generate_locus_of_3d_points(10, xc=0.05, yc=0.05, zc=0.05)
	sample2 = generate_locus_of_3d_points(10, xc=0.95, yc=0.95, zc=0.95)

	rp_max, pi_max = 0.2, 0.2
	iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=[1, 1, 1])
	assert np.all(iso == False)

	marks1 = np.ones(len(sample1))
	marks2 = np.zeros(len(sample2))
	
	cond_func = 2
	marked_iso2a = conditional_cylindrical_isolation(sample1, sample2, 
		rp_max, pi_max, marks1, marks2, cond_func, period=1)
	marked_iso2b = conditional_cylindrical_isolation(sample1, sample2, 
		rp_max, pi_max, marks2, marks1, cond_func, period=1)
	assert np.all(marked_iso2a == True)
	assert np.all(marked_iso2b == False)

def test_conditional_cylindrical_isolation_cond_func3():

	sample1 = generate_locus_of_3d_points(10, xc=0.05, yc=0.05, zc=0.05)
	sample2 = generate_locus_of_3d_points(10, xc=0.95, yc=0.95, zc=0.95)

	rp_max, pi_max = 0.2, 0.2
	iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=[1, 1, 1])
	assert np.all(iso == False)

	marks1 = np.ones(len(sample1))
	marks2 = np.zeros(len(sample2))
	
	cond_func = 3
	marked_iso3 = conditional_cylindrical_isolation(sample1, sample2, 
		rp_max, pi_max, marks1, marks2, cond_func, period=1)
	assert np.all(marked_iso3 == True)

def test_conditional_cylindrical_isolation_cond_func4():
	sample1 = generate_locus_of_3d_points(10, xc=0.05, yc=0.05, zc=0.05)
	sample2 = generate_locus_of_3d_points(10, xc=0.95, yc=0.95, zc=0.95)
	
	rp_max, pi_max = 0.2, 0.2
	iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=[1, 1, 1])
	assert np.all(iso == False)

	marks1 = np.ones(len(sample1))
	marks2 = np.zeros(len(sample2))

	cond_func = 4
	marked_iso4 = conditional_cylindrical_isolation(sample1, sample2, 
		rp_max, pi_max, marks1, marks1, cond_func, period=1)
	assert np.all(marked_iso4 == True)

def test_conditional_cylindrical_isolation_cond_func5():
	sample1 = generate_locus_of_3d_points(10, xc=0.05, yc=0.05, zc=0.05)
	sample2 = generate_locus_of_3d_points(10, xc=0.95, yc=0.95, zc=0.95)

	# First verify that the two points are not isolated when ignoring the marks
	rp_max, pi_max = 0.2, 0.2
	iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=[1, 1, 1])
	assert np.all(iso == False)

	# The first sample1 mark is 1, the second sample1 mark is also 1
	# All sample2 marks are 0
	# Thus w_1[0] > (w_2[0]+w_1[1]) NEVER holds, 
	# and so the marked isolation should always be True
	marks1a, marks1b = np.ones(len(sample1)), np.ones(len(sample1)) 
	marks1 = np.vstack([marks1a, marks1b]).T 
	marks2a, marks2b = np.zeros(len(sample2)), np.zeros(len(sample2)) 
	marks2 = np.vstack([marks2a, marks2b]).T 
	
	cond_func = 5
	marked_iso5 = conditional_cylindrical_isolation(sample1, sample2, 
		rp_max, pi_max, marks1, marks2, cond_func, period=1)
	assert np.all(marked_iso5 == True)

	# The first sample1 mark is 1, the second sample1 mark is 0
	# All sample2 marks are 0
	# Thus w_1[0] > (w_2[0]+w_1[1]) ALWAYS holds, 
	# and so the marked isolation should be equivalent to the unmarked isolation 
	marks1a, marks1b = np.ones(len(sample1)), np.zeros(len(sample1)) 
	marks1 = np.vstack([marks1a, marks1b]).T 
	marks2a, marks2b = np.zeros(len(sample2)), np.zeros(len(sample2)) 
	marks2 = np.vstack([marks2a, marks2b]).T 
	
	cond_func = 5
	marked_iso5 = conditional_cylindrical_isolation(sample1, sample2, 
		rp_max, pi_max, marks1, marks2, cond_func, period=1)
	assert np.all(marked_iso5 == False)




