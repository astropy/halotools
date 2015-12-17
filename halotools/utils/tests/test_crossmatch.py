#!/usr/bin/env python
import numpy as np 
import pytest

from ..match import crossmatch
from ...custom_exceptions import HalotoolsError

__all__ = ['test_crossmatch']

def test_crossmatch():
	""" Function providing tests of `~halotools.utils.crossmatch`. 

	* Enforce that the returned index arrays always correspond to exact matches regardless of the domain of *x* and *y*

	* Enforce that when *x* and *y* contain the same (shuffled) points, every point is matched

	* Enforce that when the domains of *x* and *y* are disjoint, there are no matches 

	* Enforce that then *y* contains repeated entries, an exception is raised. 
	"""
	num_pts_x = 1000
	num_pts_y = 100

	# x contains a unique set of entries with a domain that spans the domain of y
	x = np.random.permutation(np.arange(num_pts_x))
	y = np.arange(0, num_pts_y)
	match_into_y, matched_y = crossmatch(x, y)
	assert np.all(x[match_into_y] == y[matched_y])

	# x and y contain the same points, so every point is matched
	x = np.random.permutation(np.arange(num_pts_x))
	y = np.arange(num_pts_x)
	match_into_y, matched_y = crossmatch(x, y)
	assert np.all(x[match_into_y] == y[matched_y])
	assert len(x[match_into_y]) == len(x)
	assert len(y[matched_y]) == len(y)

	# x contains repeated entries 
	# the domain of x is disjoint from the domain of y, 
	# so there should be zero matches
	x = np.random.random_integers(0, 10, num_pts_x)
	y = np.arange(-num_pts_y)
	match_into_y, matched_y = crossmatch(x, y)
	assert np.all(x[match_into_y] == y[matched_y])
	assert len(x[match_into_y]) == 0
	assert len(y[matched_y]) == 0

	# Verify that if y has repeated entries an exception is raised
	y = np.ones(num_pts_y)
	with pytest.raises(HalotoolsError) as err:
		idx = crossmatch(x, y)
	substr = 'The second array must only contain unique entries.'
	assert substr in err.value.message 
	