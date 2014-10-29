# -*- coding: utf-8 -*-
"""

Modules performing small, commonly used tasks throughout the package.

"""

__all__ = ['initialize_numpy_array']

import numpy as np

import collections

def initialize_matching_length_1d_array(x):
	""" Simple method to return a zero-valued 1-D numpy array 
	with the length of the input x. 

	Parameters 
	----------
	x : array_like
		Can be an iterable such as a list or non-iterable such as a float. 

	y : array
		1-D array with the same length as x.

	Notes 
	----- 
	Simple workaround of an awkward feature of numpy. It is common to desire 
	an initialized numpy array, y, of the same length as some other array_like object, x, 
	for example before looping over the initialized array. But non-iterables such as 
	float return a TypeError when trying to evaluate the len() function on them, yet 
	we would like to avoid having to write IF statements over and over again that account 
	for this annoying case of not knowing whether x is iterable. This simple method solves 
	this nuisance once and for all.

	"""

	if isinstance(x, collections.Iterable):
		try:
			array_length = len(x)
			y = np.zeros(array_length)
		except TypeError:
			y = 0
	else:
		y = 0

	return y

