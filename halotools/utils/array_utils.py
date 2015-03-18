# -*- coding: utf-8 -*-
"""

Modules performing small, commonly used tasks throughout the package.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['array_like_length']

import numpy as np

import collections

def array_like_length(x):
	""" Simple method to return a zero-valued 1-D numpy array 
	with the length of the input x. 

	Parameters 
	----------
	x : array_like
		Can be an iterable such as a list or non-iterable such as a float. 

	Returns 
	-------
	array_length : int 
		length of x

	Notes 
	----- 
	Simple workaround of an awkward feature of numpy. When evaluating 
	the built-in len() function on non-iterables such as a 
	float or int, len() returns a TypeError, rather than unity. 
	Most annoyingly, the same is true on an object such as x=numpy.array(4), 
	even though such an object formally counts as an Iterable, being an ndarray. 
	This nuisance is so ubiquitous that it's convenient to have a single 
	line of code that replaces the default python len() function with sensible behavior.
	"""

	try:
		array_length = len(x)
	except TypeError:
		array_length = 1

	return array_length

