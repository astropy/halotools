# cython: language_level=2
# cython: profile=False
"""
Custom weighting function.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
cimport numpy as cnp

__author__ = ["Duncan Campbell"]

cdef cnp.float64_t custom_func(cnp.float64_t* w1, cnp.float64_t* w2):
    """
    Modify and use this function with weight_func_id=0 to get a custom function.
    """
    return w1[0]*w2[0]
