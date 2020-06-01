# cython: language_level=2
# cython: profile=False
"""
objective weighting functions.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import sys
cimport cython
import numpy as np
cimport numpy as np

__author__ = ["Duncan Campbell"]

cdef double custom_func(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift):
    """
    use this function to edit and compile to get a custom function
    """
    return w1[0]*w2[0]
