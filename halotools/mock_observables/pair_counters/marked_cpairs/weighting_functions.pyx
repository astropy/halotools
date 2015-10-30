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
from libc.math cimport fabs

__author__ = ["Duncan Campbell"]

cdef double mweights(np.float64_t w1, np.float64_t w2, np.float64_t r1, np.float64_t r2):
    """
    multiplicative weights
    return w1*w2
    id: 1
    """
    return w1*w2


cdef double sweights(np.float64_t w1, np.float64_t w2, np.float64_t r1, np.float64_t r2):
    """
    summed weights
    return w1+w2
    id: 2
    """
    return w1+w2


cdef double eqweights(np.float64_t w1, np.float64_t w2, np.float64_t r1, np.float64_t r2):
    """
    equality weights
    return r1*r2 if w1==w2
    id: 3
    """
    if w1==w2: return r1*r2
    else: return 0.0


cdef double gweights(np.float64_t w1, np.float64_t w2, np.float64_t r1, np.float64_t r2):
    """
    greater than weights
    return r1*r2 if w2>w1
    id: 4
    """
    if w2>w1: return r1*r2
    else: return 0.0


cdef double lweights(np.float64_t w1, np.float64_t w2, np.float64_t r1, np.float64_t r2):
    """
    less than weights
    return r1*r2 if w2<w1
    id: 5
    """
    if w2<w1: return r1*r2
    else: return 0.0


cdef double tgweights(np.float64_t w1, np.float64_t w2, np.float64_t r1, np.float64_t r2):
    """
    greater than tolerance weights
    return r2 if w2>(w1+r1)
    id: 6
    """
    if w2>(w1+r1): return r2
    else: return 0.0


cdef double tlweights(np.float64_t w1, np.float64_t w2, np.float64_t r1, np.float64_t r2):
    """
    less than tolerance weights
    return r2 if w2<(w1-r1)
    id: 7
    """
    if w2<(w1+r1): return r2
    else: return 0.0


cdef double tweights(np.float64_t w1, np.float64_t w2, np.float64_t r1, np.float64_t r2):
    """
    tolerance weights
    return r2 if |w1-w2|<r1
    id: 8
    """
    if fabs(w1-w2)<r1: return r2
    else: return 0.0


cdef double exweights(np.float64_t w1, np.float64_t w2, np.float64_t r1, np.float64_t r2):
    """
    exclusion weights
    return r2 if |w1-w2|>r1
    id: 9
    """
    if fabs(w1-w2)>r1: return r2
    else: return 0.0


