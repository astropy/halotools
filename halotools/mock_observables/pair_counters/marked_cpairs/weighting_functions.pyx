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

cdef double mweights(np.float64_t* w1, np.float64_t* w2):
    """
    multiplicative weights
    return w1[0]*w2[0]
    id: 1
    """
    return w1[0]*w2[0]


cdef double sweights(np.float64_t* w1, np.float64_t* w2):
    """
    summed weights
    return w1[0]+w2[0]
    id: 2
    """
    return w1[0]+w2[0]


cdef double eqweights(np.float64_t* w1, np.float64_t* w2):
    """
    equality weights
    return w1[1]*w2[1] if w1[0]==w2[0]
    id: 3
    """
    if w1[0]==w2[0]: return w1[1]*w2[1]
    else: return 0.0


cdef double gweights(np.float64_t* w1, np.float64_t* w2):
    """
    greater than weights
    return w1[1]*w2[1] if w2[0]>w1[0]
    id: 4
    """
    if w2[0]>w1[0]: return w1[1]*w2[1]
    else: return 0.0


cdef double lweights(np.float64_t* w1, np.float64_t* w2):
    """
    less than weights
    return w1[1]*w2[1] if w2[0]<w1[0]
    id: 5
    """
    if w2[0]<w1[0]: return w1[1]*w2[1]
    else: return 0.0


cdef double tgweights(np.float64_t* w1, np.float64_t* w2):
    """
    greater than tolerance weights
    return w2[1] if w2[0]>(w1[0]+w1[1])
    id: 6
    """
    if w2[0]>(w1[0]+w1[1]): return w2[1]
    else: return 0.0


cdef double tlweights(np.float64_t* w1, np.float64_t* w2):
    """
    less than tolerance weights
    return w2[1] if w2[0]<(w1[0]-w1[1])
    id: 7
    """
    if w2[0]<(w1[0]+w1[1]): return w2[1]
    else: return 0.0


cdef double tweights(np.float64_t* w1, np.float64_t* w2):
    """
    tolerance weights
    return w2[1] if |w1[0]-w2[0]|<w1[1]
    id: 8
    """
    if fabs(w1[0]-w2[0])<w1[1]: return w2[1]
    else: return 0.0


cdef double exweights(np.float64_t* w1, np.float64_t* w2):
    """
    exclusion weights
    return w2[1] if |w1[0]-w2[0]|>w1[1]
    id: 9
    """
    if fabs(w1[0]-w2[0])>w1[1]: return w2[1]
    else: return 0.0


