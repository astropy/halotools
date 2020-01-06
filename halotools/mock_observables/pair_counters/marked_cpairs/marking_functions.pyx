# cython: language_level=2
# cython: profile=False
"""
Marking function definitions.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
cimport numpy as cnp
from libc.math cimport fabs as c_fabs

__author__ = ["Duncan Campbell"]

cdef cnp.float64_t mweights(cnp.float64_t* w1, cnp.float64_t* w2):
    """
    multiplicative weights
    return w1[0]*w2[0]
    id: 1
    expects length 1 arrays
    """
    return w1[0]*w2[0]


cdef cnp.float64_t sweights(cnp.float64_t* w1, cnp.float64_t* w2):
    """
    summed weights
    return w1[0]+w2[0]
    id: 2
    expects length 1 arrays
    """
    return w1[0]+w2[0]


cdef cnp.float64_t eqweights(cnp.float64_t* w1, cnp.float64_t* w2):
    """
    equality weights
    return w1[1]*w2[1] if w1[0]==w2[0]
    id: 3
    expects length 2 arrays
    """
    if w1[0]==w2[0]:
        return w1[1]*w2[1]
    else:
        return 0.0


cdef cnp.float64_t ineqweights(cnp.float64_t* w1, cnp.float64_t* w2):
    """
    equality weights
    return w1[1]*w2[1] if w1[0]!=w2[0]
    id: 4
    expects length 2 arrays
    """
    if w1[0]!=w2[0]:
        return w1[1]*w2[1]
    else:
        return 0.0


cdef cnp.float64_t gweights(cnp.float64_t* w1, cnp.float64_t* w2):
    """
    greater than weights
    return w1[1]*w2[1] if w2[0]>w1[0]
    id: 5
    expects length 2 arrays
    """
    if w2[0]>w1[0]:
        return w1[1]*w2[1]
    else:
        return 0.0


cdef cnp.float64_t lweights(cnp.float64_t* w1, cnp.float64_t* w2):
    """
    less than weights
    return w1[1]*w2[1] if w2[0]<w1[0]
    id: 6
    expects length 2 arrays
    """
    if w2[0]<w1[0]:
        return w1[1]*w2[1]
    else:
        return 0.0


cdef cnp.float64_t tgweights(cnp.float64_t* w1, cnp.float64_t* w2):
    """
    greater than tolerance weights
    return w2[1] if w2[0]>(w1[0]+w1[1])
    id: 7
    expects length 2 arrays
    """
    if w2[0]>(w1[0]+w1[1]):
        return w2[1]
    else:
        return 0.0


cdef cnp.float64_t tlweights(cnp.float64_t* w1, cnp.float64_t* w2):
    """
    less than tolerance weights
    return w2[1] if w2[0]<(w1[0]-w1[1])
    id: 8
    expects length 2 arrays
    """
    if w2[0]<(w1[0]+w1[1]):
        return w2[1]
    else:
        return 0.0


cdef cnp.float64_t tweights(cnp.float64_t* w1, cnp.float64_t* w2):
    """
    tolerance weights
    return w2[1] if |w1[0]-w2[0]|<w1[1]
    id: 9
    expects length 2 arrays
    """
    if c_fabs(w1[0]-w2[0])<w1[1]:
        return w2[1]
    else:
        return 0.0


cdef cnp.float64_t exweights(cnp.float64_t* w1, cnp.float64_t* w2):
    """
    exclusion weights
    return w2[1] if |w1[0]-w2[0]|>w1[1]
    id: 10
    expects length 2 arrays
    """
    if c_fabs(w1[0]-w2[0])>w1[1]:
        return w2[1]
    else:
        return 0.0


cdef cnp.float64_t ratio_weights(cnp.float64_t* w1, cnp.float64_t* w2):
    """
    ratio weights
    return w2[1] if w2[0]>w1[1]*w1[0], 0 otherwise
    id: 11
    expects length 2 arrays
    """
    if w2[0] > w1[0]*w1[1]:
        return w2[1]
    else:
        return 0.0
