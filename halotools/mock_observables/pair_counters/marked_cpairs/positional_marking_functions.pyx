# cython: profile=False
"""
Marking function definitions that take the 3D position of each point as an argument
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
cimport numpy as cnp
from libc.math cimport fabs as c_fabs

__author__ = ["Duncan Campbell"]

cdef cnp.float64_t pos_shape_dot_product_func(cnp.float64_t* w1, cnp.float64_t*, w2, x1, y1, z1, x2, y2, z2):
    """
    vector dot product of w1 along s, the vector connection point 1 and point 2
    This function assumes w1 and w2 have been normalized
    """
    cdef cnp.float64_t x, y, z, rsq
    
    x = (x2-x1)
    y = (y2-y1)
    z = (z2-z1)
    rsq = x*x + y*y + z*z

    return (w1[0]*x + w1[1]*y + w1[2]*z)/c_sqrt(rsq)