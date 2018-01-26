# cython: profile=False
"""
Marking function definitions that take the 3D position of each point as an argument
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
cimport numpy as cnp
from libc.math cimport fabs as c_fabs
from libc.math cimport sqrt as c_sqrt
from libc.math cimport cos as c_cos
from libc.math cimport acos as c_acos
from libc.math cimport sin as c_sin
from libc.math cimport fmin as c_min

__author__ = ["Duncan Campbell"]


cdef cnp.float64_t pos_shape_dot_product_func(cnp.float64_t* w1, cnp.float64_t* w2,
            cnp.float64_t x1, cnp.float64_t y1, cnp.float64_t z1,
            cnp.float64_t x2, cnp.float64_t y2, cnp.float64_t z2, cnp.float64_t rsq):
    """
    vector dot product of w1 along s, the vector connection point 1 and point 2
    This function assumes w1 and w2 have been normalized
    """
    cdef cnp.float64_t x, y, z

    if rsq>0:
        x = (x2-x1)
        y = (y2-y1)
        z = (z2-z1)
        return (w1[0]*x + w1[1]*y + w1[2]*z)/c_sqrt(rsq)
    else:
        return 0.0


cdef cnp.float64_t gamma_plus_func(cnp.float64_t* w1, cnp.float64_t* w2,
            cnp.float64_t x1, cnp.float64_t y1, cnp.float64_t z1,
            cnp.float64_t x2, cnp.float64_t y2, cnp.float64_t z2, cnp.float64_t rsq):
    """
    """
    cdef cnp.float64_t x, y, z
    cdef cnp.float64_t costheta, gamma

    if rsq>0:
        x = (x2-x1)
        y = (y2-y1)
        z = (z2-z1)
        costheta = (w1[0]*x + w1[1]*y + w1[2]*z)/c_sqrt(rsq)
        gamma = c_cos(2.0*c_acos(costheta))
        return gamma
    else:
        return 0.0


cdef cnp.float64_t gamma_cross_func(cnp.float64_t* w1, cnp.float64_t* w2,
            cnp.float64_t x1, cnp.float64_t y1, cnp.float64_t z1,
            cnp.float64_t x2, cnp.float64_t y2, cnp.float64_t z2, cnp.float64_t rsq):
    """
    """
    cdef cnp.float64_t x, y, z
    cdef cnp.float64_t costheta, gamma

    if rsq>0:
        x = (x2-x1)
        y = (y2-y1)
        z = (z2-z1)
        costheta = (w1[0]*x + w1[1]*y + w1[2]*z)/c_sqrt(rsq)
        gamma = c_sin(2.0*c_acos(costheta))
        return gamma
    else:
        return 0.0

cdef cnp.float64_t squareddot_func(cnp.float64_t* w1, cnp.float64_t* w2,
            cnp.float64_t x1, cnp.float64_t y1, cnp.float64_t z1,
            cnp.float64_t x2, cnp.float64_t y2, cnp.float64_t z2, cnp.float64_t rsq):
    """
    calculate the squared dot product between a normalized vaector and the normalized direction between points
    """
    cdef cnp.float64_t x, y, z
    cdef cnp.float64_t costheta

    if rsq>0:
        x = (x2-x1)
        y = (y2-y1)
        z = (z2-z1)
        costheta = c_min((w1[0]*x + w1[1]*y + w1[2]*z)/c_sqrt(rsq), 1.0)
        return costheta*costheta
    else:
        return 0.0



