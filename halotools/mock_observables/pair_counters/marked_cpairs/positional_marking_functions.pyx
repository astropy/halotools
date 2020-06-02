# cython: language_level=2
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
from libc.math cimport fmax as c_max


__author__ = ["Duncan Campbell"]


cdef cnp.float64_t pos_shape_dot_product_func(cnp.float64_t* w1, cnp.float64_t* w2,
            cnp.float64_t x1, cnp.float64_t y1, cnp.float64_t z1,
            cnp.float64_t x2, cnp.float64_t y2, cnp.float64_t z2, cnp.float64_t rsq):
    """
    retrun the vector dot product of w1 along s, the vector connectinng point 1 and point 2
    it is assumed that w1 and w2 have been normalized.
    """
    cdef cnp.float64_t x, y, z
    cdef cnp.float64_t costheta

    if rsq>0:
        x = (x2-x1)
        y = (y2-y1)
        z = (z2-z1)
        costheta = (w1[1]*x + w1[2]*y + w1[3]*z)/c_sqrt(rsq)
        return w1[0]*w2[0]*costheta
    else:
        return 0.0


cdef cnp.float64_t gamma_plus_func(cnp.float64_t* w1, cnp.float64_t* w2,
            cnp.float64_t x1, cnp.float64_t y1, cnp.float64_t z1,
            cnp.float64_t x2, cnp.float64_t y2, cnp.float64_t z2, cnp.float64_t rsq):
    """
    return cos(2phi), where phi is the angle between w1 and s, the vector connecting point 1 and point 2
    it is assumed that w1 and w2 have been normalized.
    """
    cdef cnp.float64_t x, y, z
    cdef cnp.float64_t costheta, gamma

    if rsq>0:
        x = (x2-x1)
        y = (y2-y1)
        z = (z2-z1)
        costheta = (w1[1]*x + w1[2]*y + w1[3]*z)/c_sqrt(rsq)
        # gamma = c_cos(2.0*c_acos(costheta))
        gamma = 2.0*costheta*costheta - 1.0
        return w1[0]*w2[0]*gamma
    else:
        return 0.0


cdef cnp.float64_t gamma_cross_func(cnp.float64_t* w1, cnp.float64_t* w2,
            cnp.float64_t x1, cnp.float64_t y1, cnp.float64_t z1,
            cnp.float64_t x2, cnp.float64_t y2, cnp.float64_t z2, cnp.float64_t rsq):
    """
    return sin(2phi), where phi is the angle between w1 and s, the vector connecting point 1 and point 2
    it is assumed that w1 and w2 have been normalized.
    """
    cdef cnp.float64_t x, y, z
    cdef cnp.float64_t costheta, gamma

    if rsq>0:
        x = (x2-x1)
        y = (y2-y1)
        z = (z2-z1)
        costheta = c_min((w1[1]*x + w1[2]*y + w1[3]*z)/c_sqrt(rsq), 1.0)
        costheta = c_max(costheta, -1.0)
        gamma = c_sin(2.0*c_acos(costheta))
        return w1[0]*w2[0]*gamma
    else:
        return 0.0


cdef cnp.float64_t squareddot_func(cnp.float64_t* w1, cnp.float64_t* w2,
            cnp.float64_t x1, cnp.float64_t y1, cnp.float64_t z1,
            cnp.float64_t x2, cnp.float64_t y2, cnp.float64_t z2, cnp.float64_t rsq):
    """
    return cos^2(phi), where phi is the angle between w1 and s, the vector connecting point 1 and point 2
    it is assumed that w1 and w2 have been normalized.
    """
    cdef cnp.float64_t x, y, z
    cdef cnp.float64_t costheta

    if rsq>0:
        x = (x2-x1)
        y = (y2-y1)
        z = (z2-z1)
        costheta = (w1[1]*x + w1[2]*y + w1[3]*z)/c_sqrt(rsq)
        return w1[0]*w2[0]*costheta*costheta
    else:
        return 0.0


cdef cnp.float64_t gamma_gamma_plus_func(cnp.float64_t* w1, cnp.float64_t* w2,
            cnp.float64_t x1, cnp.float64_t y1, cnp.float64_t z1,
            cnp.float64_t x2, cnp.float64_t y2, cnp.float64_t z2, cnp.float64_t rsq):
    """
    return cos(2phi_1)*cos(2phi_2), where phi_1 is the angle between w1 and s,
    phi_2 is the angle between w2 and s, and
    the vector connecting point 1 and point 2
    it is assumed that w1 and w2 have been normalized.
    """
    cdef cnp.float64_t x, y, z
    cdef cnp.float64_t costheta, gamma1, gamma2

    if rsq>0:
        x = (x2-x1)
        y = (y2-y1)
        z = (z2-z1)
        costheta = (w1[1]*x + w1[2]*y + w1[3]*z)/c_sqrt(rsq)
        gamma1 = 2.0*costheta*costheta - 1.0
        costheta = -1.0*(w2[1]*x + w2[2]*y + w2[3]*z)/c_sqrt(rsq)
        gamma2 = 2.0*costheta*costheta - 1.0
        return w1[0]*w2[0]*gamma1*gamma2
    else:
        return 0.0


cdef cnp.float64_t gamma_gamma_cross_func(cnp.float64_t* w1, cnp.float64_t* w2,
            cnp.float64_t x1, cnp.float64_t y1, cnp.float64_t z1,
            cnp.float64_t x2, cnp.float64_t y2, cnp.float64_t z2, cnp.float64_t rsq):
    """
    return sin(2phi_1)*sin(2phi_2), where phi_1 is the angle between w1 and s,
    phi_2 is the angle between w2 and s, and
    the vector connecting point 1 and point 2
    it is assumed that w1 and w2 have been normalized.
    """
    cdef cnp.float64_t x, y, z
    cdef cnp.float64_t costheta, gamma1, gamma2

    if rsq>0:
        x = (x2-x1)
        y = (y2-y1)
        z = (z2-z1)
        costheta = c_min((w1[1]*x + w1[2]*y + w1[3]*z)/c_sqrt(rsq), 1.0)
        costheta = c_max(costheta, -1.0)
        gamma1 = c_sin(2.0*c_acos(costheta))
        costheta = c_min(-1.0*(w2[1]*x + w2[2]*y + w2[3]*z)/c_sqrt(rsq), 1.0)
        costheta = c_max(costheta, -1.0)
        gamma2 = c_sin(2.0*c_acos(costheta))
        return w1[0]*w2[0]*gamma1*gamma2
    else:
        return 0.0


cdef cnp.float64_t squareddot_eq_func(cnp.float64_t* w1, cnp.float64_t* w2,
            cnp.float64_t x1, cnp.float64_t y1, cnp.float64_t z1,
            cnp.float64_t x2, cnp.float64_t y2, cnp.float64_t z2, cnp.float64_t rsq):
    """
    return cos^2(phi), where phi is the angle between w1 and s, the vector connecting point 1 and point 2
    it is assumed that w1 and w2 have been normalized.
    """
    cdef cnp.float64_t x, y, z
    cdef cnp.float64_t costheta

    if (rsq>0) & (w1[4]==w2[1]):
        x = (x2-x1)
        y = (y2-y1)
        z = (z2-z1)
        costheta = (w1[1]*x + w1[2]*y + w1[3]*z)/c_sqrt(rsq)
        return w1[0]*w2[0]*costheta*costheta
    else:
        return 0.0


cdef cnp.float64_t squareddot_ineq_func(cnp.float64_t* w1, cnp.float64_t* w2,
            cnp.float64_t x1, cnp.float64_t y1, cnp.float64_t z1,
            cnp.float64_t x2, cnp.float64_t y2, cnp.float64_t z2, cnp.float64_t rsq):
    """
    return cos^2(phi), where phi is the angle between w1 and s, the vector connecting point 1 and point 2
    it is assumed that w1 and w2 have been normalized.
    """
    cdef cnp.float64_t x, y, z
    cdef cnp.float64_t costheta

    if (rsq>0) & (w1[4]!=w2[1]):
        x = (x2-x1)
        y = (y2-y1)
        z = (z2-z1)
        costheta = (w1[1]*x + w1[2]*y + w1[3]*z)/c_sqrt(rsq)
        return w1[0]*w2[0]*costheta*costheta
    else:
        return 0.0



