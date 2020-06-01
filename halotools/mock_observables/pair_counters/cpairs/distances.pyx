# cython: language_level=2
# cython: profile=False

"""
distance calculations
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport fabs as c_fabs

__all__=['periodic_square_distance','square_distance','perp_square_distance',\
         'para_square_distance','periodic_perp_square_distance',\
         'periodic_para_square_distance']
__author__=['Duncan Campbell']

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double periodic_square_distance(np.float64_t x1,\
                                     np.float64_t y1,\
                                     np.float64_t z1,\
                                     np.float64_t x2,\
                                     np.float64_t y2,\
                                     np.float64_t z2,\
                                     np.float64_t* period):
    """
    Calculate the 3D square cartesian distance between two sets of points with periodic
    boundary conditions.
    """

    cdef double dx, dy, dz

    dx = c_fabs(x1 - x2)
    dy = c_fabs(y1 - y2)
    dz = c_fabs(z1 - z2)
    if dx > period[0]/2.: dx = period[0] - dx
    if dy > period[1]/2.: dy = period[1] - dy
    if dz > period[2]/2.: dz = period[2] - dz
    return dx*dx+dy*dy+dz*dz

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double square_distance(np.float64_t x1, np.float64_t y1, np.float64_t z1,\
                            np.float64_t x2, np.float64_t y2, np.float64_t z2):
    """
    Calculate the 3D square cartesian distance between two sets of points.
    """

    cdef double dx, dy, dz

    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    return dx*dx+dy*dy+dz*dz

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double perp_square_distance(np.float64_t x1, np.float64_t y1,\
                                 np.float64_t x2, np.float64_t y2):
    """
    Calculate the projected square cartesian distance between two sets of points.
    e.g. r_p
    """

    cdef double dx, dy

    dx = x1 - x2
    dy = y1 - y2
    return dx*dx+dy*dy

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double para_square_distance(np.float64_t z1, np.float64_t z2):
    """
    Calculate the parallel square cartesian distance between two sets of points.
    e.g. pi
    """

    cdef double dz

    dz = z1 - z2
    return dz*dz

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double periodic_perp_square_distance(np.float64_t x1, np.float64_t y1,\
                                          np.float64_t x2, np.float64_t y2,\
                                          np.float64_t* period):
    """
    Calculate the projected square cartesian distance between two sets of points with
    periodic boundary conditions.
    e.g. r_p
    """

    cdef double dx, dy

    dx = c_fabs(x1 - x2)
    dy = c_fabs(y1 - y2)
    if dx > period[0]/2.: dx = period[0] - dx
    if dy > period[1]/2.: dy = period[1] - dy
    return dx*dx+dy*dy

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double periodic_para_square_distance(np.float64_t z1, np.float64_t z2,\
                                          np.float64_t* period):
    """
    Calculate the parallel square cartesian distance between two sets of points with
    periodic boundary conditions.
    e.g. pi
    """

    cdef double dz

    dz = c_fabs(z1 - z2)
    if dz > period[2]/2.: dz = period[2] - dz
    return dz*dz


