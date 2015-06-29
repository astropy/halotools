# cython: profile=False

"""
distance calculations
"""

from __future__ import print_function, division
cimport cython
from libc.math cimport fabs, fmin

__author__=['Duncan Campbell']


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
    
    dx = fabs(x1 - x2)
    dx = fmin(dx, period[0] - dx)
    dy = fabs(y1 - y2)
    dy = fmin(dy, period[1] - dy)
    dz = fabs(z1 - z2)
    dz = fmin(dz, period[2] - dz)
    return dx*dx+dy*dy+dz*dz


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


cdef double para_square_distance(np.float64_t z1, np.float64_t z2):
    """
    Calculate the parallel square cartesian distance between two sets of points.
    e.g. pi
    """
    
    cdef double dz
    
    dz = z1 - z2
    return dz*dz


cdef double periodic_perp_square_distance(np.float64_t x1, np.float64_t y1,\
                                                 np.float64_t x2, np.float64_t y2,\
                                                 np.float64_t* period):
    """
    Calculate the projected square cartesian distance between two sets of points with 
    periodic boundary conditions.
    e.g. r_p
    """
    
    cdef double dx, dy
    
    dx = fabs(x1 - x2)
    dx = fmin(dx, period[0] - dx)
    dy = fabs(y1 - y2)
    dy= fmin(dy, period[1] - dy)
    return dx*dx+dy*dy


cdef double periodic_para_square_distance(np.float64_t z1, np.float64_t z2,\
                                                 np.float64_t* period):
    """
    Calculate the parallel square cartesian distance between two sets of points with 
    periodic boundary conditions.
    e.g. pi
    """
    
    cdef double dz
    
    dz = fabs(z1 - z2)
    dz = fmin(dz, period[2] - dz)
    return dz*dz

