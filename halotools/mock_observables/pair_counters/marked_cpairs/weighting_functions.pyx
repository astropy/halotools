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
from libc.math cimport fabs, sqrt

__author__ = ["Duncan Campbell"]

cdef double mweights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift):
    """
    multiplicative weights
    return w1[0]*w2[0]
    id: 1
    expects length 1 arrays
    """
    return w1[0]*w2[0]


cdef double sweights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift):
    """
    summed weights
    return w1[0]+w2[0]
    id: 2
    expects length 1 arrays
    """
    return w1[0]+w2[0]


cdef double eqweights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift):
    """
    equality weights
    return w1[1]*w2[1] if w1[0]==w2[0]
    id: 3
    expects length 2 arrays
    """
    if w1[0]==w2[0]: return w1[1]*w2[1]
    else: return 0.0


cdef double ineqweights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift):
    """
    equality weights
    return w1[1]*w2[1] if w1[0]!=w2[0]
    id: 3
    expects length 2 arrays
    """
    if w1[0]!=w2[0]: return w1[1]*w2[1]
    else: return 0.0


cdef double gweights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift):
    """
    greater than weights
    return w1[1]*w2[1] if w2[0]>w1[0]
    id: 4
    expects length 2 arrays
    """
    if w2[0]>w1[0]: return w1[1]*w2[1]
    else: return 0.0


cdef double lweights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift):
    """
    less than weights
    return w1[1]*w2[1] if w2[0]<w1[0]
    id: 5
    expects length 2 arrays
    """
    if w2[0]<w1[0]: return w1[1]*w2[1]
    else: return 0.0


cdef double tgweights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift):
    """
    greater than tolerance weights
    return w2[1] if w2[0]>(w1[0]+w1[1])
    id: 6
    expects length 2 arrays
    """
    if w2[0]>(w1[0]+w1[1]): return w2[1]
    else: return 0.0


cdef double tlweights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift):
    """
    less than tolerance weights
    return w2[1] if w2[0]<(w1[0]-w1[1])
    id: 7
    expects length 2 arrays
    """
    if w2[0]<(w1[0]+w1[1]): return w2[1]
    else: return 0.0


cdef double tweights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift):
    """
    tolerance weights
    return w2[1] if |w1[0]-w2[0]|<w1[1]
    id: 8
    expects length 2 arrays
    """
    if fabs(w1[0]-w2[0])<w1[1]: return w2[1]
    else: return 0.0


cdef double exweights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift):
    """
    exclusion weights
    return w2[1] if |w1[0]-w2[0]|>w1[1]
    id: 9
    expects length 2 arrays
    """
    if fabs(w1[0]-w2[0])>w1[1]: return w2[1]
    else: return 0.0


cdef double radial_velocity_weights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift):
    """
    expects length 6 arrays
    
    w[0:2] is the position vector
    w[3:] is the velocity vector
    """
    
    #calculate radial vector between points
    cdef float rx = w1[0] - (w2[0] + shift[0])
    cdef float ry = w1[1] - (w2[1] + shift[1])
    cdef float rz = w1[2] - (w2[2] + shift[2])
    cdef float norm = sqrt(rx*rx + ry*ry + rz*rz)
    if norm==0: return 0.0
    
    #if shift[i]<0 or shift[i]>0 return -1, else return 1
    #looks weird, but it works! -DC
    cdef float xshift = -1.0*(shift[0]<0.0) - (shift[0]>0.0) + (shift[0]==0.0)
    cdef float yshift = -1.0*(shift[1]<0.0) - (shift[1]>0.0) + (shift[1]==0.0)
    cdef float zshift = -1.0*(shift[2]<0.0) - (shift[2]>0.0) + (shift[2]==0.0)
    
    cdef float dvx = xshift*(w1[3] - w2[3])
    cdef float dvy = yshift*(w1[4] - w2[4])
    cdef float dvz = zshift*(w1[5] - w2[5])
    
    return (dvx*rx + dvy*ry + dvz*rz)/norm


cdef double velocity_dot_weights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift):
    """
    expects length 3 arrays

    w is the velocity vector
    """
    
    #if shift[i]<0 or shift[i]>0 return -1, else return 1
    #looks weird, but it works! -DC
    cdef float xshift = -1.0*(shift[0]<0.0) - (shift[0]>0.0) + (shift[0]==0.0)
    cdef float yshift = -1.0*(shift[1]<0.0) - (shift[1]>0.0) + (shift[1]==0.0)
    cdef float zshift = -1.0*(shift[2]<0.0) - (shift[2]>0.0) + (shift[2]==0.0)
    
    cdef float v1_dot_v2 = (w1[0] + xshift*w2[0]) + (w1[1] + yshift*w2[1]) + (w1[2] + zshift*w2[2])
    
    return v1_dot_v2


cdef double velocity_angle_weights(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift):
    """
    expects length 3 arrays

    w is the velocity vector
    """
    
    #if shift[i]<0 or shift[i]>0 return -1, else return 1
    #looks weird, but it works! -DC
    cdef float xshift = -1.0*(shift[0]<0.0) - (shift[0]>0.0) + (shift[0]==0.0)
    cdef float yshift = -1.0*(shift[1]<0.0) - (shift[1]>0.0) + (shift[1]==0.0)
    cdef float zshift = -1.0*(shift[2]<0.0) - (shift[2]>0.0) + (shift[2]==0.0)
    
    cdef float v1_dot_v2 = (w1[0] + xshift*w2[0]) + (w1[1] + yshift*w2[1]) + (w1[2] + zshift*w2[2])
    norm = sqrt(w1[0]*w1[0] + w1[1]*w1[1] + w1[2]*w1[2])*sqrt(w2[0]*w2[0] + w2[1]*w2[1] + w2[2]*w2[2])
    
    return v1_dot_v2/norm


