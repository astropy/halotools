# cython: profile=False

"""
calculate and return the pairwise distances between two sets of points.
"""

from __future__ import print_function, division
import sys
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport fabs, fmin
from distances cimport *

__all__ = ['pairwise_distance_no_pbc']
__author__=['Duncan Campbell']


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def pairwise_distance_no_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                             np.ndarray[np.float64_t, ndim=1] y_icell1,
                             np.ndarray[np.float64_t, ndim=1] z_icell1,
                             np.ndarray[np.float64_t, ndim=1] x_icell2,
                             np.ndarray[np.float64_t, ndim=1] y_icell2,
                             np.ndarray[np.float64_t, ndim=1] z_icell2,
                             np.float64_t max_r):
    
    """
    real-space pairwise distance calculator.
    """
    
    #c definitions
    cdef int max_n =10
    cdef np.ndarray[np.float64_t, ndim=1] distances = np.zeros((max_n,), dtype=np.float64)
    cdef np.ndarray[np.int_t, ndim=1] i_ind = np.zeros((max_n,), dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=1] j_ind = np.zeros((max_n,), dtype=np.int)
    cdef double d
    cdef int i, j, n
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    
    #loop over points in grid1's cells
    n=0
    for i in range(0,Ni):
        #loop over points in grid2's cells
        for j in range(0,Nj):
                        
            #calculate the square distance
            d = square_distance(x_icell1[i],y_icell1[i],z_icell1[i],\
                                x_icell2[j],y_icell2[j],z_icell2[j])
                        
            #add distance to result
            if d<=max_r:
                if n==max_n: #resize arrays if N pairs exceeds length of storage arrays
                    max_n = max_n*2
                    distances.resize(max_n)
                    i_ind.resize(max_n)
                    j_ind.resize(max_n)
                distance[n] = d
                ind_i[n] = i
                ind_j[n] = j
                n = n+1
    
    #trim result arrays
    distance.resize(n)
    i_ind.resize(n)
    j_ind.resize(n)
        
    return distance, ind_i, ind_j