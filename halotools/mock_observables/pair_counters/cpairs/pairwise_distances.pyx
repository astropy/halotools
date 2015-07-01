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
from libcpp.vector cimport vector

__all__ = ['pairwise_distance_no_pbc', 'pairwise_distance_pbc',\
           'pairwise_xy_z_distance_no_pbc', 'pairwise_xy_z_distance_pbc']
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
    cdef vector[np.int_t] i_ind
    cdef vector[np.int_t] j_ind
    cdef vector[np.float64_t] distances
    cdef double d
    cdef np.int_t i, j, n
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
                distances.push_back(d)
                i_ind.push_back(i)
                j_ind.push_back(j)
                n = n+1
    
    return np.sqrt(distances).astype(float), np.array(i_ind).astype(int),\
           np.array(j_ind).astype(int)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def pairwise_distance_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                             np.ndarray[np.float64_t, ndim=1] y_icell1,
                             np.ndarray[np.float64_t, ndim=1] z_icell1,
                             np.ndarray[np.float64_t, ndim=1] x_icell2,
                             np.ndarray[np.float64_t, ndim=1] y_icell2,
                             np.ndarray[np.float64_t, ndim=1] z_icell2,
                             np.ndarray[np.float64_t, ndim=1] period,
                             np.float64_t max_r):
    
    """
    real-space pairwise distance calculator.
    """
    
    #c definitions
    cdef vector[np.int_t] i_ind
    cdef vector[np.int_t] j_ind
    cdef vector[np.float64_t] distances
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
            d = periodic_square_distance(x_icell1[i],y_icell1[i],z_icell1[i],\
                                         x_icell2[j],y_icell2[j],z_icell2[j],\
                                         <np.float64_t*> period.data)
                        
            #add distance to result
            if d<=max_r:
                distances.push_back(d)
                i_ind.push_back(i)
                j_ind.push_back(j)
                n = n+1
    
    return np.sqrt(distances).astype(float), np.array(i_ind).astype(int),\
           np.array(j_ind).astype(int)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def pairwise_xy_z_distance_no_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                                  np.ndarray[np.float64_t, ndim=1] y_icell1,
                                  np.ndarray[np.float64_t, ndim=1] z_icell1,
                                  np.ndarray[np.float64_t, ndim=1] x_icell2,
                                  np.ndarray[np.float64_t, ndim=1] y_icell2,
                                  np.ndarray[np.float64_t, ndim=1] z_icell2,
                                  np.float64_t max_rp, np.float64_t max_pi):
    
    """
    2+1D pairwise distance calculator.
    """
    
    #c definitions
    cdef vector[np.int_t] i_ind
    cdef vector[np.int_t] j_ind
    cdef vector[np.float64_t] para_distances
    cdef vector[np.float64_t] perp_distances
    cdef double d_perp, d_para
    cdef int i, j, n
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    
    #loop over points in grid1's cells
    n=0
    for i in range(0,Ni):
        #loop over points in grid2's cells
        for j in range(0,Nj):
                        
            #calculate the square distance
            d_perp = perp_square_distance(x_icell1[i], y_icell1[i],\
                                          x_icell2[j], y_icell2[j])
            d_para = para_square_distance(z_icell1[i], z_icell2[j])
                        
            #add distance to result
            if (d_perp<=max_rp) & (d_para<=max_pi):
                perp_distances.push_back(d_perp)
                para_distances.push_back(d_para)
                i_ind.push_back(i)
                j_ind.push_back(j)
                n = n+1
    
    return np.sqrt(perp_distances).astype(float), np.sqrt(para_distances).astype(float),\
           np.array(i_ind).astype(int), np.array(j_ind).astype(int)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def pairwise_xy_z_distance_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                               np.ndarray[np.float64_t, ndim=1] y_icell1,
                               np.ndarray[np.float64_t, ndim=1] z_icell1,
                               np.ndarray[np.float64_t, ndim=1] x_icell2,
                               np.ndarray[np.float64_t, ndim=1] y_icell2,
                               np.ndarray[np.float64_t, ndim=1] z_icell2,
                               np.ndarray[np.float64_t, ndim=1] period,
                               np.float64_t max_rp, np.float64_t max_pi):
    
    """
    2+1D pairwise distance calculator.
    """
    
    #c definitions
    cdef vector[np.int_t] i_ind
    cdef vector[np.int_t] j_ind
    cdef vector[np.float64_t] para_distances
    cdef vector[np.float64_t] perp_distances
    cdef double d_perp, d_para
    cdef int i, j, n
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    
    #loop over points in grid1's cells
    n=0
    for i in range(0,Ni):
        #loop over points in grid2's cells
        for j in range(0,Nj):
                        
            #calculate the square distance
            d_perp = periodic_perp_square_distance(x_icell1[i],y_icell1[i],\
                                                   x_icell2[j],y_icell2[j],\
                                                   <np.float64_t*>period.data)
            d_para = periodic_para_square_distance(z_icell1[i],\
                                                   z_icell2[j],\
                                                   <np.float64_t*>period.data)
                        
            #add distance to result
            if (d_perp<=max_rp) & (d_para<=max_pi):
                perp_distances.push_back(d_perp)
                para_distances.push_back(d_para)
                i_ind.push_back(i)
                j_ind.push_back(j)
                n = n+1
    
    return np.sqrt(perp_distances).astype(float), np.sqrt(para_distances).astype(float),\
           np.array(i_ind).astype(int), np.array(j_ind).astype(int)
