# cython: profile=False

"""
objective pair counter.
"""


from __future__ import print_function, division
import sys
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport fabs, fmin
from objective_weights cimport *


__author__ = ['Duncan Campbell']
__all__ = ['obj_wnpairs_no_pbc', 'obj_wnpairs_pbc']


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def obj_wnpairs_no_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                       np.ndarray[np.float64_t, ndim=1] y_icell1,
                       np.ndarray[np.float64_t, ndim=1] z_icell1,
                       np.ndarray[np.float64_t, ndim=1] x_icell2,
                       np.ndarray[np.float64_t, ndim=1] y_icell2,
                       np.ndarray[np.float64_t, ndim=1] z_icell2,
                       np.ndarray[np.float64_t, ndim=1] w_icell1,
                       np.ndarray[np.float64_t, ndim=1] w_icell2,
                       np.ndarray[np.float64_t, ndim=1] r_icell1,
                       np.ndarray[np.float64_t, ndim=1] r_icell2,
                       np.ndarray[np.float64_t, ndim=1] rbins,
                       np.int_t weight_func_id
                       ):
    """
    weighted real-space pair counter without periodic boundary conditions (no PBCs).
    Calculate the weighted number of pairs with separations less than or equal to 
    rbins[i].
    """
    
    #c definitions
    cdef int nbins = len(rbins)
    cdef int nbins_minus_one = len(rbins) -1
    cdef np.ndarray[np.float64_t, ndim=1] counts = np.zeros((nbins,), dtype=np.float64)
    cdef double d
    cdef int i, j
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    
    #loop over points in grid1's cell
    for i in range(0,len(x_icell1)):
                
        #loop over points in grid2's cell
        for j in range(0,len(x_icell2)):
                    
            #calculate the square distance
            d = square_distance(x_icell1[i],y_icell1[i],z_icell1[i],\
                                x_icell2[j],y_icell2[j],z_icell2[j])
            
            #calculate counts in bins
            if weight_func_id==0:
                radial_wbinning_0(<np.float64_t*>counts.data,\
                                  <np.float64_t*>rbins.data, d, nbins_minus_one,\
                                  w_icell1[i], w_icell2[j], r_icell1[i], r_icell2[j])
            elif weight_func_id==1:
                radial_wbinning_1(<np.float64_t*>counts.data,\
                                  <np.float64_t*>rbins.data, d, nbins_minus_one,\
                                  w_icell1[i], w_icell2[j], r_icell1[i], r_icell2[j])
            elif weight_func_id==2:
                radial_wbinning_2(<np.float64_t*>counts.data,\
                                  <np.float64_t*>rbins.data, d, nbins_minus_one,\
                                  w_icell1[i], w_icell2[j], r_icell1[i], r_icell2[j])
            elif weight_func_id==3:
                radial_wbinning_3(<np.float64_t*>counts.data,\
                                  <np.float64_t*>rbins.data, d, nbins_minus_one,\
                                  w_icell1[i], w_icell2[j], r_icell1[i], r_icell2[j])
            elif weight_func_id==4:
                radial_wbinning_4(<np.float64_t*>counts.data,\
                                  <np.float64_t*>rbins.data, d, nbins_minus_one,\
                                  w_icell1[i], w_icell2[j], r_icell1[i], r_icell2[j])
            elif weight_func_id==5:
                radial_wbinning_5(<np.float64_t*>counts.data,\
                                  <np.float64_t*>rbins.data, d, nbins_minus_one,\
                                  w_icell1[i], w_icell2[j], r_icell1[i], r_icell2[j])
            elif weight_func_id==6:
                radial_wbinning_6(<np.float64_t*>counts.data,\
                                  <np.float64_t*>rbins.data, d, nbins_minus_one,\
                                  w_icell1[i], w_icell2[j], r_icell1[i], r_icell2[j])
            elif weight_func_id==7:
                radial_wbinning_7(<np.float64_t*>counts.data,\
                                  <np.float64_t*>rbins.data, d, nbins_minus_one,\
                                  w_icell1[i], w_icell2[j], r_icell1[i], r_icell2[j])
            elif weight_func_id==8:
                radial_wbinning_8(<np.float64_t*>counts.data,\
                                  <np.float64_t*>rbins.data, d, nbins_minus_one,\
                                  w_icell1[i], w_icell2[j], r_icell1[i], r_icell2[j])
            elif weight_func_id==9:
                radial_wbinning_9(<np.float64_t*>counts.data,\
                                  <np.float64_t*>rbins.data, d, nbins_minus_one,\
                                  w_icell1[i], w_icell2[j], r_icell1[i], r_icell2[j])
    
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def obj_wnpairs_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                   np.ndarray[np.float64_t, ndim=1] y_icell1,
                   np.ndarray[np.float64_t, ndim=1] z_icell1,
                   np.ndarray[np.float64_t, ndim=1] x_icell2,
                   np.ndarray[np.float64_t, ndim=1] y_icell2,
                   np.ndarray[np.float64_t, ndim=1] z_icell2,
                   np.ndarray[np.float64_t, ndim=1] w_icell1,
                   np.ndarray[np.float64_t, ndim=1] w_icell2,
                   np.ndarray[np.float64_t, ndim=1] r_icell1,
                   np.ndarray[np.float64_t, ndim=1] r_icell2,
                   np.ndarray[np.float64_t, ndim=1] rbins,
                   np.ndarray[np.float64_t, ndim=1] period,
                   np.int_t weight_func_id):
    """
    weighted real-space pair counter with periodic boundary conditions (PBCs).
    Calculate the weighted number of pairs with separations less than or equal to 
    rbins[i].
    """
    
    #c definitions
    cdef int nbins = len(rbins)
    cdef int nbins_minus_one = len(rbins) -1
    cdef np.ndarray[np.float64_t, ndim=1] counts = np.zeros((nbins,), dtype=np.float64)
    cdef double d
    cdef int i, j
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    
    #loop over points in grid1's cell
    for i in range(0,len(x_icell1)):
                
        #loop over points in grid2's cell
        for j in range(0,len(x_icell2)):
                    
            #calculate the square distance
            d = periodic_square_distance(x_icell1[i],y_icell1[i],z_icell1[i],\
                                         x_icell2[j],y_icell2[j],z_icell2[j],\
                                         <np.float64_t*>period.data)
                    
            #calculate counts in bins
            if weight_func_id==0:
                radial_wbinning_0(<np.float64_t*>counts.data,\
                                  <np.float64_t*>rbins.data, d, nbins_minus_one,\
                                  w_icell1[i], w_icell2[j], r_icell1[i], r_icell2[j])
            elif weight_func_id==1:
                radial_wbinning_1(<np.float64_t*>counts.data,\
                                  <np.float64_t*>rbins.data, d, nbins_minus_one,\
                                  w_icell1[i], w_icell2[j], r_icell1[i], r_icell2[j])
            elif weight_func_id==2:
                radial_wbinning_2(<np.float64_t*>counts.data,\
                                  <np.float64_t*>rbins.data, d, nbins_minus_one,\
                                  w_icell1[i], w_icell2[j], r_icell1[i], r_icell2[j])
            elif weight_func_id==3:
                radial_wbinning_3(<np.float64_t*>counts.data,\
                                  <np.float64_t*>rbins.data, d, nbins_minus_one,\
                                  w_icell1[i], w_icell2[j], r_icell1[i], r_icell2[j])
            elif weight_func_id==4:
                radial_wbinning_4(<np.float64_t*>counts.data,\
                                  <np.float64_t*>rbins.data, d, nbins_minus_one,\
                                  w_icell1[i], w_icell2[j], r_icell1[i], r_icell2[j])
            elif weight_func_id==5:
                radial_wbinning_5(<np.float64_t*>counts.data,\
                                  <np.float64_t*>rbins.data, d, nbins_minus_one,\
                                  w_icell1[i], w_icell2[j], r_icell1[i], r_icell2[j])
            elif weight_func_id==6:
                radial_wbinning_6(<np.float64_t*>counts.data,\
                                  <np.float64_t*>rbins.data, d, nbins_minus_one,\
                                  w_icell1[i], w_icell2[j], r_icell1[i], r_icell2[j])
            elif weight_func_id==7:
                radial_wbinning_7(<np.float64_t*>counts.data,\
                                  <np.float64_t*>rbins.data, d, nbins_minus_one,\
                                  w_icell1[i], w_icell2[j], r_icell1[i], r_icell2[j])
            elif weight_func_id==8:
                radial_wbinning_8(<np.float64_t*>counts.data,\
                                  <np.float64_t*>rbins.data, d, nbins_minus_one,\
                                  w_icell1[i], w_icell2[j], r_icell1[i], r_icell2[j])
            elif weight_func_id==9:
                radial_wbinning_9(<np.float64_t*>counts.data,\
                                  <np.float64_t*>rbins.data, d, nbins_minus_one,\
                                  w_icell1[i], w_icell2[j], r_icell1[i], r_icell2[j])
    
    return counts


cdef inline double periodic_square_distance(np.float64_t x1,\
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


cdef inline double square_distance(np.float64_t x1, np.float64_t y1, np.float64_t z1,\
                                   np.float64_t x2, np.float64_t y2, np.float64_t z2):
    """
    Calculate the 3D square cartesian distance between two sets of points.
    """
    
    cdef double dx, dy, dz
    
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    return dx*dx+dy*dy+dz*dz


cdef inline radial_wbinning_0(np.float64_t* counts, np.float64_t* bins,\
                             np.float64_t d, np.int_t k,\
                             np.float64_t w1, np.float64_t w2,
                             np.float64_t r1, np.float64_t r2):
    """
    real space radial weighted binning function
    """
    
    while d<=bins[k]:
        counts[k] += 0
        k=k-1
        if k<0: break


cdef inline radial_wbinning_1(np.float64_t* counts, np.float64_t* bins,\
                             np.float64_t d, np.int_t k,\
                             np.float64_t w1, np.float64_t w2,
                             np.float64_t r1, np.float64_t r2):
    """
    real space radial weighted binning function
    """
    
    while d<=bins[k]:
        counts[k] += mweights(w1,w2,r1,r2)
        k=k-1
        if k<0: break


cdef inline radial_wbinning_2(np.float64_t* counts, np.float64_t* bins,\
                             np.float64_t d, np.int_t k,\
                             np.float64_t w1, np.float64_t w2,
                             np.float64_t r1, np.float64_t r2):
    """
    real space radial weighted binning function
    """
    
    while d<=bins[k]:
        counts[k] += sweights(w1,w2,r1,r2)
        k=k-1
        if k<0: break


cdef inline radial_wbinning_3(np.float64_t* counts, np.float64_t* bins,\
                             np.float64_t d, np.int_t k,\
                             np.float64_t w1, np.float64_t w2,
                             np.float64_t r1, np.float64_t r2):
    """
    real space radial weighted binning function
    """
    
    while d<=bins[k]:
        counts[k] += eqweights(w1,w2,r1,r2)
        k=k-1
        if k<0: break


cdef inline radial_wbinning_4(np.float64_t* counts, np.float64_t* bins,\
                             np.float64_t d, np.int_t k,\
                             np.float64_t w1, np.float64_t w2,
                             np.float64_t r1, np.float64_t r2):
    """
    real space radial weighted binning function
    """
    
    while d<=bins[k]:
        counts[k] += gweights(w1,w2,r1,r2)
        k=k-1
        if k<0: break


cdef inline radial_wbinning_5(np.float64_t* counts, np.float64_t* bins,\
                             np.float64_t d, np.int_t k,\
                             np.float64_t w1, np.float64_t w2,
                             np.float64_t r1, np.float64_t r2):
    """
    real space radial weighted binning function
    """
    
    while d<=bins[k]:
        counts[k] += lweights(w1,w2,r1,r2)
        k=k-1
        if k<0: break
    

cdef inline radial_wbinning_6(np.float64_t* counts, np.float64_t* bins,\
                             np.float64_t d, np.int_t k,\
                             np.float64_t w1, np.float64_t w2,
                             np.float64_t r1, np.float64_t r2):
    """
    real space radial weighted binning function
    """
    
    while d<=bins[k]:
        counts[k] += tgweights(w1,w2,r1,r2)
        k=k-1
        if k<0: break
    

cdef inline radial_wbinning_7(np.float64_t* counts, np.float64_t* bins,\
                             np.float64_t d, np.int_t k,\
                             np.float64_t w1, np.float64_t w2,
                             np.float64_t r1, np.float64_t r2):
    """
    real space radial weighted binning function
    """
    
    while d<=bins[k]:
        counts[k] += tlweights(w1,w2,r1,r2)
        k=k-1
        if k<0: break
    

cdef inline radial_wbinning_8(np.float64_t* counts, np.float64_t* bins,\
                             np.float64_t d, np.int_t k,\
                             np.float64_t w1, np.float64_t w2,
                             np.float64_t r1, np.float64_t r2):
    """
    real space radial weighted binning function
    """
    
    while d<=bins[k]:
        counts[k] += tweights(w1,w2,r1,r2)
        k=k-1
        if k<0: break


cdef inline radial_wbinning_9(np.float64_t* counts, np.float64_t* bins,\
                             np.float64_t d, np.int_t k,\
                             np.float64_t w1, np.float64_t w2,
                             np.float64_t r1, np.float64_t r2):
    """
    real space radial weighted binning function
    """
    
    while d<=bins[k]:
        counts[k] += exweights(w1,w2,r1,r2)
        k=k-1
        if k<0: break
    
    
