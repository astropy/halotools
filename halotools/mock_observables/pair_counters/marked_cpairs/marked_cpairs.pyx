# cython: profile=False

"""
objective pair counter.
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import sys
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport fabs, fmin

from .weighting_functions cimport *
from .custom_weighting_func cimport *
ctypedef double (*f_type)(np.float64_t* w1, np.float64_t* w2)

__author__ = ['Duncan Campbell']
__all__ = ['marked_npairs_no_pbc', 'marked_npairs_pbc']


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def marked_npairs_no_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                         np.ndarray[np.float64_t, ndim=1] y_icell1,
                         np.ndarray[np.float64_t, ndim=1] z_icell1,
                         np.ndarray[np.float64_t, ndim=1] x_icell2,
                         np.ndarray[np.float64_t, ndim=1] y_icell2,
                         np.ndarray[np.float64_t, ndim=1] z_icell2,
                         np.ndarray[np.float64_t, ndim=2] w_icell1,
                         np.ndarray[np.float64_t, ndim=2] w_icell2,
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
    cdef int n_weights1 = np.shape(w_icell1)[1]
    cdef int n_weights2 = np.shape(w_icell2)[1]
    cdef int nbins_minus_one = len(rbins) -1
    cdef np.ndarray[np.float64_t, ndim=1] counts = np.zeros((nbins,), dtype=np.float64)
    cdef double d
    cdef int i, j
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    cdef f_type wfunc
    
    cdef np.ndarray[np.float64_t, ndim=1] subw1 = np.zeros((n_weights1,), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] subw2 = np.zeros((n_weights2,), dtype=np.float64)
    
    #choose weighting function
    wfunc = return_weighting_function(weight_func_id)
    
    #loop over points in grid1's cell
    for i in range(0,Ni):
                
        #loop over points in grid2's cell
        for j in range(0,Nj):
                    
            #calculate the square distance
            d = square_distance(x_icell1[i],y_icell1[i],z_icell1[i],\
                                x_icell2[j],y_icell2[j],z_icell2[j])
            
            #get the weighting vectors for each point
            subw1 = w_icell1[i,:]
            subw2 = w_icell2[j,:]
            
            radial_wbinning(<np.float64_t*>counts.data,\
                            <np.float64_t*>rbins.data, d, nbins_minus_one,\
                            <np.float64_t*>subw1.data, <np.float64_t*>subw2.data,\
                            <f_type>wfunc)
    
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def marked_npairs_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                      np.ndarray[np.float64_t, ndim=1] y_icell1,
                      np.ndarray[np.float64_t, ndim=1] z_icell1,
                      np.ndarray[np.float64_t, ndim=1] x_icell2,
                      np.ndarray[np.float64_t, ndim=1] y_icell2,
                      np.ndarray[np.float64_t, ndim=1] z_icell2,
                      np.ndarray[np.float64_t, ndim=2] w_icell1,
                      np.ndarray[np.float64_t, ndim=2] w_icell2,
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
    cdef int n_weights1 = np.shape(w_icell1)[1]
    cdef int n_weights2 = np.shape(w_icell2)[1]
    cdef int nbins_minus_one = len(rbins) -1
    cdef np.ndarray[np.float64_t, ndim=1] counts = np.zeros((nbins,), dtype=np.float64)
    cdef double d
    cdef int i, j
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    cdef f_type wfunc
    
    cdef np.ndarray[np.float64_t, ndim=1] subw1 = np.zeros((n_weights1,), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] subw2 = np.zeros((n_weights2,), dtype=np.float64)
    
    #choose weighting function
    wfunc = return_weighting_function(weight_func_id)
    
    #loop over points in grid1's cell
    for i in range(0,Ni):
                
        #loop over points in grid2's cell
        for j in range(0,Nj):
                    
            #calculate the square distance
            d = periodic_square_distance(x_icell1[i],y_icell1[i],z_icell1[i],\
                                         x_icell2[j],y_icell2[j],z_icell2[j],\
                                         <np.float64_t*>period.data)
            
            #get the weighting vectors for each point
            subw1 = w_icell1[i,:]
            subw2 = w_icell2[j,:]
            
            radial_wbinning(<np.float64_t*>counts.data,\
                            <np.float64_t*>rbins.data, d, nbins_minus_one,\
                            <np.float64_t*>subw1.data, <np.float64_t*>subw2.data,\
                            <f_type>wfunc)
    
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


cdef inline radial_wbinning(np.float64_t* counts, np.float64_t* bins,\
                            np.float64_t d, np.int_t k,\
                            np.float64_t* w1, np.float64_t* w2, f_type wfunc):
    """
    real space radial weighted binning function
    """
    
    while d<=bins[k]:
        counts[k] += wfunc(w1, w2)
        k=k-1
        if k<0: break


cdef f_type return_weighting_function(weight_func_id):
    """
    returns a pointer to the user-specified weighting function
    """
    
    if weight_func_id==0:
        return custom_func
    elif weight_func_id==1:
        return mweights
    elif weight_func_id==2:
        return sweights
    elif weight_func_id==3:
        return eqweights
    elif weight_func_id==4:
        return gweights
    elif weight_func_id==5:
        return lweights
    elif weight_func_id==6:
        return tgweights
    elif weight_func_id==7:
        return tlweights
    elif weight_func_id==8:
        return tweights
    elif weight_func_id==9:
        return exweights
    else:
        raise ValueError('weighting function does not exist')


