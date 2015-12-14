# cython: profile=False

"""
brute force pair counters returing the binned pair counts per point.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import sys
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport fabs, fmin, sqrt

from .distances cimport *

__all__ = ['per_object_npairs_no_pbc']
__author__=['Duncan Campbell']

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def per_object_npairs_no_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                             np.ndarray[np.float64_t, ndim=1] y_icell1,
                             np.ndarray[np.float64_t, ndim=1] z_icell1,
                             np.ndarray[np.float64_t, ndim=1] x_icell2,
                             np.ndarray[np.float64_t, ndim=1] y_icell2,
                             np.ndarray[np.float64_t, ndim=1] z_icell2,
                             np.ndarray[np.float64_t, ndim=1] rbins):
    """
    Calculate the number of pairs with seperations greater than or equal to r per object, :math:`N_i(>r)`.
    
    This can be used for pair coutning with PBCs if the pointes are pre-shifted to 
    account for the PBC.
    
    Parameters
    ----------
    x_icell1 : numpy.array
         array of x positions of lenght N1 (data1)
    
    y_icell1 : numpy.array
         array of y positions of lenght N1 (data1)
    
    z_icell1 : numpy.array
         array of z positions of lenght N1 (data1)
    
    x_icell2 : numpy.array
         array of x positions of lenght N2 (data2)
    
    y_icell2 : numpy.array
         array of y positions of lenght N2 (data2)
    
    z_icell2 : numpy.array
         array of z positions of lenght N2 (data2)
    
    rbins : numpy.array
         array defining radial bins in which to sum the pair counts
    
    Returns
    -------
    result :  numpy.ndarray
        2-D array of pair counts per object in data1 in radial bins defined by ``rbins``.
    
    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a 
    unit cube. 
    
    >>> Npts = 1000
    >>> Lbox = 1.0
    >>> period = np.array([Lbox,Lbox,Lbox])
    
    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)
    
    Count the number of pairs that can be formed amongst these random points:
    
    >>> rbins = np.linspace(0,0.5,10)
    >>> counts = per_object_npairs_no_pbc(x,y,z,x,y,z,rbins)
    """
    
    #c definitions
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    cdef int nbins = len(rbins)
    cdef int nbins_minus_one = len(rbins) -1
    cdef np.ndarray[np.int_t, ndim=1] inner_counts = np.zeros((nbins,), dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=2] outer_counts = np.zeros((Ni, nbins), dtype=np.int)
    cdef double d
    cdef int i, j, k
    
    #square the distance bins to avoid taking a square root in a tight loop
    rbins = rbins**2
    
    #loop over points in grid1's cells
    for i in range(0,Ni):
        
        #loop over points in grid2's cells
        for j in range(0,Nj):
            
            #calculate the square distance
            d = square_distance(x_icell1[i],y_icell1[i],z_icell1[i],\
                                x_icell2[j],y_icell2[j],z_icell2[j])
                        
            #calculate counts in bins
            radial_binning(<np.int_t*> inner_counts.data,\
                           <np.float64_t*> rbins.data, d,\
                           nbins_minus_one)
        
        #update the outer counts
        for k in range(0, nbins):
             outer_counts[i, k] = inner_counts[k]
             inner_counts[k] = 0 #re-zero the inner counts
    
    return outer_counts


cdef inline radial_binning(np.int_t* counts, np.float64_t* bins,\
                           np.float64_t d, np.int_t k):
    """
    real space radial binning function
    """
    
    while d<=bins[k]:
        counts[k] += 1
        k=k-1
        if k<0: break


