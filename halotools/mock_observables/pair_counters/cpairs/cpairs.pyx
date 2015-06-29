# cython: profile=False

"""
optimized cython pair counters.  These are called by "rect_cuboid_pairs" module as the 
engine to actually calculate the pair-wise distances and do the binning.  These functions 
should be used with care as there are no 'checks' preformed to ensure the arguments are 
of the correct format.
"""

from __future__ import print_function, division
import sys
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport fabs, fmin
from distances cimport *

__all__ = ['npairs_no_pbc', 'npairs_pbc', 'wnpairs_no_pbc', 'wnpairs_pbc',\
           'jnpairs_no_pbc', 'jnpairs_pbc',\
           'xy_z_npairs_no_pbc', 'xy_z_npairs_pbc', 'xy_z_wnpairs_no_pbc', 'xy_z_wnpairs_pbc',\
           'xy_z_jnpairs_no_pbc', 'xy_z_jnpairs_pbc']
__author__=['Duncan Campbell']

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def npairs_no_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                  np.ndarray[np.float64_t, ndim=1] y_icell1,
                  np.ndarray[np.float64_t, ndim=1] z_icell1,
                  np.ndarray[np.float64_t, ndim=1] x_icell2,
                  np.ndarray[np.float64_t, ndim=1] y_icell2,
                  np.ndarray[np.float64_t, ndim=1] z_icell2,
                  np.ndarray[np.float64_t, ndim=1] rbins):
    """
    real-space pair counter without periodic boundary conditions (no PBCs).
    Calculate the number of pairs with separations less than or equal to rbins[i].
    """
    
    #c definitions
    cdef int nbins = len(rbins)
    cdef int nbins_minus_one = len(rbins) -1
    cdef np.ndarray[np.int_t, ndim=1] counts = np.zeros((nbins,), dtype=np.int)
    cdef double d
    cdef int i, j
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    
    #loop over points in grid1's cells
    for i in range(0,Ni):
        #loop over points in grid2's cells
        for j in range(0,Nj):
                        
            #calculate the square distance
            d = square_distance(x_icell1[i],y_icell1[i],z_icell1[i],\
                                x_icell2[j],y_icell2[j],z_icell2[j])
                        
            #calculate counts in bins
            radial_binning(<np.int_t*> counts.data,\
                           <np.float64_t*> rbins.data, d, nbins_minus_one)
        
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def npairs_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
               np.ndarray[np.float64_t, ndim=1] y_icell1,
               np.ndarray[np.float64_t, ndim=1] z_icell1,
               np.ndarray[np.float64_t, ndim=1] x_icell2,
               np.ndarray[np.float64_t, ndim=1] y_icell2,
               np.ndarray[np.float64_t, ndim=1] z_icell2,
               np.ndarray[np.float64_t, ndim=1] rbins,
               np.ndarray[np.float64_t, ndim=1] period):
    """
    real-space pair counter with periodic boundary conditions (PBCs).
    Calculate the number of pairs with separations less than or equal to rbins[i].
    """
    
    #c definitions
    cdef int nbins = len(rbins)
    cdef int nbins_minus_one = len(rbins) -1
    cdef np.ndarray[np.int_t, ndim=1] counts = np.zeros((nbins,), dtype=np.int)
    cdef double d
    cdef int i, j
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    
    #loop over points in grid1's cells
    for i in range(0,Ni):
        #loop over points in grid2's cells
        for j in range(0,Nj):
                        
            #calculate the square distance
            d = periodic_square_distance(x_icell1[i],y_icell1[i],z_icell1[i],\
                                         x_icell2[j],y_icell2[j],z_icell2[j],\
                                         <np.float64_t*> period.data)
                        
            #calculate counts in bins
            radial_binning(<np.int_t*> counts.data,\
                           <np.float64_t*> rbins.data, d, nbins_minus_one)
        
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def wnpairs_no_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                   np.ndarray[np.float64_t, ndim=1] y_icell1,
                   np.ndarray[np.float64_t, ndim=1] z_icell1,
                   np.ndarray[np.float64_t, ndim=1] x_icell2,
                   np.ndarray[np.float64_t, ndim=1] y_icell2,
                   np.ndarray[np.float64_t, ndim=1] z_icell2,
                   np.ndarray[np.float64_t, ndim=1] w_icell1,
                   np.ndarray[np.float64_t, ndim=1] w_icell2,
                   np.ndarray[np.float64_t, ndim=1] rbins):
    """
    weighted real-space pair counter without periodic boundary conditions (no PBCs)..
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
            radial_wbinning(<np.float64_t*>counts.data,\
                            <np.float64_t*>rbins.data, d, nbins_minus_one,\
                            w_icell1[i], w_icell2[j])
    
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def wnpairs_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                np.ndarray[np.float64_t, ndim=1] y_icell1,
                np.ndarray[np.float64_t, ndim=1] z_icell1,
                np.ndarray[np.float64_t, ndim=1] x_icell2,
                np.ndarray[np.float64_t, ndim=1] y_icell2,
                np.ndarray[np.float64_t, ndim=1] z_icell2,
                np.ndarray[np.float64_t, ndim=1] w_icell1,
                np.ndarray[np.float64_t, ndim=1] w_icell2,
                np.ndarray[np.float64_t, ndim=1] rbins,
                np.ndarray[np.float64_t, ndim=1] period):
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
            radial_wbinning(<np.float64_t*>counts.data,\
                            <np.float64_t*>rbins.data, d, nbins_minus_one,\
                            w_icell1[i], w_icell2[j])
    
    return counts

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def jnpairs_no_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                   np.ndarray[np.float64_t, ndim=1] y_icell1,
                   np.ndarray[np.float64_t, ndim=1] z_icell1,
                   np.ndarray[np.float64_t, ndim=1] x_icell2,
                   np.ndarray[np.float64_t, ndim=1] y_icell2,
                   np.ndarray[np.float64_t, ndim=1] z_icell2,
                   np.ndarray[np.float64_t, ndim=1] w_icell1,
                   np.ndarray[np.float64_t, ndim=1] w_icell2,
                   np.ndarray[np.int_t, ndim=1] j_icell1,
                   np.ndarray[np.int_t, ndim=1] j_icell2,
                   np.int_t N_samples,
                   np.ndarray[np.float64_t, ndim=1] rbins):
    """
    jackknife weighted real-space pair counter.
    Calculate the weighted number of pairs with separations less than or equal to rbins[i]
    for a jackknife sample.
    """
    
    #c definitions
    cdef int nbins = len(rbins)
    cdef int nbins_minus_one = len(rbins) -1
    cdef np.ndarray[np.float64_t, ndim=2] counts = np.zeros((N_samples,nbins), dtype=np.float64)
    cdef double d
    cdef int i, j
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    
    #loop over points in grid1's cell
    for i in range(0,Ni):
        #loop over points in grid2's cell
        for j in range(0,Nj):
                        
            #calculate the square distance
            d = square_distance(x_icell1[i],y_icell1[i],z_icell1[i],\
                                x_icell2[j],y_icell2[j],z_icell2[j])
                        
            #calculate counts in bins
            radial_jbinning(<np.float64_t*>counts.data, <np.float64_t*>rbins.data,\
                            d, nbins_minus_one, N_samples,\
                            w_icell1[i], w_icell2[j],\
                            j_icell1[i], j_icell2[j])
        
    return counts

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def jnpairs_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                np.ndarray[np.float64_t, ndim=1] y_icell1,
                np.ndarray[np.float64_t, ndim=1] z_icell1,
                np.ndarray[np.float64_t, ndim=1] x_icell2,
                np.ndarray[np.float64_t, ndim=1] y_icell2,
                np.ndarray[np.float64_t, ndim=1] z_icell2,
                np.ndarray[np.float64_t, ndim=1] w_icell1,
                np.ndarray[np.float64_t, ndim=1] w_icell2,
                np.ndarray[np.int_t, ndim=1] j_icell1,
                np.ndarray[np.int_t, ndim=1] j_icell2,
                np.int_t N_samples,
                np.ndarray[np.float64_t, ndim=1] rbins,
                np.ndarray[np.float64_t, ndim=1] period):
    """
    jackknife weighted real-space pair counter.
    Calculate the weighted number of pairs with separations less than or equal to rbins[i]
    for a jackknife sample.
    """
    
    #c definitions
    cdef int nbins = len(rbins)
    cdef int nbins_minus_one = len(rbins) -1
    cdef np.ndarray[np.float64_t, ndim=2] counts =\
        np.zeros((N_samples, nbins), dtype=np.float64)
    cdef double d
    cdef int i, j
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    
    #loop over points in grid1's cell
    for i in range(0,Ni):
        #loop over points in grid2's cell
        for j in range(0,Nj):
            
            #calculate the square distance
            d = periodic_square_distance(x_icell1[i],y_icell1[i],z_icell1[i],\
                                         x_icell2[j],y_icell2[j],z_icell2[j],\
                                         <np.float64_t*>period.data)
            
            #calculate counts in bins
            radial_jbinning(<np.float64_t*>counts.data, <np.float64_t*>rbins.data,\
                            d, nbins_minus_one, N_samples,\
                            w_icell1[i], w_icell2[j],\
                            j_icell1[i], j_icell2[j])

    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def xy_z_npairs_no_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                       np.ndarray[np.float64_t, ndim=1] y_icell1,
                       np.ndarray[np.float64_t, ndim=1] z_icell1,
                       np.ndarray[np.float64_t, ndim=1] x_icell2,
                       np.ndarray[np.float64_t, ndim=1] y_icell2,
                       np.ndarray[np.float64_t, ndim=1] z_icell2,
                       np.ndarray[np.float64_t, ndim=1] rp_bins,
                       np.ndarray[np.float64_t, ndim=1] pi_bins):
    """
    2+1D pair counter without periodic boundary conditions (no PBCs).
    Calculate the number of pairs with separations in the x-y plane less than or equal 
    to rp_bins[i], and separations in the z coordinate less than or equal to pi_bins[i].
    """
    
    #c definitions
    cdef int nrp_bins = len(rp_bins)
    cdef int npi_bins = len(pi_bins)
    cdef int nrp_bins_minus_one = len(rp_bins) -1
    cdef int npi_bins_minus_one = len(pi_bins) -1
    cdef np.ndarray[np.int_t, ndim=2] counts =\
        np.zeros((nrp_bins, npi_bins), dtype=np.int)
    cdef double d_perp, d_para
    cdef int i, j
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    
    #loop over points in grid1's cell
    for i in range(0,Ni):
                
        #loop over points in grid2's cell
        for j in range(0,Nj):
                    
            #calculate the square distance
            d_perp = perp_square_distance(x_icell1[i], y_icell1[i],\
                                          x_icell2[j], y_icell2[j])
            d_para = para_square_distance(z_icell1[i], z_icell2[j])
                        
            #calculate counts in bins
            xy_z_binning(<np.int_t*>counts.data,\
                         <np.float64_t*>rp_bins.data,\
                         <np.float64_t*>pi_bins.data,\
                         d_perp, d_para, nrp_bins_minus_one, npi_bins_minus_one)
        
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def xy_z_npairs_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                    np.ndarray[np.float64_t, ndim=1] y_icell1,
                    np.ndarray[np.float64_t, ndim=1] z_icell1,
                    np.ndarray[np.float64_t, ndim=1] x_icell2,
                    np.ndarray[np.float64_t, ndim=1] y_icell2,
                    np.ndarray[np.float64_t, ndim=1] z_icell2,
                    np.ndarray[np.float64_t, ndim=1] rp_bins,
                    np.ndarray[np.float64_t, ndim=1] pi_bins,
                    np.ndarray[np.float64_t, ndim=1] period):
    """
    2+1D pair counter without periodic boundary conditions (no PBCs).
    Calculate the number of pairs with separations in the x-y plane less than or equal 
    to rp_bins[i], and separations in the z coordinate less than or equal to pi_bins[i].
    """
    
    #c definitions
    cdef int nrp_bins = len(rp_bins)
    cdef int npi_bins = len(pi_bins)
    cdef int nrp_bins_minus_one = len(rp_bins) -1
    cdef int npi_bins_minus_one = len(pi_bins) -1
    cdef np.ndarray[np.int_t, ndim=2] counts =\
        np.zeros((nrp_bins, npi_bins), dtype=np.int)
    cdef double d_perp, d_para
    cdef int i, j
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    
    #loop over points in grid1's cell
    for i in range(0,Ni):
                
        #loop over points in grid2's cell
        for j in range(0,Nj):
                    
            #calculate the square distance
            d_perp = periodic_perp_square_distance(x_icell1[i],y_icell1[i],\
                                                   x_icell2[j],y_icell2[j],\
                                                   <np.float64_t*>period.data)
            d_para = periodic_para_square_distance(z_icell1[i],\
                                                   z_icell2[j],\
                                                   <np.float64_t*>period.data)
                        
            #calculate counts in bins
            xy_z_binning(<np.int_t*>counts.data,\
                         <np.float64_t*>rp_bins.data,\
                         <np.float64_t*>pi_bins.data,\
                         d_perp, d_para, nrp_bins_minus_one, npi_bins_minus_one)
        
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def xy_z_wnpairs_no_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                        np.ndarray[np.float64_t, ndim=1] y_icell1,
                        np.ndarray[np.float64_t, ndim=1] z_icell1,
                        np.ndarray[np.float64_t, ndim=1] x_icell2,
                        np.ndarray[np.float64_t, ndim=1] y_icell2,
                        np.ndarray[np.float64_t, ndim=1] z_icell2,
                        np.ndarray[np.float64_t, ndim=1] w_icell1,
                        np.ndarray[np.float64_t, ndim=1] w_icell2,
                        np.ndarray[np.float64_t, ndim=1] rp_bins,
                        np.ndarray[np.float64_t, ndim=1] pi_bins):
    """
    2+1D pair counter without periodic boundary conditions (no PBCs).
    Calculate the number of pairs with separations in the x-y plane less than or equal 
    to rp_bins[i], and separations in the z coordinate less than or equal to pi_bins[i].
    """
    
    #c definitions
    cdef int nrp_bins = len(rp_bins)
    cdef int npi_bins = len(pi_bins)
    cdef int nrp_bins_minus_one = len(rp_bins) -1
    cdef int npi_bins_minus_one = len(pi_bins) -1
    cdef np.ndarray[np.float64_t, ndim=2] counts =\
        np.zeros((nrp_bins, npi_bins), dtype=np.float64)
    cdef double d_perp, d_para
    cdef int i, j
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    
    #loop over points in grid1's cell
    for i in range(0,Ni):
                
        #loop over points in grid2's cell
        for j in range(0,Nj):
                    
            #calculate the square distance
            d_perp = perp_square_distance(x_icell1[i], y_icell1[i],\
                                          x_icell2[j], y_icell2[j])
            d_para = para_square_distance(z_icell1[i], z_icell2[j])
                        
            #calculate counts in bins
            xy_z_wbinning(<np.float64_t*>counts.data,\
                          <np.float64_t*>rp_bins.data,\
                          <np.float64_t*>pi_bins.data,\
                          d_perp, d_para, nrp_bins_minus_one, npi_bins_minus_one,\
                          w_icell1[i], w_icell2[j])
        
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def xy_z_wnpairs_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                     np.ndarray[np.float64_t, ndim=1] y_icell1,
                     np.ndarray[np.float64_t, ndim=1] z_icell1,
                     np.ndarray[np.float64_t, ndim=1] x_icell2,
                     np.ndarray[np.float64_t, ndim=1] y_icell2,
                     np.ndarray[np.float64_t, ndim=1] z_icell2,
                     np.ndarray[np.float64_t, ndim=1] w_icell1,
                     np.ndarray[np.float64_t, ndim=1] w_icell2,
                     np.ndarray[np.float64_t, ndim=1] rp_bins,
                     np.ndarray[np.float64_t, ndim=1] pi_bins,
                     np.ndarray[np.float64_t, ndim=1] period):
    """
    2+1D pair counter without periodic boundary conditions (no PBCs).
    Calculate the number of pairs with separations in the x-y plane less than or equal 
    to rp_bins[i], and separations in the z coordinate less than or equal to pi_bins[i].
    """
    
    #c definitions
    cdef int nrp_bins = len(rp_bins)
    cdef int npi_bins = len(pi_bins)
    cdef int nrp_bins_minus_one = len(rp_bins) -1
    cdef int npi_bins_minus_one = len(pi_bins) -1
    cdef np.ndarray[np.float64_t, ndim=2] counts =\
        np.zeros((nrp_bins, npi_bins), dtype=np.float64)
    cdef double d_perp, d_para
    cdef int i, j
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    
    #loop over points in grid1's cell
    for i in range(0,Ni):
                
        #loop over points in grid2's cell
        for j in range(0,Nj):
                    
            #calculate the square distance
            d_perp = periodic_perp_square_distance(x_icell1[i],y_icell1[i],\
                                                   x_icell2[j],y_icell2[j],\
                                                   <np.float64_t*>period.data)
            d_para = periodic_para_square_distance(z_icell1[i],\
                                                   z_icell2[j],\
                                                   <np.float64_t*>period.data)
                        
            #calculate counts in bins
            xy_z_wbinning(<np.float64_t*>counts.data,\
                          <np.float64_t*>rp_bins.data,\
                          <np.float64_t*>pi_bins.data,\
                          d_perp, d_para, nrp_bins_minus_one, npi_bins_minus_one,
                          w_icell1[i], w_icell2[j])
        
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def xy_z_jnpairs_no_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                        np.ndarray[np.float64_t, ndim=1] y_icell1,
                        np.ndarray[np.float64_t, ndim=1] z_icell1,
                        np.ndarray[np.float64_t, ndim=1] x_icell2,
                        np.ndarray[np.float64_t, ndim=1] y_icell2,
                        np.ndarray[np.float64_t, ndim=1] z_icell2,
                        np.ndarray[np.float64_t, ndim=1] w_icell1,
                        np.ndarray[np.float64_t, ndim=1] w_icell2,
                        np.ndarray[np.int_t, ndim=1] j_icell1,
                        np.ndarray[np.int_t, ndim=1] j_icell2,
                        np.int_t N_samples,
                        np.ndarray[np.float64_t, ndim=1] rp_bins,
                        np.ndarray[np.float64_t, ndim=1] pi_bins):
    """
    2+1D pair counter without periodic boundary conditions (no PBCs).
    Calculate the number of pairs with separations in the x-y plane less than or equal 
    to rp_bins[i], and separations in the z coordinate less than or equal to pi_bins[i].
    """
    
    #c definitions
    cdef int nrp_bins = len(rp_bins)
    cdef int npi_bins = len(pi_bins)
    cdef int nrp_bins_minus_one = len(rp_bins) -1
    cdef int npi_bins_minus_one = len(pi_bins) -1
    cdef np.ndarray[np.float64_t, ndim=3] counts =\
        np.zeros((N_samples, nrp_bins, npi_bins), dtype=np.float64)
    cdef double d_perp, d_para
    cdef int i, j
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    
    #loop over points in grid1's cell
    for i in range(0,Ni):
                
        #loop over points in grid2's cell
        for j in range(0,Nj):
                    
            #calculate the square distance
            d_perp = perp_square_distance(x_icell1[i], y_icell1[i],\
                                          x_icell2[j], y_icell2[j])
            d_para = para_square_distance(z_icell1[i], z_icell2[j])
                        
            #calculate counts in bins
            xy_z_jbinning(<np.float64_t*>counts.data,\
                          <np.float64_t*>rp_bins.data,\
                          <np.float64_t*>pi_bins.data,\
                          d_perp, d_para,\
                          nrp_bins_minus_one, npi_bins_minus_one, N_samples,\
                          w_icell1[i], w_icell2[j], j_icell1[i], j_icell2[j])
        
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def xy_z_jnpairs_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                     np.ndarray[np.float64_t, ndim=1] y_icell1,
                     np.ndarray[np.float64_t, ndim=1] z_icell1,
                     np.ndarray[np.float64_t, ndim=1] x_icell2,
                     np.ndarray[np.float64_t, ndim=1] y_icell2,
                     np.ndarray[np.float64_t, ndim=1] z_icell2,
                     np.ndarray[np.float64_t, ndim=1] w_icell1,
                     np.ndarray[np.float64_t, ndim=1] w_icell2,
                     np.ndarray[np.int_t, ndim=1] j_icell1,
                     np.ndarray[np.int_t, ndim=1] j_icell2,
                     np.int_t N_samples,
                     np.ndarray[np.float64_t, ndim=1] rp_bins,
                     np.ndarray[np.float64_t, ndim=1] pi_bins,
                     np.ndarray[np.float64_t, ndim=1] period):
    """
    2+1D pair counter without periodic boundary conditions (no PBCs).
    Calculate the number of pairs with separations in the x-y plane less than or equal 
    to rp_bins[i], and separations in the z coordinate less than or equal to pi_bins[i].
    """
    
    #c definitions
    cdef int nrp_bins = len(rp_bins)
    cdef int npi_bins = len(pi_bins)
    cdef int nrp_bins_minus_one = len(rp_bins) -1
    cdef int npi_bins_minus_one = len(pi_bins) -1
    cdef np.ndarray[np.float64_t, ndim=3] counts =\
        np.zeros((N_samples, nrp_bins, npi_bins), dtype=np.float64)
    cdef double d_perp, d_para
    cdef int i, j
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    
    #loop over points in grid1's cell
    for i in range(0,Ni):
                
        #loop over points in grid2's cell
        for j in range(0,Nj):
                    
            #calculate the square distance
            d_perp = periodic_perp_square_distance(x_icell1[i],y_icell1[i],\
                                                   x_icell2[j],y_icell2[j],\
                                                   <np.float64_t*>period.data)
            d_para = periodic_para_square_distance(z_icell1[i],\
                                                   z_icell2[j],\
                                                   <np.float64_t*>period.data)
                        
            #calculate counts in bins
            xy_z_jbinning(<np.float64_t*>counts.data,\
                          <np.float64_t*>rp_bins.data,\
                          <np.float64_t*>pi_bins.data,\
                          d_perp, d_para,\
                          nrp_bins_minus_one, npi_bins_minus_one, N_samples,\
                          w_icell1[i], w_icell2[j], j_icell1[i], j_icell2[j])
        
    return counts


cdef inline radial_binning(np.int_t* counts, np.float64_t* bins,\
                           np.float64_t d, np.int_t k):
    """
    real space radial binning function
    """
    
    while d<=bins[k]:
        counts[k] += 1
        k=k-1
        if k<0: break


cdef inline radial_wbinning(np.float64_t* counts, np.float64_t* bins,\
                            np.float64_t d, np.int_t k,\
                            np.float64_t w1, np.float64_t w2):
    """
    real space radial weighted binning function
    """
    
    while d<=bins[k]:
        counts[k] += w1*w2
        k=k-1
        if k<0: break


cdef inline radial_jbinning(np.float64_t* counts, np.float64_t* bins,\
                            np.float64_t d,\
                            np.int_t nbins_minus_one,\
                            np.int_t N_samples,\
                            np.float64_t w1, np.float64_t w2,\
                            np.int_t j1, np.int_t j2):
    """
    real space radial jackknife binning function
    """
    cdef int k, l
    cdef int max_l = nbins_minus_one+1
    
    for l in range(0,N_samples):
        k = nbins_minus_one
        while d<=bins[k]:
            #counts[l,k] += jweight(l, j1, j2, w1, w2)
            counts[l*max_l+k] += jweight(l, j1, j2, w1, w2)
            k=k-1
            if k<0: break


cdef inline xy_z_binning(np.int_t* counts, np.float64_t* rp_bins,\
                         np.float64_t* pi_bins, np.float64_t d_perp,\
                         np.float64_t d_para, np.int_t k,\
                         np.int_t npi_bins_minus_one):
    """
    2D+1 binning function
    """
    cdef int g
    cdef int max_k = npi_bins_minus_one+1
    
    while d_perp<=rp_bins[k]:
        g = npi_bins_minus_one
        while d_para<=pi_bins[g]:
            #counts[k,g] += 1
            counts[k*max_k+g] += 1
            g=g-1
            if g<0: break
        k=k-1
        if k<0: break


cdef inline xy_z_wbinning(np.float64_t* counts, np.float64_t* rp_bins,\
                          np.float64_t* pi_bins, np.float64_t d_perp,\
                          np.float64_t d_para, np.int_t k,\
                          np.int_t npi_bins_minus_one, np.float64_t w1, np.float64_t w2):
    """
    2D+1 weighted binning function
    """
    cdef int g
    cdef int max_k = npi_bins_minus_one+1
    
    while d_perp<=rp_bins[k]:
        g = npi_bins_minus_one
        while d_para<=pi_bins[g]:
            #counts[k,g] += w1*w2
            counts[k*max_k+g] += w1*w2
            g=g-1
            if g<0: break
        k=k-1
        if k<0: break


cdef inline xy_z_jbinning(np.float64_t* counts, np.float64_t* rp_bins,\
                          np.float64_t* pi_bins, np.float64_t d_perp,\
                          np.float64_t d_para,\
                          np.int_t nrp_bins_minus_one,\
                          np.int_t npi_bins_minus_one,\
                          np.int_t N_samples,\
                          np.float64_t w1, np.float64_t w2,\
                          np.int_t j1, np.int_t j2):
    """
    2D+1 jackknife binning function
    """
    cdef int l, k, g
    cdef int max_l = nrp_bins_minus_one+1
    cdef int max_k = npi_bins_minus_one+1
    
    for l in range(0,N_samples): #loop over jackknife samples
            k = nrp_bins_minus_one
            while d_perp<=rp_bins[k]: #loop over rp bins
                g = npi_bins_minus_one
                while d_para<=pi_bins[g]: #loop over pi bins
                    #counts[l,k,g] += jweight(l, j1, j2, w1, w2)
                    counts[l*max_l*max_k+k*max_k+g] += jweight(l, j1, j2, w1, w2)
                    g=g-1
                    if g<0: break
                k=k-1
                if k<0: break


cdef inline double jweight(np.int_t j, np.int_t j1, np.int_t j2,\
                           np.float64_t w1, np.float64_t w2):
    """
    return jackknife weighted counts
    
    parameters
    ----------
    j: jackknife subsample
    j1: jackknife sample 1 tag
    j2: jackknife sample 2 tag
    w1: weight1
    w2: weight2
    
    notes
    -----
    if sample j==0, do no jackknife weighting.  i.e. reserve this for the full sample.
    if both points are inside the sample, return w1*w2
    if both points are outside the sample, return 0.0
    if one point is within and one point is outside the sample, return 0.5*w1*w2
    """
    
    if j==0: return (w1 * w2)
    # both outside the sub-sample
    elif (j1 == j2) & (j1 == j): return 0.0
    # both inside the sub-sample
    elif (j1 != j) & (j2 != j): return (w1 * w2)
    # only one inside the sub-sample
    elif (j1 != j2) & ((j1 == j) | (j2 == j)): return 0.5*(w1 * w2)


