# cython: profile=False

"""
brute force pair counters
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import sys
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport fabs, fmin, sqrt
from .distances cimport *

__all__ = ['npairs_no_pbc',\
           'npairs_pbc',\
           'wnpairs_no_pbc',\
           'wnpairs_pbc',\
           'jnpairs_no_pbc',\
           'jnpairs_pbc',\
           'xy_z_npairs_no_pbc',\
           'xy_z_npairs_pbc',\
           'xy_z_wnpairs_no_pbc',\
           'xy_z_wnpairs_pbc',\
           'xy_z_jnpairs_no_pbc',\
           'xy_z_jnpairs_pbc',\
           's_mu_npairs_no_pbc',\
           's_mu_npairs_pbc']

__author__=['Duncan Campbell']


###########################
####   pair counters   ####
###########################

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
    Calculate the number of pairs with seperations greater than or equal to r, :math:`N(>r)`.
    
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
    result :  numpy.array
        array of pair counts in radial bins defined by ``rbins``.
    
    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a 
    unit cube. 
    
    >>> Npts = 1000
    
    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)
    
    Count the number of pairs that can be formed amongst these random points:
    
    >>> rbins = np.linspace(0,0.5,10)
    >>> counts = npairs_no_pbc(x,y,z,x,y,z,rbins)
    """
    
    #c definitions
    cdef int nbins = len(rbins)
    cdef int nbins_minus_one = len(rbins) -1
    cdef np.ndarray[np.int_t, ndim=1] counts = np.zeros((nbins,), dtype=np.int)
    cdef double d
    cdef int i, j
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    
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
    Calculate the number of pairs with seperations greater than or equal to r, :math:`N(>r)`, with periodic boundary conditions (PBC).
    
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
    
    period : numpy.array
        array defining axis-aligned periodic boundary conditions.
    
    Returns
    -------
    result :  numpy.array
        array of pair counts in radial bins defined by ``rbins``.
    
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
    
    Count the number of weighted pairs that can be formed amongst these random points:
    
    >>> rbins = np.linspace(0,0.5,10)
    >>> counts = npairs_pbc(x,y,z,x,y,z,rbins,period)
    """
    
    #c definitions
    cdef int nbins = len(rbins)
    cdef int nbins_minus_one = len(rbins) -1
    cdef np.ndarray[np.int_t, ndim=1] counts = np.zeros((nbins,), dtype=np.int)
    cdef double d
    cdef int i, j
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    
    #square the distance bins to avoid taking a square root in a tight loop
    rbins = rbins**2
    
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
    Calculate the weighted number of pairs with seperations greater than or equal to r, :math:`W(>r)`.
    
    :math:`W(>r)` is incremented by :math:`w_1 \\times w_2` if two pints have seperations 
    greater than or equal to r, where :math:`w_1` and :math:`w_2` the associated wieghts.
    
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
    
    w_icell1 : numpy.array
        array of weight floats of length N1
    
    w_icell1 : numpy.array
        array of weight floats of length N2
    
    rbins : numpy.array
         array defining radial bins in which to sum the pair counts
    
    Returns
    -------
    result :  numpy.array
        array of weighted pair counts in radial bins defined by ``rbins``.
    
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
    
    Assign random weights:
    
    >>> weights = np.random.random(Npts)
    
    Count the number of weighted pairs that can be formed amongst these random points:
    
    >>> rbins = np.linspace(0,0.5,10)
    >>> counts = wnpairs_no_pbc(x,y,z,x,y,z,weights,weights,rbins)
    """
    
    #c definitions
    cdef int nbins = len(rbins)
    cdef int nbins_minus_one = len(rbins) -1
    cdef np.ndarray[np.float64_t, ndim=1] counts = np.zeros((nbins,), dtype=np.float64)
    cdef double d
    cdef int i, j
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    
    #square the distance bins to avoid taking a square root in a tight loop
    rbins = rbins**2
    
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
    Calculate the weighted number of pairs with seperations greater than or equal to r, :math:`W(>r)`, with periodic boundary conditions (PBC).
    
    :math:`W(>r)` is incremented by :math:`w_1 \\times w_2` if two pints have seperations 
    greater than or equal to r, where :math:`w_1` and :math:`w_2` the associated wieghts.
    
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
    
    w_icell1 : numpy.array
        array of weight floats of length N1
    
    w_icell1 : numpy.array
        array of weight floats of length N2
    
    rbins : numpy.array
         array defining radial bins in which to sum the pair counts
    
    Returns
    -------
    result :  numpy.array
        array of weighted pair counts in radial bins defined by ``rbins``.
    
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
    
    Assign random weights:
    
    >>> weights = np.random.random(Npts)
    
    Count the number of weighted pairs that can be formed amongst these random points:
    
    >>> rbins = np.linspace(0,0.5,10)
    >>> counts = wnpairs_no_pbc(x,y,z,x,y,z,weights,weights,rbins)
    """
    
    #c definitions
    cdef int nbins = len(rbins)
    cdef int nbins_minus_one = len(rbins) -1
    cdef np.ndarray[np.float64_t, ndim=1] counts = np.zeros((nbins,), dtype=np.float64)
    cdef double d
    cdef int i, j
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    
    #square the distance bins to avoid taking a square root in a tight loop
    rbins = rbins**2
    
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
    Calculate the jackknife weighted number of pairs with seperations greater than or equal to r, :math:`N(>r)`.
    
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
    
    w_icell1 : numpy.array
        array of weight floats of length N1
    
    w_icell2 : numpy.array
        array of weight floats of length N2
    
    j_icell1 : numpy.array
        array of integer subsample labels of length N1
        
    j_icell2 : numpy.array
        array of integer subsample labels of length N2
        
    N_samples : int
        total number of subsamples
    
    rbins : numpy.array
         array defining radial bins in which to sum the pair counts
    
    Returns
    -------
    result :  numpy.ndarray
        2-D array of pair counts of length ``N_samples`` in radial bins defined by ``rbins``.
    """
    
    #c definitions
    cdef int nbins = len(rbins)
    cdef int nbins_minus_one = len(rbins) -1
    cdef np.ndarray[np.float64_t, ndim=2] counts = np.zeros((N_samples,nbins), dtype=np.float64)
    cdef double d
    cdef int i, j
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    
    #square the distance bins to avoid taking a square root in a tight loop
    rbins = rbins**2
    
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
    Calculate the jackknife weighted number of pairs with seperations greater than or equal to r, :math:`N(>r)`, with periodic boundary conditions (PBC).
    
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
    
    w_icell1 : numpy.array
        array of weight floats of length N1
    
    w_icell2 : numpy.array
        array of weight floats of length N2
    
    j_icell1 : numpy.array
        array of integer subsample labels of length N1
        
    j_icell2 : numpy.array
        array of integer subsample labels of length N2
        
    N_samples : int
        total number of subsamples
    
    rbins : numpy.array
         array defining radial bins in which to sum the pair counts
    
    period : numpy.array
        array defining axis-aligned periodic boundary conditions.
    
    Returns
    -------
    result :  numpy.ndarray
        2-D array of pair counts of length ``N_samples`` in radial bins defined by ``rbins``.
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
    
    #square the distance bins to avoid taking a square root in a tight loop
    rbins = rbins**2
    
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
    Calculate the number of pairs with seperations greater than or equal to :math:`r_{\\perp}` and :math:`r_{\\parallel}`, :math:`N(>r_{\\perp},>r_{\\parallel})`.
    
    :math:`r_{\\perp}` and :math:`r_{\\parallel}` are defined wrt the z-direction.
    
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
    
    rp_bins : numpy.array
        array defining projected seperation in which to sum the pair counts
    
    pi_bins : numpy.array
        array defining parallel seperation in which to sum the pair counts
    
    Returns
    -------
    result : numpy.ndarray
        2-D array of pair counts of bins defined by ``rp_bins`` and ``pi_bins``.
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
    
    #square the distance bins to avoid taking a square root in a tight loop
    rp_bins = rp_bins**2
    pi_bins = pi_bins**2
    
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
    Calculate the number of pairs with seperations greater than or equal to :math:`r_{\\perp}` and :math:`r_{\\parallel}`, :math:`N(>r_{\\perp},>r_{\\parallel})`, with periodic boundary conditions (PBC).
    
    :math:`r_{\\perp}` and :math:`r_{\\parallel}` are defined wrt the z-direction.
    
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
    
    rp_bins : numpy.array
        array defining projected seperation in which to sum the pair counts
    
    pi_bins : numpy.array
        array defining parallel seperation in which to sum the pair counts
    
    period : numpy.array
        array defining axis-aligned periodic boundary conditions.
    
    Returns
    -------
    result : numpy.ndarray
        2-D array of pair counts of bins defined by ``rp_bins`` and ``pi_bins``.
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
    
    #square the distance bins to avoid taking a square root in a tight loop
    rp_bins = rp_bins**2
    pi_bins = pi_bins**2
    
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
    Calculate the weighted number of pairs with seperations greater than or equal to :math:`r_{\\perp}` and :math:`r_{\\parallel}`, :math:`N(>r_{\\perp},>r_{\\parallel})`.
    
    :math:`r_{\\perp}` and :math:`r_{\\parallel}` are defined wrt the z-direction.
    
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
    
    w_icell1 : numpy.array
        array of weight floats of length N1
    
    w_icell2 : numpy.array
        array of weight floats of length N2
    
    rp_bins : numpy.array
        array defining projected seperation in which to sum the pair counts
    
    pi_bins : numpy.array
        array defining parallel seperation in which to sum the pair counts
    
    Returns
    -------
    result : numpy.ndarray
        2-D array of weighted pair counts of bins defined by ``rp_bins`` and ``pi_bins``.
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
    
    #square the distance bins to avoid taking a square root in a tight loop
    rp_bins = rp_bins**2
    pi_bins = pi_bins**2
    
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
    Calculate the weighted number of pairs with seperations greater than or equal to :math:`r_{\\perp}` and :math:`r_{\\parallel}`, :math:`N(>r_{\\perp},>r_{\\parallel})`, with periodic boundary conditions (PBC).
    
    :math:`r_{\\perp}` and :math:`r_{\\parallel}` are defined wrt the z-direction.
    
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
    
    w_icell1 : numpy.array
        array of weight floats of length N1
    
    w_icell2 : numpy.array
        array of weight floats of length N2
    
    rp_bins : numpy.array
        array defining projected seperation in which to sum the pair counts
    
    pi_bins : numpy.array
        array defining parallel seperation in which to sum the pair counts
    
    period : numpy.array
        array defining axis-aligned periodic boundary conditions.
    
    Returns
    -------
    result : numpy.ndarray
        2-D array of weighted pair counts of bins defined by ``rp_bins`` and ``pi_bins``.
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
    
    #square the distance bins to avoid taking a square root in a tight loop
    rp_bins = rp_bins**2
    pi_bins = pi_bins**2
    
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
    Calculate the jackknife weighted number of pairs with seperations greater than or equal to :math:`r_{\\perp}` and :math:`r_{\\parallel}`, :math:`N(>r_{\\perp},>r_{\\parallel})`.
    
    :math:`r_{\\perp}` and :math:`r_{\\parallel}` are defined wrt the z-direction.
    
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
    
    w_icell1 : numpy.array
        array of weight floats of length N1
    
    w_icell2 : numpy.array
        array of weight floats of length N2
    
    j_icell1 : numpy.array
        array of integer subsample labels of length N1
        
    j_icell2 : numpy.array
        array of integer subsample labels of length N2
        
    N_samples : int
        total number of subsamples
    
    rp_bins : numpy.array
        array defining projected seperation in which to sum the pair counts
    
    pi_bins : numpy.array
        array defining parallel seperation in which to sum the pair counts
    
    Returns
    -------
    result : numpy.ndarray
        3-D array of weighted jackknife pair counts in bins defined by ``rp_bins`` and ``pi_bins``.
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
    
    #square the distance bins to avoid taking a square root in a tight loop
    rp_bins = rp_bins**2
    pi_bins = pi_bins**2
    
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
    Calculate the jackknife weighted number of pairs with seperations greater than or equal to :math:`r_{\\perp}` and :math:`r_{\\parallel}`, :math:`N(>r_{\\perp},>r_{\\parallel})`, with periodic boundary conditions (PBC).
    
    :math:`r_{\\perp}` and :math:`r_{\\parallel}` are defined wrt the z-direction.
    
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
    
    w_icell1 : numpy.array
        array of weight floats of length N1
    
    w_icell2 : numpy.array
        array of weight floats of length N2
    
    j_icell1 : numpy.array
        array of integer subsample labels of length N1
        
    j_icell2 : numpy.array
        array of integer subsample labels of length N2
        
    N_samples : int
        total number of subsamples
    
    rp_bins : numpy.array
        array defining projected seperation in which to sum the pair counts
    
    pi_bins : numpy.array
        array defining parallel seperation in which to sum the pair counts
    
    period : numpy.array
        array defining axis-aligned periodic boundary conditions.
    
    Returns
    -------
    result : numpy.ndarray
        3-D array of weighted jackknife pair counts in bins defined by ``rp_bins`` and ``pi_bins``.
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
    
    #square the distance bins to avoid taking a square root in a tight loop
    rp_bins = rp_bins**2
    pi_bins = pi_bins**2
    
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def s_mu_npairs_no_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                       np.ndarray[np.float64_t, ndim=1] y_icell1,
                       np.ndarray[np.float64_t, ndim=1] z_icell1,
                       np.ndarray[np.float64_t, ndim=1] x_icell2,
                       np.ndarray[np.float64_t, ndim=1] y_icell2,
                       np.ndarray[np.float64_t, ndim=1] z_icell2,
                       np.ndarray[np.float64_t, ndim=1] s_bins,
                       np.ndarray[np.float64_t, ndim=1] mu_bins):
    """
    Calculate the number of pairs with seperations greater than or equal to :math:`s` and :math:`\\mu`, :math:`N(>s,>\\mu)`.
    
    :math:`s` is the radial seperation, and :math:`\\mu` is sine of the angle wrt the z-direction.
    
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
    
    s_bins : numpy.array
         array defining :math:`s` bins in which to sum the pair counts
    
    mu_bins : numpy.array
         array defining :math:`mu` bins in which to sum the pair counts
    
    Returns
    -------
    result : numpy.ndarray
        2-D array of pair counts in bins defined by ``s_bins`` and ``mu_bins``.
    """
    
    #c definitions
    cdef int ns_bins = len(s_bins)
    cdef int nmu_bins = len(mu_bins)
    cdef int ns_bins_minus_one = len(s_bins) -1
    cdef int nmu_bins_minus_one = len(mu_bins) -1
    cdef np.ndarray[np.int_t, ndim=2] counts =\
        np.zeros((ns_bins, nmu_bins), dtype=np.int)
    cdef double d_perp, d_para, s, mu
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
                        
            #transform to s and mu
            s = sqrt(d_perp + d_para)
            if s!=0: mu = sqrt(d_para)/s
            else: mu=0.0
            
            #calculate counts in bins
            xy_z_binning(<np.int_t*>counts.data,\
                         <np.float64_t*>s_bins.data,\
                         <np.float64_t*>mu_bins.data,\
                         s, mu, ns_bins_minus_one, nmu_bins_minus_one)
        
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def s_mu_npairs_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                    np.ndarray[np.float64_t, ndim=1] y_icell1,
                    np.ndarray[np.float64_t, ndim=1] z_icell1,
                    np.ndarray[np.float64_t, ndim=1] x_icell2,
                    np.ndarray[np.float64_t, ndim=1] y_icell2,
                    np.ndarray[np.float64_t, ndim=1] z_icell2,
                    np.ndarray[np.float64_t, ndim=1] s_bins,
                    np.ndarray[np.float64_t, ndim=1] mu_bins,
                    np.ndarray[np.float64_t, ndim=1] period):
    """
    Calculate the number of pairs with seperations greater than or equal to :math:`s` and :math:`\\mu`, :math:`N(>s,>\\mu)`, with periodic boundary conditions (PBC).
    
    :math:`s` is the radial seperation, and :math:`\\mu` is sine of the angle wrt the z-direction.
    
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
    
    s_bins : numpy.array
         array defining :math:`s` bins in which to sum the pair counts
    
    mu_bins : numpy.array
         array defining :math:`mu` bins in which to sum the pair counts
    
    period : numpy.array
        array defining axis-aligned periodic boundary conditions.
    
    Returns
    -------
    result : numpy.ndarray
        2-D array of pair counts in bins defined by ``s_bins`` and ``mu_bins``.
    """
    
    #c definitions
    cdef int ns_bins = len(s_bins)
    cdef int nmu_bins = len(mu_bins)
    cdef int ns_bins_minus_one = len(s_bins) -1
    cdef int nmu_bins_minus_one = len(mu_bins) -1
    cdef np.ndarray[np.int_t, ndim=2] counts =\
        np.zeros((ns_bins, nmu_bins), dtype=np.int)
    cdef double d, d_perp, d_para, s, mu
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
            
            #transform to s and mu
            s = sqrt(d_perp + d_para)
            #if s!=0: mu = sqrt(d_para)/s
            if s!=0: mu = sqrt(d_perp)/s
            else: mu=0.0
            
            #calculate counts in bins
            xy_z_binning(<np.int_t*>counts.data,\
                         <np.float64_t*>s_bins.data,\
                         <np.float64_t*>mu_bins.data,\
                         s, mu, ns_bins_minus_one, nmu_bins_minus_one)
        
    return counts


###########################
#### binning functions ####
###########################

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


###########################
###########################
###########################

cdef inline double jweight(np.int_t j, np.int_t j1, np.int_t j2,\
                           np.float64_t w1, np.float64_t w2):
    """
    Return the jackknife weighted count.
    
    parameters
    ----------
    j : int
        subsample being removed
    
    j1 : int
        integer label indicating which subsample point 1 occupies
    
    j2 : int
        integer label indicating which subsample point 2 occupies
    
    w1 : float
        weight associated with point 1
    
    w2 : float
        weight associated with point 2
    
    Returns
    -------
    w : double
        0.0, w1*w2*0.5, or w1*w2
    
    Notes
    -----
    We use the tag '0' to indicated we want to use the entire sample, i.e. no subsample
    should be labeled with a '0'.
    
    jackknife wiehgt is caclulated as follows:
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


