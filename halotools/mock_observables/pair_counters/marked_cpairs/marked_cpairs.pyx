# cython: profile=False

"""
generalized weighted brute force pair counters.
"""


from __future__ import (absolute_import, division, print_function, unicode_literals)
import sys
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport fabs, fmin
from .weighting_functions cimport *
from .custom_weighting_func cimport *
from .pairwise_velocity_funcs cimport *
from .distances cimport *

#definition of weighting function types (necessary so we can pass the functions around)
#standard marking functions
ctypedef double (*f_type)(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift)
#velocity marking functions
ctypedef void (*ff_type)(np.float64_t* w1, np.float64_t* w2, np.float64_t* shift, double *result1, double *result2, double *result3)

__author__ = ['Duncan Campbell']
__all__ = ['marked_npairs_no_pbc',\
           'xy_z_marked_npairs_no_pbc',\
           'velocity_marked_npairs_no_pbc',\
           'xy_z_velocity_marked_npairs_no_pbc']


#############################
#### pair counting funcs ####
#############################

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
                         np.int_t weight_func_id,
                         np.ndarray[np.float64_t, ndim=1] shift
                         ):
    """
    Calculate the number of weighted pairs with seperations greater than or equal to r, :math:`W(>r)`.
    
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
    
    w_icell1 : numpy.ndarray
        2-D array of weights of length N1 and depth >=1 (dependent on wfunc)
    
    w_icell2 : numpy.ndarray
        2-D array of weights of length N2 and depth >=1 (dependent on wfunc)
    
    rbins : numpy.array
         array defining radial binsin wich to sum the pair counts
    
    weight_func_id : int
        integer ID of weighting function to use.
    
    shift : numpy.array
         Length-3 vector indicating the amount the points in data2 have been shifted in 
         each dimension to faciliate the use with PBCs (usually 0.0 or +-Lbox).
    
    Returns
    -------
    result :  numpy.array
        weighted pair counts in radial bins defined by ``rbins``.
        The exact values depend on ``weight_func_id`` 
        (which weighting function was chosen).
    
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
    
    >>> weights = np.random.random(Npts)
    
    Count the number of weighted pairs that can be formed amongst these random points:
    
    >>> rbins = np.linspace(0,0.5,10)
    >>> shift = np.array([0.0,0.0,0.0])
    >>> counts = marked_npairs_no_pbc(x,y,z,x,y,z,rbins,weights,weights,1,shift)
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
    
    #square the distance bins to avoid taking a square root in a tight loop
    rbins = rbins**2
    
    #choose weighting function
    wfunc = return_weighting_function(weight_func_id)
    
    #loop over points in grid1's cell
    for i in range(0,Ni):
                
        #loop over points in grid2's cell
        for j in range(0,Nj):
            
            #calculate the square distance
            d = square_distance(x_icell1[i],y_icell1[i],z_icell1[i],\
                                x_icell2[j],y_icell2[j],z_icell2[j])
            
            radial_wbinning(<np.float64_t*>counts.data,\
                            <np.float64_t*>rbins.data, d, nbins_minus_one,\
                            &w_icell1[i,0],&w_icell2[j,0],\
                            <f_type>wfunc, <np.float64_t*>shift.data)
    
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def xy_z_marked_npairs_no_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                              np.ndarray[np.float64_t, ndim=1] y_icell1,
                              np.ndarray[np.float64_t, ndim=1] z_icell1,
                              np.ndarray[np.float64_t, ndim=1] x_icell2,
                              np.ndarray[np.float64_t, ndim=1] y_icell2,
                              np.ndarray[np.float64_t, ndim=1] z_icell2,
                              np.ndarray[np.float64_t, ndim=2] w_icell1,
                              np.ndarray[np.float64_t, ndim=2] w_icell2,
                              np.ndarray[np.float64_t, ndim=1] rp_bins,
                              np.ndarray[np.float64_t, ndim=1] pi_bins,
                              np.int_t weight_func_id,
                              np.ndarray[np.float64_t, ndim=1] shift
                             ):
    """    
    Calculate the number of weighted pairs with seperations greater than or equal to :math:`r_{\\perp}` and :math:`r_{\\parallel}`, :math:`W(>r_{\\perp},>r_{\\parallel})`.
    
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
    
    w_icell1 : numpy.ndarray
        2-D array of weights of length N1 and depth >=1 (dependent on wfunc)
    
    w_icell2 : numpy.ndarray
        2-D array of weights of length N2 and depth >=1 (dependent on wfunc)
    
    rp_bins : numpy.array
        array defining projected seperation in which to sum the pair counts
    
    pi_bins : numpy.array
        array defining parallel seperation in which to sum the pair counts
    
    weight_func_id : int
        integer ID of weighting function to use.
    
    shift : numpy.array
         Length-3 vector indicating the amount the points in data2 have been shifted in 
         each dimension to faciliate the use with PBCs (usually 0.0 or +-Lbox).
    
    Returns
    -------
    result :  numpy.array
        2-D array of weighted pair counts of bins defined by ``rp_bins`` and ``pi_bins``.
        The exact values depend on ``weight_func_id`` 
        (which weighting function was chosen).
    
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
    
    >>> weights = np.random.random(Npts)
    
    Count the number of weighted pairs that can be formed amongst these random points:
    
    >>> rp_bins = np.linspace(0,0.25,10)
    >>> pi_bins = np.linspace(0,0.5,10)
    >>> shift = np.array([0.0,0.0,0.0])
    >>> counts = xy_z_marked_npairs_no_pbc(x,y,z,x,y,z,rp_bins,pi_bins,weights,weights,1,shift)
    """
    
    #c definitions
    cdef int nrp_bins = len(rp_bins)
    cdef int npi_bins = len(pi_bins)
    cdef int nrp_bins_minus_one = len(rp_bins) -1
    cdef int npi_bins_minus_one = len(pi_bins) -1
    cdef int n_weights1 = np.shape(w_icell1)[1]
    cdef int n_weights2 = np.shape(w_icell2)[1]
    cdef np.ndarray[np.float64_t, ndim=2] counts =\
        np.zeros((nrp_bins, npi_bins), dtype=np.float64)
    cdef double d_perp, d_para
    cdef int i, j
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    cdef f_type wfunc
    
    #square the distance bins to avoid taking a square root in a tight loop
    rp_bins = rp_bins**2
    pi_bins = pi_bins**2
    
    #choose weighting function
    wfunc = return_weighting_function(weight_func_id)
    
    #loop over points in grid1's cell
    for i in range(0,Ni):
        
        #loop over points in grid2's cell
        for j in range(0,Nj):
            
            #calculate the square distance
            d_perp = perp_square_distance(x_icell1[i], y_icell1[i],\
                                          x_icell2[j], y_icell2[j])
            d_para = para_square_distance(z_icell1[i], z_icell2[j])
            
            xy_z_wbinning(<np.float64_t*>counts.data,\
                          <np.float64_t*>rp_bins.data,\
                          <np.float64_t*>pi_bins.data,
                          d_perp, d_para,\
                          nrp_bins_minus_one, npi_bins_minus_one,\
                          &w_icell1[i,0],&w_icell2[j,0],\
                          <f_type>wfunc, <np.float64_t*>shift.data)
    
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def velocity_marked_npairs_no_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                                  np.ndarray[np.float64_t, ndim=1] y_icell1,
                                  np.ndarray[np.float64_t, ndim=1] z_icell1,
                                  np.ndarray[np.float64_t, ndim=1] x_icell2,
                                  np.ndarray[np.float64_t, ndim=1] y_icell2,
                                  np.ndarray[np.float64_t, ndim=1] z_icell2,
                                  np.ndarray[np.float64_t, ndim=2] w_icell1,
                                  np.ndarray[np.float64_t, ndim=2] w_icell2,
                                  np.ndarray[np.float64_t, ndim=1] rbins,
                                  np.int_t weight_func_id,
                                  np.ndarray[np.float64_t, ndim=1] shift
                                  ):
    """
    Calculate the number of velocity weighted pairs with seperations greater than or equal to r, :math:`W(>r)`.
    
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
    
    w_icell1 : numpy.ndarray
        2-D array of weights of length N1 and depth >=1 (dependent on wfunc)
    
    w_icell2 : numpy.ndarray
        2-D array of weights of length N2 and depth >=1 (dependent on wfunc)
    
    rbins : numpy.array
         array defining radial binsin wich to sum the pair counts
    
    weight_func_id : int
        integer ID of weighting function to use.
    
    shift : numpy.array
         Length-3 vector indicating the amount the points in data2 have been shifted in 
         each dimension to faciliate the use with PBCs (usually 0.0 or +-Lbox).
    
    Returns
    -------
    result1 : numpy.ndarray
        array of weighted pair counts in radial bins defined by ``rbins``.
        The exact values depend on ``weight_func_id`` 
        (which weighting function was chosen).
        
    result2 : numpy.ndarray
        array of weighted pair counts in radial bins defined by ``rbins``.
        The exact values depend on ``weight_func_id`` 
        (which weighting function was chosen).
        
    result3 : numpy.ndarray
        array of weighted pair counts in radial bins defined by ``rbins``.
        The exact values depend on ``weight_func_id`` 
        (which weighting function was chosen).
    
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
    
    Assign random velocities to each point:
    
    >>> velocities = np.random.random((Npts,3))
    
    Calculate the sum of the radial velocity between these points:
    
    >>> rbins = np.linspace(0,0.5,10)
    >>> weights = velocities
    >>> vr, dummy, counts = marked_npairs_no_pbc(x,y,z,x,y,z,rbins,weights,weights,11,shift)
    """
    
    #c definitions
    cdef int nbins = len(rbins)
    cdef int n_weights1 = np.shape(w_icell1)[1]
    cdef int n_weights2 = np.shape(w_icell2)[1]
    cdef int nbins_minus_one = len(rbins) -1
    cdef np.ndarray[np.float64_t, ndim=1] counts1 = np.zeros((nbins,), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] counts2 = np.zeros((nbins,), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] counts3 = np.zeros((nbins,), dtype=np.float64)
    cdef double d
    cdef int i, j
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    cdef ff_type wfunc
    
    #square the distance bins to avoid taking a square root in a tight loop
    rbins = rbins**2
    
    #choose weighting function
    wfunc = return_velocity_weighting_function(weight_func_id)
    
    #loop over points in grid1's cell
    for i in range(0,Ni):
                
        #loop over points in grid2's cell
        for j in range(0,Nj):
            
            #calculate the square distance
            d = square_distance(x_icell1[i],y_icell1[i],z_icell1[i],\
                                x_icell2[j],y_icell2[j],z_icell2[j])
            
            radial_velocity_wbinning(<np.float64_t*>counts1.data,\
                                   <np.float64_t*>counts2.data,\
                                   <np.float64_t*>counts3.data,\
                                   <np.float64_t*>rbins.data, d, nbins_minus_one,\
                                   &w_icell1[i,0],&w_icell2[j,0],\
                                   <ff_type>wfunc, <np.float64_t*>shift.data)
    
    return counts1, counts2, counts3


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def xy_z_velocity_marked_npairs_no_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                                       np.ndarray[np.float64_t, ndim=1] y_icell1,
                                       np.ndarray[np.float64_t, ndim=1] z_icell1,
                                       np.ndarray[np.float64_t, ndim=1] x_icell2,
                                       np.ndarray[np.float64_t, ndim=1] y_icell2,
                                       np.ndarray[np.float64_t, ndim=1] z_icell2,
                                       np.ndarray[np.float64_t, ndim=2] w_icell1,
                                       np.ndarray[np.float64_t, ndim=2] w_icell2,
                                       np.ndarray[np.float64_t, ndim=1] rp_bins,
                                       np.ndarray[np.float64_t, ndim=1] pi_bins,
                                       np.int_t weight_func_id,
                                       np.ndarray[np.float64_t, ndim=1] shift
                                       ):
    """
    Calculate the number of velocity weighted pairs with seperations greater than or equal to :math:`r_{\\perp}` and :math:`r_{\\parallel}`, :math:`W(>r_{\\perp},>r_{\\parallel})`.
    
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
    
    w_icell1 : numpy.ndarray
        2-D array of weights of length N1 and depth >=1 (dependent on wfunc)
    
    w_icell2 : numpy.ndarray
        2-D array of weights of length N2 and depth >=1 (dependent on wfunc)
    
    rp_bins : numpy.array
         array defining projected seperation in which to sum the pair counts
    
    pi_bins : numpy.array
         array defining parallel seperation in which to sum the pair counts
    
    weight_func_id : int
        integer ID of weighting function to use.
    
    shift : numpy.array
         Length-3 vector indicating the amount the points in data2 have been shifted in 
         each dimension to faciliate the use with PBCs (usually 0.0 or +-Lbox).
    
    Returns
    -------
    result1 : numpy.ndarray
        2-D array of weighted velocity counts in ``rp_bins`` and ``pi_bins``.
        The exact values depend on ``weight_func_id`` 
        (which weighting function was chosen).
        
    result2 : numpy.ndarray
        2-D array of weighted velocity counts in ``rp_bins`` and ``pi_bins``.
        The exact values depend on ``weight_func_id`` 
        (which weighting function was chosen).
        
    result3 : numpy.ndarray
        2-D array of weighted velocity counts in ``rp_bins`` and ``pi_bins``.
        The exact values depend on ``weight_func_id`` 
        (which weighting function was chosen).
        
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
    
    Assign random velocities to each point:
    
    >>> velocities = np.random.random((Npts,3))
    
    Calculate the sum of the radial velocities between these points in bins:
    
    >>> rp_bins = np.linspace(0,0.5,10)
    >>> pi_bins = np.linspace(0,0.5,10)
    >>> weights = velocities
    >>> vr, dummy, counts = marked_npairs_no_pbc(x,y,z,x,y,z,rp_bins,pi_bins,weights,weights,11,shift)
    """
    
    #c definitions
    cdef int n_weights1 = np.shape(w_icell1)[1]
    cdef int n_weights2 = np.shape(w_icell2)[1]
    cdef int nrp_bins = len(rp_bins)
    cdef int npi_bins = len(pi_bins)
    cdef int nrp_bins_minus_one = len(rp_bins) -1
    cdef int npi_bins_minus_one = len(pi_bins) -1
    cdef np.ndarray[np.float64_t, ndim=2] counts1 =\
        np.zeros((nrp_bins, npi_bins), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] counts2 =\
        np.zeros((nrp_bins, npi_bins), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] counts3 =\
        np.zeros((nrp_bins, npi_bins), dtype=np.float64)
    cdef double d_perp, d_para
    cdef int i, j
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)
    cdef ff_type wfunc
    
    #square the distance bins to avoid taking a square root in a tight loop
    rp_bins = rp_bins**2
    pi_bins = pi_bins**2
    
    #choose weighting function
    wfunc = return_velocity_weighting_function(weight_func_id)
    
    #loop over points in grid1's cell
    for i in range(0,Ni):
                
        #loop over points in grid2's cell
        for j in range(0,Nj):
            
            #calculate the square distance
            d_perp = perp_square_distance(x_icell1[i], y_icell1[i],\
                                          x_icell2[j],y_icell2[j])
            d_para = para_square_distance(z_icell1[i], z_icell2[j],)
            
            xy_z_velocity_wbinning(<np.float64_t*>counts1.data,\
                                   <np.float64_t*>counts2.data,\
                                   <np.float64_t*>counts3.data,\
                                   <np.float64_t*>rp_bins.data,\
                                   <np.float64_t*>pi_bins.data,\
                                   d_perp, d_para,\
                                   nrp_bins_minus_one, npi_bins_minus_one,\
                                   &w_icell1[i,0],&w_icell2[j,0],\
                                   <ff_type>wfunc, <np.float64_t*>shift.data)
    return counts1, counts2, counts3


###########################
#### binning functions ####
###########################

cdef inline radial_wbinning(np.float64_t* counts, np.float64_t* bins,\
                            np.float64_t d, np.int_t k,\
                            np.float64_t* w1, np.float64_t* w2, f_type wfunc,\
                            np.float64_t* shift):
    """
    real space radial weighted binning function
    """
    
    cdef double holder = wfunc(w1, w2, shift)
    while d<=bins[k]:
        counts[k] += holder
        k=k-1
        if k<0: break


cdef inline radial_velocity_wbinning(np.float64_t* counts1,\
                                     np.float64_t* counts2,\
                                     np.float64_t* counts3,\
                                     np.float64_t* bins, np.float64_t d, np.int_t k,\
                                     np.float64_t* w1, np.float64_t* w2, ff_type wfunc,\
                                     np.float64_t* shift):
    """
    real space radial weighted binning function for pairwise velocity weights
    """
    
    cdef double holder1, holder2, holder3 
    wfunc(w1, w2, shift, &holder1, &holder2, &holder3)
    while d<=bins[k]:
        counts1[k] += holder1
        counts2[k] += holder2
        counts3[k] += 1
        k=k-1
        if k<0: break


cdef inline xy_z_wbinning(np.float64_t* counts,
                          np.float64_t* rp_bins,
                          np.float64_t* pi_bins,
                          np.float64_t d_perp,
                          np.float64_t d_para,
                          np.int_t k,
                          np.int_t npi_bins_minus_one,
                          np.float64_t* w1,
                          np.float64_t* w2,
                          f_type wfunc,\
                          np.float64_t* shift):
    """
    2D+1 binning function
    """
    cdef int g
    cdef int max_k = npi_bins_minus_one+1
    
    cdef double holder = wfunc(w1, w2, shift)
    
    while d_perp<=rp_bins[k]:
        g = npi_bins_minus_one
        while d_para<=pi_bins[g]:
            counts[k*max_k+g] += holder
            g=g-1
            if g<0: break
        k=k-1
        if k<0: break


cdef inline xy_z_velocity_wbinning(np.float64_t* counts1,
                                   np.float64_t* counts2,
                                   np.float64_t* counts3,
                                   np.float64_t* rp_bins,
                                   np.float64_t* pi_bins,
                                   np.float64_t d_perp,
                                   np.float64_t d_para,
                                   np.int_t k,
                                   np.int_t npi_bins_minus_one,
                                   np.float64_t* w1,
                                   np.float64_t* w2,
                                   ff_type wfunc,\
                                   np.float64_t* shift):
    """
    2D+1 binning function
    """
    cdef int g
    cdef int max_k = npi_bins_minus_one+1
    
    cdef double holder1, holder2, holder3 
    wfunc(w1, w2, shift, &holder1, &holder2, &holder3)
    
    while d_perp<=rp_bins[k]:
        g = npi_bins_minus_one
        while d_para<=pi_bins[g]:
            counts1[k*max_k+g] += holder1
            counts2[k*max_k+g] += holder2
            counts3[k*max_k+g] += holder3
            g=g-1
            if g<0: break
        k=k-1
        if k<0: break


###########################
####  helper functions ####
###########################

cdef f_type return_weighting_function(weight_func_id):
    """
    returns a pointer to the user-specified weighting function.
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
        return ineqweights
    elif weight_func_id==5:
        return gweights
    elif weight_func_id==6:
        return lweights
    elif weight_func_id==7:
        return tgweights
    elif weight_func_id==8:
        return tlweights
    elif weight_func_id==9:
        return tweights
    elif weight_func_id==10:
        return exweights
    else:
        raise ValueError('weighting function does not exist')

cdef ff_type return_velocity_weighting_function(weight_func_id):
    """
    returns a pointer to the user-specified pairwise velocity weighting function.
    """
    
    if weight_func_id==11:
        return relative_radial_velocity_weights
    if weight_func_id==12:
        return radial_velocity_weights
    if weight_func_id==13:
        return radial_velocity_variance_counter_weights
    if weight_func_id==14:
        return relative_los_velocity_weights
    if weight_func_id==15:
        return los_velocity_weights
    if weight_func_id==16:
        return los_velocity_variance_counter_weights
    else:
        raise ValueError('weighting function does not exist')


