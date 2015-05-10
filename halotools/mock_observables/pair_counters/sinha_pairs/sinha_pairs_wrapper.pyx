from countpairs_pbc cimport countpairs_pbc as countpairs_pbc
from countpairs_nopbc cimport countpairs_nopbc as countpairs_nopbc

import ctypes
from libc.stdlib cimport malloc, free
import numpy as np
import sys

__all__ = ['npairs']

def npairs(data1, data2, bins, period=None):
    """
    Calculate the number of pairs with separations less than or equal to rbins[i].
    
    Parameters
    ----------
    data1: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period
            
    data2: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period
            
    rbins : array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
        len(rbins) = Nrbins + 1.
            
    period: array_like, optional
        length k array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.
            
    Returns
    -------
    N_pairs : array of length len(rbins)
        number counts of pairs
     
    """
    
    #work with arrays!
    data1 = np.asarray(data1)
    if data1.ndim ==1: data1 = np.array([data1])
    data2 = np.asarray(data2)
    if data2.ndim ==1: data2 = np.array([data2])
    bins = np.asarray(bins)
    if bins.size ==1: bins = np.array([bins])
    
    #Check to make sure both data sets have the same dimension. Otherwise, throw an error!
    if np.shape(data1)[-1]!=np.shape(data2)[-1]:
        raise ValueError("data1 and data2 inputs do not have the same dimension.")
        return None
        
    #Process period entry and check for consistency.
    if period is None:
            period = np.array([np.inf]*np.shape(data1)[-1])
    else:
        period = np.asarray(period).astype("float64")
        if np.shape(period) == ():
            period = np.array([period]*np.shape(data1)[-1])
        elif np.shape(period)[0] != np.shape(data1)[-1]:
            raise ValueError("period should have len == dimension of points")
            return None
    
    if np.all(period==None):
        PBCs = False
    elif np.all(period==np.inf):
        PBCs = False
    else:
        PBCs = True
    
    len_data1 = len(data1)

    #sinha pair counter only works with 3D points
    if np.shape(data1)[1] !=3:
        raise ValueError("sinha pairs only works with 3D points")
    if np.shape(data2)[1] !=3:
        raise ValueError("sinha pairs only works with 3D points")

    #sinha pair counter only works with cubic boxes
    Lbox = period[0]

    #sinha pair counter returns counts in bins.
    #add 0 so we can get counts between 0 and rbins
    bins = np.sort(bins)
    if not 0.0 in bins:
        np.insert(bins,0,0.0)
    
    #if data1==data2, set data2 to None and do autocorrelation
    if np.all(data1==data2):
        autocorr = 1
    else:
        autocorr = 0

    #allocate memory for C arrays for first set of points
    cdef double *X1
    cdef double *Y1
    cdef double *Z1
    X1 = <double *>malloc(len_data1*sizeof(double))
    for i in range(len_data1):
        X1[i] = data1[i][0]
    Y1 = <double *>malloc(len_data1*sizeof(double))
    for i in range(len_data1):
        Y1[i] = data1[i][1]
    Z1 = <double *>malloc(len_data1*sizeof(double))
    for i in range(len_data1):
        Z1[i] = data1[i][2]

    #allocate memory for C arrays for first set of points
    cdef double *X2
    cdef double *Y2
    cdef double *Z2

    if (autocorr==1):
        len_data2 = len_data1
        X2 = <double *>malloc(len_data1*sizeof(double))
        for i in range(len_data1):
            X2[i] = data1[i][0]
        Y2 = <double *>malloc(len_data1*sizeof(double))
        for i in range(len_data1):
            Y2[i] = data1[i][1]
        Z2 = <double *>malloc(len_data1*sizeof(double))
        for i in range(len_data1):
            Z2[i] = data1[i][2]
    else:
        len_data2 = len(data2)
        X2 = <double *>malloc(len_data2*sizeof(double))
        for i in range(len_data2):
            X2[i] = data2[i][0]
        Y2 = <double *>malloc(len_data2*sizeof(double))
        for i in range(len_data2):
            Y2[i] = data2[i][1]
        Z2 = <double *>malloc(len_data2*sizeof(double))
        for i in range(len_data2):
            Z2[i] = data2[i][2]
    
    #define constants used by sinha pair counter
    cdef double xmin, ymin, zmin, xmax, ymax, zmax, max_bin
    if PBCs==True:
        xmin=0.0
        ymin=0.0
        zmin=0.0
        xmax=float(Lbox)
        ymax=float(Lbox)
        zmax=float(Lbox)
    elif PBCs==False:
        xmin = float(np.amin(np.hstack((data1[:,0],data2[:,0]))))
        ymin = float(np.amin(np.hstack((data1[:,1],data2[:,1]))))
        zmin = float(np.amin(np.hstack((data1[:,2],data2[:,2]))))
        xmax = float(np.amax(np.hstack((data1[:,0],data2[:,0]))))
        ymax = float(np.amax(np.hstack((data1[:,1],data2[:,1]))))
        zmax = float(np.amax(np.hstack((data1[:,2],data2[:,2]))))
        xyz_min = float(np.amin([xmin, ymin, zmin]))
        xmin = xyz_min
        ymin = xyz_min
        zmin = xyz_min
        xyz_max = float(np.amax([xmax, ymax, zmax]))
        xmax = xyz_max
        ymax = xyz_max
        zmax = xyz_max
    max_bin = float(np.amax(bins))
    len_bins = len(bins)
    
    #process bins
    cdef double *rbins
    rbins = <double *>malloc(len_bins*sizeof(double))
    for i in range(len_bins):
        rbins[i] = bins[i]

    #create list of pair counts
    cdef int* c_paircounts 
    c_paircounts= <int *>malloc(len_bins*sizeof(int))

    if PBCs==True:
        countpairs_pbc(len_data1, X1, Y1, Z1, len_data2, X2, Y2, Z2, xmin, xmax, ymin, ymax, zmin, zmax, autocorr, max_bin, len_bins, rbins, &c_paircounts)
    elif PBCs==False:
        countpairs_nopbc(len_data1, X1, Y1, Z1, len_data2, X2, Y2, Z2, xmin, xmax, ymin, ymax, zmin, zmax, autocorr, max_bin, len_bins, rbins, &c_paircounts)

    #convert c array back into python list
    paircounts =[]
    for i in range(len_bins):
        paircounts.append(c_paircounts[i])
    free(c_paircounts)
    paircounts = np.cumsum(paircounts)
    if autocorr == 1:
        paircounts[0] += len_data1
    return np.array(paircounts)





