#!/usr/bin/env python
# cython: profile=False

#Duncan Campbell
#August 22, 2014
#Yale University
#Calculate the number of pairs with separations less than r. This code has been optimized 
#for speed, so it may not be the most readable thing in the whole world.  See pairs.py for
#a pretty clear pure python version.

from __future__ import print_function, division
cimport cython
cimport numpy as np
import numpy as np

__all__=['npairs','wnpairs','pairwise_distances']

from libc.math cimport sqrt, floor, fabs, fmin
cdef np.float64_t infinity = np.inf

@cython.boundscheck(False)
@cython.wraparound(False)
def npairs(data1, data2, rbins, period=None):
    """
    Calculate the number of pairs with separations less than or equal to rbins[i].
    
    Parameters
    ----------
    data1: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period. This cython implementation requires data1.ndim==2.
            
    data2: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period. This cython implementation requires data2.ndim==2.
            
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
        
    Notes
    -----
    This code executes a brute force method to calculate pair counts. All distances 
    between all pairs are calculated. Depending on the use-case, this may not be the most 
    efficient approach. Consider a tree based pair counting algorithm.
     
    """
    
    #Check to make sure both data sets have the same dimension. Otherwise, throw an error!
    ii = len(np.shape(data1))-1
    jj = len(np.shape(data1))-1
    if np.shape(data1)[ii]!=np.shape(data2)[jj]:
        raise ValueError("data1 and data2 inputs do not have the same dimension.")
        
    #Process period entry and check for consistency.
    if period is None:
            period = np.array([np.inf]*np.shape(data1)[ii])
            PBCs = False #do non-periodic calc
    else:
        period = np.asarray(period).astype(np.float64)
        PBCs = True #do periodic calc
        if np.shape(period) == ():
            period = np.array([period]*np.shape(data1)[ii])
        elif np.shape(period)[0] != np.shape(data1)[ii]:
            raise ValueError("period should have len == dimension of points")
    
    #square the radial bins, so I don't have to take the squareroot in the dist calc
    if np.shape(rbins)==(): rbins = np.array([rbins])
    else: rbins = np.array(rbins)
    if rbins.ndim != 1: raise ValueError('rbins must be a 1-D array')
    rbins = rbins**2.0
    
    #static type and fix efficient indexing.
    cdef np.ndarray[np.float64_t, ndim=2] cdata1 = np.ascontiguousarray(data1,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] cdata2 = np.ascontiguousarray(data2,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] crbins = np.ascontiguousarray(rbins,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cperiod = np.ascontiguousarray(period,dtype=np.float64)
    #static type and fix efficient indexing.
    cdef int M1 = data1.shape[0]
    cdef int M2 = data2.shape[0]
    cdef int N = data1.shape[1]
    cdef int nbins = rbins.shape[0]
    cdef double tmp, d
    cdef np.ndarray[np.int_t, ndim=1] counts = np.zeros((nbins,), dtype=np.int)
    cdef int i,j,k
    
    if PBCs==True:
        #loop through all pairs
        for i in range(M1):
            for j in range(M2):
                d = 0.0
                for k in range(N):
                    tmp = fabs(cdata1[i, k] - cdata2[j, k])
                    tmp = fmin(tmp, cperiod[k] - tmp)
                    #tmp = cdata1[i, k] - cdata2[j, k] #non-periodic dist calc
                    d += tmp * tmp
                #d = sqrt(d)
                k = nbins-1
                while d<=crbins[k]:
                    counts[k] += 1
                    k=k-1
                    if k<0: break
    else:
        #loop through all pairs
        for i in range(M1):
            for j in range(M2):
                d = 0.0
                for k in range(N):
                    #tmp = fabs(cdata1[i, k] - cdata2[j, k]) #PBCs dist cals
                    #tmp = fmin(tmp, cperiod[k] - tmp)
                    tmp = cdata1[i, k] - cdata2[j, k]
                    d += tmp * tmp
                #d = sqrt(d)
                k = nbins-1
                while d<=crbins[k]:
                    counts[k] += 1
                    k=k-1
                    if k<0: break
                
    return np.asarray(counts)

@cython.boundscheck(False)
@cython.wraparound(False)
def wnpairs(data1, data2, rbins, period=None, weights1=None, weights2=None):
    """
    Calculate the weighted number of pairs with separations less than or equal to rbins[i].
    
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
            
    weights1: array_like, optional
        length N1 array containing weights used for weighted pair counts, w1*w2.
            
    weights2: array_like, optional
        length N2 array containing weights used for weighted pair counts, w1*w2.
            
    Returns
    -------
    wN_pairs : array of length len(rbins)
        weighted number counts of pairs
    
    Notes
    -----
    This code executes a brute force method to calculate pair counts. All distances 
    between all pairs are calculated. Depending on the use-case, this may not be the most 
    efficient approach. Consider a tree based pair counting algorithm.
     
    """
    
    #Check to make sure both data sets have the same dimension. Otherwise, throw an error!
    ii = len(np.shape(data1))-1
    jj = len(np.shape(data1))-1
    if np.shape(data1)[ii]!=np.shape(data2)[jj]:
        raise ValueError("data1 and data2 inputs do not have the same dimension.")
        
    #Process period entry and check for consistency.
    if period is None:
            period = np.array([np.inf]*np.shape(data1)[ii])
            PBCs=False
    else:
        period = np.asarray(period).astype(np.float64)
        PBCs=True
        if np.shape(period) == ():
            period = np.array([period]*np.shape(data1)[ii])
        elif np.shape(period)[0] != np.shape(data1)[ii]:
            raise ValueError("period should have len == dimension of points")
    
    #Process weights1 entry and check for consistency.
    if weights1 is None:
            weights1 = np.array([1.0]*np.shape(data1)[0], dtype=np.float64)
    else:
        weights1 = np.asarray(weights1).astype("float64")
        if np.shape(weights1)[0] != np.shape(data1)[0]:
            raise ValueError("weights1 should have same len as data1")
    #Process weights2 entry and check for consistency.
    if weights2 is None:
            weights2 = np.array([1.0]*np.shape(data2)[0], dtype=np.float64)
    else:
        weights2 = np.asarray(weights2).astype("float64")
        if np.shape(weights2)[0] != np.shape(data2)[0]:
            raise ValueError("weights2 should have same len as data2")
    
    #square the radial bins, so I don't have to take the squareroot in the dist calc
    if np.shape(rbins)==(): rbins = np.array([rbins])
    else: rbins = np.array(rbins)
    if rbins.ndim != 1: raise ValueError('rbins must be a 1-D array')
    rbins = rbins**2.0
    
    #static type and fix efficient indexing.
    cdef np.ndarray[np.float64_t, ndim=2] cdata1 = np.ascontiguousarray(data1,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] cdata2 = np.ascontiguousarray(data2,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cweights1 = np.ascontiguousarray(weights1,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cweights2 = np.ascontiguousarray(weights2,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] crbins = np.ascontiguousarray(rbins,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cperiod = np.ascontiguousarray(period,dtype=np.float64)
    #static type and fix efficient indexing.
    cdef int M1 = data1.shape[0]
    cdef int M2 = data2.shape[0]
    cdef int N = data1.shape[1]
    cdef int nbins = rbins.shape[0]
    cdef double tmp, d
    cdef np.ndarray[np.float64_t, ndim=1] counts = np.zeros((nbins,), dtype=np.int)
    cdef int i,j,k
    
    if PBCs==True:
        #loop through all pairs
        for i in range(M1):
            for j in range(M2):
                d = 0.0
                for k in range(N):
                    tmp = fabs(cdata1[i, k] - cdata2[j, k]) #PBC dist calc
                    tmp = fmin(tmp, cperiod[k] - tmp)
                    #tmp = cdata1[i, k] - cdata2[j, k]
                    d += tmp * tmp
                #d = sqrt(d)
                k = nbins-1
                while d<=crbins[k]:
                    counts[k] += cweights1[i]*cweights2[j]
                    k=k-1
                    if k<0: break
    else:
        #loop through all pairs
        for i in range(M1):
            for j in range(M2):
                d = 0.0
                for k in range(N):
                    #tmp = fabs(cdata1[i, k] - cdata2[j, k])
                    #tmp = fmin(tmp, cperiod[k] - tmp)
                    tmp = cdata1[i, k] - cdata2[j, k] #non-periodic dist calc
                    d += tmp * tmp
                #d = sqrt(d)
                k = nbins-1
                while d<=crbins[k]:
                    counts[k] += cweights1[i]*cweights2[j]
                    k=k-1
                    if k<0: break
                
    return np.asarray(counts)


@cython.boundscheck(False)
@cython.wraparound(False)
def pairwise_distances(double[:, ::1] data1, period=None):
    """
    Calculate the distance between all pairs of points.
    
    Parameters
    ----------
    data1: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period
            
    period: array_like, optional
        length k array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.
            
    Returns
    -------
    D: numpy.matrix
        N by N matrix with D[i,j] being the distance between points i and j.
    
    Notes
    -----
    The resulting matrix has shape (N,N), and for large N, the array may be too large to 
    store in memory.
     
    """
    
    #Process period entry and check for consistency.
    ii = len(np.shape(data1))-1
    if period is None:
            period = np.array([np.inf]*np.shape(data1)[ii])
            PBCs=False
    else:
        period = np.asarray(period).astype(np.float64)
        PBCs=True
        if np.shape(period) == ():
            period = np.array([period]*np.shape(data1)[ii])
        elif np.shape(period)[0] != np.shape(data1)[ii]:
            raise ValueError("period should have len == dimension of points")
    
    cdef np.ndarray[np.float64_t, ndim=1] cperiod = np.ascontiguousarray(period,dtype=np.float64)
    cdef int M = data1.shape[0]
    cdef int N = data1.shape[1]
    cdef double tmp, d
    cdef double[:, ::1] D = np.empty((M, M), dtype=np.float64)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = fabs(data1[i, k] - data1[j, k])
                tmp = fmin(tmp, cperiod[k] - tmp)
                #tmp = data1[i, k] - data1[j, k] #non-periodic dist calc
                d += tmp * tmp
            D[i, j] = sqrt(d)
    return np.asmatrix(D)