#!/usr/bin/env python

#Duncan Campbell
#Yale University
#July 24, 2014
#calculate pair counts with dumb brute force method as a sanity check.


from __future__ import division, print_function
from halotools.mock_observables.spatial.distances import euclidean_distance as distance
import numpy as np

__all__=['npairs','wnpairs','pairs']


def npairs(data1, data2, rbins, period=None):
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
    rbins = np.asarray(rbins)
    if rbins.size ==1: rbins = np.array([rbins])
    
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
    
    N1 = len(data1)
    N2 = len(data2)
    dd = np.zeros((N1*N2,)) #store radial pair seperations 
    for i in range(0,N1): #calculate distance between every point and every other point
        x1 = data1[i,:]
        x2 = data2
        dd[i*N2:i*N2+N2] = distance(x1, x2, period)
        
    #sort results
    dd.sort()
    #count number less than r
    n = np.zeros((rbins.size,), dtype=np.int)
    for i in range(rbins.size): #this is ugly... is there a sexier way?
        if rbins[i]>np.min(period)/2.0:
            print("Warning: counting pairs with seperations larger than period/2 is awkward.")
            print("r=", rbins[i], "  min(period)/2=",np.min(period)/2.0)
        n[i] = len(np.where(dd<=rbins[i])[0])
    
    return n


def wnpairs(data1, data2, r, period=None, weights1=None, weights2=None):
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
     
    """
    
    #work with arrays!
    data1 = np.asarray(data1)
    if data1.ndim ==1: data1 = np.array([data1])
    data2 = np.asarray(data2)
    if data2.ndim ==1: data2 = np.array([data2])
    r = np.asarray(r)
    if r.size == 1: r = np.array([r])
    
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
        if np.shape(period)[0] != np.shape(data1)[-1]:
            raise ValueError("period should have len == dimension of points")
            return None
    
    #Process weights1 entry and check for consistency.
    if weights1 is None:
            weights1 = np.array([1.0]*np.shape(data1)[0], dtype=np.float64)
    else:
        weights1 = np.asarray(weights1).astype("float64")
        if np.shape(weights1)[0] != np.shape(data1)[0]:
            raise ValueError("weights1 should have same len as data1")
            return None
    #Process weights2 entry and check for consistency.
    if weights2 is None:
            weights2 = np.array([1.0]*np.shape(data2)[0], dtype=np.float64)
    else:
        weights2 = np.asarray(weights2).astype("float64")
        if np.shape(weights2)[0] != np.shape(data2)[0]:
            raise ValueError("weights2 should have same len as data2")
            return None
    
    N1 = len(data1)
    N2 = len(data2)
    dd = np.zeros((N1,N2), dtype=np.float64) #store radial pair seperations 
    for i in range(0,N1): #calculate distance between every point and every other point
        x1 = data1[i,:]
        x2 = data2
        dd[i,:] = distance(x1, x2, period)
        
    #count number less than r
    n = np.zeros((r.size,), dtype=np.float64)
    for i in range(r.size): #this is ugly... is there a sexier way?
        if r[i]>np.min(period)/2:
            print("Warning: counting pairs with seperations larger than period/2 is awkward.")
            print("r=", r[i], "  min(period)/2=",np.min(period)/2)
        for j in range(N1):
            n[i] += np.sum(np.extract(dd[j,:]<=r[i],weights2))*weights1[j]
    
    return n


def pairs(data1, r, data2=None, period=None):
    """
    Calculate the pairs with separations less than or equal to rbins[i].
    
    Parameters
    ----------
    data1: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period
            
    r : float
        radius for which pairs are counted. 
        
    data2: array_like, optional
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period
            
    period: array_like, optional
        length k array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.
            
    Returns
    -------
    pairs : Set of pairs (i,j), with i < j
     
    """
    
    #work with arrays!
    data1 = np.asarray(data1)
    if data2==None:
        data2 = np.asarray(data1)
        self_match=False
    else:
        data2 = np.asarray(data2)
        self_match=True
    
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
        if np.shape(period)[0] != np.shape(data1)[-1]:
            raise ValueError("period should have len == dimension of points")
            return None
    
    N1 = len(data1)
    N2 = len(data2)
    dd = np.zeros((N1,N2)) #store radial pair seperations 
    for i in range(0,N1): #calculate distance between every point and every other point
        x1 = data1[i,:]
        x2 = data2
        dd[i,:] = distance(x1, x2, period)
    
    pairs = np.argwhere(dd<=r)
    
    spairs = set()
    for i in range(len(pairs)):
        if self_match==False:
            if pairs[i,0] != pairs[i,1]:
                spairs.add((min(pairs[i]),max(pairs[i])))
        if self_match==True:
            spairs.add((min(pairs[i]),max(pairs[i])))
    
    return spairs


