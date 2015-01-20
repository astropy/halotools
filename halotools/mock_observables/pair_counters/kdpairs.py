#!/usr/bin/env python

#Duncan Campbell
#Yale University
#October 6, 2014
#Calculate pair counts with a kdtree structure.


from __future__ import division, print_function
from spatial.kdtrees.ckdtree import cKDTree
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
    
    tree_1 = cKDTree(data1)
    tree_2 = cKDTree(data2)
    
    n = tree_1.count_neighbors(tree_2,rbins,period=period)
    
    return n


def wnpairs(data1, data2, rbins, period=None, weights1=None, weights2=None):
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
    
    tree_1 = cKDTree(data1)
    tree_2 = cKDTree(data2)
    
    n = tree_1.wcount_neighbors(tree_2, rbins, period=period, \
                                sweight=weights1, oweight=weights2)
    
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
    
    tree_1 = cKDTree(data1)
    
    pairs = tree_1.query_ball_tree(tree_2, r, period=period)
    
    spairs = set()
    for i in range(len(pairs)):
        if self_match==False:
            if pairs[i,0] != pairs[i,1]:
                spairs.add((min(pairs[i]),max(pairs[i])))
        if self_match==True:
            spairs.add((min(pairs[i]),max(pairs[i])))
    
    return spairs


def jnpairs(data1, data2, rbins, period=None, weights1=None,  weights2=None, N_vol_elements=None):
    """
    Calculate the number of jackknife pairs with separations less than or equal to rbins[i].
    
    Parameters
    ----------
    data1: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period
            
    data2: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period
    
    weights1: array_like
        lenght N numpy array of integer sub volume labels. Should be between integer in
        range [1,N_vol_elemtns] 
            
    weights2: array_like
        length N numpy array of integer sub volume labels. Should be between integer in 
        range [1,N_vol_elemtns] 
            
    rbins : array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
        len(rbins) = Nrbins + 1.
            
    period: array_like, optional
        length k array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.
    
    N_vol_elements: int
        number of jackknife samples
            
    Returns
    -------
    N_pairs : array of length shape(N_vol_elements+1,len(rbins))
        jackknife weighted number counts of pairs
     
    """
    
    wdim = N_vol_elements+1
    
    #work with arrays!
    data1 = np.asarray(data1)
    if data1.ndim ==1: data1 = np.array([data1])
    data2 = np.asarray(data2)
    if data2.ndim ==1: data2 = np.array([data2])
    rbins = np.asarray(rbins)
    if rbins.size ==1: rbins = np.array([rbins])
    
    #Check to make sure both data sets have the same dimension. Otherwise, throw an error!
    if np.shape(data1)[-1]!=np.shape(data2)[-1]:
        print(np.shape(data1),np.shape(data2))
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
    
    tree_1 = cKDTree(data1)
    tree_2 = cKDTree(data2)
    
    n = tree_1.wcount_neighbors_custom_2D(tree_2, rbins, period=period, \
                                 sweights=weights1, oweights=weights2, w=None, wdim=wdim)
    
    return n

