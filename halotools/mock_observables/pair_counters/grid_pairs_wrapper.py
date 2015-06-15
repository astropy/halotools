
"""
pair counting functions
"""


import numpy as np
from grid_pairs import *
from time import time


def npairs(data1, data2, rbins, Lbox=None, period=None, verbose=False):
    """
    real-space pair counter.
    Calculate the number of pairs with separations less than or equal to rbins[i].
    
    Parameters
    ----------
    data1: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period. This cython implementation requires data1.ndim==2.
            
    data2: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period. This cython implementation requires data2.ndim==2.
            
    rbins: array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
        len(rbins) = Nrbins + 1.
    
    Lbox: array_like, optional
        length of cube sides which encloses data1 and data2.
    
    period: array_like, optional
        length k array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.  If True, period is set to be Lbox
            
    Returns
    -------
    N_pairs : array of length len(rbins)
        number counts of pairs
    """
    
    #process input
    data1 = np.array(data1)
    data2 = np.array(data2)
    rbins = np.array(rbins)
    if np.all(period==np.inf): period=None
    
    #enforce shape requirements on input
    if (np.shape(data1)[1]!=3) | (data1.ndim>2):
        raise ValueError("data1 must be of shape (N,3)")
    if (np.shape(data2)[1]!=3) | (data2.ndim>2):
        raise ValueError("data2 must be of shape (N,3)")
    if rbins.ndim != 1:
        raise ValueError("rbins must be a 1D array")
    
    #process Lbox parameter
    if (Lbox is None) & (period is None): 
        data1, data2, Lbox = _enclose_in_box(data1, data2)
    elif (Lbox is None) & (period is not None):
        Lbox = period
    elif np.shape(Lbox)==():
        Lbox = np.array([Lbox]*3)
    elif np.shape(Lbox)==(1,):
        Lbox = np.array([Lbox[0]]*3)
    else: Lbox = np.array(Lbox)
    if np.shape(Lbox) != (3,):
        raise ValueError("Lbox must be an array of length 3, or number indicating the \
                          length of one side of a cube")
    
    #are we working with periodic boundary conditions (PBCs)?
    if period is None: 
        PBCs = False
    elif np.shape(period) == (3,):
        PBCs = True
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif np.shape(period) == (1,):
        period = np.array([period[0]]*3)
        PBCs = True
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif isinstance(period, (int, long, float, complex)):
        period = np.array([period]*3)
        PBCs = True
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif (period == True) & (Lbox is not None):
        PBCs = True
        period = Lbox
    elif (period == True) & (Lbox is None):
        raise ValueError("If period is set to True, Lbox must be defined.")
    else: PBCs=True
    
    #check to see we dont count pairs more than once
    if (PBCs==True) & np.any(np.max(rbins)>Lbox/2.0):
        raise ValueError('cannot count pairs with seperations \
                          larger than Lbox/2 with PBCs')
    
    #use cython functions to do pair counting
    if PBCs==False:
        counts = npairs_no_pbc(data1, data2, rbins, Lbox, verbose)
    else: #PBCs==True
        counts = npairs_pbc(data1, data2, rbins, Lbox, period, verbose)
    
    return counts
    

def xy_z_npairs(data1, data2, rp_bins, pi_bins, Lbox=None, period=None, verbose=False):
    """
    2+1D pair counter.
    Calculate the number of pairs with separations in the x-y plane less than or equal 
    to rp_bins[i], and separations in the z coordinate less than or equal to pi_bins[i].
    
    Parameters
    ----------
    data1: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period. This cython implementation requires data1.ndim==2.
            
    data2: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period. This cython implementation requires data2.ndim==2.
            
    rp_bins: array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
        len(rp_bins) = Nrp_bins + 1.
    
    pi_bins: array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
        len(pi_bins) = Npi_bins + 1.
    
    Lbox: array_like
        length of box sides.
    
    period: array_like, optional
        length k array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity. If True, period is set to be Lbox
            
    Returns
    -------
    N_pairs : ndarray of shape (len(rp_bins), len(pi_bins))
        number counts of pairs
    """
    
    #process input
    data1 = np.array(data1)
    data2 = np.array(data2)
    rp_bins = np.array(rp_bins)
    pi_bins = np.array(pi_bins)
    if np.all(period==np.inf): period=None
    
    #enforce shape requirements on input
    if (np.shape(data1)[1]!=3) | (data1.ndim>2):
        raise ValueError("data1 must be of shape (N,3)")
    if (np.shape(data2)[1]!=3) | (data2.ndim>2):
        raise ValueError("data2 must be of shape (N,3)")
    if rp_bins.ndim != 1:
        raise ValueError("rp_bins must be a 1D array")
    if pi_bins.ndim != 1:
        raise ValueError("pi_bins must be a 1D array")
    
    #process Lbox parameter
    if (Lbox is None) & (period is None): 
        data1, data2, Lbox = _enclose_in_box(data1, data2)
    elif (Lbox is None) & (period is not None):
        Lbox = period
    elif np.shape(Lbox)==():
        Lbox = np.array([Lbox]*3)
    elif np.shape(Lbox)==(1,):
        Lbox = np.array([Lbox[0]]*3)
    else: Lbox = np.array(Lbox)
    if np.shape(Lbox) != (3,):
        raise ValueError("Lbox must be an array of length 3, or number indicating the \
                          length of one side of a cube")
    
    #are we working with periodic boundary conditions (PBCs)?
    if period is None: 
        PBCs = False
    elif np.shape(period) == (3,):
        PBCs = True
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif np.shape(period) == (1,):
        period = np.array([period[0]]*3)
        PBCs = True
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif isinstance(period, (int, long, float, complex)):
        period = np.array([period]*3)
        PBCs = True
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif (period == True) & (Lbox is not None):
        PBCs = True
        period = Lbox
    elif (period == True) & (Lbox is None):
        raise ValueError("If period is set to True, Lbox must be defined.")
    else: PBCs=True
    
    #check to see we dont count pairs more than once    
    if (PBCs==True) & np.any(np.max(rp_bins)>Lbox[0:2]/2.0):
        raise ValueError('grid_pairs pair counter cannot count pairs with seperations\
                          larger than Lbox/2 with PBCs')
    if (PBCs==True) & np.any(np.max(pi_bins)>Lbox[2]/2.0):
        raise ValueError('grid_pairs pair counter cannot count pairs with seperations\
                          larger than Lbox/2 with PBCs')
    
    #use cython functions to do pair counting
    if PBCs==False:
        counts = xy_z_npairs_no_pbc(data1, data2, rp_bins, pi_bins, Lbox, verbose)
    else: #PBCs==True
        counts = xy_z_npairs_pbc(data1, data2, rp_bins, pi_bins, Lbox, period, verbose)
    
    return counts


def wnpairs(data1, data2, rbins, Lbox=None, period=None, weights1=None, weights2=None, verbose=False):
    """
    weighted real-space pair counter.
    Calculate the weighted number of pairs with separations less than or equal to 
    rbins[i].
    
    Parameters
    ----------
    data1: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period. This cython implementation requires data1.ndim==2.
            
    data2: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period. This cython implementation requires data2.ndim==2.
            
    rbins: array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
        len(rbins) = Nrbins + 1.
    
    Lbox: array_like, optional
        length of cube sides which encloses data1 and data2.
    
    period: array_like, optional
        length k array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.  If True, period is set to be Lbox
            
    Returns
    -------
    N_pairs : array of length len(rbins)
        number counts of pairs
    """
    
    #process input
    data1 = np.array(data1)
    data2 = np.array(data2)
    rbins = np.array(rbins)
    if np.all(period==np.inf): period=None
    
    #enforce shape requirements on input
    if (np.shape(data1)[1]!=3) | (data1.ndim>2):
        raise ValueError("data1 must be of shape (N,3)")
    if (np.shape(data2)[1]!=3) | (data2.ndim>2):
        raise ValueError("data2 must be of shape (N,3)")
    if rbins.ndim != 1:
        raise ValueError("rbins must be a 1D array")
    
    #process Lbox parameter
    if (Lbox is None) & (period is None): 
        data1, data2, Lbox = _enclose_in_box(data1, data2)
    elif (Lbox is None) & (period is not None):
        Lbox = period
    elif np.shape(Lbox)==():
        Lbox = np.array([Lbox]*3)
    elif np.shape(Lbox)==(1,):
        Lbox = np.array([Lbox[0]]*3)
    else: Lbox = np.array(Lbox)
    if np.shape(Lbox) != (3,):
        raise ValueError("Lbox must be an array of length 3, or number indicating the \
                          length of one side of a cube")
    
    #are we working with periodic boundary conditions (PBCs)?
    if period is None: 
        PBCs = False
    elif np.shape(period) == (3,):
        PBCs = True
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif np.shape(period) == (1,):
        period = np.array([period[0]]*3)
        PBCs = True
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif isinstance(period, (int, long, float, complex)):
        period = np.array([period]*3)
        PBCs = True
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif (period == True) & (Lbox is not None):
        PBCs = True
        period = Lbox
    elif (period == True) & (Lbox is None):
        raise ValueError("If period is set to True, Lbox must be defined.")
    else: PBCs=True
    
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
    
    #check to see we dont count pairs more than once
    if (PBCs==True) & np.any(np.max(rbins)>Lbox/2.0):
        raise ValueError('cannot count pairs with seperations \
                          larger than Lbox/2 with PBCs')
    
    #use cython functions to do pair counting
    if PBCs==False:
        counts = wnpairs_no_pbc(data1, data2, rbins, Lbox, weights1, weights2, verbose)
    else: #PBCs==True
        counts = wnpairs_pbc(data1, data2, rbins, Lbox, period, weights1, weights2, verbose)
    
    return counts


def xy_z_wnpairs(data1, data2, rp_bins, pi_bins, Lbox=[1.0,1.0,1.0], period=None, weights1=None, weights2=None, verbose=False):
    """
    weighted 2+1D pair counter.
    Calculate the weighted number of pairs with separations in the x-y plane less than or 
    equal to rp_bins[i], and separations in the z coordinate less than or equal to 
    pi_bins[i].
    
    Parameters
    ----------
    data1: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period. This cython implementation requires data1.ndim==2.
            
    data2: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period. This cython implementation requires data2.ndim==2.
            
    rp_bins: array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
        len(rp_bins) = Nrp_bins + 1.
    
    pi_bins: array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
        len(pi_bins) = Npi_bins + 1.
    
    Lbox: array_like
        length of box sides.
    
    period: array_like, optional
        length k array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity. If True, period is set to be Lbox
            
    Returns
    -------
    N_pairs : ndarray of shape (len(rp_bins), len(pi_bins))
        number counts of pairs
    """
    
    #process input
    data1 = np.array(data1)
    data2 = np.array(data2)
    rp_bins = np.array(rp_bins)
    pi_bins = np.array(pi_bins)
    if np.all(period==np.inf): period=None
    
    #enforce shape requirements on input
    if (np.shape(data1)[1]!=3) | (data1.ndim>2):
        raise ValueError("data1 must be of shape (N,3)")
    if (np.shape(data2)[1]!=3) | (data2.ndim>2):
        raise ValueError("data2 must be of shape (N,3)")
    if rp_bins.ndim != 1:
        raise ValueError("rp_bins must be a 1D array")
    if pi_bins.ndim != 1:
        raise ValueError("pi_bins must be a 1D array")
    
    #process Lbox parameter
    if (Lbox is None) & (period is None): 
        data1, data2, Lbox = _enclose_in_box(data1, data2)
    elif (Lbox is None) & (period is not None):
        Lbox = period
    elif np.shape(Lbox)==():
        Lbox = np.array([Lbox]*3)
    elif np.shape(Lbox)==(1,):
        Lbox = np.array([Lbox[0]]*3)
    else: Lbox = np.array(Lbox)
    if np.shape(Lbox) != (3,):
        raise ValueError("Lbox must be an array of length 3, or number indicating the \
                          length of one side of a cube")
    
    #are we working with periodic boundary conditions (PBCs)?
    if period is None: 
        PBCs = False
    elif np.shape(period) == (3,):
        PBCs = True
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif np.shape(period) == (1,):
        period = np.array([period[0]]*3)
        PBCs = True
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif isinstance(period, (int, long, float, complex)):
        period = np.array([period]*3)
        PBCs = True
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif (period == True) & (Lbox is not None):
        PBCs = True
        period = Lbox
    elif (period == True) & (Lbox is None):
        raise ValueError("If period is set to True, Lbox must be defined.")
    else: PBCs=True
    
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
    
    #check to see we dont count pairs more than once    
    if (PBCs==True) & np.any(np.max(rp_bins)>Lbox[0:2]/2.0):
        raise ValueError('grid_pairs pair counter cannot count pairs with seperations\
                          larger than Lbox/2 with PBCs')
    if (PBCs==True) & np.any(np.max(pi_bins)>Lbox[2]/2.0):
        raise ValueError('grid_pairs pair counter cannot count pairs with seperations\
                          larger than Lbox/2 with PBCs')
    
    #use cython functions to do pair counting
    if PBCs==False:
        counts = xy_z_wnpairs_no_pbc(data1, data2, rp_bins, pi_bins, Lbox, weights1,\
                                  weights2, verbose)
    else: #PBCs==True
        counts = xy_z_wnpairs_pbc(data1, data2, rp_bins, pi_bins, Lbox, period, weights1,\
                                  weights2, verbose)
    
    return counts


def jnpairs(data1, data2, rbins, Lbox=None, period=None, weights1=None, weights2=None,\
            jtags1=None, jtags2=None, N_samples=1, verbose=False):
    """
    jackknife weighted real-space pair counter.
    Calculate the weighted number of pairs with separations less than or equal to rbins[i]
    for a jackknife sample.
    
    Parameters
    ----------
    data1: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period. This cython implementation requires data1.ndim==2.
            
    data2: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period. This cython implementation requires data2.ndim==2.
            
    rbins: array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
        len(rbins) = Nrbins + 1.
    
    Lbox: array_like, optional
        length of cube sides which encloses data1 and data2.
    
    period: array_like, optional
        length k array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.  If True, period is set to be Lbox
    
    weights1: array_like, optional
        length N1 array containing weights used for weighted pair counts
        
    weights2: array_like, optional
        length N2 array containing weights used for weighted pair counts.
    
    jtags1: array_like, optional
        length N1 array containing integer tags used to define jackknife sample membership
        
    jtags2: array_like, optional
        length N2 array containing integer tags used to define jackknife sample membership
    
    N_samples: int, optional
        number of jackknife samples
        
    Returns
    -------
    N_pairs : ndarray of shape (N_samples+1,len(rbins))
        number counts of pairs with seperations <=rbins[i]
    """
    
    #process input
    data1 = np.array(data1)
    data2 = np.array(data2)
    rbins = np.array(rbins)
    if np.all(period==np.inf): period=None
    
    #enforce shape requirements on input
    if (np.shape(data1)[1]!=3) | (data1.ndim>2):
        raise ValueError("data1 must be of shape (N,3)")
    if (np.shape(data2)[1]!=3) | (data2.ndim>2):
        raise ValueError("data2 must be of shape (N,3)")
    if rbins.ndim != 1:
        raise ValueError("rbins must be a 1D array")
    
    #process Lbox parameter
    if (Lbox is None) & (period is None): 
        data1, data2, Lbox = _enclose_in_box(data1, data2)
    elif (Lbox is None) & (period is not None):
        Lbox = period
    elif np.shape(Lbox)==():
        Lbox = np.array([Lbox]*3)
    elif np.shape(Lbox)==(1,):
        Lbox = np.array([Lbox[0]]*3)
    else: Lbox = np.array(Lbox)
    if np.shape(Lbox) != (3,):
        raise ValueError("Lbox must be an array of length 3, or number indicating the \
                          length of one side of a cube")
    
    #are we working with periodic boundary conditions (PBCs)?
    if period is None: 
        PBCs = False
    elif np.shape(period) == (3,):
        PBCs = True
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif np.shape(period) == (1,):
        period = np.array([period[0]]*3)
        PBCs = True
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif isinstance(period, (int, long, float, complex)):
        period = np.array([period]*3)
        PBCs = True
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif (period == True) & (Lbox is not None):
        PBCs = True
        period = Lbox
    elif (period == True) & (Lbox is None):
        raise ValueError("If period is set to True, Lbox must be defined.")
    else: PBCs=True
    
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
    
    #Process jtags_1 entry and check for consistency.
    if jtags1 is None:
            jtags1 = np.array([0]*np.shape(data1)[0], dtype=np.int)
    else:
        jtags1 = np.asarray(jtags1).astype("int")
        if np.shape(jtags1)[0] != np.shape(data1)[0]:
            raise ValueError("jtags1 should have same len as data1")
    #Process jtags_2 entry and check for consistency.
    if jtags2 is None:
            jtags2 = np.array([0]*np.shape(data2)[0], dtype=np.int)
    else:
        jtags2 = np.asarray(jtags2).astype("int")
        if np.shape(jtags2)[0] != np.shape(data2)[0]:
            raise ValueError("jtags2 should have same len as data2")
    
    if type(N_samples) is not int: 
        raise ValueError("There must be an integer number of jackknife samples")
    if np.max(jtags1)>N_samples:
        raise ValueError("There are more jackknife samples than indicated by N_samples")
    if np.max(jtags2)>N_samples:
        raise ValueError("There are more jackknife samples than indicated by N_samples")
    
    #use cython functions to do pair counting
    if PBCs==False:
        counts = jnpairs_no_pbc(data1, data2, rbins, Lbox, weights1, weights2,\
                             jtags1, jtags2, N_samples, verbose)
    else: #PBCs==True
        counts = jnpairs_pbc(data1, data2, rbins, Lbox, period, weights1, weights2,\
                             jtags1, jtags2, N_samples, verbose)
    
    return counts


def xy_z_jnpairs(data1, data2, rp_bins, pi_bins, Lbox=[1.0,1.0,1.0], period=None,\
                 weights1=None, weights2=None, jtags1=None, jtags2=None, N_samples=1, verbose=False):
    """
    jackknife weighted 2+1D pair counter.
    Calculate the weighted number of pairs with separations in the x-y plane less than or 
    equal to rp_bins[i], and separations in the z coordinate less than or equal to 
    pi_bins[i] for a jackknife sample.
    
    Parameters
    ----------
    data1: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period. This cython implementation requires data1.ndim==2.
            
    data2: array_like
        N by k numpy array of k-dimensional positions. Should be between zero and 
        period. This cython implementation requires data2.ndim==2.
            
    rp_bins: array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
        len(rp_bins) = Nrp_bins + 1.
    
    pi_bins: array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
        len(pi_bins) = Npi_bins + 1.
    
    Lbox: array_like
        length of box sides.
    
    period: array_like, optional
        length k array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity. If True, period is set to be Lbox
    
    weights1: array_like, optional
        length N1 array containing weights used for weighted pair counts
        
    weights2: array_like, optional
        length N2 array containing weights used for weighted pair counts.
    
    jtags1: array_like, optional
        length N1 array containing integer tags used to define jackknife sample membership
        
    jtags2: array_like, optional
        length N2 array containing integer tags used to define jackknife sample membership
    
    N_samples: int, optional
        number of jackknife samples
    
    Returns
    -------
    N_pairs : ndarray of shape (N_samples+1, len(rp_bins), len(pi_bins))
        number counts of pairs with separations <= rp_bins[i], pi_bins[j]
    """
    
    #process input
    data1 = np.array(data1)
    data2 = np.array(data2)
    rp_bins = np.array(rp_bins)
    pi_bins = np.array(pi_bins)
    if np.all(period==np.inf): period=None
    
    #enforce shape requirements on input
    if (np.shape(data1)[1]!=3) | (data1.ndim>2):
        raise ValueError("data1 must be of shape (N,3)")
    if (np.shape(data2)[1]!=3) | (data2.ndim>2):
        raise ValueError("data2 must be of shape (N,3)")
    if rp_bins.ndim != 1:
        raise ValueError("rp_bins must be a 1D array")
    if pi_bins.ndim != 1:
        raise ValueError("pi_bins must be a 1D array")
    
    #process Lbox parameter
    if (Lbox is None) & (period is None): 
        data1, data2, Lbox = _enclose_in_box(data1, data2)
    elif (Lbox is None) & (period is not None):
        Lbox = period
    elif np.shape(Lbox)==():
        Lbox = np.array([Lbox]*3)
    elif np.shape(Lbox)==(1,):
        Lbox = np.array([Lbox[0]]*3)
    else: Lbox = np.array(Lbox)
    if np.shape(Lbox) != (3,):
        raise ValueError("Lbox must be an array of length 3, or number indicating the \
                          length of one side of a cube")
    
    #are we working with periodic boundary conditions (PBCs)?
    if period is None: 
        PBCs = False
    elif np.shape(period) == (3,):
        PBCs = True
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif np.shape(period) == (1,):
        period = np.array([period[0]]*3)
        PBCs = True
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif isinstance(period, (int, long, float, complex)):
        period = np.array([period]*3)
        PBCs = True
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif (period == True) & (Lbox is not None):
        PBCs = True
        period = Lbox
    elif (period == True) & (Lbox is None):
        raise ValueError("If period is set to True, Lbox must be defined.")
    else: PBCs=True
    
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
    
    #Process jtags_1 entry and check for consistency.
    if jtags1 is None:
            jtags1 = np.array([0]*np.shape(data1)[0], dtype=np.int)
    else:
        jtags1 = np.asarray(jtags1).astype("int")
        if np.shape(jtags1)[0] != np.shape(data1)[0]:
            raise ValueError("jtags1 should have same len as data1")
    #Process jtags_2 entry and check for consistency.
    if jtags2 is None:
            jtags2 = np.array([0]*np.shape(data2)[0], dtype=np.int)
    else:
        jtags2 = np.asarray(jtags2).astype("int")
        if np.shape(jtags2)[0] != np.shape(data2)[0]:
            raise ValueError("jtags2 should have same len as data2")
    
    if type(N_samples) is not int: 
        raise ValueError("There must be an integer number of jackknife samples")
    if np.max(jtags1)>N_samples:
        raise ValueError("There are more jackknife samples than indicated by N_samples")
    if np.max(jtags2)>N_samples:
        raise ValueError("There are more jackknife samples than indicated by N_samples")
    
    #use cython functions to do pair counting
    if PBCs==False:
        counts = xy_z_jnpairs_no_pbc(data1, data2, rp_bins, pi_bins, Lbox,\
                                     weights1, weights2, jtags1, jtags2, N_samples,\
                                     verbose)
    else: #PBCs==True
        counts = xy_z_jnpairs_pbc(data1, data2, rp_bins, pi_bins, Lbox, period,\
                                  weights1, weights2, jtags1, jtags2, N_samples,\
                                  verbose)
    
    return counts


def _enclose_in_box(data1, data2):
    """
    build axis aligned box which encloses all points. 
    shift points so cube's origin is at 0,0,0.
    """
    xmin = np.min([np.min(data1[:,0]),np.min(data2[:,0])])
    ymin = np.min([np.min(data1[:,1]),np.min(data2[:,1])])
    zmin = np.min([np.min(data1[:,2]),np.min(data2[:,2])])
    xmax = np.max([np.max(data1[:,0]),np.max(data2[:,0])])
    ymax = np.max([np.max(data1[:,1]),np.max(data2[:,1])])
    zmax = np.max([np.max(data1[:,2]),np.max(data2[:,2])])
    xyzmin = np.min([xmin,ymin,zmin])
    xyzmax = np.min([xmax,ymax,zmax])-xyzmin
    data1 = data1-xyzmin
    data2 = data2-xyzmin
    Lbox = np.array([xyzmax]*3)
    
    return data1, data2, Lbox

##########################################################################################
def main():
    """
    run this main program to get timed tests of the pair counter.
    """
    
    _test_npairs_speed()
    _test_wnpairs_speed()
    _test_xy_z_npairs_speed()
    _test_xy_z_wnpairs_speed()
    _test_jnpairs_speed()


def _test_npairs_speed():

    "bolshoi like test out to ~20 Mpc"
    Npts = 1e5
    Lbox = [250.0,250.0,250.0]
    period = np.array(Lbox)
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    
    rbins = np.logspace(-2,1.3)
    
    print("##########npairs##########")
    print("running speed test with {0} points".format(Npts))
    print("in {0} x {1} x {2} box.".format(Lbox[0],Lbox[1],Lbox[2]))
    print("to maximum seperation {0}".format(np.max(rbins)))

    #w/ PBCs
    start = time()
    result = npairs(data1, data1, rbins, Lbox=Lbox, period=period)
    end = time()
    runtime = end-start
    print("Total runtime (PBCs) = %.1f seconds" % runtime)

    #w/o PBCs
    start = time()
    result = npairs(data1, data1, rbins, Lbox=Lbox, period=None)
    end = time()
    runtime = end-start
    print("Total runtime (no PBCs) = %.1f seconds" % runtime)
    print("########################## \n")


def _test_wnpairs_speed():

    "bolshoi like test out to ~20 Mpc"
    Npts = 1e5
    Lbox = [250.0,250.0,250.0]
    period = np.array(Lbox)
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    weights1 = np.random.random(Npts)
    
    rbins = np.logspace(-2,1.3)
    
    print("##########wnpairs##########")
    print("running speed test with {0} points".format(Npts))
    print("in {0} x {1} x {2} box.".format(Lbox[0],Lbox[1],Lbox[2]))
    print("to maximum seperation {0}".format(np.max(rbins)))

    #w/ PBCs
    start = time()
    result = wnpairs(data1, data1, rbins, Lbox=Lbox, period=period, weights1=weights1, weights2=weights1)
    end = time()
    runtime = end-start
    print("Total runtime (PBCs) = %.1f seconds" % runtime)

    #w/o PBCs
    start = time()
    result = wnpairs(data1, data1, rbins, Lbox=Lbox, period=None, weights1=weights1, weights2=weights1)
    end = time()
    runtime = end-start
    print("Total runtime (no PBCs) = %.1f seconds" % runtime)
    print("########################### \n")


def _test_xy_z_npairs_speed():

    "bolshoi like test out to ~20 Mpc"
    Npts = 1e5
    Lbox = [250.0,250.0,250.0]
    period = np.array(Lbox)
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    
    rp_bins = np.logspace(-2,1.3)
    pi_bins = np.linspace(0,50,20)
    
    print("##########xy_z_npairs##########")
    print("running speed test with {0} points".format(Npts))
    print("in {0} x {1} x {2} box.".format(Lbox[0],Lbox[1],Lbox[2]))
    print("to maximum seperation {0}, {1}".format(np.max(rp_bins),np.max(pi_bins)))

    #w/ PBCs
    start = time()
    result = xy_z_npairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=period)
    end = time()
    runtime = end-start
    print("Total runtime (PBCs) = %.1f seconds" % runtime)

    #w/o PBCs
    start = time()
    result = xy_z_npairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=None)
    end = time()
    runtime = end-start
    print("Total runtime (no PBCs) = %.1f seconds" % runtime)
    print("############################### \n")


def _test_xy_z_wnpairs_speed():

    "bolshoi like test out to ~20 Mpc"
    Npts = 1e5
    Lbox = [250.0,250.0,250.0]
    period = np.array(Lbox)
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    weights1 = np.random.random(Npts)
    
    rp_bins = np.logspace(-2,1.3)
    pi_bins = np.linspace(0,50,20)
    
    print("##########xy_z_wnpairs##########")
    print("running speed test with {0} points".format(Npts))
    print("in {0} x {1} x {2} box.".format(Lbox[0],Lbox[1],Lbox[2]))
    print("to maximum seperation {0}, {1}".format(np.max(rp_bins),np.max(pi_bins)))

    #w/ PBCs
    start = time()
    result = xy_z_wnpairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=period,\
                         weights1=weights1, weights2=weights1)
    end = time()
    runtime = end-start
    print("Total runtime (PBCs) = %.1f seconds" % runtime)

    #w/o PBCs
    start = time()
    result = xy_z_wnpairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=None,\
                          weights1=weights1, weights2=weights1)
    end = time()
    runtime = end-start
    print("Total runtime (no PBCs) = %.1f seconds" % runtime)
    print("################################ \n")


def _test_jnpairs_speed():

    "bolshoi like test out to ~20 Mpc"
    Npts = 1e5
    Nsamples=5*5*5
    Lbox = [250.0,250.0,250.0]
    period = np.array(Lbox)
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    weights1 = np.random.random(Npts)
    jtags1 = np.sort(np.random.random_integers(1, Nsamples, size=Npts))
    
    rbins = np.logspace(-2,1.3)
    
    print("##########jnpairs##########")
    print("running speed test with {0} points".format(Npts))
    print("{0} jackknife samples".format(Npts))
    print("in {0} x {1} x {2} box.".format(Lbox[0],Lbox[1],Lbox[2]))
    print("to maximum seperation {0}".format(np.max(rbins)))

    #w/ PBCs
    start = time()
    result = jnpairs(data1, data1, rbins, Lbox=Lbox, period=period, weights1=weights1, weights2=weights1, jtags1=jtags1, jtags2=jtags1, N_samples=Nsamples)
    end = time()
    runtime = end-start
    print("Total runtime (PBCs) = %.1f seconds" % runtime)

    #w/o PBCs
    start = time()
    result = jnpairs(data1, data1, rbins, Lbox=Lbox, period=None, weights1=weights1, weights2=weights1, jtags1=jtags1, jtags2=jtags1, N_samples=Nsamples)
    end = time()
    runtime = end-start
    print("Total runtime (no PBCs) = %.1f seconds" % runtime)
    print("########################### \n")

if __name__ == '__main__':
    main()
