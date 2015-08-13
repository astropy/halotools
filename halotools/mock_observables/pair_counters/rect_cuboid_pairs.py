# -*- coding: utf-8 -*-

"""
rectangular Cuboid Pair Counter. 

This module contains pair counting functions used to count the number of pairs with 
separations less than or equal to r, optimized for simulation boxes.

This module also contains a 'main' function which runs speed tests.
"""

from __future__ import print_function, division
import numpy as np
from rect_cuboid import *
from cpairs import *
from time import time
import sys
import multiprocessing
from functools import partial


__all__=['npairs', 'wnpairs', 'jnpairs', 'xy_z_npairs', 'xy_z_wnpairs', 'xy_z_jnpairs']
__author__=['Duncan Campbell']


def npairs(data1, data2, rbins, Lbox=None, period=None, verbose=False, N_threads=1):
    """
    real-space pair counter.
    
    Count the number of pairs (x1,x2) that can be formed, with x1 drawn from data1 and x2
    drawn from data2, and where distance(x1, x2) <= rbins[i]. 
    
    Parameters
    ----------
    data1: array_like
        N1 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period.
            
    data2: array_like
        N2 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period.
            
    rbins: array_like
        numpy array of boundaries defining the bins in which pairs are counted.
    
    Lbox: array_like, optional
        length of cube sides which encloses data1 and data2.
    
    period: array_like, optional
        length 3 array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*3).
        If none, PBCs are set to infinity.  If True, period is set to be Lbox
    
    verbose: Boolean, optional
        If True, print out information and progress.
    
    N_threads: int, optional
        number of 'threads' to use in the pair counting.  if set to 'max', use all 
        available cores.  N_threads=0 is the default.
    
    Returns
    -------
    N_pairs : array of length len(rbins)
        number of pairs
    """
    
    if N_threads is not 1:
        if N_threads=='max':
            N_threads = multiprocessing.cpu_count()
        if isinstance(N_threads,int):
            pool = multiprocessing.Pool(N_threads)
        else: return ValueError("N_threads argument must be an integer number or 'max'")
    
    #process input
    data1 = np.array(data1)
    data2 = np.array(data2)
    rbins = np.array(rbins)
    if np.all(period==np.inf): period=None
    
    #enforce shape requirements on input
    if (np.shape(data1)[1]!=3) | (data1.ndim>2):
        raise ValueError("data1 must be of shape (Npts,3)")
    if (np.shape(data2)[1]!=3) | (data2.ndim>2):
        raise ValueError("data2 must be of shape (Npts,3)")
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
    
    #build grids for data1 and data2
    cell_size = np.array([np.max(rbins)]*3)
    grid1 = rect_cuboid_cells(data1[:,0], data1[:,1], data1[:,2], Lbox, cell_size)
    grid2 = rect_cuboid_cells(data2[:,0], data2[:,1], data2[:,2], Lbox, cell_size)
    
    #square radial bins to make distance calculation cheaper
    rbins = rbins**2.0
    
    #print come information
    if verbose==True:
        print("running grid pairs with {0} by {1} points".format(len(data1),len(data2)))
        print("cell size= {0}".format(grid1.dL))
        print("number of cells = {0}".format(np.prod(grid1.num_divs)))
    
    #number of cells
    Ncell1 = np.prod(grid1.num_divs)
    
    #create a function to call with only one argument
    engine = partial(_npairs_engine, grid1, grid2, rbins, period, PBCs)
    
    #do the pair counting
    if N_threads>1:
        counts = np.sum(pool.map(engine,range(Ncell1)),axis=0)
        pool.close()
    if N_threads==1:
        counts = np.sum(map(engine,range(Ncell1)),axis=0)


    
    return counts


def _npairs_engine(grid1, grid2, rbins, period, PBCs, icell1):
    """
    pair counting engine for npairs function.  This code calls a cython function.
    """
    
    counts = np.zeros(len(rbins))
    
    #extract the points in the cell
    x_icell1, y_icell1, z_icell1 = (grid1.x[grid1.slice_array[icell1]],\
                                    grid1.y[grid1.slice_array[icell1]],\
                                    grid1.z[grid1.slice_array[icell1]])
        
    #get the list of neighboring cells
    ix1, iy1, iz1 = np.unravel_index(icell1,(grid1.num_divs[0],\
                                             grid1.num_divs[1],\
                                             grid1.num_divs[2]))
    adj_cell_arr = grid1.adjacent_cells(ix1, iy1, iz1)
            
    #Loop over each of the (up to) 27 subvolumes neighboring, including the current cell.
    for icell2 in adj_cell_arr:
                
        #extract the points in the cell
        x_icell2 = grid2.x[grid2.slice_array[icell2]]
        y_icell2 = grid2.y[grid2.slice_array[icell2]]
        z_icell2 = grid2.z[grid2.slice_array[icell2]]
            
        #use cython functions to do pair counting
        if PBCs==False:
            counts += npairs_no_pbc(x_icell1, y_icell1, z_icell1,\
                                    x_icell2, y_icell2, z_icell2,\
                                    rbins)
        else: #PBCs==True
            counts += npairs_pbc(x_icell1, y_icell1, z_icell1,\
                                 x_icell2, y_icell2, z_icell2,\
                                 rbins, period)
    return counts



def wnpairs(data1, data2, rbins, Lbox=None, period=None, weights1=None, weights2=None,\
            verbose=False, N_threads=1):
    """
    weighted real-space pair counter.
    
    Count the weighted number of pairs (x1,x2) that can be formed, with x1 drawn from 
    data1 and x2 drawn from data2, and where distance(x1, x2) <= rbins[i].  Weighted 
    counts are calculated as w1*w2.
    
    Parameters
    ----------
    data1: array_like
        N1 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period. This cython implementation requires data1.ndim==2.
            
    data2: array_like
        N2 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period. This cython implementation requires data2.ndim==2.
            
    rbins: array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
    
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
    
    verbose: Boolean, optional
        If True, print out information and progress.
    
    N_threads: int, optional
        number of 'threads' to use in the pair counting.  if set to 'max', use all 
        available cores.  N_threads=0 is the default.
        
    Returns
    -------
    N_pairs : array of length len(rbins)
        number counts of pairs
    """
    
    if N_threads is not 1:
        if N_threads=='max':
            N_threads = multiprocessing.cpu_count()
        if isinstance(N_threads,int):
            pool = multiprocessing.Pool(N_threads)
        else: return ValueError("N_threads argument must be an integer number or 'max'")
    
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
    
    #build grids for data1 and data2
    cell_size = np.array([np.max(rbins)]*3)
    grid1 = rect_cuboid_cells(data1[:,0], data1[:,1], data1[:,2], Lbox, cell_size)
    grid2 = rect_cuboid_cells(data2[:,0], data2[:,1], data2[:,2], Lbox, cell_size)
    
    #sort the weights arrays
    weights1 = weights1[grid1.idx_sorted]
    weights2 = weights2[grid2.idx_sorted]
    
    #square radial bins to make distance calculation cheaper
    rbins = rbins**2.0
    
    #number of cells
    Ncell1 = np.prod(grid1.num_divs)
    
    #create a function to call with only one argument
    engine = partial(_wnpairs_engine, grid1, grid2, weights1, weights2, rbins, period, PBCs)
    
    #do the pair counting
    if N_threads>1:
        counts = np.sum(pool.map(engine,range(Ncell1)),axis=0)
    if N_threads==1:
        counts = np.sum(map(engine,range(Ncell1)),axis=0)
    
    return counts


def _wnpairs_engine(grid1, grid2, weights1, weights2, rbins, period, PBCs, icell1):
    
    counts = np.zeros(len(rbins))
    
    #extract the points in the cell
    x_icell1, y_icell1, z_icell1 = (grid1.x[grid1.slice_array[icell1]],\
                                    grid1.y[grid1.slice_array[icell1]],\
                                    grid1.z[grid1.slice_array[icell1]])
        
    #extract the weights in the cell
    w_icell1 = weights1[grid1.slice_array[icell1]]
        
    #get the list of neighboring cells
    ix1, iy1, iz1 = np.unravel_index(icell1,(grid1.num_divs[0],\
                                             grid1.num_divs[1],\
                                             grid1.num_divs[2]))
    adj_cell_arr = grid1.adjacent_cells(ix1, iy1, iz1)
        
    #Loop over each of the 27 subvolumes neighboring, including the current cell.
    for icell2 in adj_cell_arr:
            
        ix2, iy2, iz2 = np.unravel_index(icell2,(grid2.num_divs[0],\
                                                 grid2.num_divs[1],\
                                                 grid2.num_divs[2]))
        
        #extract the points in the cell
        x_icell2 = grid2.x[grid2.slice_array[icell2]]
        y_icell2 = grid2.y[grid2.slice_array[icell2]]
        z_icell2 = grid2.z[grid2.slice_array[icell2]]
        
        #extract the weights in the cell
        w_icell2 = weights2[grid2.slice_array[icell2]]
        
        #use cython functions to do pair counting
        if PBCs==False:
            counts += wnpairs_no_pbc(x_icell1, y_icell1, z_icell1,\
                                     x_icell2, y_icell2, z_icell2,\
                                     w_icell1, w_icell2,\
                                     rbins)
        else: #PBCs==True
            counts += wnpairs_pbc(x_icell1, y_icell1, z_icell1,\
                                  x_icell2, y_icell2, z_icell2,\
                                  w_icell1, w_icell2,\
                                  rbins, period)
    return counts


def jnpairs(data1, data2, rbins, Lbox=None, period=None, weights1=None, weights2=None,\
            jtags1=None, jtags2=None, N_samples=0, verbose=False, N_threads=1):
    """
    jackknife weighted real-space pair counter.
    
    Count the weighted number of pairs (x1,x2) that can be formed, with x1 drawn from 
    data1 and x2 drawn from data2, and where distance(x1, x2) <= rbins[i].  Weighted 
    counts are calculated as w1*w2. Jackknife sampled pair counts are returned.
    
    Parameters
    ----------
    data1: array_like
        N1 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period. This cython implementation requires data1.ndim==2.
            
    data2: array_like
        N2 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period. This cython implementation requires data2.ndim==2.
            
    rbins: array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
    
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
        length N1 array containing integer tags used to define jackknife sample 
        membership. Tags are in the range [1,N_samples]. '0' is a reserved tag and should 
        not be used.
        
    jtags2: array_like, optional
        length N2 array containing integer tags used to define jackknife sample 
        membership. Tags are in the range [1,N_samples]. '0' is a reserved tag and should 
        not be used.
    
    N_samples: int, optional
        number of jackknife samples
    
    verbose: Boolean, optional
        If True, print out information and progress.
    
    N_threads: int, optional
        number of 'threads' to use in the pair counting.  If set to 'max', use all 
        available cores.  N_threads=0 is the default.
        
    Returns
    -------
    N_pairs : ndarray of shape (N_samples+1,len(rbins))
        number counts of pairs with seperations <=rbins[i]
    
    Notes
    -----
    Jackknife weights are calculated using a weighting function.
    
    if both points are outside the sample, return 0.0
    if both points are inside the sample, return (w1 * w2)
    if one point is inside, and the other is outside return 0.5*(w1 * w2)
    """
    
    if N_threads is not 1:
        if N_threads=='max':
            N_threads = multiprocessing.cpu_count()
        if isinstance(N_threads,int):
            pool = multiprocessing.Pool(N_threads)
        else: return ValueError("N_threads argument must be an integer number or 'max'")
    
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
                          length of one side of a rectangular cuboid")
    
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
    
    #Check bounds of jackknife tags
    if np.min(jtags1)<1: raise ValueError("jtags1 must be >=1")
    if np.min(jtags2)<1: raise ValueError("jtags2 must be >=1")
    if np.max(jtags1)>N_samples: raise ValueError("jtags1 must be <=N_samples")
    if np.max(jtags2)>N_samples: raise ValueError("jtags2 must be <=N_samples")
    
    #throw warning if some tags do not exist
    if not np.array_equal(np.unique(jtags1),np.arange(1,N_samples+1)):
        print("Warning: data1 does not contain points in every jackknife sample.")
    if not np.array_equal(np.unique(jtags1),np.arange(1,N_samples+1)):
        print("Warning: data2 does not contain points in every jackknife sample.")
    
    if type(N_samples) is not int: 
        raise ValueError("There must be an integer number of jackknife samples")
    if np.max(jtags1)>N_samples:
        raise ValueError("There are more jackknife samples than indicated by N_samples")
    if np.max(jtags2)>N_samples:
        raise ValueError("There are more jackknife samples than indicated by N_samples")
    
    #build grids for data1 and data2
    cell_size = np.array([np.max(rbins)]*3)
    grid1 = rect_cuboid_cells(data1[:,0], data1[:,1], data1[:,2], Lbox, cell_size)
    grid2 = rect_cuboid_cells(data2[:,0], data2[:,1], data2[:,2], Lbox, cell_size)
    
    #sort the weights arrays
    weights1 = weights1[grid1.idx_sorted]
    weights2 = weights2[grid2.idx_sorted]
        
    #sort the jackknife tag arrays
    jtags1 = jtags1[grid1.idx_sorted]
    jtags2 = jtags2[grid2.idx_sorted]
    
    #square radial bins to make distance calculation cheaper
    rbins = rbins**2.0
    
    #Loop over all subvolumes in grid1
    Ncell1 = np.prod(grid1.num_divs)
    
    #create a function to call with only one argument
    engine = partial(_jnpairs_engine, grid1, grid2, weights1, weights2, jtags1, jtags2,\
                     N_samples, rbins, period, PBCs)
    
    #do the pair counting
    if N_threads>1:
        counts = np.sum(pool.map(engine,range(Ncell1)),axis=0)
    if N_threads==1:
        counts = np.sum(map(engine,range(Ncell1)),axis=0)
    
    return counts


def _jnpairs_engine(grid1, grid2, weights1, weights2, jtags1, jtags2, N_samples, rbins,\
                    period, PBCs, icell1):
    
    counts = np.zeros((N_samples+1,len(rbins)))
    
    #extract the points in the cell
    x_icell1, y_icell1, z_icell1 = (grid1.x[grid1.slice_array[icell1]],\
                                    grid1.y[grid1.slice_array[icell1]],\
                                    grid1.z[grid1.slice_array[icell1]])
        
    #extract the weights in the cell
    w_icell1 = weights1[grid1.slice_array[icell1]]
        
    #extract the jackknife tags in the cell
    j_icell1 = jtags1[grid1.slice_array[icell1]]
        
    #get the list of neighboring cells
    ix1, iy1, iz1 = np.unravel_index(icell1,(grid1.num_divs[0],\
                                             grid1.num_divs[1],\
                                             grid1.num_divs[2]))
    adj_cell_arr = grid1.adjacent_cells(ix1, iy1, iz1)
        
    #Loop over each of the 27 subvolumes neighboring, including the current cell.
    for icell2 in adj_cell_arr:
            
        ix2, iy2, iz2 = np.unravel_index(icell2,(grid2.num_divs[0],\
                                                 grid2.num_divs[1],\
                                                 grid2.num_divs[2]))
            
        #extract the points in the cell
        x_icell2 = grid2.x[grid2.slice_array[icell2]]
        y_icell2 = grid2.y[grid2.slice_array[icell2]]
        z_icell2 = grid2.z[grid2.slice_array[icell2]]
            
        #extract the weights in the cell
        w_icell2 = weights2[grid2.slice_array[icell2]]
            
        #extract the jackknife tags in the cell
        j_icell2 = jtags2[grid2.slice_array[icell2]]
            
        #use cython functions to do pair counting
        if PBCs==False:
            counts += jnpairs_no_pbc(x_icell1, y_icell1, z_icell1,\
                                     x_icell2, y_icell2, z_icell2,\
                                     w_icell1, w_icell2,\
                                     j_icell1, j_icell2, N_samples+1,\
                                     rbins)
        else: #PBCs==True
            counts += jnpairs_pbc(x_icell1, y_icell1, z_icell1,\
                                  x_icell2, y_icell2, z_icell2,\
                                  w_icell1, w_icell2,\
                                  j_icell1, j_icell2, N_samples+1,\
                                  rbins, period)
    
    return counts


def xy_z_npairs(data1, data2, rp_bins, pi_bins, Lbox=None, period=None, verbose=False, N_threads=1):
    """
    real-space pair counter.
    
    Count the number of pairs (x1,x2) that can be formed, with x1 drawn from data1 and x2
    drawn from data2, and where distance(x1, x2) <= rbins[i]. 
    
    Parameters
    ----------
    data1: array_like
        N1 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period.
            
    data2: array_like
        N2 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period.
            
    rp_bins: array_like
        numpy array of boundaries defining the radial projected bins in which pairs are 
        counted.
    
    pi_bins: array_like
        numpy array of boundaries defining the parallel bins in which pairs are counted.
    
    Lbox: array_like, optional
        length of cube sides which encloses data1 and data2.
    
    period: array_like, optional
        length k array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.  If True, period is set to be Lbox
    
    verbose: Boolean, optional
        If True, print out information and progress.
    
    N_threads: int, optional
        number of 'threads' to use in the pair counting.  if set to 'max', use all 
        available cores.  N_threads=0 is the default.
    
    Returns
    -------
    N_pairs : array of length len(rbins)
        number of pairs
    """
    
    if N_threads is not 1:
        if N_threads=='max':
            N_threads = multiprocessing.cpu_count()
        if isinstance(N_threads,int):
            pool = multiprocessing.Pool(N_threads)
        else: return ValueError("N_threads argument must be an integer number or 'max'")
    
    #process input
    data1 = np.array(data1)
    data2 = np.array(data2)
    rp_bins = np.array(rp_bins)
    pi_bins = np.array(pi_bins)
    if np.all(period==np.inf): period=None
    
    #enforce shape requirements on input
    if (np.shape(data1)[1]!=3) | (data1.ndim>2):
        raise ValueError("data1 must be of shape (Npts,3)")
    if (np.shape(data2)[1]!=3) | (data2.ndim>2):
        raise ValueError("data2 must be of shape (Npts,3)")
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
    
    #build grids for data1 and data2
    cell_size = np.array([np.max(rp_bins),np.max(rp_bins),np.max(pi_bins)])
    grid1 = rect_cuboid_cells(data1[:,0], data1[:,1], data1[:,2], Lbox, cell_size)
    grid2 = rect_cuboid_cells(data2[:,0], data2[:,1], data2[:,2], Lbox, cell_size)
    
    #square radial bins to make distance calculation cheaper
    rp_bins = rp_bins**2.0
    pi_bins = pi_bins**2.0
    
    #print come information
    if verbose==True:
        print("running grid pairs with {0} by {1} points".format(len(data1),len(data2)))
        print("cell size= {0}".format(grid1.dL))
        print("number of cells = {0}".format(np.prod(grid1.num_divs)))
    
    #number of cells
    Ncell1 = np.prod(grid1.num_divs)
    
    #create a function to call with only one argument
    engine = partial(_xy_z_npairs_engine, grid1, grid2, rp_bins, pi_bins, period, PBCs)
    
    #do the pair counting
    if N_threads>1:
        counts = np.sum(pool.map(engine,range(Ncell1)),axis=0)
    if N_threads==1:
        counts = np.sum(map(engine,range(Ncell1)),axis=0)
    
    return counts


def _xy_z_npairs_engine(grid1, grid2, rp_bins, pi_bins, period, PBCs, icell1):
    """
    pair counting engine for npairs function.  This code calls a cython function.
    """
    
    counts = np.zeros((len(rp_bins),len(pi_bins)))
    
    #extract the points in the cell
    x_icell1, y_icell1, z_icell1 = (grid1.x[grid1.slice_array[icell1]],\
                                        grid1.y[grid1.slice_array[icell1]],\
                                        grid1.z[grid1.slice_array[icell1]])
        
    #get the list of neighboring cells
    ix1, iy1, iz1 = np.unravel_index(icell1,(grid1.num_divs[0],\
                                                 grid1.num_divs[1],\
                                                 grid1.num_divs[2]))
    adj_cell_arr = grid1.adjacent_cells(ix1, iy1, iz1)
            
    #Loop over each of the (up to) 27 subvolumes neighboring, including the current cell.
    for icell2 in adj_cell_arr:
                
        #extract the points in the cell
        x_icell2 = grid2.x[grid2.slice_array[icell2]]
        y_icell2 = grid2.y[grid2.slice_array[icell2]]
        z_icell2 = grid2.z[grid2.slice_array[icell2]]
            
        #use cython functions to do pair counting
        if PBCs==False:
            counts += xy_z_npairs_no_pbc(x_icell1, y_icell1, z_icell1,\
                                         x_icell2, y_icell2, z_icell2,\
                                         rp_bins, pi_bins)
        else: #PBCs==True
            counts += xy_z_npairs_pbc(x_icell1, y_icell1, z_icell1,\
                                      x_icell2, y_icell2, z_icell2,\
                                      rp_bins, pi_bins, period)
    return counts


def s_mu_npairs(data1, data2, s_bins, mu_bins, Lbox=None, period=None, verbose=False, N_threads=1):
    """
    real-space pair counter.
    
    Count the number of pairs (x1,x2) that can be formed, with x1 drawn from data1 and x2
    drawn from data2, and where distance(x1, x2) <= rbins[i]. 
    
    Parameters
    ----------
    data1: array_like
        N1 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period.
            
    data2: array_like
        N2 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period.
            
    s_bins: array_like
        numpy array of boundaries defining the radial bins in which pairs are counted.
    
    mu_bins: array_like
        numpy array of boundaries defining sin(angle) from the line of sight that pairs 
        are counted in.
    
    Lbox: array_like, optional
        length of cube sides which encloses data1 and data2.
    
    period: array_like, optional
        length k array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.  If True, period is set to be Lbox
    
    verbose: Boolean, optional
        If True, print out information and progress.
    
    N_threads: int, optional
        number of 'threads' to use in the pair counting.  if set to 'max', use all 
        available cores.  N_threads=0 is the default.
    
    Returns
    -------
    N_pairs: np.ndarray
        array of shape len(s_bins) x len(mu_bins) with the number of pairs with 
        separations less than or equal to s_bins[i], mu_bins[j].
    """
    
    if N_threads is not 1:
        if N_threads=='max':
            N_threads = multiprocessing.cpu_count()
        if isinstance(N_threads,int):
            pool = multiprocessing.Pool(N_threads)
        else: return ValueError("N_threads argument must be an integer number or 'max'")
    
    #process input
    data1 = np.array(data1)
    data2 = np.array(data2)
    s_bins = np.array(s_bins)
    mu_bins = np.array(mu_bins)
    if np.all(period==np.inf): period=None
    
    #enforce shape requirements on input
    if (np.shape(data1)[1]!=3) | (data1.ndim>2):
        raise ValueError("data1 must be of shape (Npts,3)")
    if (np.shape(data2)[1]!=3) | (data2.ndim>2):
        raise ValueError("data2 must be of shape (Npts,3)")
    if s_bins.ndim != 1:
        raise ValueError("s_bins must be a 1D array")
    if mu_bins.ndim != 1:
        raise ValueError("mu_bins must be a 1D array")
    
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
    if (PBCs==True) & np.any(np.max(s_bins)>Lbox/2.0):
        raise ValueError('grid_pairs pair counter cannot count pairs with seperations\
                          larger than Lbox/2 with PBCs')
    
    #build grids for data1 and data2
    cell_size = np.array([np.max(s_bins),np.max(s_bins),np.max(s_bins)])
    grid1 = rect_cuboid_cells(data1[:,0], data1[:,1], data1[:,2], Lbox, cell_size)
    grid2 = rect_cuboid_cells(data2[:,0], data2[:,1], data2[:,2], Lbox, cell_size)
    
    #do not square s and mu bins!
    
    #print come information
    if verbose==True:
        print("running grid pairs with {0} by {1} points".format(len(data1),len(data2)))
        print("cell size= {0}".format(grid1.dL))
        print("number of cells = {0}".format(np.prod(grid1.num_divs)))
    
    #number of cells
    Ncell1 = np.prod(grid1.num_divs)
    
    #create a function to call with only one argument
    engine = partial(_s_mu_npairs_engine, grid1, grid2, s_bins, mu_bins, period, PBCs)
    
    #do the pair counting
    if N_threads>1:
        counts = np.sum(pool.map(engine,range(Ncell1)),axis=0)
    if N_threads==1:
        counts = np.sum(map(engine,range(Ncell1)),axis=0)
    
    return counts


def _s_mu_npairs_engine(grid1, grid2, s_bins, mu_bins, period, PBCs, icell1):
    """
    pair counting engine for npairs function.  This code calls a cython function.
    """
    
    counts = np.zeros((len(s_bins),len(mu_bins)))
    
    #extract the points in the cell
    x_icell1, y_icell1, z_icell1 = (grid1.x[grid1.slice_array[icell1]],\
                                    grid1.y[grid1.slice_array[icell1]],\
                                    grid1.z[grid1.slice_array[icell1]])
        
    #get the list of neighboring cells
    ix1, iy1, iz1 = np.unravel_index(icell1,(grid1.num_divs[0],\
                                             grid1.num_divs[1],\
                                             grid1.num_divs[2]))
    adj_cell_arr = grid1.adjacent_cells(ix1, iy1, iz1)
            
    #Loop over each of the (up to) 27 subvolumes neighboring, including the current cell.
    for icell2 in adj_cell_arr:
                
        #extract the points in the cell
        x_icell2 = grid2.x[grid2.slice_array[icell2]]
        y_icell2 = grid2.y[grid2.slice_array[icell2]]
        z_icell2 = grid2.z[grid2.slice_array[icell2]]
            
        #use cython functions to do pair counting
        if PBCs==False:
            counts += s_mu_npairs_no_pbc(x_icell1, y_icell1, z_icell1,\
                                         x_icell2, y_icell2, z_icell2,\
                                         s_bins, mu_bins)
        else: #PBCs==True
            counts += s_mu_npairs_pbc(x_icell1, y_icell1, z_icell1,\
                                      x_icell2, y_icell2, z_icell2,\
                                      s_bins, mu_bins, period)
    return counts



def xy_z_wnpairs(data1, data2, rp_bins, pi_bins, Lbox=None, period=None, weights1=None, weights2=None,\
            verbose=False, N_threads=1):
    """
    weighted real-space pair counter.
    
    Count the weighted number of pairs (x1,x2) that can be formed, with x1 drawn from 
    data1 and x2 drawn from data2, and where distance(x1, x2) <= rbins[i].  Weighted 
    counts are calculated as w1*w2.
    
    Parameters
    ----------
    data1: array_like
        N1 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period. This cython implementation requires data1.ndim==2.
            
    data2: array_like
        N2 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period. This cython implementation requires data2.ndim==2.
            
    rp_bins: array_like
        numpy array of boundaries defining the radial projected bins in which pairs are 
        counted.
    
    pi_bins: array_like
        numpy array of boundaries defining the parallel bins in which pairs are counted. 
    
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
    
    verbose: Boolean, optional
        If True, print out information and progress.
    
    N_threads: int, optional
        number of 'threads' to use in the pair counting.  if set to 'max', use all 
        available cores.  N_threads=0 is the default.
        
    Returns
    -------
    N_pairs : array of length len(rbins)
        number counts of pairs
    """
    
    if N_threads is not 1:
        if N_threads=='max':
            N_threads = multiprocessing.cpu_count()
        if isinstance(N_threads,int):
            pool = multiprocessing.Pool(N_threads)
        else: return ValueError("N_threads argument must be an integer number or 'max'")
    
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
    
    #build grids for data1 and data2
    cell_size = np.array([np.max(rp_bins),np.max(rp_bins),np.max(pi_bins)])
    grid1 = rect_cuboid_cells(data1[:,0], data1[:,1], data1[:,2], Lbox, cell_size)
    grid2 = rect_cuboid_cells(data2[:,0], data2[:,1], data2[:,2], Lbox, cell_size)
    
    #sort the weights arrays
    weights1 = weights1[grid1.idx_sorted]
    weights2 = weights2[grid2.idx_sorted]
    
    #square radial bins to make distance calculation cheaper
    rp_bins = rp_bins**2.0
    pi_bins = pi_bins**2.0
    
    #number of cells
    Ncell1 = np.prod(grid1.num_divs)
    
    #create a function to call with only one argument
    engine = partial(_xy_z_wnpairs_engine, grid1, grid2, weights1, weights2, rp_bins, pi_bins, period, PBCs)
    
    #do the pair counting
    if N_threads>1:
        counts = np.sum(pool.map(engine,range(Ncell1)),axis=0)
    if N_threads==1:
        counts = np.sum(map(engine,range(Ncell1)),axis=0)
    
    return counts


def _xy_z_wnpairs_engine(grid1, grid2, weights1, weights2, rp_bins, pi_bins, period, PBCs, icell1):
    
    counts = np.zeros((len(rp_bins),len(pi_bins)))
    
    #extract the points in the cell
    x_icell1, y_icell1, z_icell1 = (grid1.x[grid1.slice_array[icell1]],\
                                    grid1.y[grid1.slice_array[icell1]],\
                                    grid1.z[grid1.slice_array[icell1]])
        
    #extract the weights in the cell
    w_icell1 = weights1[grid1.slice_array[icell1]]
        
    #get the list of neighboring cells
    ix1, iy1, iz1 = np.unravel_index(icell1,(grid1.num_divs[0],\
                                             grid1.num_divs[1],\
                                             grid1.num_divs[2]))
    adj_cell_arr = grid1.adjacent_cells(ix1, iy1, iz1)
        
    #Loop over each of the 27 subvolumes neighboring, including the current cell.
    for icell2 in adj_cell_arr:
            
        ix2, iy2, iz2 = np.unravel_index(icell2,(grid2.num_divs[0],\
                                                 grid2.num_divs[1],\
                                                 grid2.num_divs[2]))
        
        #extract the points in the cell
        x_icell2 = grid2.x[grid2.slice_array[icell2]]
        y_icell2 = grid2.y[grid2.slice_array[icell2]]
        z_icell2 = grid2.z[grid2.slice_array[icell2]]
        
        #extract the weights in the cell
        w_icell2 = weights2[grid2.slice_array[icell2]]
        
        #use cython functions to do pair counting
        if PBCs==False:
            counts += xy_z_wnpairs_no_pbc(x_icell1, y_icell1, z_icell1,\
                                          x_icell2, y_icell2, z_icell2,\
                                          w_icell1, w_icell2,\
                                          rp_bins, pi_bins)
        else: #PBCs==True
            counts += xy_z_wnpairs_pbc(x_icell1, y_icell1, z_icell1,\
                                       x_icell2, y_icell2, z_icell2,\
                                       w_icell1, w_icell2,\
                                       rp_bins, pi_bins, period)
    return counts


def xy_z_jnpairs(data1, data2, rp_bins, pi_bins, Lbox=None, period=None, weights1=None, weights2=None,\
            jtags1=None, jtags2=None, N_samples=0, verbose=False, N_threads=1):
    """
    jackknife weighted real-space pair counter.
    
    Count the weighted number of pairs (x1,x2) that can be formed, with x1 drawn from 
    data1 and x2 drawn from data2, and where distance(x1, x2) <= rbins[i].  Weighted 
    counts are calculated as w1*w2. Jackknife sampled pair counts are returned.
    
    Parameters
    ----------
    data1: array_like
        N1 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period. This cython implementation requires data1.ndim==2.
            
    data2: array_like
        N2 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period. This cython implementation requires data2.ndim==2.
            
    rp_bins: array_like
        numpy array of boundaries defining the radial projected bins in which pairs are 
        counted.
    
    pi_bins: array_like
        numpy array of boundaries defining the parallel bins in which pairs are counted.
    
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
        length N1 array containing integer tags used to define jackknife sample 
        membership. Tags are in the range [1,N_samples]. '0' is a reserved tag and should 
        not be used.
        
    jtags2: array_like, optional
        length N2 array containing integer tags used to define jackknife sample 
        membership. Tags are in the range [1,N_samples]. '0' is a reserved tag and should 
        not be used.
    
    N_samples: int, optional
        number of jackknife samples
    
    verbose: Boolean, optional
        If True, print out information and progress.
    
    N_threads: int, optional
        number of 'threads' to use in the pair counting.  If set to 'max', use all 
        available cores.  N_threads=0 is the default.
        
    Returns
    -------
    N_pairs : ndarray of shape (N_samples+1,len(rbins))
        number counts of pairs with seperations <=rbins[i]
    
    Notes
    -----
    Jackknife weights are calculated using a weighting function.
    
    if both points are outside the sample, return 0.0
    if both points are inside the sample, return (w1 * w2)
    if one point is inside, and the other is outside return 0.5*(w1 * w2)
    """
    
    if N_threads is not 1:
        if N_threads=='max':
            N_threads = multiprocessing.cpu_count()
        if isinstance(N_threads,int):
            pool = multiprocessing.Pool(N_threads)
        else: return ValueError("N_threads argument must be an integer number or 'max'")
    
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
                          length of one side of a rectangular cuboid")
    
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
    
    #Check bounds of jackknife tags
    if np.min(jtags1)<1: raise ValueError("jtags1 must be >=1")
    if np.min(jtags2)<1: raise ValueError("jtags2 must be >=1")
    if np.max(jtags1)>N_samples: raise ValueError("jtags1 must be <=N_samples")
    if np.max(jtags2)>N_samples: raise ValueError("jtags2 must be <=N_samples")
    
    #throw warning if some tags do not exist
    if not np.array_equal(np.unique(jtags1),np.arange(1,N_samples+1)):
        print("Warning: data1 does not contain points in every jackknife sample.")
    if not np.array_equal(np.unique(jtags1),np.arange(1,N_samples+1)):
        print("Warning: data2 does not contain points in every jackknife sample.")
    
    if type(N_samples) is not int: 
        raise ValueError("There must be an integer number of jackknife samples")
    if np.max(jtags1)>N_samples:
        raise ValueError("There are more jackknife samples than indicated by N_samples")
    if np.max(jtags2)>N_samples:
        raise ValueError("There are more jackknife samples than indicated by N_samples")
    
    #check to see we dont count pairs more than once    
    if (PBCs==True) & np.any(np.max(rp_bins)>Lbox[0:2]/2.0):
        raise ValueError('grid_pairs pair counter cannot count pairs with seperations\
                          larger than Lbox/2 with PBCs')
    if (PBCs==True) & np.any(np.max(pi_bins)>Lbox[2]/2.0):
        raise ValueError('grid_pairs pair counter cannot count pairs with seperations\
                          larger than Lbox/2 with PBCs')
    
    #build grids for data1 and data2
    cell_size = np.array([np.max(rp_bins),np.max(rp_bins),np.max(pi_bins)])
    grid1 = rect_cuboid_cells(data1[:,0], data1[:,1], data1[:,2], Lbox, cell_size)
    grid2 = rect_cuboid_cells(data2[:,0], data2[:,1], data2[:,2], Lbox, cell_size)
    
    #sort the weights arrays
    weights1 = weights1[grid1.idx_sorted]
    weights2 = weights2[grid2.idx_sorted]
        
    #sort the jackknife tag arrays
    jtags1 = jtags1[grid1.idx_sorted]
    jtags2 = jtags2[grid2.idx_sorted]
    
    #square radial bins to make distance calculation cheaper
    rp_bins = rp_bins**2.0
    pi_bins = pi_bins**2.0
    
    #Loop over all subvolumes in grid1
    Ncell1 = np.prod(grid1.num_divs)
    
    #create a function to call with only one argument
    engine = partial(_xy_z_jnpairs_engine, grid1, grid2, weights1, weights2, jtags1, jtags2,\
                     N_samples, rp_bins, pi_bins, period, PBCs)
    
    #do the pair counting
    if N_threads>1:
        counts = np.sum(pool.map(engine,range(Ncell1)),axis=0)
    if N_threads==1:
        counts = np.sum(map(engine,range(Ncell1)),axis=0)
    
    return counts


def _xy_z_jnpairs_engine(grid1, grid2, weights1, weights2, jtags1, jtags2, N_samples, rp_bins, pi_bins,\
                         period, PBCs, icell1):
    
    counts = np.zeros((N_samples+1,len(rp_bins),len(pi_bins)))
    
    #extract the points in the cell
    x_icell1, y_icell1, z_icell1 = (grid1.x[grid1.slice_array[icell1]],\
                                    grid1.y[grid1.slice_array[icell1]],\
                                    grid1.z[grid1.slice_array[icell1]])
        
    #extract the weights in the cell
    w_icell1 = weights1[grid1.slice_array[icell1]]
        
    #extract the jackknife tags in the cell
    j_icell1 = jtags1[grid1.slice_array[icell1]]
        
    #get the list of neighboring cells
    ix1, iy1, iz1 = np.unravel_index(icell1,(grid1.num_divs[0],\
                                             grid1.num_divs[1],\
                                             grid1.num_divs[2]))
    adj_cell_arr = grid1.adjacent_cells(ix1, iy1, iz1)
        
    #Loop over each of the 27 subvolumes neighboring, including the current cell.
    for icell2 in adj_cell_arr:
            
        ix2, iy2, iz2 = np.unravel_index(icell2,(grid2.num_divs[0],\
                                                 grid2.num_divs[1],\
                                                 grid2.num_divs[2]))
            
        #extract the points in the cell
        x_icell2 = grid2.x[grid2.slice_array[icell2]]
        y_icell2 = grid2.y[grid2.slice_array[icell2]]
        z_icell2 = grid2.z[grid2.slice_array[icell2]]
            
        #extract the weights in the cell
        w_icell2 = weights2[grid2.slice_array[icell2]]
            
        #extract the jackknife tags in the cell
        j_icell2 = jtags2[grid2.slice_array[icell2]]
            
        #use cython functions to do pair counting
        if PBCs==False:
            counts += xy_z_jnpairs_no_pbc(x_icell1, y_icell1, z_icell1,\
                                          x_icell2, y_icell2, z_icell2,\
                                          w_icell1, w_icell2,\
                                          j_icell1, j_icell2, N_samples+1,\
                                          rp_bins, pi_bins)
        else: #PBCs==True
            counts += xy_z_jnpairs_pbc(x_icell1, y_icell1, z_icell1,\
                                       x_icell2, y_icell2, z_icell2,\
                                       w_icell1, w_icell2,\
                                       j_icell1, j_icell2, N_samples+1,\
                                       rp_bins, pi_bins, period)
    
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
    
    # real space pair counter speed tests
    _test_npairs_speed()
    _test_wnpairs_speed()
    _test_jnpairs_speed()
    
    # 2D+1 space pair counter speed tests
    _test_xy_z_npairs_speed()
    _test_xy_z_wnpairs_speed()
    _test_xy_z_jnpairs_speed()


def _test_npairs_speed():

    "bolshoi like test out to ~20 Mpc"
    N_threads=4
    Npts = 1e5
    Lbox = [250.0,250.0,250.0]
    period = np.array(Lbox)
    
    x = np.random.uniform(0, Lbox[0], Npts)
    y = np.random.uniform(0, Lbox[1], Npts)
    z = np.random.uniform(0, Lbox[2], Npts)
    data1 = np.vstack((x,y,z)).T
    
    rbins = np.logspace(-2,1.3)
    
    print("##########npairs##########")
    print("running with {0}/{1} cores".format(N_threads,multiprocessing.cpu_count()))
    print("running speed test with {0} points".format(Npts))
    print("in {0} x {1} x {2} box.".format(Lbox[0],Lbox[1],Lbox[2]))
    print("to maximum seperation {0}".format(np.max(rbins)))

    #w/ PBCs
    start = time()
    result = npairs(data1, data1, rbins, Lbox=Lbox, period=period, verbose=False, N_threads=N_threads)
    end = time()
    runtime = end-start
    print("Total runtime (PBCs) = %.1f seconds" % runtime)

    #w/o PBCs
    start = time()
    result = npairs(data1, data1, rbins, Lbox=Lbox, period=None, verbose=False, N_threads=N_threads)
    end = time()
    runtime = end-start
    print("Total runtime (no PBCs) = %.1f seconds" % runtime)
    print("########################## \n")


def _test_wnpairs_speed():

    "bolshoi like test out to ~20 Mpc"
    N_threads=4
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
    print("running with {0}/{1} cores".format(N_threads,multiprocessing.cpu_count()))
    print("running speed test with {0} points".format(Npts))
    print("in {0} x {1} x {2} box.".format(Lbox[0],Lbox[1],Lbox[2]))
    print("to maximum seperation {0}".format(np.max(rbins)))

    #w/ PBCs
    start = time()
    result = wnpairs(data1, data1, rbins, Lbox=Lbox, period=period,\
                     weights1=weights1, weights2=weights1, verbose=True,\
                     N_threads=N_threads)
    end = time()
    runtime = end-start
    print("Total runtime (PBCs) = %.1f seconds" % runtime)

    #w/o PBCs
    start = time()
    result = wnpairs(data1, data1, rbins, Lbox=Lbox, period=None,\
                     weights1=weights1, weights2=weights1, verbose=True,\
                     N_threads=N_threads)
    end = time()
    runtime = end-start
    print("Total runtime (no PBCs) = %.1f seconds" % runtime)
    print("########################### \n")


def _test_jnpairs_speed():

    "bolshoi like test out to ~20 Mpc"
    N_threads=4
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
    print("running with {0}/{1} cores".format(N_threads,multiprocessing.cpu_count()))
    print("running speed test with {0} points".format(Npts))
    print("{0} jackknife samples".format(Nsamples))
    print("in {0} x {1} x {2} box.".format(Lbox[0],Lbox[1],Lbox[2]))
    print("to maximum seperation {0}".format(np.max(rbins)))

    #w/ PBCs
    start = time()
    result = jnpairs(data1, data1, rbins, Lbox=Lbox, period=period,\
                     weights1=weights1, weights2=weights1, jtags1=jtags1, jtags2=jtags1,\
                     N_samples=Nsamples, verbose=True, N_threads=N_threads)
    end = time()
    runtime = end-start
    print("Total runtime (PBCs) = %.1f seconds" % runtime)

    #w/o PBCs
    start = time()
    result = jnpairs(data1, data1, rbins, Lbox=Lbox, period=None,\
                     weights1=weights1, weights2=weights1, jtags1=jtags1, jtags2=jtags1,\
                     N_samples=Nsamples, verbose=True, N_threads=N_threads)
    end = time()
    runtime = end-start
    print("Total runtime (no PBCs) = %.1f seconds" % runtime)
    print("########################### \n")


def _test_xy_z_npairs_speed():

    "bolshoi like test out to ~20 Mpc"
    N_threads=4
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
    print("running with {0}/{1} cores".format(N_threads,multiprocessing.cpu_count()))
    print("running speed test with {0} points".format(Npts))
    print("in {0} x {1} x {2} box.".format(Lbox[0],Lbox[1],Lbox[2]))
    print("to maximum seperation {0}, {1}".format(np.max(rp_bins),np.max(pi_bins)))

    #w/ PBCs
    start = time()
    result = xy_z_npairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=period, verbose=False, N_threads=N_threads)
    end = time()
    runtime = end-start
    print("Total runtime (PBCs) = %.1f seconds" % runtime)

    #w/o PBCs
    start = time()
    result = xy_z_npairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=None, verbose=False, N_threads=N_threads)
    end = time()
    runtime = end-start
    print("Total runtime (no PBCs) = %.1f seconds" % runtime)
    print("############################### \n")


def _test_xy_z_wnpairs_speed():

    "bolshoi like test out to ~20 Mpc"
    N_threads=4
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
    print("running with {0}/{1} cores".format(N_threads,multiprocessing.cpu_count()))
    print("running speed test with {0} points".format(Npts))
    print("in {0} x {1} x {2} box.".format(Lbox[0],Lbox[1],Lbox[2]))
    print("to maximum seperation {0}, {1}".format(np.max(rp_bins),np.max(pi_bins)))

    #w/ PBCs
    start = time()
    result = xy_z_wnpairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=period,\
                     weights1=weights1, weights2=weights1, verbose=True,\
                     N_threads=N_threads)
    end = time()
    runtime = end-start
    print("Total runtime (PBCs) = %.1f seconds" % runtime)

    #w/o PBCs
    start = time()
    result = xy_z_wnpairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=None,\
                     weights1=weights1, weights2=weights1, verbose=True,\
                     N_threads=N_threads)
    end = time()
    runtime = end-start
    print("Total runtime (no PBCs) = %.1f seconds" % runtime)
    print("################################ \n")


def _test_xy_z_jnpairs_speed():

    "bolshoi like test out to ~20 Mpc"
    N_threads=4
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
    
    rp_bins = np.logspace(-2,1.3)
    pi_bins = np.linspace(0,50,20)
    
    print("##########xy_z_jnpairs##########")
    print("running with {0}/{1} cores".format(N_threads,multiprocessing.cpu_count()))
    print("running speed test with {0} points".format(Npts))
    print("{0} jackknife samples".format(Nsamples))
    print("in {0} x {1} x {2} box.".format(Lbox[0],Lbox[1],Lbox[2]))
    print("to maximum seperation {0}, {1}".format(np.max(rp_bins),np.max(pi_bins)))

    #w/ PBCs
    start = time()
    result = xy_z_jnpairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=period,\
                     weights1=weights1, weights2=weights1, jtags1=jtags1, jtags2=jtags1,\
                     N_samples=Nsamples, verbose=True, N_threads=N_threads)
    end = time()
    runtime = end-start
    print("Total runtime (PBCs) = %.1f seconds" % runtime)

    #w/o PBCs
    start = time()
    result = xy_z_jnpairs(data1, data1, rp_bins, pi_bins, Lbox=Lbox, period=None,\
                     weights1=weights1, weights2=weights1, jtags1=jtags1, jtags2=jtags1,\
                     N_samples=Nsamples, verbose=True, N_threads=N_threads)
    end = time()
    runtime = end-start
    print("Total runtime (no PBCs) = %.1f seconds" % runtime)
    print("################################ \n")


if __name__ == '__main__':
    main()
