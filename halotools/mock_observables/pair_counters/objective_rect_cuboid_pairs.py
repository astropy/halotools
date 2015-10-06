# -*- coding: utf-8 -*-

"""
Rectangular Cuboid Objective Weighted Pair Counter. 

This module contains pair counting function(s) used to count the number of pairs with 
separations less than or equal to r, optimized for simulation boxes.

The weighting is done using special user specified objective weighting functions.
"""

from __future__ import print_function, division
import numpy as np
from time import time
import sys
import multiprocessing
from functools import partial

from .rect_cuboid import *
from .objective_cpairs import *

__all__=['obj_wnpairs']
__author__=['Duncan Campbell']


def obj_wnpairs(data1, data2, rbins, Lbox=None, period=None,\
                weights1=None, weights2=None, aux1=None, aux2=None,\
                wfunc=0, verbose=False, N_threads=1):
    """
    weighted real-space pair counter.
    
    Count the weighted number of pairs (x1,x2) that can be formed, with x1 drawn from 
    data1 and x2 drawn from data2, and where distance(x1, x2) <= rbins[i].  Weighted 
    counts are calculated as f(w1,w2,r1,r2)
    
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
    
    aux1: array_like, optional
        length N1 array containing auxiliary weights used for weighted pair counts
        
    aux2: array_like, optional
        length N2 array containing auxiliary weights used for weighted pair counts.
    
    wfunc: int, optional
        weighting function ID.
    
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
    
    if type(wfunc) is not int:
        raise ValueError("wfunc ID must be an integer")
    if (wfunc<0 | wfunc>9):
        raise ValueError("wfunc ID does not exist.  Availabel wfunc are:", list_weighting_functions())
    
    if verbose==True:
        print("Using wfunc: {0}".format(wfunc))
    
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
    
    #Process weights1 entry and check for consistency.
    if aux1 is None:
            aux1 = np.array([1.0]*np.shape(data1)[0], dtype=np.float64)
    else:
        aux1 = np.asarray(aux1).astype("float64")
        if np.shape(aux1)[0] != np.shape(data1)[0]:
            raise ValueError("aux1 should have same len as data1")
    #Process weights2 entry and check for consistency.
    if aux2 is None:
        aux2 = np.array([1.0]*np.shape(data2)[0], dtype=np.float64)
    else:
        aux2 = np.asarray(aux2).astype("float64")
        if np.shape(aux2)[0] != np.shape(data2)[0]:
            raise ValueError("aux2 should have same len as data2")
    
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
    engine = partial(_wnpairs_engine, grid1, grid2, weights1, weights2, aux1, aux2, rbins, period, PBCs, wfunc)
    
    #do the pair counting
    if N_threads>1:
        counts = np.sum(pool.map(engine,range(Ncell1)),axis=0)
        pool.close()
    if N_threads==1:
        counts = np.sum(map(engine,range(Ncell1)),axis=0)
    
    return counts


def _wnpairs_engine(grid1, grid2, weights1, weights2, aux1, aux2, rbins, period, PBCs, wfunc, icell1):
    
    counts = np.zeros(len(rbins))
    
    #extract the points in the cell
    x_icell1, y_icell1, z_icell1 = (grid1.x[grid1.slice_array[icell1]],\
                                    grid1.y[grid1.slice_array[icell1]],\
                                    grid1.z[grid1.slice_array[icell1]])
        
    #extract the weights in the cell
    w_icell1 = weights1[grid1.slice_array[icell1]]
    
    #extract the weights in the cell
    r_icell1 = aux1[grid1.slice_array[icell1]]
        
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
        
        #extract the weights in the cell
        r_icell2 = aux2[grid2.slice_array[icell2]]
        
        #use cython functions to do pair counting
        if PBCs==False:
            counts += obj_wnpairs_no_pbc(x_icell1, y_icell1, z_icell1,\
                                     x_icell2, y_icell2, z_icell2,\
                                     w_icell1, w_icell2, r_icell1, r_icell2,\
                                     rbins, wfunc)
        else: #PBCs==True
            counts += obj_wnpairs_pbc(x_icell1, y_icell1, z_icell1,\
                                  x_icell2, y_icell2, z_icell2,\
                                  w_icell1, w_icell2, r_icell1, r_icell2,\
                                  rbins, period, wfunc)
    return counts


def list_weighting_functions():
    """
    Print the available weighting functions for this module.
    
    Weighting functions take 2 floats attached to each data1 and data2
    and return a double
    """
    
    print("func ID 0: custom user-defined  and compiled weighting function")
    print("func ID 1: multiplicative weights, return w1*w2")
    print("func ID 2: summed weights, return w1+w2")
    print("func ID 3: equality weights, return r1*r2 if w1==w2")
    print("func ID 4: greater than weights, return r1*r2 if w2>w1")
    print("func ID 5: less than weights, return r1*r2 if w2<w1")
    print("func ID 6: greater than tolerance weights, return r2 if w2>(w1+r1)")
    print("func ID 7: less than tolerance weights, return r2 if w2<(w1-r1)")
    print("func ID 8: tolerance weights, return r2 if |w1-w2|<r1")
    print("func ID 9: exclusion weights, return r2 if |w1-w2|>r1")

if __name__ == '__main__':
    main()
