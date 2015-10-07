# -*- coding: utf-8 -*-

"""
Cuboid FoF pair search
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from time import time
import sys
import multiprocessing
from functools import partial
from scipy.sparse import coo_matrix

from .rect_cuboid import *
from .cpairs.pairwise_distances import *

__all__=['fof_pairs', 'xy_z_fof_pairs']
__author__=['Duncan Campbell']


def fof_pairs(data1, data2, r_max, Lbox=None, period=None, verbose=False, N_threads=1):
    """
    real-space FoF pair finder.
    
    return the pairs wich have separations <= r_max
    
    Parameters
    ----------
    data1: array_like
        N1 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period.
            
    data2: array_like
        N2 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period.
            
    r_max: float
        maximum distance to connect pairs
    
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
    dists : scipy.sparse.coo_matrix
        N1 x N2 sparse matrix in COO format containing distances between points.
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
    if np.all(period==np.inf): period=None
    
    #enforce shape requirements on input
    if (np.shape(data1)[1]!=3) | (data1.ndim>2):
        raise ValueError("data1 must be of shape (Npts,3)")
    if (np.shape(data2)[1]!=3) | (data2.ndim>2):
        raise ValueError("data2 must be of shape (Npts,3)")
    
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
    if (PBCs==True) & np.any(np.max(r_max)>Lbox/2.0):
        raise ValueError('cannot count pairs with seperations \
                          larger than Lbox/2 with PBCs')
    
    #choose grid size along each dimension.
    #too small of a grid size is inefficient.
    use_max = (Lbox/r_max) > 10
    cell_size = np.array([np.max(r_max)]*3)
    cell_size[use_max] = Lbox[use_max]/10.0
    #cell shouldn't be bigger than the box
    too_big = (cell_size>Lbox)
    cell_size[too_big] = Lbox[too_big]
    
    #build grids for data1 and data2
    grid1 = rect_cuboid_cells(data1[:,0], data1[:,1], data1[:,2], Lbox, cell_size)
    grid2 = rect_cuboid_cells(data2[:,0], data2[:,1], data2[:,2], Lbox, cell_size)
    
    #square radial bins to make distance calculation cheaper
    r_max = r_max**2.0
    
    #print come information
    if verbose==True:
        print("running for pairs with {0} by {1} points".format(len(data1),len(data2)))
        print("cell size= {0}".format(grid1.dL))
        print("number of cells = {0}".format(np.prod(grid1.num_divs)))
    
    #number of cells
    Ncell1 = np.prod(grid1.num_divs)
    
    #create a function to call with only one argument
    engine = partial(_fof_pairs_engine, grid1, grid2, r_max, period, PBCs)
    
    #do the pair counting
    if N_threads>1:
        result = pool.map(engine,range(Ncell1))
        pool.close()
    if N_threads==1:
        result = map(engine,range(Ncell1))
    
    #arrays to store result
    d = np.zeros((0,), dtype='float')
    i_inds = np.zeros((0,), dtype='int')
    j_inds = np.zeros((0,), dtype='int')
    
    #unpack the results
    for i in range(len(result)):
        d = np.append(d,result[i][0])
        i_inds = np.append(i_inds,result[i][1])
        j_inds = np.append(j_inds,result[i][2])
    
    #resort the result (it was sorted to make in continuous over the cell structure)
    i_inds = grid1.idx_sorted[i_inds]
    j_inds = grid2.idx_sorted[j_inds]

    return coo_matrix((d, (i_inds, j_inds)))


def _fof_pairs_engine(grid1, grid2, r_max, period, PBCs, icell1):
    """
    pair counting engine for npairs function.  This code calls a cython function.
    """
    
    d = np.zeros((0,), dtype='float')
    i_inds = np.zeros((0,), dtype='int')
    j_inds = np.zeros((0,), dtype='int')
    
    #extract the points in the cell
    x_icell1, y_icell1, z_icell1 = (grid1.x[grid1.slice_array[icell1]],\
                                    grid1.y[grid1.slice_array[icell1]],\
                                    grid1.z[grid1.slice_array[icell1]])
    
    i_min = grid1.slice_array[icell1].start
    
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
        
        j_min = grid2.slice_array[icell2].start
        
        #use cython functions to do pair counting
        if PBCs==False:
            dd, ii_inds, jj_inds = pairwise_distance_no_pbc(x_icell1, y_icell1, z_icell1,\
                                                            x_icell2, y_icell2, z_icell2,\
                                                            r_max)
        else: #PBCs==True
            dd, ii_inds, jj_inds = pairwise_distance_pbc(x_icell1, y_icell1, z_icell1,\
                                                         x_icell2, y_icell2, z_icell2,\
                                                         period, r_max)
        
        ii_inds = ii_inds+i_min
        jj_inds = jj_inds+j_min
        
        #update storage arrays
        d = np.concatenate((d,dd))
        i_inds = np.concatenate((i_inds,ii_inds))
        j_inds = np.concatenate((j_inds,jj_inds))
        
    return d, i_inds, j_inds


def xy_z_fof_pairs(data1, data2, rp_max, pi_max, Lbox=None, period=None, verbose=False,\
                   N_threads=1):
    """
    redshift-space FoF pair finder.
    
    return the pairs wich have separations <= rp_max and <=pi_max
    
    Parameters
    ----------
    data1: array_like
        N1 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period.
            
    data2: array_like
        N2 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period.
            
    r_max: float
        maximum distance to connect pairs
    
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
    dists : scipy.sparse.coo_matrix
        N1 x N2 sparse matrix in COO format containing distances between points.
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
    if np.all(period==np.inf): period=None
    
    #enforce shape requirements on input
    if (np.shape(data1)[1]!=3) | (data1.ndim>2):
        raise ValueError("data1 must be of shape (Npts,3)")
    if (np.shape(data2)[1]!=3) | (data2.ndim>2):
        raise ValueError("data2 must be of shape (Npts,3)")
    
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
    if (PBCs==True) & np.any(rp_max>Lbox[0:2]/2.0):
        raise ValueError('grid_pairs pair counter cannot count pairs with seperations\
                          larger than Lbox/2 with PBCs')
    if (PBCs==True) & np.any(pi_max>Lbox[2]/2.0):
        raise ValueError('grid_pairs pair counter cannot count pairs with seperations\
                          larger than Lbox/2 with PBCs')
    
    #choose grid size along each dimension.
    #too small of a grid size is inefficient.
    cell_size = np.zeros((3,))
    cell_size[0:2] = np.array([rp_max]*2)
    cell_size[2] = pi_max
    use_max = (Lbox/cell_size) > 10
    cell_size[use_max] = Lbox[use_max]/10.0
    #cells shouldn't be bigger than the box
    too_big = (cell_size>Lbox)
    cell_size[too_big] = Lbox[too_big]
    
    #build grids for data1 and data2
    grid1 = rect_cuboid_cells(data1[:,0], data1[:,1], data1[:,2], Lbox, cell_size)
    grid2 = rect_cuboid_cells(data2[:,0], data2[:,1], data2[:,2], Lbox, cell_size)
    
    #square radial bins to make distance calculation cheaper
    rp_max = rp_max**2.0
    pi_max = pi_max**2.0
    
    #print come information
    if verbose==True:
        print("running for pairs with {0} by {1} points".format(len(data1),len(data2)))
        print("cell size= {0}".format(grid1.dL))
        print("number of cells = {0}".format(np.prod(grid1.num_divs)))
    
    #number of cells
    Ncell1 = np.prod(grid1.num_divs)
    
    #create a function to call with only one argument
    engine = partial(_xy_z_fof_pairs_engine, grid1, grid2, rp_max, pi_max, period, PBCs)
    
    #do the pair counting
    if N_threads>1:
        result = pool.map(engine,range(Ncell1))
        pool.close()
    if N_threads==1:
        result = map(engine,range(Ncell1))
    
    #arrays to store result
    d_perp = np.zeros((0,), dtype='float')
    d_para = np.zeros((0,), dtype='float')
    i_inds = np.zeros((0,), dtype='int')
    j_inds = np.zeros((0,), dtype='int')
    
    #unpack the results
    for i in range(len(result)):
        d_perp = np.append(d_perp,result[i][0])
        d_para = np.append(d_para,result[i][1])
        i_inds = np.append(i_inds,result[i][2])
        j_inds = np.append(j_inds,result[i][3])
    
    #resort the result (it was sorted to make in continuous over the cell structure)
    i_inds = grid1.idx_sorted[i_inds]
    j_inds = grid2.idx_sorted[j_inds]
    
    return coo_matrix((d_perp, (i_inds, j_inds))), coo_matrix((d_para, (i_inds, j_inds)))


def _xy_z_fof_pairs_engine(grid1, grid2, rp_max, pi_max, period, PBCs, icell1):
    """
    pair counting engine for npairs function.  This code calls a cython function.
    """
    
    d_perp = np.zeros((0,), dtype='float')
    d_para = np.zeros((0,), dtype='float')
    i_inds = np.zeros((0,), dtype='int')
    j_inds = np.zeros((0,), dtype='int')
    
    #extract the points in the cell
    x_icell1, y_icell1, z_icell1 = (grid1.x[grid1.slice_array[icell1]],\
                                    grid1.y[grid1.slice_array[icell1]],\
                                    grid1.z[grid1.slice_array[icell1]])
    
    i_min = grid1.slice_array[icell1].start
    
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
        
        j_min = grid2.slice_array[icell2].start
        
        #use cython functions to do pair counting
        if PBCs==False:
            dd_perp, dd_para, ii_inds, jj_inds = pairwise_xy_z_distance_no_pbc(x_icell1, y_icell1, z_icell1,\
                                                            x_icell2, y_icell2, z_icell2,\
                                                            rp_max, pi_max)
        else: #PBCs==True
            dd_perp, dd_para, ii_inds, jj_inds = pairwise_xy_z_distance_pbc(x_icell1, y_icell1, z_icell1,\
                                                         x_icell2, y_icell2, z_icell2,\
                                                         period, rp_max, pi_max)
        
        ii_inds = ii_inds+i_min
        jj_inds = jj_inds+j_min
        
        #update storage arrays
        d_perp = np.concatenate((d_perp,dd_perp))
        d_para = np.concatenate((d_para,dd_para))
        i_inds = np.concatenate((i_inds,ii_inds))
        j_inds = np.concatenate((j_inds,jj_inds))
        
    return d_perp, d_para, i_inds, j_inds


