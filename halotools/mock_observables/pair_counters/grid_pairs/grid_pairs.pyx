#!/usr/bin/env python
# cython: profile=False

from __future__ import print_function, division
__all__ = ['npairs', 'xy_z_npairs', 'wnpairs', 'xy_z_wnpairs', 'jnpairs']

import sys
import numpy as np
cimport cython
cimport numpy as np
from libc.math cimport fabs, fmin

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def npairs(data1, data2, rbins, Lbox=None, period=None):
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
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif np.shape(period) == (1,):
        period = np.array([period[0]]*3)
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif isinstance(period, (int, long, float, complex)):
        period = np.array([period]*3)
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
    
    #c definitions
    cdef int nbins = len(rbins)
    cdef np.ndarray[np.float64_t, ndim=1] crbins = \
        np.ascontiguousarray(rbins,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cperiod = \
        np.ascontiguousarray(period,dtype=np.float64)
    cdef np.ndarray[np.int_t, ndim=1] counts = \
        np.zeros((nbins,), dtype=np.int)
    
    #build grids for data1 and data2
    cell_size = np.array([np.max(rbins)]*3)
    grid1 = cube_grid(data1[:,0], data1[:,1], data1[:,2], Lbox, cell_size)
    grid2 = cube_grid(data2[:,0], data2[:,1], data2[:,2], Lbox, cell_size)
    
    #square radial bins to make distance calculation cheaper
    crbins = crbins**2.0
    
    #print come information
    #print("running grid pairs with {0} by {1} points".format(len(data1),len(data2)))
    #print("cell size= {0}".format(grid1.dL))
    #print("number of cells = {0}".format(np.prod(grid1.num_divs)))
    #print("PBCs= {0}".format(PBCs))
    
    #more c definitions used inside loop
    cdef int i, j, k
    cdef int icell1,icell2
    cdef int ix1, iy1, iz1, ix2, iy2, iz2
    cdef double d
    cdef np.ndarray[np.float64_t, ndim=1] x_icell1, y_icell1, z_icell1
    cdef np.ndarray[np.float64_t, ndim=1] x_icell2, y_icell2, z_icell2
    cdef np.ndarray[np.int_t, ndim=1] adj_cell_arr
    
    #Loop over all subvolumes in grid1
    for icell1 in range(np.prod(grid1.num_divs)):
        #calculate progress
        progress = icell1/(np.prod(grid1.num_divs))*100
        print("    {0:.2f} %%".format(progress),end='\r')
        sys.stdout.flush()
        
        #extract the points in the cell
        x_icell1, y_icell1, z_icell1 = (grid1.x[grid1.slice_array[icell1]],\
                                        grid1.y[grid1.slice_array[icell1]],\
                                        grid1.z[grid1.slice_array[icell1]])
        
        #get the list of neighboring cells
        ix1, iy1, iz1 = np.unravel_index(icell1,(grid1.num_divs[0],\
                                                 grid1.num_divs[1],\
                                                 grid1.num_divs[2]))
        adj_cell_arr = grid1.adjacent_cells(ix1, iy1, iz1)
        
        if PBCs==True:
            #Loop over each of the 27 subvolumes neighboring, including the current cell.
            for icell2 in adj_cell_arr:
            
                ix2, iy2, iz2 = np.unravel_index(icell2,(grid2.num_divs[0],\
                                                         grid2.num_divs[1],\
                                                         grid2.num_divs[2]))
            
                #extract the points in the cell
                x_icell2 = grid2.x[grid2.slice_array[icell2]]
                y_icell2 = grid2.y[grid2.slice_array[icell2]]
                z_icell2 = grid2.z[grid2.slice_array[icell2]]
                
                #loop over points in grid1's cell
                for i in range(0,len(x_icell1)):
                    #loop over points in grid2's cell
                    for j in range(0,len(x_icell2)):
                        #calculate the square distance
                        d = periodic_square_distance(x_icell1[i],y_icell1[i],z_icell1[i],\
                                                     x_icell2[j],y_icell2[j],z_icell2[j],\
                                                     <np.float64_t*>cperiod.data)
                        #calculate counts in bins
                        k = nbins-1
                        while d<=crbins[k]:
                            counts[k] += 1
                            k=k-1
                            if k<0: break
        elif PBCs==False:
            #Loop over each of the 27 subvolumes neighboring, including the current cell.
            for icell2 in adj_cell_arr:
                #extract the points in the cell
                x_icell2 = grid2.x[grid2.slice_array[icell2]]
                y_icell2 = grid2.y[grid2.slice_array[icell2]]
                z_icell2 = grid2.z[grid2.slice_array[icell2]]
                #loop over points in grid1's cell
                for i in range(0,len(x_icell1)):
                    #loop over points in grid2's cell
                    for j in range(0,len(x_icell2)):
                        #calculate the square distance
                        d = square_distance(x_icell1[i],y_icell1[i],z_icell1[i],\
                                            x_icell2[j],y_icell2[j],z_icell2[j])
                        #calculate counts in bins
                        k = nbins-1
                        while d<=crbins[k]:
                            counts[k] += 1
                            k=k-1
                            if k<0: break
        
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def xy_z_npairs(data1, data2, rp_bins, pi_bins, Lbox=[1.0,1.0,1.0], period=None):
    """
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
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif np.shape(period) == (1,):
        period = np.array([period[0]]*3)
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif isinstance(period, (int, long, float, complex)):
        period = np.array([period]*3)
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
    
    #c definitions
    cdef int nrp_bins = len(rp_bins)
    cdef int npi_bins = len(pi_bins)
    cdef np.ndarray[np.float64_t, ndim=1] crp_bins = \
        np.ascontiguousarray(rp_bins,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cpi_bins = \
        np.ascontiguousarray(pi_bins,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cperiod = \
        np.ascontiguousarray(period,dtype=np.float64)
    cdef np.ndarray[np.int_t, ndim=2] counts = \
        np.zeros((nrp_bins, npi_bins), dtype=np.int)
    
    #build grids for data1 and data2
    cell_size = np.array([np.max(rp_bins),np.max(rp_bins),np.max(pi_bins)])
    grid1 = cube_grid(data1[:,0], data1[:,1], data1[:,2], Lbox, cell_size)
    grid2 = cube_grid(data2[:,0], data2[:,1], data2[:,2], Lbox, cell_size)
    
    #square radial bins to make distance calculation cheaper
    crp_bins = crp_bins**2.0
    cpi_bins = cpi_bins**2.0
    
    #print come information
    #print("running grid pairs with {0} by {1} points".format(len(data1),len(data2)))
    #print("cell size= {0}".format(grid1.dL))
    #print("number of cells = {0}".format(np.prod(grid1.num_divs)))
    #print("PBCs= {0}".format(PBCs))
    
    #more c definitions used inside loop
    cdef int i, j, k, g
    cdef int icell1,icell2
    cdef int ix1, iy1, iz1, ix2, iy2, iz2
    cdef double d_perp, d_para
    cdef np.ndarray[np.float64_t, ndim=1] x_icell1, y_icell1, z_icell1
    cdef np.ndarray[np.float64_t, ndim=1] x_icell2, y_icell2, z_icell2
    cdef np.ndarray[np.int_t, ndim=1] adj_cell_arr
    
    #Loop over all subvolumes in grid1
    for icell1 in range(np.prod(grid1.num_divs)):
        #calculate progress
        progress = icell1/(np.prod(grid1.num_divs))*100
        print("    {0:.2f} %%".format(progress),end='\r')
        sys.stdout.flush()
        
        #extract the points in the cell
        x_icell1, y_icell1, z_icell1 = (grid1.x[grid1.slice_array[icell1]],\
                                        grid1.y[grid1.slice_array[icell1]],\
                                        grid1.z[grid1.slice_array[icell1]])
        
        #get the list of neighboring cells
        ix1, iy1, iz1 = np.unravel_index(icell1,(grid1.num_divs[0],\
                                                 grid1.num_divs[1],\
                                                 grid1.num_divs[2]))
        adj_cell_arr = grid1.adjacent_cells(ix1, iy1, iz1)
        
        if PBCs==True:
            #Loop over each of the 27 subvolumes neighboring, including the current cell.
            for icell2 in adj_cell_arr:
            
                ix2, iy2, iz2 = np.unravel_index(icell2,(grid2.num_divs[0],\
                                                         grid2.num_divs[1],\
                                                         grid2.num_divs[2]))
            
                #extract the points in the cell
                x_icell2 = grid2.x[grid2.slice_array[icell2]]
                y_icell2 = grid2.y[grid2.slice_array[icell2]]
                z_icell2 = grid2.z[grid2.slice_array[icell2]]
                
                #loop over points in grid1's cell
                for i in range(0,len(x_icell1)):
                    #loop over points in grid2's cell
                    for j in range(0,len(x_icell2)):
                        #calculate the square distance
                        d_perp = periodic_perp_square_distance(x_icell1[i],y_icell1[i],\
                                                               x_icell2[j],y_icell2[j],\
                                                               <np.float64_t*>cperiod.data)
                        d_para = periodic_para_square_distance(z_icell1[i],\
                                                               z_icell2[j],\
                                                               <np.float64_t*>cperiod.data)
                        #calculate counts in bins
                        k = nrp_bins-1
                        while d_perp<=crp_bins[k]:
                            g = npi_bins-1
                            while d_para<=cpi_bins[g]:
                                counts[k,g] += 1
                                g=g-1
                                if g<0: break
                            k=k-1
                            if k<0: break
        elif PBCs==False:
            #Loop over each of the 27 subvolumes neighboring, including the current cell.
            for icell2 in adj_cell_arr:
                
                #extract the points in the cell
                x_icell2 = grid2.x[grid2.slice_array[icell2]]
                y_icell2 = grid2.y[grid2.slice_array[icell2]]
                z_icell2 = grid2.z[grid2.slice_array[icell2]]
                
                #loop over points in grid1's cell
                for i in range(0,len(x_icell1)):
                    #loop over points in grid2's cell
                    for j in range(0,len(x_icell2)):
                        #calculate the square distance
                        d_perp = perp_square_distance(x_icell1[i], y_icell1[i],\
                                                      x_icell2[j], y_icell2[j])
                        d_para = para_square_distance(z_icell1[i], z_icell2[j])
                        
                        #calculate counts in bins
                        k = nrp_bins-1
                        while d_perp<=crp_bins[k]:
                            g = npi_bins-1
                            while d_para<=cpi_bins[g]:
                                counts[k,g] += 1
                                g=g-1
                                if g<0: break
                            k=k-1
                            if k<0: break
        
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def wnpairs(data1, data2, rbins, Lbox=None, period=None, weights1=None, weights2=None):
    """
    Calculate the weighted number of pairs with separations less than or equal to rbins[i].
    
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
    
    #are we working with periodic boundary conditions (PBCs)?
    if period is None: 
        PBCs = False
    elif np.shape(period) == (3,):
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif np.shape(period) == (1,):
        period = np.array([period[0]]*3)
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif isinstance(period, (int, long, float, complex)):
        period = np.array([period]*3)
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
    
    #c definitions
    cdef int nbins = len(rbins)
    cdef np.ndarray[np.float64_t, ndim=1] crbins = \
        np.ascontiguousarray(rbins,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cperiod = \
        np.ascontiguousarray(period,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] counts = \
        np.zeros((nbins,), dtype=np.float64)
    
    #build grids for data1 and data2
    cell_size = np.array([np.max(rbins)]*3)
    grid1 = cube_grid(data1[:,0], data1[:,1], data1[:,2], Lbox, cell_size)
    grid2 = cube_grid(data2[:,0], data2[:,1], data2[:,2], Lbox, cell_size)
    
    #sort the weights arrays
    weights1 = weights1[grid1.idx_sorted]
    weights2 = weights2[grid2.idx_sorted]
    cdef np.ndarray[np.float64_t, ndim=1] cweights1 = \
        np.ascontiguousarray(weights1,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cweights2 = \
        np.ascontiguousarray(weights2,dtype=np.float64)
    
    #square radial bins to make distance calculation cheaper
    crbins = crbins**2.0
    
    #print come information
    #print("running grid pairs with {0} by {1} points".format(len(data1),len(data2)))
    #print("cell size= {0}".format(grid1.dL))
    #print("number of cells = {0}".format(np.prod(grid1.num_divs)))
    #print("PBCs= {0}".format(PBCs))
    
    #more c definitions used inside loop
    cdef int i, j, k
    cdef int icell1,icell2
    cdef int ix1, iy1, iz1, ix2, iy2, iz2
    cdef double d
    cdef np.ndarray[np.float64_t, ndim=1] x_icell1, y_icell1, z_icell1
    cdef np.ndarray[np.float64_t, ndim=1] x_icell2, y_icell2, z_icell2
    cdef np.ndarray[np.float64_t, ndim=1] w_icell1, w_icell2
    cdef np.ndarray[np.int_t, ndim=1] adj_cell_arr
    
    #Loop over all subvolumes in grid1
    for icell1 in range(np.prod(grid1.num_divs)):
        #calculate progress
        progress = icell1/(np.prod(grid1.num_divs))*100
        print("    {0:.2f} %%".format(progress),end='\r')
        sys.stdout.flush()
        
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
        
        if PBCs==True:
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
                
                #loop over points in grid1's cell
                for i in range(0,len(x_icell1)):
                    #loop over points in grid2's cell
                    for j in range(0,len(x_icell2)):
                        #calculate the square distance
                        d = periodic_square_distance(x_icell1[i],y_icell1[i],z_icell1[i],\
                                                     x_icell2[j],y_icell2[j],z_icell2[j],\
                                                     <np.float64_t*>cperiod.data)
                        #calculate counts in bins
                        k = nbins-1
                        while d<=crbins[k]:
                            counts[k] += w_icell1[i]*w_icell2[j]
                            k=k-1
                            if k<0: break
        elif PBCs==False:
            #Loop over each of the 27 subvolumes neighboring, including the current cell.
            for icell2 in adj_cell_arr:
                
                #extract the points in the cell
                x_icell2 = grid2.x[grid2.slice_array[icell2]]
                y_icell2 = grid2.y[grid2.slice_array[icell2]]
                z_icell2 = grid2.z[grid2.slice_array[icell2]]
                
                #extract the weights in the cell
                w_icell2 = weights2[grid2.slice_array[icell2]]
                
                #loop over points in grid1's cell
                for i in range(0,len(x_icell1)):
                    #loop over points in grid2's cell
                    for j in range(0,len(x_icell2)):
                        #calculate the square distance
                        d = square_distance(x_icell1[i],y_icell1[i],z_icell1[i],\
                                            x_icell2[j],y_icell2[j],z_icell2[j])
                        #calculate counts in bins
                        k = nbins-1
                        while d<=crbins[k]:
                            counts[k] += w_icell1[i]*w_icell2[j]
                            k=k-1
                            if k<0: break
        
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def xy_z_wnpairs(data1, data2, rp_bins, pi_bins, Lbox=[1.0,1.0,1.0], period=None, weights1=None, weights2=None):
    """
    Calculate the weighted number of pairs with separations in the x-y plane less than or equal 
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
    
    #are we working with periodic boundary conditions (PBCs)?
    if period is None: 
        PBCs = False
    elif np.shape(period) == (3,):
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif np.shape(period) == (1,):
        period = np.array([period[0]]*3)
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif isinstance(period, (int, long, float, complex)):
        period = np.array([period]*3)
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
    
    #c definitions
    cdef int nrp_bins = len(rp_bins)
    cdef int npi_bins = len(pi_bins)
    cdef np.ndarray[np.float64_t, ndim=1] crp_bins = \
        np.ascontiguousarray(rp_bins,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cpi_bins = \
        np.ascontiguousarray(pi_bins,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cperiod = \
        np.ascontiguousarray(period,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] counts = \
        np.zeros((nrp_bins, npi_bins), dtype=np.float64)
    
    #build grids for data1 and data2
    cell_size = np.array([np.max(rp_bins),np.max(rp_bins),np.max(pi_bins)])
    grid1 = cube_grid(data1[:,0], data1[:,1], data1[:,2], Lbox, cell_size)
    grid2 = cube_grid(data2[:,0], data2[:,1], data2[:,2], Lbox, cell_size)
    
    #sort the weights arrays
    weights1 = weights1[grid1.idx_sorted]
    weights2 = weights2[grid2.idx_sorted]
    cdef np.ndarray[np.float64_t, ndim=1] cweights1 = \
        np.ascontiguousarray(weights1,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cweights2 = \
        np.ascontiguousarray(weights2,dtype=np.float64)
    
    #square radial bins to make distance calculation cheaper
    crp_bins = crp_bins**2.0
    cpi_bins = cpi_bins**2.0
    
    #print come information
    #print("running grid pairs with {0} by {1} points".format(len(data1),len(data2)))
    #print("cell size= {0}".format(grid1.dL))
    #print("number of cells = {0}".format(np.prod(grid1.num_divs)))
    #print("PBCs= {0}".format(PBCs))
    
    #more c definitions used inside loop
    cdef int i, j, k, g
    cdef int icell1,icell2
    cdef int ix1, iy1, iz1, ix2, iy2, iz2
    cdef double d_perp, d_para
    cdef np.ndarray[np.float64_t, ndim=1] x_icell1, y_icell1, z_icell1
    cdef np.ndarray[np.float64_t, ndim=1] x_icell2, y_icell2, z_icell2
    cdef np.ndarray[np.float64_t, ndim=1] w_icell1, w_icell2
    cdef np.ndarray[np.int_t, ndim=1] adj_cell_arr
    
    #Loop over all subvolumes in grid1
    for icell1 in range(np.prod(grid1.num_divs)):
        #calculate progress
        progress = icell1/(np.prod(grid1.num_divs))*100
        print("    {0:.2f} %%".format(progress),end='\r')
        sys.stdout.flush()
        
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
        
        if PBCs==True:
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
                
                #loop over points in grid1's cell
                for i in range(0,len(x_icell1)):
                    #loop over points in grid2's cell
                    for j in range(0,len(x_icell2)):
                        #calculate the square distance
                        d_perp = periodic_perp_square_distance(x_icell1[i],y_icell1[i],\
                                                               x_icell2[j],y_icell2[j],\
                                                               <np.float64_t*>cperiod.data)
                        d_para = periodic_para_square_distance(z_icell1[i],\
                                                               z_icell2[j],\
                                                               <np.float64_t*>cperiod.data)
                        #calculate counts in bins
                        k = nrp_bins-1
                        while d_perp<=crp_bins[k]:
                            g = npi_bins-1
                            while d_para<=cpi_bins[g]:
                                counts[k,g] += w_icell1[i]*w_icell2[j]
                                g=g-1
                                if g<0: break
                            k=k-1
                            if k<0: break
        elif PBCs==False:
            #Loop over each of the 27 subvolumes neighboring, including the current cell.
            for icell2 in adj_cell_arr:
                
                #extract the points in the cell
                x_icell2 = grid2.x[grid2.slice_array[icell2]]
                y_icell2 = grid2.y[grid2.slice_array[icell2]]
                z_icell2 = grid2.z[grid2.slice_array[icell2]]
                
                #extract the weights in the cell
                w_icell2 = weights2[grid2.slice_array[icell2]]
                
                #loop over points in grid1's cell
                for i in range(0,len(x_icell1)):
                    #loop over points in grid2's cell
                    for j in range(0,len(x_icell2)):
                        #calculate the square distance
                        d_perp = perp_square_distance(x_icell1[i], y_icell1[i],\
                                                      x_icell2[j], y_icell2[j])
                        d_para = para_square_distance(z_icell1[i], z_icell2[j])
                        
                        #calculate counts in bins
                        k = nrp_bins-1
                        while d_perp<=crp_bins[k]:
                            g = npi_bins-1
                            while d_para<=cpi_bins[g]:
                                counts[k,g] += w_icell1[i]*w_icell2[j]
                                g=g-1
                                if g<0: break
                            k=k-1
                            if k<0: break
        
    return counts


def jnpairs(data1, data2, rbins, Lbox=None, period=None, weights1=None, weights2=None,\
            jtags1=None, jtags2=None, N_samples=1):
    """
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
    
    #are we working with periodic boundary conditions (PBCs)?
    if period is None: 
        PBCs = False
    elif np.shape(period) == (3,):
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif np.shape(period) == (1,):
        period = np.array([period[0]]*3)
        if np.any(period!=Lbox):
            raise ValueError("period must == Lbox") 
    elif isinstance(period, (int, long, float, complex)):
        period = np.array([period]*3)
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
    
    #c definitions
    cdef int nbins = len(rbins)
    cdef np.ndarray[np.float64_t, ndim=1] crbins = \
        np.ascontiguousarray(rbins,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cperiod = \
        np.ascontiguousarray(period,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] counts = \
        np.zeros((N_samples+1, nbins), dtype=np.float64)
    
    #build grids for data1 and data2
    cell_size = np.array([np.max(rbins)]*3)
    grid1 = cube_grid(data1[:,0], data1[:,1], data1[:,2], Lbox, cell_size)
    grid2 = cube_grid(data2[:,0], data2[:,1], data2[:,2], Lbox, cell_size)
    
    #sort the weights arrays
    weights1 = weights1[grid1.idx_sorted]
    weights2 = weights2[grid2.idx_sorted]
    cdef np.ndarray[np.float64_t, ndim=1] cweights1 = \
        np.ascontiguousarray(weights1,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cweights2 = \
        np.ascontiguousarray(weights2,dtype=np.float64)
        
    #sort the jackknife tag arrays
    jtags1 = jtags1[grid1.idx_sorted]
    jtags2 = jtags2[grid2.idx_sorted]
    cdef np.ndarray[np.int_t, ndim=1] cjtags1 = \
        np.ascontiguousarray(jtags1,dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=1] cjtags2 = \
        np.ascontiguousarray(jtags2,dtype=np.int)
    
    #square radial bins to make distance calculation cheaper
    crbins = crbins**2.0
    
    #print come information
    #print("running grid pairs with {0} by {1} points".format(len(data1),len(data2)))
    #print("cell size= {0}".format(grid1.dL))
    #print("number of cells = {0}".format(np.prod(grid1.num_divs)))
    #print("PBCs= {0}".format(PBCs))
    
    #more c definitions used inside loop
    cdef int i, j, k, l
    cdef int cN_samples = N_samples+1
    cdef int icell1,icell2
    cdef int ix1, iy1, iz1, ix2, iy2, iz2
    cdef double d
    cdef np.ndarray[np.float64_t, ndim=1] x_icell1, y_icell1, z_icell1
    cdef np.ndarray[np.float64_t, ndim=1] x_icell2, y_icell2, z_icell2
    cdef np.ndarray[np.float64_t, ndim=1] w_icell1, w_icell2
    cdef np.ndarray[np.int_t, ndim=1] j_icell1, j_icell2
    cdef np.ndarray[np.int_t, ndim=1] adj_cell_arr
    
    #Loop over all subvolumes in grid1
    for icell1 in range(np.prod(grid1.num_divs)):
        #calculate progress
        progress = icell1/(np.prod(grid1.num_divs))*100
        print("    {0:.2f} %%".format(progress),end='\r')
        sys.stdout.flush()
        
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
        
        if PBCs==True:
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
                
                #loop over points in grid1's cell
                for i in range(0,len(x_icell1)):
                    #loop over points in grid2's cell
                    for j in range(0,len(x_icell2)):
                        #calculate the square distance
                        d = periodic_square_distance(x_icell1[i],y_icell1[i],z_icell1[i],\
                                                     x_icell2[j],y_icell2[j],z_icell2[j],\
                                                     <np.float64_t*>cperiod.data)
                        #calculate counts in bins
                        for l in range(0,cN_samples):
                            k = nbins-1
                            while d<=crbins[k]:
                                counts[l,k] += jweight(l, j_icell1[i],j_icell2[j],\
                                                       w_icell1[i],w_icell2[j])
                                k=k-1
                                if k<0: break
        elif PBCs==False:
            #Loop over each of the 27 subvolumes neighboring, including the current cell.
            for icell2 in adj_cell_arr:
                
                #extract the points in the cell
                x_icell2 = grid2.x[grid2.slice_array[icell2]]
                y_icell2 = grid2.y[grid2.slice_array[icell2]]
                z_icell2 = grid2.z[grid2.slice_array[icell2]]
                
                #extract the weights in the cell
                w_icell2 = weights2[grid2.slice_array[icell2]]
        
                #extract the jackknife tags in the cell
                j_icell2 = jtags2[grid2.slice_array[icell2]]
                
                #loop over points in grid1's cell
                for i in range(0,len(x_icell1)):
                    #loop over points in grid2's cell
                    for j in range(0,len(x_icell2)):
                        #calculate the square distance
                        d = square_distance(x_icell1[i],y_icell1[i],z_icell1[i],\
                                            x_icell2[j],y_icell2[j],z_icell2[j])
                        #calculate counts in bins
                        for l in range(0,cN_samples):
                            k = nbins-1
                            while d<=crbins[k]:
                                counts[l,k] += jweight(l, j_icell1[i],j_icell2[j],\
                                                       w_icell1[i],w_icell2[j])
                                k=k-1
                                if k<0: break
        
    return counts


cdef inline double periodic_square_distance(np.float64_t x1, np.float64_t y1, np.float64_t z1,\
                                          np.float64_t x2, np.float64_t y2, np.float64_t z2,\
                                          np.float64_t* period):
    """
    Calculate the 3D square cartesian distance between two sets of points with periodic
    boundary conditions.
    """
    
    cdef double dx, dy, dz
    
    dx = fabs(x1 - x2)
    dx = fmin(dx, period[0] - dx)
    dy = fabs(y1 - y2)
    dy = fmin(dy, period[1] - dy)
    dz = fabs(z1 - z2)
    dz = fmin(dz, period[2] - dz)
    return dx*dx+dy*dy+dz*dz


cdef inline double square_distance(np.float64_t x1, np.float64_t y1, np.float64_t z1,\
                                   np.float64_t x2, np.float64_t y2, np.float64_t z2):
    """
    Calculate the 3D square cartesian distance between two sets of points.
    """
    
    cdef double dx, dy, dz
    
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    return dx*dx+dy*dy+dz*dz


cdef inline double perp_square_distance(np.float64_t x1, np.float64_t y1,\
                                        np.float64_t x2, np.float64_t y2):
    """
    Calculate the projected square cartesian distance between two sets of points.
    e.g. r_p
    """
    
    cdef double dx, dy
    
    dx = x1 - x2
    dy = y1 - y2
    return dx*dx+dy*dy


cdef inline double para_square_distance(np.float64_t z1, np.float64_t z2):
    """
    Calculate the parallel square cartesian distance between two sets of points.
    e.g. pi
    """
    
    cdef double dz
    
    dz = z1 - z2
    return dz*dz


cdef inline double periodic_perp_square_distance(np.float64_t x1, np.float64_t y1,\
                                                 np.float64_t x2, np.float64_t y2,\
                                                 np.float64_t* period):
    """
    Calculate the projected square cartesian distance between two sets of points with 
    periodic boundary conditions.
    e.g. r_p
    """
    
    cdef double dx, dy
    
    dx = fabs(x1 - x2)
    dx = fmin(dx, period[0] - dx)
    dy = fabs(y1 - y2)
    dy= fmin(dy, period[1] - dy)
    return dx*dx+dy*dy


cdef inline double periodic_para_square_distance(np.float64_t z1, np.float64_t z2,\
                                                 np.float64_t* period):
    """
    Calculate the parallel square cartesian distance between two sets of points with 
    periodic boundary conditions.
    e.g. pi
    """
    
    cdef double dz
    
    dz = fabs(z1 - z2)
    dz = fmin(dz, period[2] - dz)
    return dz*dz


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
class cube_grid():

    def __init__(self, x, y, z, Lbox, cell_size):
        """
        Initialize the grid. 

        Parameters 
        ----------
        x, y, z : arrays
            Length-Npts arrays containing the spatial position of the Npts points. 
        
        Lbox : float
            Length scale defining the periodic boundary conditions

        cell_size : float 
            The approximate cell size into which the box will be divided. 
        """

        self.cell_size = cell_size.astype(np.float)
        self.Lbox = Lbox.astype(np.float)
        self.num_divs = np.floor(Lbox/cell_size).astype(int)
        self.dL = Lbox/self.num_divs
        
        #build grid tree
        idx_sorted, slice_array = self.compute_cell_structure(x, y, z)
        self.x = np.ascontiguousarray(x[idx_sorted],dtype=np.float64)
        self.y = np.ascontiguousarray(y[idx_sorted],dtype=np.float64)
        self.z = np.ascontiguousarray(z[idx_sorted],dtype=np.float64)
        self.slice_array = slice_array
        self.idx_sorted = idx_sorted

    def compute_cell_structure(self, x, y, z):
        """ 
        Method divides the periodic box into regular, cubical subvolumes, and assigns a 
        subvolume index to each point.  The returned arrays can be used to efficiently 
        access only those points in a given subvolume. 

        Parameters 
        ----------
        x, y, z : arrays
            Length-Npts arrays containing the spatial position of the Npts points. 

        Returns 
        -------
        idx_sorted : array
            Array of indices that sort the points according to the dictionary 
            order of the 3d subvolumes. 

        slice_array : array 
            array of slice objects used to access the elements of x, y, and z 
            of points residing in a given subvolume. 

        Notes 
        -----
        The dictionary ordering of 3d cells where :math:`dL = L_{\\rm box} / 2` 
        is defined as follows:

            * (0, 0, 0) <--> 0

            * (0, 0, 1) <--> 1

            * (0, 1, 0) <--> 2

            * (0, 1, 1) <--> 3

        And so forth. Each of the Npts thus has a unique triplet, 
        or equivalently, unique integer specifying the subvolume containing the point. 
        The unique integer is called the *cellID*. 
        In order to access the *x* positions of the points lying in subvolume *i*, 
        x[idx_sort][slice_array[i]]. 

        In practice, because fancy indexing with `idx_sort` is not instantaneous, 
        it will be more efficient to use `idx_sort` once to sort the x, y, and z arrays 
        in-place, and then access the sorted arrays with the relevant slice_array element. 
        This is the strategy used in the `retrieve_tree` method. 

        """

        ix = np.floor(x/self.dL[0]).astype(int)
        iy = np.floor(y/self.dL[1]).astype(int)
        iz = np.floor(z/self.dL[2]).astype(int)
        
        #take care of points right on the boundary
        inds = np.where(ix>=self.num_divs[0])[0]
        ix[inds]= self.num_divs[0]-1
        inds = np.where(iy>=self.num_divs[1])[0]
        iy[inds]= self.num_divs[1]-1
        inds = np.where(iz>=self.num_divs[2])[0]
        iz[inds]= self.num_divs[2]-1

        particle_indices = np.ravel_multi_index((ix, iy, iz),\
                                               (self.num_divs[0],\
                                                self.num_divs[1],\
                                                self.num_divs[2]))
        
        idx_sorted = np.argsort(particle_indices)
        bin_indices = np.searchsorted(particle_indices[idx_sorted], 
                                      np.arange(np.prod(self.num_divs)))
        bin_indices = np.append(bin_indices, None)
        
        slice_array = np.empty(np.prod(self.num_divs), dtype=object)
        for icell in range(np.prod(self.num_divs)):
            slice_array[icell] = slice(bin_indices[icell], bin_indices[icell+1], 1)
            
        return idx_sorted, slice_array
    
    
    def adjacent_cells(self, *args):
        """ 
        Given a subvolume specified by the input arguments,  
        return the length-27 array of cellIDs of the neighboring cells. 
        The input subvolume can be specified either by its ix, iy, iz triplet, 
        or by its cellID. 
        Parameters 
        ----------
        ix, iy, iz : int, optional
            Integers specifying the ix, iy, and iz triplet of the subvolume. 
            If ix, iy, and iz are not passed, then ic must be passed. 
        ic : int, optional
            Integer specifying the cellID of the input subvolume
            If ic is not passed, the ix, iy, and iz must be passed. 
        Returns 
        -------
        result : int array
            Length-27 array of cellIDs of neighboring subvolumes. 
        Notes 
        -----
        If one argument is passed to `adjacent_cells`, this argument will be 
        interpreted as the cellID of the input subvolume. 
        If three arguments are passed, these will be interpreted as 
        the ix, iy, iz triplet of the input subvolume. 
        """

        ixgen, iygen, izgen = np.unravel_index(np.arange(3**3), (3, 3, 3)) 

        if len(args) >= 3:
            ix, iy, iz = args[0], args[1], args[2]
        elif len(args) == 1:
            ic = args[0]
            ix, iy, iz = np.unravel_index(ic, (self.num_divs, self.num_divs, self.num_divs))

        ixgen = (ixgen + ix - 1) % self.num_divs[0]
        iygen = (iygen + iy - 1) % self.num_divs[1]
        izgen = (izgen + iz - 1) % self.num_divs[2]

        return np.unique(np.ravel_multi_index((ixgen, iygen, izgen), 
            (self.num_divs[0], self.num_divs[1], self.num_divs[2])))


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


cdef inline double jweight(np.int_t j, np.int_t j1, np.int_t j2,\
                           np.float64_t w1, np.float64_t w2):
    """
    return jackknife weighted counts
    
    parameters
    ----------
    j: jackknife subsample
    j1: jackknife sample 1 tag
    j2: jackknife sample 2 tag
    w1: weight1
    w2: weight2
    
    notes
    -----
    if sample j==0, do no jackknife weighting.  i.e. reserve this for the full sample.
    if both points are inside the sample, return w1*w2
    if both points are outside the sample, return 0.0
    if one point is within and one point is outside the sample, return 0.5*w1*w2
    """
    
    if j==0: return (w1 * w2)
    # both outside the sub-sample
    elif (j1 == j2) & (j1 == j): return 0.0
    # both inside the sub-sample 
    elif (j1 == j2): return (w1 * w2)
    # only one inside the sub-sample
    elif (j1 != j2) & ((j1 == j) or (j2 == j)): return 0.5*(w1 * w2)
    # both inside the sub-sample
    elif (j1 != j2) & (j1 != j) & (j2 != j): return (w1 * w2)
    
    