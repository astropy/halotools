#!/usr/bin/env python
# cython: profile=False

"""
pair counters optimized to run on simulation boxes.
"""

from __future__ import print_function, division
__all__ = ['npairs_no_pbc', 'npairs_pbc', 'xy_z_npairs_no_pbc', 'xy_z_npairs_pbc',\
           'wnpairs_no_pbc', 'wnpairs_pbc', 'xy_z_wnpairs_no_pbc', 'xy_z_wnpairs_pbc',\
           'jnpairs_no_pbc', 'jnpairs_pbc', 'xy_z_jnpairs_no_pbc', 'xy_z_jnpairs_pbc']

import sys
import numpy as np
from cube_grid import cube_grid
cimport cython
cimport numpy as np
from libc.math cimport fabs, fmin

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def npairs_no_pbc(data1, data2, rbins, Lbox, verbose=False):
    """
    real-space pair counter without periodic boundary conditions (no PBCs).
    Calculate the number of pairs with separations less than or equal to rbins[i].
    """
    
    #c definitions
    cdef int nbins = len(rbins)
    cdef int nbins_minus_one = len(rbins) -1  #to make this a 0-indexed counter
    cdef np.ndarray[np.float64_t, ndim=1] crbins = \
        np.ascontiguousarray(rbins,dtype=np.float64)
    cdef np.ndarray[np.int_t, ndim=1] counts = \
        np.zeros((nbins,), dtype=np.int)
    
    #build grids for data1 and data2
    cell_size = np.array([np.max(rbins)]*3)
    grid1 = cube_grid(data1[:,0], data1[:,1], data1[:,2], Lbox, cell_size)
    grid2 = cube_grid(data2[:,0], data2[:,1], data2[:,2], Lbox, cell_size)
    
    #square radial bins to make distance calculation cheaper
    crbins = crbins**2.0
    
    #print come information
    if verbose==True:
        print("running grid pairs with {0} by {1} points".format(len(data1),len(data2)))
        print("cell size= {0}".format(grid1.dL))
        print("number of cells = {0}".format(np.prod(grid1.num_divs)))
    
    #more c definitions used inside loop
    cdef int i, j, k
    cdef int icell1,icell2
    cdef int ix1, iy1, iz1, ix2, iy2, iz2
    cdef double d, progress
    cdef np.ndarray[np.float64_t, ndim=1] x_icell1, y_icell1, z_icell1
    cdef np.ndarray[np.float64_t, ndim=1] x_icell2, y_icell2, z_icell2
    cdef np.ndarray[np.int_t, ndim=1] adj_cell_arr
    cdef int Ncell1 = np.prod(grid1.num_divs)
    
    #Loop over all subvolumes in grid1
    for icell1 in range(Ncell1):
        
        #calculate progress
        if verbose==True:
            progress = icell1/Ncell1*100
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
            
        #Loop over each of the 27 subvolumes neighboring, including the current cell.
        for icell2 in adj_cell_arr:
                
            #extract the points in the cell
            x_icell2 = grid2.x[grid2.slice_array[icell2]]
            y_icell2 = grid2.y[grid2.slice_array[icell2]]
            z_icell2 = grid2.z[grid2.slice_array[icell2]]
                
            #loop over points in grid1's cells
            for i in range(0,len(x_icell1)):
                    
                #loop over points in grid2's cells
                for j in range(0,len(x_icell2)):
                        
                    #calculate the square distance
                    d = square_distance(x_icell1[i],y_icell1[i],z_icell1[i],\
                                        x_icell2[j],y_icell2[j],z_icell2[j])
                        
                    #calculate counts in bins
                    radial_binning(<np.int_t*> counts.data,\
                                   <np.float64_t*> crbins.data, d, nbins_minus_one)
        
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def npairs_pbc(data1, data2, rbins, Lbox, period, verbose=False):
    """
    real-space pair counter with periodic boundary conditions (PBCs).
    Calculate the number of pairs with separations less than or equal to rbins[i].
    """
    
    #c definitions
    cdef int nbins = len(rbins)
    cdef int nbins_minus_one = len(rbins) -1  #to make this a 0-indexed counter
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
    if verbose==True:
        print("running grid pairs with {0} by {1} points".format(len(data1),len(data2)))
        print("cell size= {0}".format(grid1.dL))
        print("number of cells = {0}".format(np.prod(grid1.num_divs)))
    
    #more c definitions used inside loop
    cdef int i, j, k
    cdef int icell1,icell2
    cdef int ix1, iy1, iz1, ix2, iy2, iz2
    cdef double d, progress
    cdef np.ndarray[np.float64_t, ndim=1] x_icell1, y_icell1, z_icell1
    cdef np.ndarray[np.float64_t, ndim=1] x_icell2, y_icell2, z_icell2
    cdef np.ndarray[np.int_t, ndim=1] adj_cell_arr
    cdef int Ncell1=np.prod(grid1.num_divs)
    
    
    #Loop over all subvolumes in grid1
    for icell1 in range(Ncell1):
        
        #calculate progress
        if verbose==True:
            progress = icell1/(Ncell1)*100
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
                    radial_binning(<np.int_t*>counts.data,\
                                   <np.float64_t*>crbins.data, d, nbins_minus_one)
    
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def xy_z_npairs_no_pbc(data1, data2, rp_bins, pi_bins, Lbox, verbose=False):
    """
    2+1D pair counter without periodic boundary conditions (no PBCs).
    Calculate the number of pairs with separations in the x-y plane less than or equal 
    to rp_bins[i], and separations in the z coordinate less than or equal to pi_bins[i].
    """
    
    #c definitions
    cdef int nrp_bins = len(rp_bins)
    cdef int npi_bins = len(pi_bins)
    cdef int nrp_bins_minus_one = len(rp_bins)-1 #to make this a 0-indexed counter
    cdef int npi_bins_minus_one = len(pi_bins)-1 #to make this a 0-indexed counter
    cdef np.ndarray[np.float64_t, ndim=1] crp_bins = \
        np.ascontiguousarray(rp_bins,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cpi_bins = \
        np.ascontiguousarray(pi_bins,dtype=np.float64)
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
    if verbose==True:
        print("running grid pairs with {0} by {1} points".format(len(data1),len(data2)))
        print("cell size= {0}".format(grid1.dL))
        print("number of cells = {0}".format(np.prod(grid1.num_divs)))
    
    #more c definitions used inside loop
    cdef int i, j, k, g
    cdef int icell1,icell2
    cdef int ix1, iy1, iz1, ix2, iy2, iz2
    cdef double d_perp, d_para, progress
    cdef np.ndarray[np.float64_t, ndim=1] x_icell1, y_icell1, z_icell1
    cdef np.ndarray[np.float64_t, ndim=1] x_icell2, y_icell2, z_icell2
    cdef np.ndarray[np.int_t, ndim=1] adj_cell_arr
    cdef int Ncell1=np.prod(grid1.num_divs)
    
    #Loop over all subvolumes in grid1
    for icell1 in range(Ncell1):
        
        #calculate progress
        if verbose==True:
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
                    xy_z_binning(<np.int_t*>counts.data,\
                                 <np.float64_t*>crp_bins.data,\
                                 <np.float64_t*>cpi_bins.data,\
                                 d_perp, d_para, nrp_bins_minus_one, npi_bins_minus_one)
        
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def xy_z_npairs_pbc(data1, data2, rp_bins, pi_bins, Lbox, period,\
                    verbose=False):
    """
    2+1D pair counter with periodic boundary conditions (PBCs).
    Calculate the number of pairs with separations in the x-y plane less than or equal 
    to rp_bins[i], and separations in the z coordinate less than or equal to pi_bins[i].
    """
    
    #c definitions
    cdef int nrp_bins = len(rp_bins)
    cdef int npi_bins = len(pi_bins)
    cdef int nrp_bins_minus_one = len(rp_bins)-1 #to make this a 0-indexed counter
    cdef int npi_bins_minus_one = len(pi_bins)-1 #to make this a 0-indexed counter
    cdef np.ndarray[np.float64_t, ndim=1] crp_bins = \
        np.ascontiguousarray(rp_bins,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cpi_bins = \
        np.ascontiguousarray(pi_bins,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cperiod = \
        np.ascontiguousarray(period,dtype=np.float64)
    cdef np.ndarray[np.int_t, ndim=2] counts = \
        np.ascontiguousarray(np.zeros((nrp_bins, npi_bins), dtype=np.int))
    
    #build grids for data1 and data2
    cell_size = np.array([np.max(rp_bins),np.max(rp_bins),np.max(pi_bins)])
    grid1 = cube_grid(data1[:,0], data1[:,1], data1[:,2], Lbox, cell_size)
    grid2 = cube_grid(data2[:,0], data2[:,1], data2[:,2], Lbox, cell_size)
    
    #square radial bins to make distance calculation cheaper
    crp_bins = crp_bins**2.0
    cpi_bins = cpi_bins**2.0
    
    #print come information
    if verbose==True:
        print("running grid pairs with {0} by {1} points".format(len(data1),len(data2)))
        print("cell size= {0}".format(grid1.dL))
        print("number of cells = {0}".format(np.prod(grid1.num_divs)))
    
    #more c definitions used inside loop
    cdef int i, j, k, g
    cdef int icell1,icell2
    cdef int ix1, iy1, iz1, ix2, iy2, iz2
    cdef double d_perp, d_para, progress
    cdef np.ndarray[np.float64_t, ndim=1] x_icell1, y_icell1, z_icell1
    cdef np.ndarray[np.float64_t, ndim=1] x_icell2, y_icell2, z_icell2
    cdef np.ndarray[np.int_t, ndim=1] adj_cell_arr
    cdef int Ncell1=np.prod(grid1.num_divs)
    
    #Loop over all subvolumes in grid1
    for icell1 in range(Ncell1):
        
        #calculate progress
        if verbose==True:
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
                    xy_z_binning(<np.int_t*>counts.data,\
                                 <np.float64_t*>crp_bins.data,\
                                 <np.float64_t*>cpi_bins.data,\
                                 d_perp, d_para, nrp_bins_minus_one, npi_bins_minus_one)
        
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def wnpairs_no_pbc(data1, data2, rbins, Lbox, weights1, weights2, verbose=False):
    """
    weighted real-space pair counter without periodic boundary conditions (no PBCs)..
    Calculate the weighted number of pairs with separations less than or equal to 
    rbins[i].
    """
    
    #c definitions
    cdef int nbins = len(rbins)
    cdef int nbins_minus_one = len(rbins) -1  #to make this a 0-indexed counter
    cdef np.ndarray[np.float64_t, ndim=1] crbins = \
        np.ascontiguousarray(rbins,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] counts = \
        np.ascontiguousarray(np.zeros((nbins,), dtype=np.float64))
    
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
    if verbose==True:
        print("running grid pairs with {0} by {1} points".format(len(data1),len(data2)))
        print("cell size= {0}".format(grid1.dL))
        print("number of cells = {0}".format(np.prod(grid1.num_divs)))
    
    #more c definitions used inside loop
    cdef int i, j, k
    cdef int icell1,icell2
    cdef int ix1, iy1, iz1, ix2, iy2, iz2
    cdef double d, progress
    cdef np.ndarray[np.float64_t, ndim=1] x_icell1, y_icell1, z_icell1
    cdef np.ndarray[np.float64_t, ndim=1] x_icell2, y_icell2, z_icell2
    cdef np.ndarray[np.float64_t, ndim=1] w_icell1, w_icell2
    cdef np.ndarray[np.int_t, ndim=1] adj_cell_arr
    cdef int Ncell1=np.prod(grid1.num_divs)
    
    #Loop over all subvolumes in grid1
    for icell1 in range(Ncell1):
        
        #calculate progress
        if verbose==True:
            progress = icell1/(Ncell1)*100
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
                    radial_wbinning(<np.float64_t*>counts.data,\
                                    <np.float64_t*>crbins.data, d, nbins_minus_one,\
                                    w_icell1[i], w_icell2[j])
        
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def wnpairs_pbc(data1, data2, rbins, Lbox, period, weights1, weights2, verbose=False):
    """
    weighted real-space pair counter with periodic boundary conditions (PBCs).
    Calculate the weighted number of pairs with separations less than or equal to 
    rbins[i].
    """
    
    #c definitions
    cdef int nbins = len(rbins)
    cdef int nbins_minus_one = len(rbins) -1  #to make this a 0-indexed counter
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
    if verbose==True:
        print("running grid pairs with {0} by {1} points".format(len(data1),len(data2)))
        print("cell size= {0}".format(grid1.dL))
        print("number of cells = {0}".format(np.prod(grid1.num_divs)))
    
    #more c definitions used inside loop
    cdef int i, j, k
    cdef int icell1,icell2
    cdef int ix1, iy1, iz1, ix2, iy2, iz2
    cdef double d, progress
    cdef np.ndarray[np.float64_t, ndim=1] x_icell1, y_icell1, z_icell1
    cdef np.ndarray[np.float64_t, ndim=1] x_icell2, y_icell2, z_icell2
    cdef np.ndarray[np.float64_t, ndim=1] w_icell1, w_icell2
    cdef np.ndarray[np.int_t, ndim=1] adj_cell_arr
    cdef int Ncell1=np.prod(grid1.num_divs)
    
    #Loop over all subvolumes in grid1
    for icell1 in range(Ncell1):
        
        #calculate progress
        if verbose==True:
            progress = icell1/(Ncell1)*100
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
                    radial_wbinning(<np.float64_t*>counts.data,\
                                    <np.float64_t*>crbins.data, d, nbins_minus_one,\
                                    w_icell1[i], w_icell2[j])
        
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def xy_z_wnpairs_no_pbc(data1, data2, rp_bins, pi_bins, Lbox, weights1, weights2,\
                        verbose=False):
    """
    weighted 2+1D pair counter.
    Calculate the weighted number of pairs with separations in the x-y plane less than or 
    equal to rp_bins[i], and separations in the z coordinate less than or equal to 
    pi_bins[i].
    """
    
    #c definitions
    cdef int nrp_bins = len(rp_bins)
    cdef int npi_bins = len(pi_bins)
    cdef int nrp_bins_minus_one = len(rp_bins)-1
    cdef int npi_bins_minus_one = len(pi_bins)-1
    cdef np.ndarray[np.float64_t, ndim=1] crp_bins = \
        np.ascontiguousarray(rp_bins,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cpi_bins = \
        np.ascontiguousarray(pi_bins,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] counts = \
        np.ascontiguousarray(np.zeros((nrp_bins, npi_bins), dtype=np.float64))
    
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
    if verbose==True:
        print("running grid pairs with {0} by {1} points".format(len(data1),len(data2)))
        print("cell size= {0}".format(grid1.dL))
        print("number of cells = {0}".format(np.prod(grid1.num_divs)))
    
    #more c definitions used inside loop
    cdef int i, j, k, g
    cdef int icell1,icell2
    cdef int ix1, iy1, iz1, ix2, iy2, iz2
    cdef double d_perp, d_para, progress
    cdef np.ndarray[np.float64_t, ndim=1] x_icell1, y_icell1, z_icell1
    cdef np.ndarray[np.float64_t, ndim=1] x_icell2, y_icell2, z_icell2
    cdef np.ndarray[np.float64_t, ndim=1] w_icell1, w_icell2
    cdef np.ndarray[np.int_t, ndim=1] adj_cell_arr
    cdef int Ncell1=np.prod(grid1.num_divs)
    
    #Loop over all subvolumes in grid1
    for icell1 in range(Ncell1):
        
        #calculate progress
        if verbose==True:
            progress = icell1/(Ncell1)*100
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
                    xy_z_wbinning(<np.float64_t*>counts.data, <np.float64_t*> crp_bins.data,\
                                  <np.float64_t*> cpi_bins.data, d_perp,\
                                  d_para, nrp_bins_minus_one,\
                                  npi_bins_minus_one, w_icell1[i], w_icell2[j])
        
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def xy_z_wnpairs_pbc(data1, data2, rp_bins, pi_bins, Lbox, period, weights1, weights2,\
                     verbose=False):
    """
    weighted 2+1D pair counter.
    Calculate the weighted number of pairs with separations in the x-y plane less than or 
    equal to rp_bins[i], and separations in the z coordinate less than or equal to 
    pi_bins[i].
    """
    
    #c definitions
    cdef int nrp_bins = len(rp_bins)
    cdef int npi_bins = len(pi_bins)
    cdef int nrp_bins_minus_one = len(rp_bins)-1
    cdef int npi_bins_minus_one = len(pi_bins)-1
    cdef np.ndarray[np.float64_t, ndim=1] crp_bins = \
        np.ascontiguousarray(rp_bins,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cpi_bins = \
        np.ascontiguousarray(pi_bins,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cperiod = \
        np.ascontiguousarray(period,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] counts = \
        np.ascontiguousarray(np.zeros((nrp_bins, npi_bins), dtype=np.float64))
    
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
    if verbose==True:
        print("running grid pairs with {0} by {1} points".format(len(data1),len(data2)))
        print("cell size= {0}".format(grid1.dL))
        print("number of cells = {0}".format(np.prod(grid1.num_divs)))
    
    #more c definitions used inside loop
    cdef int i, j, k, g
    cdef int icell1,icell2
    cdef int ix1, iy1, iz1, ix2, iy2, iz2
    cdef double d_perp, d_para, progress
    cdef np.ndarray[np.float64_t, ndim=1] x_icell1, y_icell1, z_icell1
    cdef np.ndarray[np.float64_t, ndim=1] x_icell2, y_icell2, z_icell2
    cdef np.ndarray[np.float64_t, ndim=1] w_icell1, w_icell2
    cdef np.ndarray[np.int_t, ndim=1] adj_cell_arr
    cdef int Ncell1=np.prod(grid1.num_divs)
    
    #Loop over all subvolumes in grid1
    for icell1 in range(Ncell1):
        
        #calculate progress
        if verbose==True:
            progress = icell1/(Ncell1)*100
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
                    xy_z_wbinning(<np.float64_t*>counts.data, <np.float64_t*> crp_bins.data,\
                                  <np.float64_t*> cpi_bins.data, d_perp,\
                                  d_para, nrp_bins_minus_one,\
                                  npi_bins_minus_one, w_icell1[i], w_icell2[j])
        
    return counts


def jnpairs_no_pbc(data1, data2, rbins, Lbox, weights1, weights2, jtags1, jtags2,\
                   N_samples, verbose=False):
    """
    jackknife weighted real-space pair counter.
    Calculate the weighted number of pairs with separations less than or equal to rbins[i]
    for a jackknife sample.
    """
    
    #c definitions
    cdef int nbins = len(rbins)
    cdef int nbins_minus_one = len(rbins)-1
    cdef np.ndarray[np.float64_t, ndim=1] crbins = \
        np.ascontiguousarray(rbins,dtype=np.float64)
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
    if verbose==True:
        print("running grid pairs with {0} by {1} points".format(len(data1),len(data2)))
        print("cell size= {0}".format(grid1.dL))
        print("number of cells = {0}".format(np.prod(grid1.num_divs)))
    
    #more c definitions used inside loop
    cdef int i, j, k, l
    cdef int cN_samples = N_samples+1
    cdef int icell1,icell2
    cdef int ix1, iy1, iz1, ix2, iy2, iz2
    cdef double d, progress
    cdef np.ndarray[np.float64_t, ndim=1] x_icell1, y_icell1, z_icell1
    cdef np.ndarray[np.float64_t, ndim=1] x_icell2, y_icell2, z_icell2
    cdef np.ndarray[np.float64_t, ndim=1] w_icell1, w_icell2
    cdef np.ndarray[np.int_t, ndim=1] j_icell1, j_icell2
    cdef np.ndarray[np.int_t, ndim=1] adj_cell_arr
    cdef int Ncell1=np.prod(grid1.num_divs)
    
    #Loop over all subvolumes in grid1
    for icell1 in range(Ncell1):
        
        #calculate progress
        if verbose==True:
            progress = icell1/(Ncell1)*100
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
                    radial_jbinning(<np.float64_t*>counts.data, <np.float64_t*> crbins.data,\
                                    d, nbins_minus_one, cN_samples,\
                                    w_icell1[i], w_icell2[j],\
                                    j_icell1[i], j_icell2[j])
        
    return counts


def jnpairs_pbc(data1, data2, rbins, Lbox, period, weights1, weights2, jtags1, jtags2,\
                N_samples, verbose=False):
    """
    jackknife weighted real-space pair counter.
    Calculate the weighted number of pairs with separations less than or equal to rbins[i]
    for a jackknife sample.
    """
    
    #c definitions
    cdef int nbins = len(rbins)
    cdef int nbins_minus_one = len(rbins)-1
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
    if verbose==True:
        print("running grid pairs with {0} by {1} points".format(len(data1),len(data2)))
        print("cell size= {0}".format(grid1.dL))
        print("number of cells = {0}".format(np.prod(grid1.num_divs)))
    
    #more c definitions used inside loop
    cdef int i, j, k, l
    cdef int cN_samples = N_samples+1
    cdef int icell1,icell2
    cdef int ix1, iy1, iz1, ix2, iy2, iz2
    cdef double d, progress
    cdef np.ndarray[np.float64_t, ndim=1] x_icell1, y_icell1, z_icell1
    cdef np.ndarray[np.float64_t, ndim=1] x_icell2, y_icell2, z_icell2
    cdef np.ndarray[np.float64_t, ndim=1] w_icell1, w_icell2
    cdef np.ndarray[np.int_t, ndim=1] j_icell1, j_icell2
    cdef np.ndarray[np.int_t, ndim=1] adj_cell_arr
    cdef int Ncell1=np.prod(grid1.num_divs)
    
    #Loop over all subvolumes in grid1
    for icell1 in range(Ncell1):
        
        #calculate progress
        if verbose==True:
            progress = icell1/(Ncell1)*100
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
                    radial_jbinning(<np.float64_t*>counts.data, <np.float64_t*> crbins.data,\
                                    d, nbins_minus_one, cN_samples,\
                                    w_icell1[i], w_icell2[j],\
                                    j_icell1[i], j_icell2[j])
        
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def xy_z_jnpairs_no_pbc(data1, data2, rp_bins, pi_bins, Lbox, weights1, weights2,\
                        jtags1, jtags2, N_samples, verbose=False):
    """
    jackknife weighted 2+1D pair counter.
    Calculate the weighted number of pairs with separations in the x-y plane less than or 
    equal to rp_bins[i], and separations in the z coordinate less than or equal to 
    pi_bins[i] for a jackknife sample.
    """
    
    #c definitions
    cdef int nrp_bins = len(rp_bins)
    cdef int npi_bins = len(pi_bins)
    cdef int nrp_bins_minus_one = len(rp_bins)-1
    cdef int npi_bins_minus_one = len(pi_bins)-1
    cdef np.ndarray[np.float64_t, ndim=1] crp_bins = \
        np.ascontiguousarray(rp_bins,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cpi_bins = \
        np.ascontiguousarray(pi_bins,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=3] counts = \
        np.zeros((N_samples+1, nrp_bins, npi_bins), dtype=np.float64)
    
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
    
    #sort the jackknife tag arrays
    jtags1 = jtags1[grid1.idx_sorted]
    jtags2 = jtags2[grid2.idx_sorted]
    cdef np.ndarray[np.int_t, ndim=1] cjtags1 = \
        np.ascontiguousarray(jtags1,dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=1] cjtags2 = \
        np.ascontiguousarray(jtags2,dtype=np.int)
    
    #square radial bins to make distance calculation cheaper
    crp_bins = crp_bins**2.0
    cpi_bins = cpi_bins**2.0
    
    #print come information
    if verbose==True:
        print("running grid pairs with {0} by {1} points".format(len(data1),len(data2)))
        print("cell size= {0}".format(grid1.dL))
        print("number of cells = {0}".format(np.prod(grid1.num_divs)))
    
    #more c definitions used inside loop
    cdef int i, j, k, g
    cdef int cN_samples = N_samples+1
    cdef int icell1,icell2
    cdef int ix1, iy1, iz1, ix2, iy2, iz2
    cdef double d_perp, d_para, progress
    cdef np.ndarray[np.float64_t, ndim=1] x_icell1, y_icell1, z_icell1
    cdef np.ndarray[np.float64_t, ndim=1] x_icell2, y_icell2, z_icell2
    cdef np.ndarray[np.float64_t, ndim=1] w_icell1, w_icell2
    cdef np.ndarray[np.int_t, ndim=1] j_icell1, j_icell2
    cdef np.ndarray[np.int_t, ndim=1] adj_cell_arr
    cdef int Ncell1=np.prod(grid1.num_divs)
    
    #Loop over all subvolumes in grid1
    for icell1 in range(np.prod(grid1.num_divs)):
        
        #calculate progress
        if verbose==True:
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
                    d_perp = perp_square_distance(x_icell1[i], y_icell1[i],\
                                                  x_icell2[j], y_icell2[j])
                    d_para = para_square_distance(z_icell1[i], z_icell2[j])
                        
                    #calculate counts in bins
                    xy_z_jbinning(<np.float64_t*>counts.data,\
                                  <np.float64_t*> crp_bins.data,\
                                  <np.float64_t*> cpi_bins.data, d_perp, d_para,\
                                  nrp_bins_minus_one, npi_bins_minus_one, cN_samples,\
                                  w_icell1[i],w_icell2[j], j_icell1[i],j_icell2[j])
        
    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def xy_z_jnpairs_pbc(data1, data2, rp_bins, pi_bins, Lbox=[1.0,1.0,1.0], period=None,\
                     weights1=None, weights2=None, jtags1=None, jtags2=None, N_samples=1, verbose=False):
    """
    jackknife weighted 2+1D pair counter.
    Calculate the weighted number of pairs with separations in the x-y plane less than or 
    equal to rp_bins[i], and separations in the z coordinate less than or equal to 
    pi_bins[i] for a jackknife sample.
    """
    
    #c definitions
    cdef int nrp_bins = len(rp_bins)
    cdef int npi_bins = len(pi_bins)
    cdef int nrp_bins_minus_one = len(rp_bins)-1
    cdef int npi_bins_minus_one = len(pi_bins)-1
    cdef np.ndarray[np.float64_t, ndim=1] crp_bins = \
        np.ascontiguousarray(rp_bins,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cpi_bins = \
        np.ascontiguousarray(pi_bins,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cperiod = \
        np.ascontiguousarray(period,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=3] counts = \
        np.zeros((N_samples+1, nrp_bins, npi_bins), dtype=np.float64)
    
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
    
    #sort the jackknife tag arrays
    jtags1 = jtags1[grid1.idx_sorted]
    jtags2 = jtags2[grid2.idx_sorted]
    cdef np.ndarray[np.int_t, ndim=1] cjtags1 = \
        np.ascontiguousarray(jtags1,dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=1] cjtags2 = \
        np.ascontiguousarray(jtags2,dtype=np.int)
    
    #square radial bins to make distance calculation cheaper
    crp_bins = crp_bins**2.0
    cpi_bins = cpi_bins**2.0
    
    #print come information
    if verbose==True:
        print("running grid pairs with {0} by {1} points".format(len(data1),len(data2)))
        print("cell size= {0}".format(grid1.dL))
        print("number of cells = {0}".format(np.prod(grid1.num_divs)))
    
    #more c definitions used inside loop
    cdef int i, j, k, g
    cdef int cN_samples = N_samples+1
    cdef int icell1,icell2
    cdef int ix1, iy1, iz1, ix2, iy2, iz2
    cdef double d_perp, d_para, progress
    cdef np.ndarray[np.float64_t, ndim=1] x_icell1, y_icell1, z_icell1
    cdef np.ndarray[np.float64_t, ndim=1] x_icell2, y_icell2, z_icell2
    cdef np.ndarray[np.float64_t, ndim=1] w_icell1, w_icell2
    cdef np.ndarray[np.int_t, ndim=1] j_icell1, j_icell2
    cdef np.ndarray[np.int_t, ndim=1] adj_cell_arr
    cdef int Ncell1=np.prod(grid1.num_divs)
    
    #Loop over all subvolumes in grid1
    for icell1 in range(np.prod(grid1.num_divs)):
        
        #calculate progress
        if verbose==True:
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
                    d_perp = periodic_perp_square_distance(x_icell1[i],y_icell1[i],\
                                                           x_icell2[j],y_icell2[j],\
                                                           <np.float64_t*>cperiod.data)
                    d_para = periodic_para_square_distance(z_icell1[i],\
                                                           z_icell2[j],\
                                                           <np.float64_t*>cperiod.data)
                        
                    #calculate counts in bins
                    xy_z_jbinning(<np.float64_t*>counts.data,\
                                  <np.float64_t*> crp_bins.data,\
                                  <np.float64_t*> cpi_bins.data, d_perp, d_para,\
                                  nrp_bins_minus_one, npi_bins_minus_one, cN_samples,\
                                  w_icell1[i],w_icell2[j], j_icell1[i],j_icell2[j])

    return counts


cdef inline radial_binning(np.int_t* counts, np.float64_t* bins,\
                           np.float64_t d, np.int_t k):
    """
    real space radial binning function
    """
    
    while d<=bins[k]:
        counts[k] += 1
        k=k-1
        if k<0: break


cdef inline radial_wbinning(np.float64_t* counts, np.float64_t* bins,\
                            np.float64_t d, np.int_t k,\
                            np.float64_t w1, np.float64_t w2):
    """
    real space radial weighted binning function
    """
    
    while d<=bins[k]:
        counts[k] += w1*w2
        k=k-1
        if k<0: break


cdef inline radial_jbinning(np.float64_t* counts, np.float64_t* bins,\
                            np.float64_t d,\
                            np.int_t nbins_minus_one,\
                            np.int_t N_samples,\
                            np.float64_t w1, np.float64_t w2,\
                            np.int_t j1, np.int_t j2):
    """
    real space radial jackknife binning function
    """
    cdef int k, l
    cdef int max_l = nbins_minus_one+1
    
    for l in range(0,N_samples):
        k = nbins_minus_one
        while d<=bins[k]:
            #counts[l,k] += jweight(l, j1, j2, w1, w2)
            counts[l*max_l+k] += jweight(l, j1, j2, w1, w2)
            k=k-1
            if k<0: break


cdef inline xy_z_binning(np.int_t* counts, np.float64_t* rp_bins,\
                         np.float64_t* pi_bins, np.float64_t d_perp,\
                         np.float64_t d_para, np.int_t k,\
                         np.int_t npi_bins_minus_one):
    """
    2D+1 binning function
    """
    cdef int g
    cdef int max_k = npi_bins_minus_one+1
    
    while d_perp<=rp_bins[k]:
        g = npi_bins_minus_one
        while d_para<=pi_bins[g]:
            #counts[k,g] += 1
            counts[k*max_k+g] += 1
            g=g-1
            if g<0: break
        k=k-1
        if k<0: break


cdef inline xy_z_wbinning(np.float64_t* counts, np.float64_t* rp_bins,\
                          np.float64_t* pi_bins, np.float64_t d_perp,\
                          np.float64_t d_para, np.int_t k,\
                          np.int_t npi_bins_minus_one, np.float64_t w1, np.float64_t w2):
    """
    2D+1 weighted binning function
    """
    cdef int g
    cdef int max_k = npi_bins_minus_one+1
    
    while d_perp<=rp_bins[k]:
        g = npi_bins_minus_one
        while d_para<=pi_bins[g]:
            #counts[k,g] += w1*w2
            counts[k*max_k+g] += w1*w2
            g=g-1
            if g<0: break
        k=k-1
        if k<0: break


cdef inline xy_z_jbinning(np.float64_t* counts, np.float64_t* rp_bins,\
                          np.float64_t* pi_bins, np.float64_t d_perp,\
                          np.float64_t d_para,\
                          np.int_t nrp_bins_minus_one,\
                          np.int_t npi_bins_minus_one,\
                          np.int_t N_samples,\
                          np.float64_t w1, np.float64_t w2,\
                          np.int_t j1, np.int_t j2):
    """
    2D+1 jackknife binning function
    """
    cdef int l, k, g
    cdef int max_l = nrp_bins_minus_one+1
    cdef int max_k = npi_bins_minus_one+1
    
    for l in range(0,N_samples): #loop over jackknife samples
            k = nrp_bins_minus_one
            while d_perp<=rp_bins[k]: #loop over rp bins
                g = npi_bins_minus_one
                while d_para<=pi_bins[g]: #loop over pi bins
                    #counts[l,k,g] += jweight(l, j1, j2, w1, w2)
                    counts[l*max_l*max_k+k*max_k+g] += jweight(l, j1, j2, w1, w2)
                    g=g-1
                    if g<0: break
                k=k-1
                if k<0: break


cdef inline double periodic_square_distance(np.float64_t x1,\
                                            np.float64_t y1,\
                                            np.float64_t z1,\
                                            np.float64_t x2,\
                                            np.float64_t y2,\
                                            np.float64_t z2,\
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


