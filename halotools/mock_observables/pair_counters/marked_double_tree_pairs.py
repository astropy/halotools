# -*- coding: utf-8 -*-

"""
double tree marked pair counter
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import sys
import multiprocessing
from functools import partial
from .double_tree import FlatRectanguloidDoubleTree
from .double_tree_helpers import *
from .marked_double_tree_helpers import *
from .marked_cpairs import *
from ...custom_exceptions import *
from ...utils.array_utils import convert_to_ndarray, array_is_monotonic

__all__ = ['marked_npairs','velocity_marked_npairs']
__author__ = ['Duncan Campbell', 'Andrew Hearin']


def marked_npairs(data1, data2, rbins, period=None, 
    weights1 = None, weights2 = None, 
    wfunc = 0, verbose = False, num_threads = 1,
    approx_cell1_size = None, approx_cell2_size = None):
    """
    weighted real-space pair counter.
    
    Count the weighted number of pairs (x1,x2) that can be formed, with x1 drawn from 
    data1 and x2 drawn from data2, and where distance(x1, x2) <= rbins[i].  Weighted 
    counts are calculated as wfunc(w1,w2,r1,r2)
    
    Parameters
    ----------
    data1 : array_like
        N1 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period. This cython implementation requires data1.ndim==2.
            
    data2 : array_like
        N2 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period. This cython implementation requires data2.ndim==2.
            
    rbins : array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
    
    period : array_like, optional
        length k array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.  If True, period is set to be Lbox
    
    weights1 : array_like, optional
        Either a 1-d array of length N1, or a 2-d array of length N1 x N_weights, 
        containing weights used for weighted pair counts
        
    weights2 : array_like, optional
        Either a 1-d array of length N2, or a 2-d array of length N2 x N_weights, 
        containing weights used for weighted pair counts
        
    wfunc : int, optional
        weighting function ID.
    
    verbose : Boolean, optional
        If True, print out information and progress.
    
    num_threads : int, optional
        number of 'threads' to use in the pair counting.  if set to 'max', use all 
        available cores.  num_threads=0 is the default.
    
    approx_cell1_size : array_like, optional 
        Length-3 array serving as a guess for the optimal manner by which 
        the `~halotools.mock_observables.pair_counters.FlatRectanguloidDoubleTree` 
        will apportion the ``data`` points into subvolumes of the simulation box. 
        The optimum choice unavoidably depends on the specs of your machine. 
        Default choice is to use 1/10 of the box size in each dimension, 
        which will return reasonable result performance for most use-cases. 
        Performance can vary sensitively with this parameter, so it is highly 
        recommended that you experiment with this parameter when carrying out  
        performance-critical calculations. 
        
    approx_cell2_size : array_like, optional 
        See comments for ``approx_cell1_size``. 
        
    Returns
    -------
    wN_pairs : array of length len(rbins)
        weighted number counts of pairs
    """
    
    ### Process the inputs with the helper function
    x1, y1, z1, x2, y2, z2, rbins, period, num_threads, PBCs = (
        _npairs_process_args(data1, data2, rbins, period, 
            verbose, num_threads, approx_cell1_size, approx_cell2_size)
        )
    xperiod, yperiod, zperiod = period 
    rmax = np.max(rbins)

    # Process the input weights and with the helper function
    weights1, weights2 = (
        _marked_npairs_process_weights(data1, data2, 
            weights1, weights2, wfunc))

    ### Compute the estimates for the cell sizes
    approx_cell1_size, approx_cell2_size = (
        _set_approximate_cell_sizes(approx_cell1_size, approx_cell2_size, rmax, period)
        )
    approx_x1cell_size, approx_y1cell_size, approx_z1cell_size = approx_cell1_size
    approx_x2cell_size, approx_y2cell_size, approx_z2cell_size = approx_cell2_size

    double_tree = FlatRectanguloidDoubleTree(
        x1, y1, z1, x2, y2, z2,  
        approx_x1cell_size, approx_y1cell_size, approx_z1cell_size, 
        approx_x2cell_size, approx_y2cell_size, approx_z2cell_size, 
        rmax, rmax, rmax, xperiod, yperiod, zperiod, PBCs=PBCs)

    #sort the weights arrays
    weights1 = np.ascontiguousarray(weights1[double_tree.tree1.idx_sorted, :])
    weights2 = np.ascontiguousarray(weights2[double_tree.tree2.idx_sorted, :])
    
    #square radial bins to make distance calculation cheaper
    rbins_squared = rbins**2.0
    
    #number of cells
    Ncell1 = double_tree.num_x1divs*double_tree.num_y1divs*double_tree.num_z1divs

    #create a function to call with only one argument
    engine = partial(_wnpairs_engine, double_tree, 
        weights1, weights2, rbins_squared, period, PBCs, wfunc)
    
    #do the pair counting
    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        counts = np.sum(pool.map(engine,range(Ncell1)),axis=0)
        pool.close()
    else:
        counts = np.sum(map(engine,range(Ncell1)),axis=0)
    
    return counts


def _wnpairs_engine(double_tree, weights1, weights2, 
                    rbins_squared, period, PBCs, wfunc, icell1):
    """
    engine that calls cython function to count pairs
    """
    
    counts = np.zeros(len(rbins_squared))
    
    #extract the points in the cell
    s1 = double_tree.tree1.slice_array[icell1]
    x_icell1, y_icell1, z_icell1 = (
        double_tree.tree1.x[s1],
        double_tree.tree1.y[s1],
        double_tree.tree1.z[s1])

    #extract the weights in the cell
    w_icell1 = weights1[s1, :]

    xsearch_length = np.sqrt(rbins_squared[-1])
    ysearch_length = np.sqrt(rbins_squared[-1])
    zsearch_length = np.sqrt(rbins_squared[-1])
    adj_cell_generator = double_tree.adjacent_cell_generator(
        icell1, xsearch_length, ysearch_length, zsearch_length)
            
    adj_cell_counter = 0
    for icell2, xshift, yshift, zshift in adj_cell_generator:
        
        #set shift array as -1,1,0 depending on direction of/if cell shifted.
        shift = np.array([xshift,zshift,yshift]).astype(float)
        
        #extract the points in the cell
        s2 = double_tree.tree2.slice_array[icell2]
        x_icell2 = double_tree.tree2.x[s2] + xshift
        y_icell2 = double_tree.tree2.y[s2] + yshift 
        z_icell2 = double_tree.tree2.z[s2] + zshift

        #extract the weights in the cell
        w_icell2 = weights2[s2, :]
            
        #use cython functions to do pair counting
        counts += marked_npairs_no_pbc(x_icell1, y_icell1, z_icell1,
                                       x_icell2, y_icell2, z_icell2,
                                       w_icell1, w_icell2, 
                                       rbins_squared, wfunc, shift)
            
    return counts


def velocity_marked_npairs(data1, data2, rbins, period=None, 
    weights1 = None, weights2 = None, 
    wfunc = 0, verbose = False, num_threads = 1,
    approx_cell1_size = None, approx_cell2_size = None):
    """
    weighted real-space pair counter..
    
    Count the weighted number of pairs (x1,x2) that can be formed, with x1 drawn from 
    data1 and x2 drawn from data2, and where distance(x1, x2) <= rbins[i].  Weighted 
    counts are calculated as wfunc(w1,w2,r1,r2), whee in this case, wfunc returns 2 floats.
    
    Parameters
    ----------
    data1 : array_like
        N1 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period. This cython implementation requires data1.ndim==2.
            
    data2 : array_like
        N2 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period. This cython implementation requires data2.ndim==2.
            
    rbins : array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
    
    period : array_like, optional
        length k array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.  If True, period is set to be Lbox
    
    weights1 : array_like, optional
        Either a 1-d array of length N1, or a 2-d array of length N1 x N_weights, 
        containing weights used for weighted pair counts
        
    weights2 : array_like, optional
        Either a 1-d array of length N2, or a 2-d array of length N2 x N_weights, 
        containing weights used for weighted pair counts
        
    wfunc : int, optional
        weighting function ID.
    
    verbose : Boolean, optional
        If True, print out information and progress.
    
    num_threads : int, optional
        number of 'threads' to use in the pair counting.  if set to 'max', use all 
        available cores.  num_threads=0 is the default.
    
    approx_cell1_size : array_like, optional 
        Length-3 array serving as a guess for the optimal manner by which 
        the `~halotools.mock_observables.pair_counters.FlatRectanguloidDoubleTree` 
        will apportion the ``data`` points into subvolumes of the simulation box. 
        The optimum choice unavoidably depends on the specs of your machine. 
        Default choice is to use 1/10 of the box size in each dimension, 
        which will return reasonable result performance for most use-cases. 
        Performance can vary sensitively with this parameter, so it is highly 
        recommended that you experiment with this parameter when carrying out  
        performance-critical calculations. 
        
    approx_cell2_size : array_like, optional 
        See comments for ``approx_cell1_size``. 
        
    Returns
    -------
    w1N_pairs : array of length len(rbins)
        weighted number counts of pairs
    
    w2N_pairs : array of length len(rbins)
        weighted number counts of pairs
    
    w3N_pairs : array of length len(rbins)
        weighted number counts of pairs
    """
    
    ### Process the inputs with the helper function
    x1, y1, z1, x2, y2, z2, rbins, period, num_threads, PBCs = (
        _npairs_process_args(data1, data2, rbins, period, 
            verbose, num_threads, approx_cell1_size, approx_cell2_size)
        )
    xperiod, yperiod, zperiod = period 
    rmax = np.max(rbins)

    # Process the input weights and with the helper function
    weights1, weights2 = (
        _velocity_marked_npairs_process_weights(data1, data2, 
            weights1, weights2, wfunc))

    ### Compute the estimates for the cell sizes
    approx_cell1_size, approx_cell2_size = (
        _set_approximate_cell_sizes(approx_cell1_size, approx_cell2_size, rmax, period)
        )
    approx_x1cell_size, approx_y1cell_size, approx_z1cell_size = approx_cell1_size
    approx_x2cell_size, approx_y2cell_size, approx_z2cell_size = approx_cell2_size

    double_tree = FlatRectanguloidDoubleTree(
        x1, y1, z1, x2, y2, z2,  
        approx_x1cell_size, approx_y1cell_size, approx_z1cell_size, 
        approx_x2cell_size, approx_y2cell_size, approx_z2cell_size, 
        rmax, rmax, rmax, xperiod, yperiod, zperiod, PBCs=PBCs)

    #sort the weights arrays
    weights1 = np.ascontiguousarray(weights1[double_tree.tree1.idx_sorted, :])
    weights2 = np.ascontiguousarray(weights2[double_tree.tree2.idx_sorted, :])
    
    #square radial bins to make distance calculation cheaper
    rbins_squared = rbins**2.0
    
    #number of cells
    Ncell1 = double_tree.num_x1divs*double_tree.num_y1divs*double_tree.num_z1divs

    #create a function to call with only one argument
    engine = partial(_velocity_wnpairs_engine, double_tree, 
        weights1, weights2, rbins_squared, period, PBCs, wfunc)
    
    #do the pair counting
    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        result = pool.map(engine,range(Ncell1))
        result = np.array(result)
        counts1, counts2, counts3 = (result[:,0],result[:,1],result[:,2])
        counts1 = np.sum(counts1,axis=0)
        counts2 = np.sum(counts2,axis=0)
        counts3 = np.sum(counts3,axis=0)
        pool.close()
    else:
        result = map(engine,range(Ncell1))
        result = np.array(result)
        counts1, counts2, counts3= (result[:,0],result[:,1],result[:,2])
        counts1 = np.sum(counts1,axis=0)
        counts2 = np.sum(counts2,axis=0)
        counts3 = np.sum(counts3,axis=0)
    
    return counts1, counts2, counts3


def _velocity_wnpairs_engine(double_tree, weights1, weights2, 
                             rbins_squared, period, PBCs, wfunc, icell1):
    """
    engine that calls cython function to count pairs
    """
    
    counts1 = np.zeros(len(rbins_squared))
    counts2 = np.zeros(len(rbins_squared))
    counts3 = np.zeros(len(rbins_squared))
    
    #extract the points in the cell
    s1 = double_tree.tree1.slice_array[icell1]
    x_icell1, y_icell1, z_icell1 = (
        double_tree.tree1.x[s1],
        double_tree.tree1.y[s1],
        double_tree.tree1.z[s1])

    #extract the weights in the cell
    w_icell1 = weights1[s1, :]

    xsearch_length = np.sqrt(rbins_squared[-1])
    ysearch_length = np.sqrt(rbins_squared[-1])
    zsearch_length = np.sqrt(rbins_squared[-1])
    adj_cell_generator = double_tree.adjacent_cell_generator(
        icell1, xsearch_length, ysearch_length, zsearch_length)
            
    adj_cell_counter = 0
    for icell2, xshift, yshift, zshift in adj_cell_generator:
        
        #set shift array as -1,1,0 depending on direction of/if cell shifted.
        shift = np.array([xshift,zshift,yshift]).astype(float)
        
        #extract the points in the cell
        s2 = double_tree.tree2.slice_array[icell2]
        x_icell2 = double_tree.tree2.x[s2] + xshift
        y_icell2 = double_tree.tree2.y[s2] + yshift 
        z_icell2 = double_tree.tree2.z[s2] + zshift

        #extract the weights in the cell
        w_icell2 = weights2[s2, :]
            
        #use cython functions to do pair counting
        holder1, holder2, holder3 = velocity_marked_npairs_no_pbc(
                                       x_icell1, y_icell1, z_icell1,
                                       x_icell2, y_icell2, z_icell2,
                                       w_icell1, w_icell2, 
                                       rbins_squared, wfunc, shift)
        counts1 += holder1 
        counts2 += holder2
        counts3 += holder3
            
    return counts1, counts2, counts3 


