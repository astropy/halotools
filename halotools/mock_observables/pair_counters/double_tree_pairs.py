# -*- coding: utf-8 -*-

"""
rectangular Cuboid Pair Counter. 
This module contains pair counting functions used to count the number of pairs with 
separations less than or equal to r, optimized for simulation boxes.
This module also contains a 'main' function which runs speed tests.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from copy import copy 

from time import time
import sys
import multiprocessing
from functools import partial

from .double_tree import FlatRectanguloidDoubleTree
from .double_tree_helpers import *

from .cpairs import *

from ...custom_exceptions import *
from ...utils.array_utils import convert_to_ndarray, array_is_monotonic

__all__ = ['double_tree_npairs']
__author__ = ['Duncan Campbell', 'Andrew Hearin']

##########################################################################

def npairs(data1, data2, rbins, period = None,\
           verbose = False, num_threads = 1,\
           approx_cell1_size = None, approx_cell2_size = None):
    """
    real-space pair counter.
    
    Count the number of pairs (x1,x2) that can be formed, with x1 drawn from data1 and x2
    drawn from data2, and where distance(x1, x2) <= rbins[i]. 
    
    Parameters
    ----------
    data1: array_like
        N1 by 3 numpy array of 3-dimensional positions. 
        Values of each dimension should be between zero and the corresponding dimension 
        of the input period.
            
    data2: array_like
        N2 by 3 numpy array of 3-dimensional positions.
        Values of each dimension should be between zero and the corresponding dimension 
        of the input period.
            
    rbins: array_like
        Boundaries defining the bins in which pairs are counted.
        
    period: array_like, optional
        Length-3 array defining the periodic boundary conditions. 
        If only one number is specified, the enclosing volume is assumed to 
        be a periodic cube (by far the most common case). 
        If period is set to None, the default option, 
        PBCs are set to infinity.  
    verbose: Boolean, optional
        If True, print out information and progress.
    
    num_threads: int, optional
        Number of CPU cores to use in the pair counting. 
        If ``num_threads`` is set to the string 'max', use all available cores. 
        Default is 1 thread for a serial calculation that 
        does not open a multiprocessing pool. 

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
    num_pairs : array of length len(rbins)
        number of pairs
    """

    ### Process the inputs with the helper function
    x1, y1, z1, x2, y2, z2, rbins, period, num_threads, PBCs = (
        _npairs_process_args(data1, data2, rbins, period, 
            verbose, num_threads, approx_cell1_size, approx_cell2_size)
        )        
    
    xperiod, yperiod, zperiod = period 
    rmax = np.max(rbins)
    
    ### Compute the estimates for the cell sizes
    approx_cell1_size, approx_cell2_size = (
        _set_approximate_cell_sizes(approx_cell1_size, approx_cell2_size, rmax)
        )
    approx_x1cell_size, approx_y1cell_size, approx_z1cell_size = approx_cell1_size
    approx_x2cell_size, approx_y2cell_size, approx_z2cell_size = approx_cell2_size

    double_tree = FlatRectanguloidDoubleTree(
        x1, y1, z1, x2, y2, z2,  
        approx_x1cell_size, approx_y1cell_size, approx_z1cell_size, 
        approx_x2cell_size, approx_y2cell_size, approx_z2cell_size, 
        rmax, rmax, rmax, xperiod, yperiod, zperiod, PBCs=PBCs)

    #square radial bins to make distance calculation cheaper
    rbins_squared = rbins**2.0
        
    #number of cells
    Ncell1 = double_tree.num_x1divs*double_tree.num_y1divs*double_tree.num_z1divs

    #create a function to call with only one argument
    engine = partial(_npairs_engine, 
        double_tree, rbins_squared, period, PBCs)
    
    #do the pair counting
    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        counts = np.sum(pool.map(engine,range(Ncell1)),axis=0)
        pool.close()
    if num_threads == 1:
        counts = np.sum(map(engine,range(Ncell1)),axis=0)

    return counts

def _npairs_engine(double_tree, rbins_squared, period, PBCs, icell1):
    """
    pair counting engine for npairs function.  This code calls a cython function.
    """
    # print("...working on icell1 = %i" % icell1)
    
    counts = np.zeros(len(rbins_squared))
    
    #extract the points in the cell
    s1 = double_tree.tree1.slice_array[icell1]
    x_icell1, y_icell1, z_icell1 = (
        double_tree.tree1.x[s1],
        double_tree.tree1.y[s1],
        double_tree.tree1.z[s1])
        
    xsearch_length = np.sqrt(rbins_squared[-1])
    ysearch_length = np.sqrt(rbins_squared[-1])
    zsearch_length = np.sqrt(rbins_squared[-1])
    adj_cell_generator = double_tree.adjacent_cell_generator(
        icell1, xsearch_length, ysearch_length, zsearch_length)
            
    adj_cell_counter = 0
    for icell2, xshift, yshift, zshift in adj_cell_generator:
                
        #extract the points in the cell
        s2 = double_tree.tree2.slice_array[icell2]
        x_icell2 = double_tree.tree2.x[s2] + xshift
        y_icell2 = double_tree.tree2.y[s2] + yshift 
        z_icell2 = double_tree.tree2.z[s2] + zshift


        #use cython functions to do pair counting
        counts += npairs_no_pbc(
            x_icell1, y_icell1, z_icell1,
            x_icell2, y_icell2, z_icell2,
            rbins_squared)
            
    return counts



##########################################################################

def jnpairs(data1, data2, rbins, period=None, weights1=None, weights2=None,
    jtags1=None, jtags2=None, N_samples=0, verbose=False, num_threads=1, 
    approx_cell1_size = None, approx_cell2_size = None):
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
    
    num_threads: int, optional
        number of 'threads' to use in the pair counting.  If set to 'max', use all 
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
    N_pairs : ndarray of shape (N_samples+1,len(rbins))
        number counts of pairs with seperations <=rbins[i]
    
    Notes
    -----
    Jackknife weights are calculated using a weighting function.
    
    if both points are outside the sample, return 0.0
    if both points are inside the sample, return (w1 * w2)
    if one point is inside, and the other is outside return 0.5*(w1 * w2)
    """
    ### Process the inputs with the helper function
    x1, y1, z1, x2, y2, z2, rbins, period, num_threads, PBCs = (
        _npairs_process_args(data1, data2, rbins, period, 
            verbose, num_threads, approx_cell1_size, approx_cell2_size)
        )
    xperiod, yperiod, zperiod = period 
    rmax = np.max(rbins)

    # Process the input weights and jackknife-tags with the helper function
    weights1, weights2, jtags1, jtags2 = (
        _jnpairs_process_weights_jtags(data1, data2, 
            weights1, weights2, jtags1, jtags2, N_samples))

    ### Compute the estimates for the cell sizes
    approx_cell1_size, approx_cell2_size = (
        _set_approximate_cell_sizes(approx_cell1_size, approx_cell2_size, rmax)
        )
    approx_x1cell_size, approx_y1cell_size, approx_z1cell_size = approx_cell1_size
    approx_x2cell_size, approx_y2cell_size, approx_z2cell_size = approx_cell2_size

    double_tree = FlatRectanguloidDoubleTree(
        x1, y1, z1, x2, y2, z2,  
        approx_x1cell_size, approx_y1cell_size, approx_z1cell_size, 
        approx_x2cell_size, approx_y2cell_size, approx_z2cell_size, 
        rmax, rmax, rmax, xperiod, yperiod, zperiod, PBCs=PBCs)


    #sort the weights arrays
    weights1 = weights1[double_tree.tree1.idx_sorted]
    weights2 = weights2[double_tree.tree2.idx_sorted]
        
    #sort the jackknife tag arrays
    jtags1 = jtags1[double_tree.tree1.idx_sorted]
    jtags2 = jtags2[double_tree.tree2.idx_sorted]

    #square radial bins to make distance calculation cheaper
    rbins_squared = rbins**2.0
        
    #number of cells
    Ncell1 = double_tree.num_x1divs*double_tree.num_y1divs*double_tree.num_z1divs


    #create a function to call with only one argument
    engine = partial(_jnpairs_engine, double_tree, 
        weights1, weights2, jtags1, jtags2, N_samples, rbins_squared, period, PBCs)
    
    #do the pair counting
    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        counts = np.sum(pool.map(engine,range(Ncell1)),axis=0)
        pool.close()
    else:
        counts = np.sum(map(engine,range(Ncell1)),axis=0)
    
    return counts


##########################################################################

def _jnpairs_engine(double_tree, weights1, weights2, jtags1, jtags2, 
    N_samples, rbins_squared, period, PBCs, icell1):
    """
    """

    counts = np.zeros((N_samples+1,len(rbins)))
    
    #extract the points in the cell
    s1 = double_tree.tree1.slice_array[icell1]
    x_icell1, y_icell1, z_icell1 = (
        double_tree.tree1.x[s1],
        double_tree.tree1.y[s1],
        double_tree.tree1.z[s1])

    #extract the weights in the cell
    w_icell1 = weights1[s1]
        
    #extract the jackknife tags in the cell
    j_icell1 = jtags1[s1]

    xsearch_length = np.sqrt(rbins_squared[-1])
    ysearch_length = np.sqrt(rbins_squared[-1])
    zsearch_length = np.sqrt(rbins_squared[-1])
    adj_cell_generator = double_tree.adjacent_cell_generator(
        icell1, xsearch_length, ysearch_length, zsearch_length)
            
    adj_cell_counter = 0
    for icell2, xshift, yshift, zshift in adj_cell_generator:
                
        #extract the points in the cell
        s2 = double_tree.tree2.slice_array[icell2]
        x_icell2 = double_tree.tree2.x[s2] + xshift
        y_icell2 = double_tree.tree2.y[s2] + yshift 
        z_icell2 = double_tree.tree2.z[s2] + zshift

        #extract the weights in the cell
        w_icell2 = weights2[s2]
            
        #extract the jackknife tags in the cell
        j_icell2 = jtags2[s2]

        #use cython functions to do pair counting
        counts += jnpairs_no_pbc(x_icell1, y_icell1, z_icell1,
            x_icell2, y_icell2, z_icell2,
            w_icell1, w_icell2, j_icell1, j_icell2, 
            N_samples+1, rbins)
            
    return counts



##########################################################################




