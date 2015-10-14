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
from time import time
import sys
import multiprocessing
from functools import partial

from .double_tree import FlatRectanguloidDoubleTree
from .cpairs import *

from ...custom_exceptions import *
from ...utils.array_utils import convert_to_ndarray, array_is_monotonic

__all__ = ['double_tree_npairs']
__author__ = ['Duncan Campbell', 'Andrew Hearin']

def double_tree_npairs(data1, data2, rbins, period = None, 
    verbose = False, num_threads = 1, approx_cell1_size = None, approx_cell2_size = None):
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
    
    Returns
    -------
    num_pairs : array of length len(rbins)
        number of pairs
    """
    
    if num_threads is not 1:
        if num_threads=='max':
            num_threads = multiprocessing.cpu_count()
        if isinstance(num_threads,int):
            pool = multiprocessing.Pool(num_threads)
        else: 
            msg = "Input ``num_threads`` argument must be an integer or 'max'"
            raise HalotoolsError(msg)
    
    # Passively enforce that we are working with ndarrays
    x1 = data1[:,0]
    y1 = data1[:,1]
    z1 = data1[:,2]
    x2 = data2[:,0]
    y2 = data2[:,1]
    z2 = data2[:,2]
    rbins = convert_to_ndarray(rbins)

    try:
        assert rbins.ndim == 1
        assert len(rbins) > 1
        if len(rbins) > 2:
            assert array_is_monotonic(rbins, strict = True) == 1
    except AssertionError:
        msg = "Input ``rbins`` must be a monotonically increasing 1D array with at least two entries"
        raise HalotoolsError(msg)

    # Set the boolean value for the PBCs variable
    if period is None:
        PBCs = False
        x1, y1, z1, x2, y2, z2, period = (
            _enclose_in_box(x1, y1, z1, x2, y2, z2))
    else:
        PBCs = True
        period = convert_to_ndarray(period)
        if len(period) == 1:
            period = np.array([period[0]]*3)
        try:
            assert np.all(period < np.inf)
            assert np.all(period > 0)
        except AssertionError:
            msg = "Input ``period`` must be a bounded positive number in all dimensions"
            raise HalotoolsError(msg)
    xperiod, yperiod, zperiod = period 

    rmax = np.max(rbins)
    
    if approx_cell1_size is None:
        approx_x1cell_size = period[0]/10.
        approx_y1cell_size = period[1]/10.
        approx_z1cell_size = period[2]/10.
    else:
        approx_cell1_size = convert_to_ndarray(approx_cell1_size)
        try:
            assert len(approx_cell1_size) == 3
            approx_x1cell_size, approx_y1cell_size, approx_z1cell_size = approx_cell1_size
        except AssertionError:
            msg = ("Input ``approx_cell1_size`` must be a length-3 sequence")
            raise HalotoolsError(msg)

    if approx_cell2_size is None:
        approx_x2cell_size = period[0]/10.
        approx_y2cell_size = period[1]/10.
        approx_z2cell_size = period[2]/10.
    else:
        approx_cell2_size = convert_to_ndarray(approx_cell2_size)
        try:
            assert len(approx_cell2_size) == 3
            approx_x2cell_size, approx_y2cell_size, approx_z2cell_size = approx_cell2_size
        except AssertionError:
            msg = ("Input ``approx_cell2_size`` must be a length-3 sequence")
            raise HalotoolsError(msg)

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
    if num_threads>1:
        counts = np.sum(pool.map(engine,range(Ncell1)),axis=0)
        pool.close()
    if num_threads==1:
        counts = np.sum(map(engine,range(Ncell1)),axis=0)

    return counts.astype(int)

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


def _enclose_in_box(x1, y1, z1, x2, y2, z2):
    """
    build axis aligned box which encloses all points. 
    shift points so cube's origin is at 0,0,0.
    """
    
    xmin = np.min([np.min(x1),np.min(x2)])
    ymin = np.min([np.min(y1),np.min(y2)])
    zmin = np.min([np.min(z1),np.min(z2)])
    xmax = np.max([np.max(x1),np.max(x2)])
    ymax = np.max([np.max(y1),np.max(y2)])
    zmax = np.max([np.max(z1),np.max(z2)])
    
    xyzmin = np.min([xmin,ymin,zmin])
    xyzmax = np.min([xmax,ymax,zmax])-xyzmin
    
    x1 = x1 - xyzmin
    y1 = y1 - xyzmin
    z1 = z1 - xyzmin
    x2 = x2 - xyzmin
    y2 = y2 - xyzmin
    z2 = z2 - xyzmin
    
    period = np.array([xyzmax, xyzmax, xyzmax])
    
    return x1, y1, z1, x2, y2, z2, period
