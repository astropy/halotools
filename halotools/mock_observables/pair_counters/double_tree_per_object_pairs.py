# -*- coding: utf-8 -*-

"""

"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from copy import copy 

import time
import sys
import multiprocessing
from multiprocessing import Value, Lock
from functools import partial
import pytest
from .double_tree import FlatRectanguloidDoubleTree
from .double_tree_helpers import *
from .cpairs import *
from ...custom_exceptions import *
from ...utils.array_utils import convert_to_ndarray, array_is_monotonic

__all__ = ['per_object_npairs']
__author__ = ['Duncan Campbell', 'Andrew Hearin']

##########################################################################

def per_object_npairs(data1, data2, rbins, period = None,\
                     verbose = False, num_threads = 1,\
                     approx_cell1_size = None, approx_cell2_size = None):
    """    
    Function counts the number of times the pair count between two samples exceeds a 
    threshold value as a function of the 3d spatial separation *r*.
    
    Parameters
    ----------
    data1 : array_like
        N1 by 3 numpy array of 3-dimensional positions. 
        Values of each dimension should be between zero and the corresponding dimension 
        of the input period.
            
    data2 : array_like
        N2 by 3 numpy array of 3-dimensional positions.
        Values of each dimension should be between zero and the corresponding dimension 
        of the input period.
            
    rbins : array_like
        Boundaries defining the bins in which pairs are counted.
    
    n_thresh : int, optional
        positive integer number indicating the threshold pair count
    
    period : array_like, optional
        Length-3 array defining the periodic boundary conditions. 
        If only one number is specified, the enclosing volume is assumed to 
        be a periodic cube (by far the most common case). 
        If period is set to None, the default option, 
        PBCs are set to infinity.  

    verbose : Boolean, optional
        If True, print out information and progress.
    
    num_threads : int, optional
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
    num_pairs : array_like 
        Numpy array of length len(rbins) storing the numbers of pairs in the input bins. 

    Examples 
    --------
    For illustration purposes, we'll create some fake data and call the pair counter:

    >>> Npts1, Npts2, Lbox = 1e3, 1e3, 250.
    >>> period = [Lbox, Lbox, Lbox]
    >>> rbins = np.logspace(-1, 1.5, 15)

    >>> x1 = np.random.uniform(0, Lbox, Npts1)
    >>> y1 = np.random.uniform(0, Lbox, Npts1)
    >>> z1 = np.random.uniform(0, Lbox, Npts1)
    >>> x2 = np.random.uniform(0, Lbox, Npts2)
    >>> y2 = np.random.uniform(0, Lbox, Npts2)
    >>> z2 = np.random.uniform(0, Lbox, Npts2)

    We transform our *x, y, z* points into the array shape used by the pair-counter by 
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation 
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> data1 = np.vstack([x1, y1, z1]).T 
    >>> data2 = np.vstack([x2, y2, z2]).T 

    >>> result = per_object_npairs(data1, data2, rbins, period = period)
    """

    ### Process the inputs with the helper function
    x1, y1, z1, x2, y2, z2, rbins, period, num_threads, PBCs = (
        _npairs_process_args(data1, data2, rbins, period, 
            verbose, num_threads, approx_cell1_size, approx_cell2_size)
        )
    
    xperiod, yperiod, zperiod = period 
    rmax = np.max(rbins)
    
    if verbose==True:
        print("running double_tree_pairs.xy_z_npairs on {0} x {1}\n"
              "points with PBCs={2}".format(len(data1), len(data2), PBCs))
        start = time.time()
    
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

    #number of cells
    Ncell1 = double_tree.num_x1divs*double_tree.num_y1divs*double_tree.num_z1divs
    Ncell1 = double_tree.num_x2divs*double_tree.num_y2divs*double_tree.num_z2divs
    
    if verbose==True:
        print("volume 1 split {0},{1},{2} times along each dimension,\n"
              "resulting in {3} cells.".format(double_tree.num_x1divs,\
              double_tree.num_y1divs,double_tree.num_z1divs,Ncell1))
        print("volume 2 split {0},{1},{2} times along each dimension,\n"
              "resulting in {3} cells.".format(double_tree.num_x2divs,\
              double_tree.num_y2divs,double_tree.num_z2divs,Ncell1))
    
    #create a function to call with only one argument
    engine = partial(_per_object_npairs_engine, 
        double_tree, rbins, period, PBCs)
    
    #do the pair counting
    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        result = pool.map(engine,range(Ncell1))
        pool.close()
        counts = np.vstack(result)
    if num_threads == 1:
        result = map(engine,range(Ncell1))
        counts = np.vstack(result)
    
    if verbose==True:
        print("total run time: {0} seconds".format(time.time()-start))
    
    return counts


def _per_object_npairs_engine(double_tree, rbins, period, PBCs, icell1):
    """
    pair counting engine for threshold_npairs function.
    This code calls a cython function.
    """
    # print("...working on icell1 = %i" % icell1)
    
    #extract the points in the cell
    s1 = double_tree.tree1.slice_array[icell1]
    x_icell1, y_icell1, z_icell1 = (
        double_tree.tree1.x[s1],
        double_tree.tree1.y[s1],
        double_tree.tree1.z[s1])
    
    counts = np.zeros((len(x_icell1), len(rbins)))
        
    xsearch_length = rbins[-1]
    ysearch_length = rbins[-1]
    zsearch_length = rbins[-1]
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
        counts += per_object_npairs_no_pbc(
            x_icell1, y_icell1, z_icell1,
            x_icell2, y_icell2, z_icell2,
            rbins)
    
    return counts
