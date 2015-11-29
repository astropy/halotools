# -*- coding: utf-8 -*-

"""
This module contains functions used to calculate 
the number of pairs of points as a function of the 
separation between the points. Many choices for the separation variable(s) are available, 
including 3-D spherical shells, `~halotools.mock_observables.pair_counters.npairs`, 
2+1-D cylindrical shells, `~halotools.mock_observables.pair_counters.xy_z_npairs`, 
and separations :math:`s + \\theta_{\\rm los}` defined by angular & line-of-sight coordinates, 
`~halotools.mock_observables.pair_counters.s_mu_npairs`. 
There is also a function `~halotools.mock_observables.pair_counters.jnpairs` used to 
provide jackknife error estimates on the pair counts. 
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

__all__ = ['npairs', 'jnpairs', 'xy_z_npairs', 's_mu_npairs']
__author__ = ['Duncan Campbell', 'Andrew Hearin']

##########################################################################

def npairs(data1, data2, rbins, period = None,\
           verbose = False, num_threads = 1,\
           approx_cell1_size = None, approx_cell2_size = None):
    """
    Function counts the number of pairs of points as a function of the 3d spatial separation *r*. 
        
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

    >>> result = npairs(data1, data2, rbins, period = period)
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

    #square radial bins to make distance calculation cheaper
    rbins_squared = rbins**2.0
        
    #number of cells
    Ncell1 = double_tree.num_x1divs*double_tree.num_y1divs*double_tree.num_z1divs
    
    if verbose==True:
        print("volume 1 split {0},{1},{2} times along each dimension,\n"
              "resulting in {3} cells.".format(double_tree.num_x1divs,\
              double_tree.num_y1divs,double_tree.num_z1divs,Ncell1))
        print("volume 2 split {0},{1},{2} times along each dimension,\n"
              "resulting in {3} cells.".format(double_tree.num_x2divs,\
              double_tree.num_y2divs,double_tree.num_z2divs,Ncell1))
    
    #create a function to call with only one argument
    engine = partial(_npairs_engine, 
        double_tree, rbins_squared, period, PBCs)
    
    #do the pair counting
    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        result = pool.map(engine,range(Ncell1))
        pool.close()
        counts = np.sum(result,axis=0)
    if num_threads == 1:
        counts = np.sum(map(engine,range(Ncell1)),axis=0)
    
    if verbose==True:
        print("total run time: {0} seconds".format(time.time()-start))
    
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
    Pair counter used to make jackknife error estimates of real-space pair counter 
    `~halotools.mock_observables.pair_counters.npairs`. 
        
    Parameters
    ----------
    data1: array_like
        N1 by 3 numpy array of 3-dimensional positions. 
        Values of each dimension should be between zero and the corresponding dimension 
        of the input period.
            
    data2: array_like
        N1 by 3 numpy array of 3-dimensional positions. 
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
    
    weights1: array_like, optional
        length N1 array containing weights used for weighted pair counts. 
        
    weights2: array_like, optional
        length N2 array containing weights used for weighted pair counts.
    
    jtags1: array_like, optional
        length N1 array containing integer tags used to define jackknife sample 
        membership. Tags are in the range [1, N_samples]. 
        The tag '0' is a reserved tag and should not be used.
        
    jtags2: array_like, optional
        length N2 array containing integer tags used to define jackknife sample 
        membership. Tags are in the range [1, N_samples]. 
        The tag '0' is a reserved tag and should not be used.
    
    N_samples: int, optional
        Total number of jackknife samples. All values of ``jtags1`` and ``jtags2`` 
        should be in the range [1, N_samples]. 
    
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
    N_pairs : array_like  
        Numpy array of shape (N_samples+1,len(rbins)). 
        The sub-array N_pairs[0, :] stores numbers of pairs 
        in the input bins for the entire sample. 
        The sub-array N_pairs[i, :] stores numbers of pairs 
        in the input bins for the :math:`i^{\\rm th}` jackknife sub-sample. 
    
    Notes
    -----
    Jackknife weights are calculated using a weighting function.
    
    If both points are outside the sample, the weighting function returns 0.
    If both points are inside the sample, the weighting function returns (w1 * w2)
    If one point is inside, and the other is outside, the weighting function returns (w1 * w2)/2

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

    Ordinarily, you would create ``jtags`` for the points by properly subdivide 
    the points into spatial sub-volumes. For illustration purposes, we'll simply 
    use randomly assigned sub-volumes as this has no impact on the calling signature:

    >>> N_samples = 10
    >>> jtags1 = np.random.random_integers(1, N_samples, Npts1)
    >>> jtags2 = np.random.random_integers(1, N_samples, Npts2)

    >>> result = jnpairs(data1, data2, rbins, period = period, jtags1=jtags1, jtags2=jtags2, N_samples = N_samples)

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
        search_volume = (2.0*rmax)**3
        total_volume  = period.prod()
        print("searching for pairs over {}% of the total volume\n"
              "for each point.".format(search_volume/total_volume))
    
    # Process the input weights and jackknife-tags with the helper function
    weights1, weights2, jtags1, jtags2 = (
        _jnpairs_process_weights_jtags(data1, data2, 
            weights1, weights2, jtags1, jtags2, N_samples))

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
    weights1 = weights1[double_tree.tree1.idx_sorted]
    weights2 = weights2[double_tree.tree2.idx_sorted]
        
    #sort the jackknife tag arrays
    jtags1 = jtags1[double_tree.tree1.idx_sorted]
    jtags2 = jtags2[double_tree.tree2.idx_sorted]

    #square radial bins to make distance calculation cheaper
    rbins_squared = rbins**2.0
        
    #number of cells
    Ncell1 = double_tree.num_x1divs*double_tree.num_y1divs*double_tree.num_z1divs
    
    if verbose==True:
        print("volume 1 split {0},{1},{2} times along each dimension,\n"
              "resulting in {3} cells.".format(double_tree.num_x1divs,\
              double_tree.num_y1divs,double_tree.num_z1divs,Ncell1))
        print("volume 2 split {0},{1},{2} times along each dimension,\n"
              "resulting in {3} cells.".format(double_tree.num_x2divs,\
              double_tree.num_y2divs,double_tree.num_z2divs,Ncell1))
    
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

    counts = np.zeros((N_samples+1,len(rbins_squared)))
    
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
            N_samples+1, rbins_squared)
            
    return counts

##########################################################################
def xy_z_npairs(data1, data2, rp_bins, pi_bins, period=None, verbose=False, num_threads=1, 
                approx_cell1_size = None, approx_cell2_size = None):

    """
    Function counts the number of pairs of points as a function of projected separation :math:`r_{\\rm p}` and line-of-sight separation :math:`\\pi`. 
        
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

    rp_bins: array_like
        numpy array of boundaries defining the projected separation 
        :math:`r_{\\rm p}` bins in which pairs are 
        counted.
    
    pi_bins: array_like
        numpy array of boundaries defining the line-of-sight separation 
        :math:`\\pi` bins in which pairs are counted.
    
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
    N_pairs : array_like 
        2-d array of length *Num_rp_bins x Num_pi_bins* storing the pair counts in each bin. 

    Examples 
    --------
    For illustration purposes, we'll create some fake data and call the pair counter:

    >>> Npts1, Npts2, Lbox = 1e3, 1e3, 1000.
    >>> period = [Lbox, Lbox, Lbox]
    >>> rp_bins = np.logspace(-1, 2, 15)
    >>> pi_bins = np.logspace(1, 2)

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

    >>> result = xy_z_npairs(data1, data2, rp_bins, pi_bins, period = period)
    """
    ### Process the inputs with the helper function
    x1, y1, z1, x2, y2, z2, rp_bins, pi_bins, period, num_threads, PBCs = (
        _xy_z_npairs_process_args(data1, data2, rp_bins, pi_bins, period, 
            verbose, num_threads, approx_cell1_size, approx_cell2_size)
        )        
    
    xperiod, yperiod, zperiod = period 
    rp_max = np.max(rp_bins)
    pi_max = np.max(pi_bins)
    
    if verbose==True:
        print("running double_tree_pairs.xy_z_npairs on {0} x {1}\n"
              "points with PBCs={2}".format(len(data1), len(data2), PBCs))
        search_volume = (2.0*rp_max)*(2.0*rp_max)*(2.0*pi_max)
        total_volume  = period.prod()
        print("searching for pairs over {}% of the total volume\n"
              "for each point.".format(search_volume/total_volume))
    
    ### Compute the estimates for the cell sizes

    result = _set_approximate_xy_z_cell_sizes(
        approx_cell1_size, approx_cell2_size, rp_max, pi_max, period)
    approx_cell1_size = result[0]
    approx_cell2_size = result[1]


    approx_x1cell_size, approx_y1cell_size = approx_cell1_size[:2]
    approx_z1cell_size = approx_cell1_size[2]

    approx_x2cell_size, approx_y2cell_size = approx_cell2_size[:2]
    approx_z2cell_size = approx_cell2_size[2]

    double_tree = FlatRectanguloidDoubleTree(
        x1, y1, z1, x2, y2, z2,  
        approx_x1cell_size, approx_y1cell_size, approx_z1cell_size, 
        approx_x2cell_size, approx_y2cell_size, approx_z2cell_size, 
        rp_max, rp_max, pi_max, xperiod, yperiod, zperiod, PBCs=PBCs)

    #square radial bins to make distance calculation cheaper
    rp_bins_squared = rp_bins**2.0
    pi_bins_squared = pi_bins**2.0

    #number of cells
    Ncell1 = double_tree.num_x1divs*double_tree.num_y1divs*double_tree.num_z1divs
    Ncell2 = double_tree.num_x2divs*double_tree.num_y2divs*double_tree.num_z2divs
    
    if verbose==True:
        print("volume 1 split {0},{1},{2} times along each dimension,\n"
              "resulting in {3} cells.".format(double_tree.num_x1divs,\
              double_tree.num_y1divs,double_tree.num_z1divs,Ncell1))
        print("volume 2 split {0},{1},{2} times along each dimension,\n"
              "resulting in {3} cells.".format(double_tree.num_x2divs,\
              double_tree.num_y2divs,double_tree.num_z2divs,Ncell1))
    
    #create a function to call with only one argument
    engine = partial(_xy_z_npairs_engine, double_tree, rp_bins_squared, 
        pi_bins_squared, period, PBCs)

    #do the pair counting
    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        counts = np.sum(pool.map(engine,range(Ncell1)),axis=0)
        pool.close()
    if num_threads == 1:
        counts = np.sum(map(engine,range(Ncell1)),axis=0)

    return counts

def _xy_z_npairs_engine(double_tree, rp_bins_squared, pi_bins_squared, period, PBCs, icell1):
    """
    pair counting engine for npairs function.  This code calls a cython function.
    """
    # print("...working on icell1 = %i" % icell1)
    
    counts = np.zeros((len(rp_bins_squared),len(pi_bins_squared)))
    
    #extract the points in the cell
    s1 = double_tree.tree1.slice_array[icell1]
    x_icell1, y_icell1, z_icell1 = (
        double_tree.tree1.x[s1],
        double_tree.tree1.y[s1],
        double_tree.tree1.z[s1])
        
    xsearch_length = np.sqrt(rp_bins_squared[-1])
    ysearch_length = np.sqrt(rp_bins_squared[-1])
    zsearch_length = np.sqrt(pi_bins_squared[-1])
    adj_cell_generator = double_tree.adjacent_cell_generator(
        icell1, xsearch_length, ysearch_length, zsearch_length)
    
    adj_cell_counter = 0
    for icell2, xshift, yshift, zshift in adj_cell_generator:
        adj_cell_counter +=1
        
        #ix, iy, iz = double_tree.tree2.cell_tuple_from_cell_idx(icell1)
        #print(adj_cell_counter, icell1, icell2, xshift, yshift, zshift)
        
        #extract the points in the cell
        s2 = double_tree.tree2.slice_array[icell2]
        x_icell2 = double_tree.tree2.x[s2] + xshift
        y_icell2 = double_tree.tree2.y[s2] + yshift 
        z_icell2 = double_tree.tree2.z[s2] + zshift

        #use cython functions to do pair counting
        counts += xy_z_npairs_no_pbc(
            x_icell1, y_icell1, z_icell1,
            x_icell2, y_icell2, z_icell2,
            rp_bins_squared, pi_bins_squared)
            
    return counts


def s_mu_npairs(data1, data2, s_bins, mu_bins, period = None,\
                verbose = False, num_threads = 1,\
                approx_cell1_size = None, approx_cell2_size = None):
    """ 
    Function counts the number of pairs as a function of radial separation, *s,* and :math:`\\mu\\equiv\\sin(\\theta_{\\rm los})`, where :math:`\\theta_{\\rm los}` is the line-of-sight angle between points and :math:`s^2 = r_{\\rm parallel}^2 + r_{\\rm perp}^2`. 
    
    
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
            
    s_bins: array_like
        numpy array of boundaries defining the radial bins in which pairs are counted.
    
    mu_bins: array_like
        numpy array of boundaries defining bins in :math:`\\sin(\\theta_{\\rm los})` 
        in which the pairs are counted in.  
        Note that using the sine is not common convention for 
        calculating the two point correlation function (see notes).
    
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
        which will result in reasonable performance for most use-cases. 
        Performance can vary sensitively with this parameter, so it is highly 
        recommended that you experiment with it when carrying out  
        performance-critical calculations. 

    approx_cell2_size : array_like, optional 
        See comments for ``approx_cell1_size``. 
    
    Returns
    -------
    num_pairs : array of length len(rbins)
        number of pairs

    Notes
    ------
    The quantity :math:`\\mu` is defined as the :math:`\\sin(\\theta_{\\rm los})` 
    and not the conventional :math:`\\cos(\\theta_{\\rm los})`. This is 
    because the pair counter has been optimized under the assumption that its 
    separation variable (in this case, :math:`\\mu`) *increases* 
    as :math:`\\theta_{\\rm los})` increases. 

    Returns
    -------
    N_pairs : array_like 
        2-d array of length *Num_rp_bins x Num_pi_bins* storing the pair counts in each bin. 

    Examples 
    --------
    For illustration purposes, we'll create some fake data and call the pair counter:

    >>> Npts1, Npts2, Lbox = 1e3, 1e3, 200.
    >>> period = [Lbox, Lbox, Lbox]
    >>> s_bins = np.logspace(-1, 1.25, 15)
    >>> mu_bins = np.linspace(-0.5, 0.5)

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

    >>> result = s_mu_npairs(data1, data2, s_bins, mu_bins, period = period)

    """
    
    #the parameters for this are similar to npairs, except mu_bins needs to be processed.
    # Process the inputs with the helper function
    x1, y1, z1, x2, y2, z2, rbins, period, num_threads, PBCs = (
        _npairs_process_args(data1, data2, s_bins, period, 
            verbose, num_threads, approx_cell1_size, approx_cell2_size)
        )        
    
    xperiod, yperiod, zperiod = period 
    rmax = np.max(s_bins)
    
    #process mu_bins parameter separately
    mu_bins = convert_to_ndarray(mu_bins)
    try:
        assert mu_bins.ndim == 1
        assert len(mu_bins) > 1
        if len(mu_bins) > 2:
            assert array_is_monotonic(mu_bins, strict = True) == 1
    except AssertionError:
        msg = ("Input ``mu_bins`` must be a monotonically increasing \n"
              "1D array with at least two entries")
        raise HalotoolsError(msg)
    
    
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

    #square radial bins to make distance calculation cheaper
    rbins_squared = rbins**2.0
        
    #number of cells
    Ncell1 = double_tree.num_x1divs*double_tree.num_y1divs*double_tree.num_z1divs

    #create a function to call with only one argument
    engine = partial(_s_mu_npairs_engine, double_tree, s_bins, mu_bins, period, PBCs)
    
    #do the pair counting
    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        counts = np.sum(pool.map(engine,range(Ncell1)),axis=0)
        pool.close()
    if num_threads == 1:
        counts = np.sum(map(engine,range(Ncell1)),axis=0)

    return counts

def _s_mu_npairs_engine(double_tree, s_bins, mu_bins, period, PBCs, icell1):
    """
    pair counting engine for npairs function.  This code calls a cython function.
    """
    # print("...working on icell1 = %i" % icell1)
    
    counts = np.zeros((len(s_bins), len(mu_bins)))
    
    #extract the points in the cell
    s1 = double_tree.tree1.slice_array[icell1]
    x_icell1, y_icell1, z_icell1 = (
        double_tree.tree1.x[s1],
        double_tree.tree1.y[s1],
        double_tree.tree1.z[s1])
        
    xsearch_length = s_bins[-1]
    ysearch_length = s_bins[-1]
    zsearch_length = s_bins[-1]
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
        counts += s_mu_npairs_no_pbc(
            x_icell1, y_icell1, z_icell1,
            x_icell2, y_icell2, z_icell2,
            s_bins, mu_bins)
            
    return counts


