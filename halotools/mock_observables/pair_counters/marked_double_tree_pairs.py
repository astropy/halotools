# -*- coding: utf-8 -*-

"""
generalized marked (weighted) pair counter

Points are partitioned using `~halotools.mock_observables.pair_counters.double_tree`
for effecient pair coutning.
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

__all__ = ['marked_npairs',\
           'xy_z_marked_npairs',\
           'velocity_marked_npairs',\
           'xy_z_velocity_marked_npairs']
__author__ = ['Duncan Campbell', 'Andrew Hearin']


def marked_npairs(data1, data2, rbins,
                  period=None, weights1 = None, weights2 = None, 
                  wfunc = 0, verbose = False, num_threads = 1,
                  approx_cell1_size = None, approx_cell2_size = None):
    """
    Calculate the number of weighted pairs with seperations greater than or equal to r, :math:`W(>r)`.
    
    The weight given to each pair is determined by the weights for a pair, 
    :math:`w_1`, :math:`w_2`, and a user-specified "weighting function", indicated 
    by the ``wfunc`` parameter, :math:`f(w_1,w_2)`.
    
    Parameters
    ----------
    data1 : array_like
        *N1* by 3 array of 3-D positions.  If the ``period`` parameter is set, each
        component of the coordinates should be bounded between zero and the corresponding
        periodic boundary.
    
    data2 : array_like
        *N2* by 3 array of 3-D positions.  If the ``period`` parameter is set, each
        component of the coordinates should be bounded between zero and the corresponding
        periodic boundary.
    
    rbins : array_like
        numpy array of length *Nrbins+1* defining the boundaries of bins in which 
        pairs are counted. 
    
    period : array_like, optional
        Length-3 array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, the period is assumed to be np.array([Lbox]*3).
    
    weights1 : array_like, optional
        Either a 1-D array of length *N1*, or a 2-D array of length *N1* x *N_weights*, 
        containing the weights used for the weighted pair counts. If this parameter is
        None, the weights are set to np.ones(*(N1,N_weights)*).
    
    weights2 : array_like, optional
        Either a 1-D array of length *N1*, or a 2-D array of length *N1* x *N_weights*, 
        containing the weights used for the weighted pair counts. If this parameter is
        None, the weights are set to np.ones(*(N1,N_weights)*).
        
    wfunc : int, optional
        weighting function integer ID. Each weighting function requires a specific 
        number of weights per point, *N_weights*.  See the Notes for a description of
        available weighting functions.
    
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
    wN_pairs : numpy.array
        array of length *Nrbins* containing the weighted number counts of pairs
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
    
    #number of cells
    Ncell1 = double_tree.num_x1divs*double_tree.num_y1divs*double_tree.num_z1divs
    Ncell2 = double_tree.num_x2divs*double_tree.num_y2divs*double_tree.num_z2divs

    #create a function to call with only one argument
    engine = partial(_marked_npairs_engine, double_tree, 
        weights1, weights2, rbins, period, PBCs, wfunc)
    
    #do the pair counting
    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        counts = np.sum(pool.map(engine,range(Ncell1)),axis=0)
        pool.close()
    else:
        counts = np.sum(map(engine,range(Ncell1)),axis=0)
    
    return counts


def _marked_npairs_engine(double_tree, weights1, weights2, 
                    rbins, period, PBCs, wfunc, icell1):
    """
    private internal function for 
    `~halotools.mock_observables.pair_counters.marked_double_tree_pairs.marked_npairs`.
    
    This is an engine that calls a cython module to count pairs
    """
    
    counts = np.zeros(len(rbins))
    
    #extract the points in the cell
    s1 = double_tree.tree1.slice_array[icell1]
    x_icell1, y_icell1, z_icell1 = (
        double_tree.tree1.x[s1],
        double_tree.tree1.y[s1],
        double_tree.tree1.z[s1])

    #extract the weights in the cell
    w_icell1 = weights1[s1, :]

    xsearch_length = rbins[-1]
    ysearch_length = rbins[-1]
    zsearch_length = rbins[-1]
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
                                       rbins, wfunc, shift)
            
    return counts


def xy_z_marked_npairs(data1, data2, rp_bins, pi_bins, period=None, 
                       weights1 = None, weights2 = None, 
                       wfunc = 0, verbose = False, num_threads = 1,
                       approx_cell1_size = None, approx_cell2_size = None):
    """
    Calculate the number of weighted pairs with seperations greater than or equal to :math:`r_{\\perp}` and :math:`r_{\\parallel}`, :math:`W(>r_{\\perp},>r_{\\parallel})`.
    
    :math:`r_{\\perp}` and :math:`r_{\\parallel}` are defined wrt the z-direction.
    
    The weight given to each pair is determined by the weights for a pair, 
    :math:`w_1`, :math:`w_2`, and a user-specified "weighting function", indicated 
    by the ``wfunc`` parameter, :math:`f(w_1,w_2)`.
    
    Parameters
    ----------
    data1 : array_like
        *N1* by 3 array of 3-D positions.  If the ``period`` parameter is set, each
        component of the coordinates should be bounded between zero and the corresponding
        periodic boundary.
    
    data2 : array_like
        *N2* by 3 array of 3-D positions.  If the ``period`` parameter is set, each
        component of the coordinates should be bounded between zero and the corresponding
        periodic boundary.
            
    rp_bins : array_like
        numpy array of length Nrp_bins+1 defining the boundaries of bins of projected 
        separation, :math:`r_{\\rm p}`, in which pairs are counted.
    
    pi_bins : array_like
        numpy array of length Npi_bins+1 defining the boundaries of bins of parallel
        separation, :math:`\\pi`, in which pairs are counted.
    
    period : array_like, optional
        Length-3 array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, the period is assumed to be np.array([Lbox]*3).
    
    weights1 : array_like, optional
        Either a 1-D array of length *N1*, or a 2-D array of length *N1* x *N_weights*, 
        containing the weights used for the weighted pair counts. If this parameter is
        None, the weights are set to np.ones(*(N1,N_weights)*).
    
    weights2 : array_like, optional
        Either a 1-D array of length *N1*, or a 2-D array of length *N1* x *N_weights*, 
        containing the weights used for the weighted pair counts. If this parameter is
        None, the weights are set to np.ones(*(N1,N_weights)*).
        
    wfunc : int, optional
        weighting function integer ID. Each weighting function requires a specific 
        number of weights per point, *N_weights*.  See the Notes for a description of
        available weighting functions.
    
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
    wN_pairs : numpy.ndarray
        2-D array of shape *(Nrp_bins,Npi_bins)* containing the weighted number 
        counts of pairs
    """
    
    ### Process the inputs with the helper function
    x1, y1, z1, x2, y2, z2, rp_bins, pi_bins, period, num_threads, PBCs = (
        _xy_z_npairs_process_args(data1, data2, rp_bins, pi_bins, period, 
            verbose, num_threads, approx_cell1_size, approx_cell2_size)
        )
    xperiod, yperiod, zperiod = period 
    rp_max = np.max(rp_bins)
    pi_max = np.max(pi_bins)
    
    # Process the input weights and with the helper function
    weights1, weights2 = (
        _marked_npairs_process_weights(data1, data2, 
            weights1, weights2, wfunc))
    
    ### Compute the estimates for the cell sizes
    approx_cell1_size, approx_cell2_size = _set_approximate_xy_z_cell_sizes(
        approx_cell1_size, approx_cell2_size, rp_max, pi_max, period)
    
    approx_x1cell_size, approx_y1cell_size, approx_z1cell_size = approx_cell1_size
    approx_x2cell_size, approx_y2cell_size, approx_z2cell_size = approx_cell2_size
    
    double_tree = FlatRectanguloidDoubleTree(
        x1, y1, z1, x2, y2, z2,  
        approx_x1cell_size, approx_y1cell_size, approx_z1cell_size, 
        approx_x2cell_size, approx_y2cell_size, approx_z2cell_size, 
        rp_max, rp_max, pi_max, xperiod, yperiod, zperiod, PBCs=PBCs)
    
    #sort the weights arrays
    weights1 = np.ascontiguousarray(weights1[double_tree.tree1.idx_sorted, :])
    weights2 = np.ascontiguousarray(weights2[double_tree.tree2.idx_sorted, :])
    
    #number of cells
    Ncell1 = double_tree.num_x1divs*double_tree.num_y1divs*double_tree.num_z1divs
    Ncell2 = double_tree.num_x2divs*double_tree.num_y2divs*double_tree.num_z2divs
    
    #create a function to call with only one argument
    engine = partial(_xy_z_marked_npairs_engine, double_tree, 
        weights1, weights2, rp_bins, pi_bins, period, PBCs, wfunc)
    
    #do the pair counting
    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        counts = np.sum(pool.map(engine,range(Ncell1)),axis=0)
        pool.close()
    else:
        counts = np.sum(map(engine,range(Ncell1)),axis=0)
    
    return counts


def _xy_z_marked_npairs_engine(double_tree, weights1, weights2, 
                               rp_bins, pi_bins, period, PBCs, wfunc, icell1):
    """
    private internal function for 
    `~halotools.mock_observables.pair_counters.marked_double_tree_pairs.xy_z_marked_npairs`.
    
    This is an engine that calls a cython module to count pairs
    """
    
    counts = np.zeros((len(rp_bins),len(pi_bins)))
    
    #extract the points in the cell
    s1 = double_tree.tree1.slice_array[icell1]
    x_icell1, y_icell1, z_icell1 = (
        double_tree.tree1.x[s1],
        double_tree.tree1.y[s1],
        double_tree.tree1.z[s1])
    
    #extract the weights in the cell
    w_icell1 = weights1[s1, :]
    
    xsearch_length = rp_bins[-1]
    ysearch_length = rp_bins[-1]
    zsearch_length = pi_bins[-1]
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
        counts += xy_z_marked_npairs_no_pbc(x_icell1, y_icell1, z_icell1,
                                            x_icell2, y_icell2, z_icell2,
                                            w_icell1, w_icell2, 
                                            rp_bins, pi_bins, wfunc, shift)
            
    return counts


def velocity_marked_npairs(data1, data2, rbins, period=None, 
    weights1 = None, weights2 = None, 
    wfunc = 0, verbose = False, num_threads = 1,
    approx_cell1_size = None, approx_cell2_size = None):
    """
    Calculate the number of velocity weighted pairs with seperations greater than or equal to r, :math:`W(>r)`.
    
    The weight given to each pair is determined by the weights for a pair, 
    :math:`w_1`, :math:`w_2`, and a user-specified "velocity weighting function", indicated 
    by the ``wfunc`` parameter, :math:`f(w_1,w_2)`.
    
    Parameters
    ----------
    data1 : array_like
        *N1* by 3 array of 3-D positions.  If the ``period`` parameter is set, each
        component of the coordinates should be bounded between zero and the corresponding
        periodic boundary.
    
    data2 : array_like
        *N2* by 3 array of 3-D positions.  If the ``period`` parameter is set, each
        component of the coordinates should be bounded between zero and the corresponding
        periodic boundary.
    
    rbins : array_like
        numpy array of length Nrbins+1 defining the boundaries of bins in which 
        pairs are counted. 
    
    period : array_like, optional
        Length-3 array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, the period is assumed to be np.array([Lbox]*3).
    
    weights1 : array_like, optional
        Either a 1-D array of length *N1*, or a 2-D array of length *N1* x *N_weights*, 
        containing the weights used for the weighted pair counts. If this parameter is
        None, the weights are set to np.ones(*(N1,N_weights)*).
    
    weights2 : array_like, optional
        Either a 1-D array of length *N1*, or a 2-D array of length *N1* x *N_weights*, 
        containing the weights used for the weighted pair counts. If this parameter is
        None, the weights are set to np.ones(*(N1,N_weights)*).
    
    wfunc : int, optional
        velocity weighting function integer ID. Each weighting function requires a specific 
        number of weights per point, *N_weights*.  See the Notes for a description of
        available weighting functions.
    
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
    w1N_pairs : numpy.array
        array of length *Nrbins* containing the weighted number counts of pairs
        The exact values depend on ``weight_func_id`` 
        (which weighting function was chosen).
    
    w2N_pairs : numpy.array
        array of length *Nrbins* containing the weighted number counts of pairs
        The exact values depend on ``weight_func_id`` 
        (which weighting function was chosen).
    
    w3N_pairs : numpy.array
        array of length *Nrbins* containing the weighted number counts of pairs
        The exact values depend on ``weight_func_id`` 
        (which weighting function was chosen).
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
    
    #number of cells
    Ncell1 = double_tree.num_x1divs*double_tree.num_y1divs*double_tree.num_z1divs
    Ncell2 = double_tree.num_x2divs*double_tree.num_y2divs*double_tree.num_z2divs

    #create a function to call with only one argument
    engine = partial(_velocity_marked_npairs_engine, double_tree, 
        weights1, weights2, rbins, period, PBCs, wfunc)
    
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


def _velocity_marked_npairs_engine(double_tree, weights1, weights2, 
                             rbins, period, PBCs, wfunc, icell1):
    """
    private internal function for 
    `~halotools.mock_observables.pair_counters.marked_double_tree_pairs.velocity_marked_npairs`.
    
    This is an engine that calls a cython module to count pairs
    """
    
    counts1 = np.zeros(len(rbins))
    counts2 = np.zeros(len(rbins))
    counts3 = np.zeros(len(rbins))
    
    #extract the points in the cell
    s1 = double_tree.tree1.slice_array[icell1]
    x_icell1, y_icell1, z_icell1 = (
        double_tree.tree1.x[s1],
        double_tree.tree1.y[s1],
        double_tree.tree1.z[s1])

    #extract the weights in the cell
    w_icell1 = weights1[s1, :]

    xsearch_length = rbins[-1]
    ysearch_length = rbins[-1]
    zsearch_length = rbins[-1]
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
                                       rbins, wfunc, shift)
        counts1 += holder1 
        counts2 += holder2
        counts3 += holder3
            
    return counts1, counts2, counts3 


def xy_z_velocity_marked_npairs(data1, data2, rp_bins, pi_bins, period=None, 
    weights1 = None, weights2 = None, 
    wfunc = 0, verbose = False, num_threads = 1,
    approx_cell1_size = None, approx_cell2_size = None):
    """
    Calculate the number of velocity weighted pairs with seperations greater than or equal to :math:`r_{\\perp}` and :math:`r_{\\parallel}`, :math:`W(>r_{\\perp},>r_{\\parallel})`.
    
    :math:`r_{\\perp}` and :math:`r_{\\parallel}` are defined wrt the z-direction.
    
    The weight given to each pair is determined by the weights for a pair, 
    :math:`w_1`, :math:`w_2`, and a user-specified "velocity weighting function", indicated 
    by the ``wfunc`` parameter, :math:`f(w_1,w_2)`.
    
    Parameters
    ----------
    data1 : array_like
        *N1* by 3 array of 3-D positions.  If the ``period`` parameter is set, each
        component of the coordinates should be bounded between zero and the corresponding
        periodic boundary.
    
    data2 : array_like
        *N2* by 3 array of 3-D positions.  If the ``period`` parameter is set, each
        component of the coordinates should be bounded between zero and the corresponding
        periodic boundary.
    
    rp_bins : array_like
        numpy array of length Nrp_bins+1 defining the boundaries of bins of projected 
        separation, :math:`r_{\\rm p}`, in which pairs are counted.
    
    pi_bins : array_like
        numpy array of length Npi_bins+1 defining the boundaries of bins of parallel
        separation, :math:`\\pi`, in which pairs are counted.
    
    period : array_like, optional
        Length-3 array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, the period is assumed to be np.array([Lbox]*3).
    
    weights1 : array_like, optional
        Either a 1-D array of length *N1*, or a 2-D array of length *N1* x *N_weights*, 
        containing the weights used for the weighted pair counts. If this parameter is
        None, the weights are set to np.ones(*(N1,N_weights)*).
    
    weights2 : array_like, optional
        Either a 1-D array of length *N1*, or a 2-D array of length *N1* x *N_weights*, 
        containing the weights used for the weighted pair counts. If this parameter is
        None, the weights are set to np.ones(*(N1,N_weights)*).
    
    wfunc : int, optional
        velocity weighting function integer ID. Each weighting function requires a specific 
        number of weights per point, *N_weights*.  See the Notes for a description of
        available weighting functions.
    
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
    w1N_pairs : numpy.array
        2-D array of shape *(Nrp_bins,Npi_bins)* containing the weighted number counts 
        of pairs. The exact values depend on ``weight_func_id`` 
        (which weighting function was chosen).
    
    w2N_pairs : numpy.array
        2-D array of shape *(Nrp_bins,Npi_bins)* containing the weighted number counts 
        of pairs. The exact values depend on ``weight_func_id`` 
        (which weighting function was chosen).
    
    w3N_pairs : numpy.array
        2-D array of shape *(Nrp_bins,Npi_bins)* containing the weighted number counts 
        of pairs. The exact values depend on ``weight_func_id`` 
        (which weighting function was chosen).
    """
    
    ### Process the inputs with the helper function
    x1, y1, z1, x2, y2, z2, rp_bins, pi_bins, period, num_threads, PBCs = (
        _xy_z_npairs_process_args(data1, data2, rp_bins, pi_bins, period, 
            verbose, num_threads, approx_cell1_size, approx_cell2_size)
        )
    xperiod, yperiod, zperiod = period 
    rp_max = np.max(rp_bins)
    pi_max = np.max(pi_bins)
    
    # Process the input weights and with the helper function
    weights1, weights2 = (
        _velocity_marked_npairs_process_weights(data1, data2, 
            weights1, weights2, wfunc))
    
    ### Compute the estimates for the cell sizes
    approx_cell1_size, approx_cell2_size = _set_approximate_xy_z_cell_sizes(
        approx_cell1_size, approx_cell2_size, rp_max, pi_max, period)
    approx_x1cell_size, approx_y1cell_size, approx_z1cell_size = approx_cell1_size
    approx_x2cell_size, approx_y2cell_size, approx_z2cell_size = approx_cell2_size
    
    double_tree = FlatRectanguloidDoubleTree(
        x1, y1, z1, x2, y2, z2,  
        approx_x1cell_size, approx_y1cell_size, approx_z1cell_size, 
        approx_x2cell_size, approx_y2cell_size, approx_z2cell_size, 
        rp_max, rp_max, pi_max, xperiod, yperiod, zperiod, PBCs=PBCs)

    #sort the weights arrays
    weights1 = np.ascontiguousarray(weights1[double_tree.tree1.idx_sorted, :])
    weights2 = np.ascontiguousarray(weights2[double_tree.tree2.idx_sorted, :])
    
    #number of cells
    Ncell1 = double_tree.num_x1divs*double_tree.num_y1divs*double_tree.num_z1divs
    Ncell2 = double_tree.num_x2divs*double_tree.num_y2divs*double_tree.num_z2divs
    
    #create a function to call with only one argument
    engine = partial(_xy_z_velocity_marked_npairs_engine, double_tree, 
        weights1, weights2, rp_bins, pi_bins, period, PBCs, wfunc)
    
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


def _xy_z_velocity_marked_npairs_engine(double_tree, weights1, weights2, 
                                  rp_bins, pi_bins, period, PBCs, wfunc, icell1):
    """
    private internal function for 
    `~halotools.mock_observables.pair_counters.marked_double_tree_pairs.xy_z_velocity_marked_npairs`.
    
    This is an engine that calls a cython module to count pairs
    """
    
    counts1 = np.zeros((len(rp_bins), len(pi_bins)))
    counts2 = np.zeros((len(rp_bins), len(pi_bins)))
    counts3 = np.zeros((len(rp_bins), len(pi_bins)))
    
    #extract the points in the cell
    s1 = double_tree.tree1.slice_array[icell1]
    x_icell1, y_icell1, z_icell1 = (
        double_tree.tree1.x[s1],
        double_tree.tree1.y[s1],
        double_tree.tree1.z[s1])

    #extract the weights in the cell
    w_icell1 = weights1[s1, :]

    xsearch_length = rp_bins[-1]
    ysearch_length = rp_bins[-1]
    zsearch_length = pi_bins[-1]
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
        holder1, holder2, holder3 = xy_z_velocity_marked_npairs_no_pbc(
                                       x_icell1, y_icell1, z_icell1,
                                       x_icell2, y_icell2, z_icell2,
                                       w_icell1, w_icell2, 
                                       rp_bins, pi_bins, wfunc, shift)
        counts1 += holder1 
        counts2 += holder2
        counts3 += holder3
            
    return counts1, counts2, counts3 

