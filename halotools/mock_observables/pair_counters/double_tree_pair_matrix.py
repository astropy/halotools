# -*- coding: utf-8 -*-

"""
caclulate the pairwise distances in cuboid volumes using 
`~halotools.mock_observables.double_tree`.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import time
import multiprocessing
from functools import partial
from scipy.sparse import coo_matrix
from .double_tree_helpers import (_set_approximate_cell_sizes, 
    _set_approximate_xy_z_cell_sizes, _enclose_in_box)
from .double_tree import *
from .cpairs.pairwise_distances import *
from .marked_cpairs.conditional_pairwise_distances import *
from ...custom_exceptions import *
from ...utils.array_utils import convert_to_ndarray, array_is_monotonic

__all__=['pair_matrix', 'xy_z_pair_matrix',\
         'conditional_pair_matrix','conditional_xy_z_pair_matrix']
__author__=['Duncan Campbell']


def pair_matrix(data1, data2, r_max, period=None, verbose=False, num_threads=1,
                approx_cell1_size = None, approx_cell2_size = None):
    """
    Calculate the distance to all pairs with seperations less than ``r_max``.
    
    Parameters
    ----------
    data1 : array_like
        N1 by 3 numpy array of 3-D positions.
            
    data2 : array_like
        N2 by 3 numpy array of 3-D positions.
            
    r_max : float
        Maximum distance to search for pairs
    
    period : array_like, optional
        length 3 array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*3).
        If none, PBCs are set to infinity.  If True, period is set to be Lbox
    
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
    dists : `~scipy.sparse.coo_matrix`
        N1 x N2 sparse matrix in COO format containing distances between points.
    
    Notes
    -----
    The distances between all points with seperations less than ``r_max`` are stored 
    and returned.  If there are many points and/or ``r_max`` is large, this can become
    very memmory intensive.
    
    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a 
    periodic unit cube. 
    
    >>> Npts = 1000
    >>> Lbox = 1.0
    >>> period = np.array([Lbox,Lbox,Lbox])
        
    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)
        
    We transform our *x, y, z* points into the array shape used by the pair-counter by 
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation 
    is used throughout the `~halotools.mock_observables` sub-package:
    
    >>> coords = np.vstack((x,y,z)).T
    
    Now, we can find the distance between all points:
    
    >>> r_max = 0.1
    >>> dists = pair_matrix(coords, coords, r_max, period=period)
    
    The diagonal of this matrix will be zeros, the distance between each point and itself.
    The off diagonal elements are the pairwise distances between points i,j in the order 
    they appear in `coords`.  In this case, the matrix will be symmetric.
    """
    
    search_dim_max = np.array([r_max, r_max, r_max])
    function_args = [data1, data2, period, num_threads, search_dim_max]
    x1, y1, z1, x2, y2, z2, period, num_threads, PBCs = _process_args(*function_args)
    #note that process_args sets period equal to Lbox is there are no PBCs
    xperiod, yperiod, zperiod = period
    r_max = float(r_max)
    
    if verbose==True:
        print("running on {0} x {1}\n"
              "points with PBCs={2}".format(len(data1), len(data2), PBCs))
        start = time.time()
    
    approx_cell1_size, approx_cell2_size = (
        _set_approximate_cell_sizes(approx_cell1_size, approx_cell2_size, r_max, period)
        )
    approx_x1cell_size, approx_y1cell_size, approx_z1cell_size = approx_cell1_size
    approx_x2cell_size, approx_y2cell_size, approx_z2cell_size = approx_cell2_size
    
    double_tree = FlatRectanguloidDoubleTree(x1, y1, z1, x2, y2, z2,
                      approx_x1cell_size, approx_y1cell_size, approx_z1cell_size, 
                      approx_x2cell_size, approx_y2cell_size, approx_z2cell_size, 
                      r_max, r_max, r_max, xperiod, yperiod, zperiod, PBCs=PBCs)
    
    #number of cells
    Ncell1 = double_tree.num_x1divs*double_tree.num_y1divs*double_tree.num_z1divs
    Ncell2 = double_tree.num_x2divs*double_tree.num_y2divs*double_tree.num_z2divs
    
    if verbose==True:
        print("volume 1 split {0},{1},{2} times along each dimension,\n"
              "resulting in {3} cells.".format(double_tree.num_x1divs,\
              double_tree.num_y1divs,double_tree.num_z1divs,Ncell1))
        print("volume 2 split {0},{1},{2} times along each dimension,\n"
              "resulting in {3} cells.".format(double_tree.num_x2divs,\
              double_tree.num_y2divs,double_tree.num_z2divs,Ncell2))
    
    #create a function to call with only one argument
    engine = partial(_pair_matrix_engine, double_tree, r_max, period, PBCs)
    
    #do the pair counting
    if num_threads>1:
        pool = multiprocessing.Pool(num_threads)
        result = pool.map(engine,range(Ncell1))
        pool.close()
    if num_threads==1:
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
    i_inds = double_tree.tree1.idx_sorted[i_inds]
    j_inds = double_tree.tree2.idx_sorted[j_inds]
    
    if verbose==True:
        print("total run time: {0} seconds".format(time.time()-start))
    
    return coo_matrix((d, (i_inds, j_inds)), shape=(len(data1),len(data2)))


def _pair_matrix_engine(double_tree, r_max, period, PBCs, icell1):
    """
    pair counting engine for npairs function.  This code calls a cython function.
    """
    
    d = np.zeros((0,), dtype='float')
    i_inds = np.zeros((0,), dtype='int')
    j_inds = np.zeros((0,), dtype='int')
    
    #extract the points in the cell
    s1 = double_tree.tree1.slice_array[icell1]
    x_icell1, y_icell1, z_icell1 = (
        double_tree.tree1.x[s1],
        double_tree.tree1.y[s1],
        double_tree.tree1.z[s1])
    
    i_min = s1.start
    
    xsearch_length = r_max
    ysearch_length = r_max
    zsearch_length = r_max
    adj_cell_generator = double_tree.adjacent_cell_generator(
        icell1, xsearch_length, ysearch_length, zsearch_length)
            
    adj_cell_counter = 0
    for icell2, xshift, yshift, zshift in adj_cell_generator:
                
        #extract the points in the cell
        s2 = double_tree.tree2.slice_array[icell2]
        x_icell2 = double_tree.tree2.x[s2] + xshift
        y_icell2 = double_tree.tree2.y[s2] + yshift 
        z_icell2 = double_tree.tree2.z[s2] + zshift
        
        j_min = s2.start
        
        dd, ii_inds, jj_inds = pairwise_distance_no_pbc(x_icell1, y_icell1, z_icell1,\
                                                        x_icell2, y_icell2, z_icell2,\
                                                        r_max)
        
        ii_inds = ii_inds+i_min
        jj_inds = jj_inds+j_min
        
        #update storage arrays
        d = np.concatenate((d,dd))
        i_inds = np.concatenate((i_inds,ii_inds))
        j_inds = np.concatenate((j_inds,jj_inds))
    
    return d, i_inds, j_inds


def xy_z_pair_matrix(data1, data2, rp_max, pi_max, period=None, verbose=False,\
                     num_threads=1, approx_cell1_size = None, approx_cell2_size = None):
    """
    Calculate the distance to all pairs with perpendicular seperations less than or 
    equal to ``rp_max`` and parallel seperations ``pi_max`` in redshift space.
    
    Parameters
    ----------
    data1 : array_like
        N1 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period.
            
    data2 : array_like
        N2 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period.
            
    rp_max : float
        maximum distance to connect pairs
    
    pi_max : float
        maximum distance to connect pairs
    
    period : array_like, optional
        length 3 array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*3).
        If none, PBCs are set to infinity.  If True, period is set to be Lbox
    
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
    perp_dists : `~scipy.sparse.coo_matrix`
        N1 x N2 sparse matrix in COO format containing perpendicular distances between points.
    
    para_dists : `~scipy.sparse.coo_matrix`
        N1 x N2 sparse matrix in COO format containing parallel distances between points.
    
    Notes
    -----
    The distances between all points with seperations that meet the secified conditions 
    are stored and returned.  If there are many points and/or ``rp_max`` and ``pi_max`` 
    are large, this can become very memmory intensive.
    
    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a 
    periodic unit cube. 
    
    >>> Npts = 1000
    >>> Lbox = 1.0
    >>> period = np.array([Lbox,Lbox,Lbox])
    
    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)
    
    We transform our *x, y, z* points into the array shape used by the pair-counter by 
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation 
    is used throughout the `~halotools.mock_observables` sub-package:
    
    >>> coords = np.vstack((x,y,z)).T
    
    Now, we can find the distance between all points:
    
    >>> rp_max = 0.1
    >>> pi_max = 0.2
    >>> d_perp, d_para = xy_z_pair_matrix(coords, coords, rp_max, pi_max, period=period)
    
    The diagonal of this matrix will be zeros, the distance between each point and itself.
    The off diagonal elements are the pairwise distances between points i,j in the order 
    they appear in `coords`.  In this case, the matrix will be symmetric.
    """
    
    search_dim_max = np.array([rp_max, rp_max, pi_max])
    function_args = [data1, data2, period, num_threads, search_dim_max]
    x1, y1, z1, x2, y2, z2, period, num_threads, PBCs = _process_args(*function_args)
    #note that process_args sets period equal to Lbox is there are no PBCs
    xperiod, yperiod, zperiod = period 
    rp_max = float(rp_max)
    pi_max = float(pi_max)
    
    if verbose==True:
        print("running on {0} x {1}\n"
              "points with PBCs={2}".format(len(data1), len(data2), PBCs))
        start = time.time()
    
    approx_cell1_size, approx_cell2_size = (
        _set_approximate_xy_z_cell_sizes(approx_cell1_size, approx_cell2_size, rp_max, pi_max, period)
        )
    approx_x1cell_size, approx_y1cell_size, approx_z1cell_size = approx_cell1_size
    approx_x2cell_size, approx_y2cell_size, approx_z2cell_size = approx_cell2_size
    
    double_tree = FlatRectanguloidDoubleTree(x1, y1, z1, x2, y2, z2,
                      approx_x1cell_size, approx_y1cell_size, approx_z1cell_size, 
                      approx_x2cell_size, approx_y2cell_size, approx_z2cell_size, 
                      rp_max, rp_max, pi_max, xperiod, yperiod,zperiod, PBCs=PBCs)
    
    #number of cells
    Ncell1 = double_tree.num_x1divs*double_tree.num_y1divs*double_tree.num_z1divs
    Ncell2 = double_tree.num_x2divs*double_tree.num_y2divs*double_tree.num_z2divs
    
    if verbose==True:
        print("volume 1 split {0},{1},{2} times along each dimension,\n"
              "resulting in {3} cells.".format(double_tree.num_x1divs,\
              double_tree.num_y1divs,double_tree.num_z1divs,Ncell1))
        print("volume 2 split {0},{1},{2} times along each dimension,\n"
              "resulting in {3} cells.".format(double_tree.num_x2divs,\
              double_tree.num_y2divs,double_tree.num_z2divs,Ncell2))
    
    #create a function to call with only one argument
    engine = partial(_xy_z_pair_matrix_engine, double_tree, rp_max, pi_max, period, PBCs)
    
    #do the pair counting
    if num_threads>1:
        pool = multiprocessing.Pool(num_threads)
        result = pool.map(engine,range(Ncell1))
        pool.close()
    if num_threads==1:
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
    i_inds = double_tree.tree1.idx_sorted[i_inds]
    j_inds = double_tree.tree2.idx_sorted[j_inds]
    
    if verbose==True:
        print("total run time: {0} seconds".format(time.time()-start))
    
    return coo_matrix((d_perp, (i_inds, j_inds)), shape=(len(data1),len(data2))),\
           coo_matrix((d_para, (i_inds, j_inds)), shape=(len(data1),len(data2)))


def _xy_z_pair_matrix_engine(double_tree, rp_max, pi_max, period, PBCs, icell1):
    """
    pair counting engine for xy_z_fof_npairs function.  This code calls a cython function.
    """
    
    d_perp = np.zeros((0,), dtype='float')
    d_para = np.zeros((0,), dtype='float')
    i_inds = np.zeros((0,), dtype='int')
    j_inds = np.zeros((0,), dtype='int')
    
    #extract the points in the cell
    s1 = double_tree.tree1.slice_array[icell1]
    x_icell1, y_icell1, z_icell1 = (
        double_tree.tree1.x[s1],
        double_tree.tree1.y[s1],
        double_tree.tree1.z[s1])
    
    i_min = s1.start
    
    xsearch_length = rp_max
    ysearch_length = rp_max
    zsearch_length = pi_max
    adj_cell_generator = double_tree.adjacent_cell_generator(
        icell1, xsearch_length, ysearch_length, zsearch_length)
    
    adj_cell_counter = 0
    for icell2, xshift, yshift, zshift in adj_cell_generator:
        adj_cell_counter +=1
        
        #extract the points in the cell
        s2 = double_tree.tree2.slice_array[icell2]
        x_icell2 = double_tree.tree2.x[s2] + xshift
        y_icell2 = double_tree.tree2.y[s2] + yshift 
        z_icell2 = double_tree.tree2.z[s2] + zshift
        
        j_min = s2.start
        
        dd_perp, dd_para, ii_inds, jj_inds = pairwise_xy_z_distance_no_pbc(\
                                                 x_icell1, y_icell1, z_icell1,\
                                                 x_icell2, y_icell2, z_icell2,\
                                                 rp_max, pi_max)
        
        ii_inds = ii_inds+i_min
        jj_inds = jj_inds+j_min
        
        #update storage arrays
        d_perp = np.concatenate((d_perp,dd_perp))
        d_para = np.concatenate((d_para,dd_para))
        i_inds = np.concatenate((i_inds,ii_inds))
        j_inds = np.concatenate((j_inds,jj_inds))
    
    return d_perp, d_para, i_inds, j_inds


def conditional_pair_matrix(data1, data2, r_max, weights1, weights2, cond_func_id,
                            period=None, verbose=False, num_threads=1,
                            approx_cell1_size = None, approx_cell2_size = None):
    """
    Calculate the distance to all pairs with seperations less than ``r_max`` that pass a user specified condition.
    
    Parameters
    ----------
    data1 : array_like
        N1 by 3 numpy array of 3-D positions.
    
    data2 : array_like
        N2 by 3 numpy array of 3-D positions.
    
    r_max : float
        Maximum distance to search for pairs
    
    weights1 : array_like, optional
        Either a 1-D array of length *N1*, or a 2-D array of length *N1* x *N_weights*, 
        containing the weights used for the conditional counts. If this parameter is
        None, the weights are set to np.ones(*(N1,N_weights)*).
    
    weights2 : array_like, optional
        Either a 1-D array of length *N1*, or a 2-D array of length *N1* x *N_weights*, 
        containing the weights used for the conditional counts. If this parameter is
        None, the weights are set to np.ones(*(N1,N_weights)*).
    
    cond_func_id : int, optional
        conditonal function integer ID. Each conditional function requires a specific 
        number of weights per point, *N_weights*.  See the Notes for a description of
        available functions.
    
    period : array_like, optional
        length 3 array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*3).
        If none, PBCs are set to infinity.  If True, period is set to be Lbox
    
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
    dists : `~scipy.sparse.coo_matrix`
        N1 x N2 sparse matrix in COO format containing distances between points.
    
    Notes
    -----
    The distances between all points with seperations less than ``r_max`` are stored 
    and returned.  If there are many points and/or ``r_max`` is large, this can become
    very memmory intensive.
    
    Points with zero seperation are considered a self-match, and do no count as neighbors.
    
    There are multiple conditonal functions available.  In general, each requires a 
    different number of marks per point, N_marks.  The conditonal function gets passed 
    two vectors per pair, w1 and w2, of length N_marks and return a float.  
    
    A pair pair is counted as a neighbor if the conditonal function evaulates as True.
    
    The available marking functions, ``cond_func`` and the associated integer 
    ID numbers are:
    
    #. greater than (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] > w_2[0] \\\\
                    False & : w_1[0] \\geq w_2[0] \\\\
                \\end{array}
                \\right.
    
    #. less than (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] < w_2[0] \\\\
                    False & : w_1[0] \\leq w_2[0] \\\\
                \\end{array}
                \\right.
    
    #. equality (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] = w_2[0] \\\\
                    False & : w_1[0] \\neq w_2[0] \\\\
                \\end{array}
                \\right.
    
    #. inequality (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] \\neq w_2[0] \\\\
                    False & : w_1[0] = w_2[0] \\\\
                \\end{array}
                \\right.
    
    #. tolerance greater than (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] > (w_2[0]+w_1[1]) \\\\
                    False & : w_1[0] \\leq (w_2[0]+w_1[1]) \\\\
                \\end{array}
                \\right.
    
    #. tolerance less than (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] < (w_2[0]+w_1[1]) \\\\
                    False & : w_1[0] \\geq (w_2[0]+w_1[1]) \\\\
                \\end{array}
                \\right.
    
    
    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a 
    periodic unit cube. 
    
    >>> Npts = 1000
    >>> Lbox = 1.0
    >>> period = np.array([Lbox,Lbox,Lbox])
        
    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)
        
    We transform our *x, y, z* points into the array shape used by the pair-counter by 
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation 
    is used throughout the `~halotools.mock_observables` sub-package:
    
    >>> coords = np.vstack((x,y,z)).T
    
    Create some random weights:
    
    >>> weights = np.random.random(Npts)
    
    Now, we can find the distance between all points:
    
    >>> r_max = 0.1
    >>> cond_func = 1
    >>> dists = conditional_pair_matrix(coords, coords, r_max, weights, weights, cond_func, period=period)
    
    The diagonal of this matrix will be zeros, the distance between each point and itself.
    The off diagonal elements are the pairwise distances between points i,j in the order 
    they appear in `coords`.  In this case, the matrix will be symmetric.
    """
    
    search_dim_max = np.array([r_max, r_max, r_max])
    function_args = [data1, data2, period, num_threads, search_dim_max]
    x1, y1, z1, x2, y2, z2, period, num_threads, PBCs = _process_args(*function_args)
    #note that process_args sets period equal to Lbox is there are no PBCs
    xperiod, yperiod, zperiod = period
    r_max = float(r_max)
    
    weights1, weights2 = _process_weights(weights1, weights2, cond_func_id, data1, data2)
    
    if verbose==True:
        print("running on {0} x {1}\n"
              "points with PBCs={2}".format(len(data1), len(data2), PBCs))
        start = time.time()
    
    approx_cell1_size, approx_cell2_size = (
        _set_approximate_cell_sizes(approx_cell1_size, approx_cell2_size, r_max, period)
        )
    approx_x1cell_size, approx_y1cell_size, approx_z1cell_size = approx_cell1_size
    approx_x2cell_size, approx_y2cell_size, approx_z2cell_size = approx_cell2_size
    
    double_tree = FlatRectanguloidDoubleTree(x1, y1, z1, x2, y2, z2,
                      approx_x1cell_size, approx_y1cell_size, approx_z1cell_size, 
                      approx_x2cell_size, approx_y2cell_size, approx_z2cell_size, 
                      r_max, r_max, r_max, xperiod, yperiod, zperiod, PBCs=PBCs)
    
    #number of cells
    Ncell1 = double_tree.num_x1divs*double_tree.num_y1divs*double_tree.num_z1divs
    Ncell2 = double_tree.num_x2divs*double_tree.num_y2divs*double_tree.num_z2divs
    
    if verbose==True:
        print("volume 1 split {0},{1},{2} times along each dimension,\n"
              "resulting in {3} cells.".format(double_tree.num_x1divs,\
              double_tree.num_y1divs,double_tree.num_z1divs,Ncell1))
        print("volume 2 split {0},{1},{2} times along each dimension,\n"
              "resulting in {3} cells.".format(double_tree.num_x2divs,\
              double_tree.num_y2divs,double_tree.num_z2divs,Ncell2))
    
    #create a function to call with only one argument
    engine = partial(_conditional_pair_matrix_engine, double_tree, weights1, weights2, r_max, period, cond_func_id, PBCs)
    
    #do the pair counting
    if num_threads>1:
        pool = multiprocessing.Pool(num_threads)
        result = pool.map(engine,range(Ncell1))
        pool.close()
    if num_threads==1:
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
    i_inds = double_tree.tree1.idx_sorted[i_inds]
    j_inds = double_tree.tree2.idx_sorted[j_inds]
    
    if verbose==True:
        print("total run time: {0} seconds".format(time.time()-start))
    
    return coo_matrix((d, (i_inds, j_inds)), shape=(len(data1),len(data2)))


def _conditional_pair_matrix_engine(double_tree, weights1, weights2, r_max, period, PBCs, cond_func_id, icell1):
    """
    pair counting engine for npairs function.  This code calls a cython function.
    """
    
    d = np.zeros((0,), dtype='float')
    i_inds = np.zeros((0,), dtype='int')
    j_inds = np.zeros((0,), dtype='int')
    
    #extract the points in the cell
    s1 = double_tree.tree1.slice_array[icell1]
    x_icell1, y_icell1, z_icell1 = (
        double_tree.tree1.x[s1],
        double_tree.tree1.y[s1],
        double_tree.tree1.z[s1])
    
    #extract the weights in the cell
    w_icell1 = weights1[s1, :]
    
    i_min = s1.start
    
    xsearch_length = r_max
    ysearch_length = r_max
    zsearch_length = r_max
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
        w_icell2 = weights2[s2, :]
        
        j_min = s2.start
        
        dd, ii_inds, jj_inds = conditional_pairwise_distance_no_pbc(x_icell1, y_icell1, z_icell1,\
                                                                    x_icell2, y_icell2, z_icell2,\
                                                                    r_max, w_icell1, w_icell2, cond_func_id)
        
        ii_inds = ii_inds+i_min
        jj_inds = jj_inds+j_min
        
        #update storage arrays
        d = np.concatenate((d,dd))
        i_inds = np.concatenate((i_inds,ii_inds))
        j_inds = np.concatenate((j_inds,jj_inds))
    
    return d, i_inds, j_inds


def conditional_xy_z_pair_matrix(data1, data2, rp_max, pi_max, weights1, weights2,
                     cond_func_id, period=None, verbose=False,\
                     num_threads=1, approx_cell1_size = None, approx_cell2_size = None):
    """
    Calculate the distance to all pairs with perpendicular seperations less than or 
    equal to ``rp_max`` and parallel seperations ``pi_max`` in redshift space  that 
    pass a user specified condition.
    
    Parameters
    ----------
    data1 : array_like
        N1 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period.
            
    data2 : array_like
        N2 by 3 numpy array of 3-dimensional positions. Should be between zero and 
        period.
            
    rp_max : float
        maximum distance to connect pairs
    
    pi_max : float
        maximum distance to connect pairs
    
    weights1 : array_like, optional
        Either a 1-D array of length *N1*, or a 2-D array of length *N1* x *N_weights*, 
        containing the weights used for the conditional counts. If this parameter is
        None, the weights are set to np.ones(*(N1,N_weights)*).
    
    weights2 : array_like, optional
        Either a 1-D array of length *N1*, or a 2-D array of length *N1* x *N_weights*, 
        containing the weights used for the conditional counts. If this parameter is
        None, the weights are set to np.ones(*(N1,N_weights)*).
    
    cond_func_id : int, optional
        conditonal function integer ID. Each conditional function requires a specific 
        number of weights per point, *N_weights*.  See the Notes for a description of
        available functions.
    
    period : array_like, optional
        length 3 array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*3).
        If none, PBCs are set to infinity.  If True, period is set to be Lbox
    
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
    perp_dists : `~scipy.sparse.coo_matrix`
        N1 x N2 sparse matrix in COO format containing perpendicular distances between points.
    
    para_dists : `~scipy.sparse.coo_matrix`
        N1 x N2 sparse matrix in COO format containing parallel distances between points.
    
    Notes
    -----
    The distances between all points with seperations that meet the secified conditions 
    are stored and returned.  If there are many points and/or ``rp_max`` and ``pi_max`` 
    are large, this can become very memmory intensive.
    
    Points with zero seperation are considered a self-match, and do no count as neighbors.
    
    There are multiple conditonal functions available.  In general, each requires a 
    different number of marks per point, N_marks.  The conditonal function gets passed 
    two vectors per pair, w1 and w2, of length N_marks and return a float.  
    
    A pair pair is counted as a neighbor if the conditonal function evaulates as True.
    
    The available marking functions, ``cond_func`` and the associated integer 
    ID numbers are:
    
    #. greater than (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] > w_2[0] \\\\
                    False & : w_1[0] \\geq w_2[0] \\\\
                \\end{array}
                \\right.
    
    #. less than (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] < w_2[0] \\\\
                    False & : w_1[0] \\leq w_2[0] \\\\
                \\end{array}
                \\right.
    
    #. equality (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] = w_2[0] \\\\
                    False & : w_1[0] \\neq w_2[0] \\\\
                \\end{array}
                \\right.
    
    #. inequality (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] \\neq w_2[0] \\\\
                    False & : w_1[0] = w_2[0] \\\\
                \\end{array}
                \\right.
    
    #. tolerance greater than (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] > (w_2[0]+w_1[1]) \\\\
                    False & : w_1[0] \\leq (w_2[0]+w_1[1]) \\\\
                \\end{array}
                \\right.
    
    #. tolerance less than (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] < (w_2[0]+w_1[1]) \\\\
                    False & : w_1[0] \\geq (w_2[0]+w_1[1]) \\\\
                \\end{array}
                \\right.
    
    
    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a 
    periodic unit cube. 
    
    >>> Npts = 1000
    >>> Lbox = 1.0
    >>> period = np.array([Lbox,Lbox,Lbox])
    
    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)
    
    We transform our *x, y, z* points into the array shape used by the pair-counter by 
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation 
    is used throughout the `~halotools.mock_observables` sub-package:
    
    >>> coords = np.vstack((x,y,z)).T
    
    Create random weights:
    
    >>> weights = np.random.random(Npts)
    
    Now, we can find the distance between all points:
    
    >>> rp_max = 0.1
    >>> pi_max = 0.2
    >>> cond_func = 1
    >>> d_perp, d_para = conditional_xy_z_pair_matrix(coords, coords, rp_max, pi_max, weights, weights, cond_func, period=period)
    
    The diagonal of this matrix will be zeros, the distance between each point and itself.
    The off diagonal elements are the pairwise distances between points i,j in the order 
    they appear in `coords`.  In this case, the matrix will be symmetric.
    """
    
    search_dim_max = np.array([rp_max, rp_max, pi_max])
    function_args = [data1, data2, period, num_threads, search_dim_max]
    x1, y1, z1, x2, y2, z2, period, num_threads, PBCs = _process_args(*function_args)
    #note that process_args sets period equal to Lbox is there are no PBCs
    xperiod, yperiod, zperiod = period 
    rp_max = float(rp_max)
    pi_max = float(pi_max)
    
    weights1, weights2 = _process_weights(weights1, weights2, cond_func_id, data1, data2)
    
    if verbose==True:
        print("running on {0} x {1}\n"
              "points with PBCs={2}".format(len(data1), len(data2), PBCs))
        start = time.time()
    
    approx_cell1_size, approx_cell2_size = (
        _set_approximate_xy_z_cell_sizes(approx_cell1_size, approx_cell2_size, rp_max, pi_max, period)
        )
    approx_x1cell_size, approx_y1cell_size, approx_z1cell_size = approx_cell1_size
    approx_x2cell_size, approx_y2cell_size, approx_z2cell_size = approx_cell2_size
    
    double_tree = FlatRectanguloidDoubleTree(x1, y1, z1, x2, y2, z2,
                      approx_x1cell_size, approx_y1cell_size, approx_z1cell_size, 
                      approx_x2cell_size, approx_y2cell_size, approx_z2cell_size, 
                      rp_max, rp_max, pi_max, xperiod, yperiod,zperiod, PBCs=PBCs)
    
    #number of cells
    Ncell1 = double_tree.num_x1divs*double_tree.num_y1divs*double_tree.num_z1divs
    Ncell2 = double_tree.num_x2divs*double_tree.num_y2divs*double_tree.num_z2divs
    
    if verbose==True:
        print("volume 1 split {0},{1},{2} times along each dimension,\n"
              "resulting in {3} cells.".format(double_tree.num_x1divs,\
              double_tree.num_y1divs,double_tree.num_z1divs,Ncell1))
        print("volume 2 split {0},{1},{2} times along each dimension,\n"
              "resulting in {3} cells.".format(double_tree.num_x2divs,\
              double_tree.num_y2divs,double_tree.num_z2divs,Ncell2))
    
    #create a function to call with only one argument
    engine = partial(_conditional_xy_z_pair_matrix_engine, double_tree, weights1, weights2, rp_max, pi_max, period, PBCs, cond_func_id)
    
    #do the pair counting
    if num_threads>1:
        pool = multiprocessing.Pool(num_threads)
        result = pool.map(engine,range(Ncell1))
        pool.close()
    if num_threads==1:
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
    i_inds = double_tree.tree1.idx_sorted[i_inds]
    j_inds = double_tree.tree2.idx_sorted[j_inds]
    
    if verbose==True:
        print("total run time: {0} seconds".format(time.time()-start))
    
    return coo_matrix((d_perp, (i_inds, j_inds)), shape=(len(data1),len(data2))),\
           coo_matrix((d_para, (i_inds, j_inds)), shape=(len(data1),len(data2)))


def _conditional_xy_z_pair_matrix_engine(double_tree, weights1, weights2, rp_max, pi_max, period, PBCs, cond_func_id, icell1):
    """
    pair counting engine for xy_z_fof_npairs function.  This code calls a cython function.
    """
    
    d_perp = np.zeros((0,), dtype='float')
    d_para = np.zeros((0,), dtype='float')
    i_inds = np.zeros((0,), dtype='int')
    j_inds = np.zeros((0,), dtype='int')
    
    #extract the points in the cell
    s1 = double_tree.tree1.slice_array[icell1]
    x_icell1, y_icell1, z_icell1 = (
        double_tree.tree1.x[s1],
        double_tree.tree1.y[s1],
        double_tree.tree1.z[s1])
    
    #extract the weights in the cell
    w_icell1 = weights1[s1, :]
    
    i_min = s1.start
    
    xsearch_length = rp_max
    ysearch_length = rp_max
    zsearch_length = pi_max
    adj_cell_generator = double_tree.adjacent_cell_generator(
        icell1, xsearch_length, ysearch_length, zsearch_length)
    
    adj_cell_counter = 0
    for icell2, xshift, yshift, zshift in adj_cell_generator:
        adj_cell_counter +=1
        
        #extract the points in the cell
        s2 = double_tree.tree2.slice_array[icell2]
        x_icell2 = double_tree.tree2.x[s2] + xshift
        y_icell2 = double_tree.tree2.y[s2] + yshift 
        z_icell2 = double_tree.tree2.z[s2] + zshift
        
        #extract the weights in the cell
        w_icell2 = weights2[s2, :]
        
        j_min = s2.start
        
        dd_perp, dd_para, ii_inds, jj_inds = conditional_pairwise_xy_z_distance_no_pbc(\
                                                 x_icell1, y_icell1, z_icell1,\
                                                 x_icell2, y_icell2, z_icell2,\
                                                 rp_max, pi_max,
                                                 w_icell1, w_icell2, cond_func_id)
        
        ii_inds = ii_inds+i_min
        jj_inds = jj_inds+j_min
        
        #update storage arrays
        d_perp = np.concatenate((d_perp,dd_perp))
        d_para = np.concatenate((d_para,dd_para))
        i_inds = np.concatenate((i_inds,ii_inds))
        j_inds = np.concatenate((j_inds,jj_inds))
    
    return d_perp, d_para, i_inds, j_inds


def _process_args(data1, data2, period, num_threads, search_dim_max):
    """
    private internal function to process the arguments of the pair matrix functions.
    """
    
    if num_threads is not 1:
        if num_threads=='max':
            num_threads = multiprocessing.cpu_count()
        if not isinstance(num_threads,int):
            msg = ("\n Input ``num_threads`` argument must \n"
                   "be an integer or the string 'max'")
            raise HalotoolsError(msg)
    
    # Passively enforce that we are working with ndarrays
    x1 = data1[:,0]
    y1 = data1[:,1]
    z1 = data1[:,2]
    x2 = data2[:,0]
    y2 = data2[:,1]
    z2 = data2[:,2]

    # Set the boolean value for the PBCs variable
    if period is None:
        PBCs = False
        x1, y1, z1, x2, y2, z2, period = (
            _enclose_in_box(x1, y1, z1, x2, y2, z2, min_size=search_dim_max*3.0))
    else:
        PBCs = True
        period = convert_to_ndarray(period).astype(float)
        if len(period) == 1:
            period = np.array([period[0]]*3)
        try:
            assert np.all(period < np.inf)
            assert np.all(period > 0)
        except AssertionError:
            msg = "Input ``period`` must be a bounded positive number in all dimensions"
            raise HalotoolsError(msg)
    
    return x1, y1, z1, x2, y2, z2, period, num_threads, PBCs


def _process_weights(weights1, weights2, cond_func, data1, data2):
    """
    private internal function to process the weights
    """
    
    correct_num_weights = _func_signature_int_from_cond_func(cond_func)
    npts_data1 = np.shape(data1)[0]
    npts_data2 = np.shape(data2)[0]
    correct_shape1 = (npts_data1, correct_num_weights)
    correct_shape2 = (npts_data2, correct_num_weights)
    
    ### Process the input weights1
    _converted_to_2d_from_1d = False
    # First convert weights1 into a 2-d ndarray
    if weights1 is None:
        weights1 = np.ones((npts_data1, 1), dtype = np.float64)
    else:
        weights1 = convert_to_ndarray(weights1)
        weights1 = weights1.astype("float64")
        if weights1.ndim == 1:
            _converted_to_2d_from_1d = True
            npts1 = len(weights1)
            weights1 = weights1.reshape((npts1, 1))
        elif weights1.ndim == 2:
            pass
        else:
            ndim1 = weights1.ndim
            msg = ("\n You must either pass in a 1-D or 2-D array \n"
                   "for the input `weights1`. Instead, an array of \n"
                   "dimension %i was received.")
            raise HalotoolsError(msg % ndim1)
    
    npts_weights1 = np.shape(weights1)[0]
    num_weights1 = np.shape(weights1)[1]
    # At this point, weights1 is guaranteed to be a 2-d ndarray
    ### now we check its shape
    if np.shape(weights1) != correct_shape1:
        if _converted_to_2d_from_1d is True:
            msg = ("\n You passed in a 1-D array for `weights1` that \n"
                   "does not have the correct length. The number of \n"
                   "points in `data1` = %i, while the number of points \n"
                   "in your input 1-D `weights1` array = %i")
            raise HalotoolsError(msg % (npts_data1, npts_weights1))
        else:
            msg = ("\n You passed in a 2-D array for `weights1` that \n"
                   "does not have a consistent shape with `data1`. \n"
                   "`data1` has length %i. The input value of `cond_func` = %i \n"
                   "For this value of `cond_func`, there should be %i weights \n"
                   "per point. The shape of your input `weights1` is (%i, %i)\n")
            raise HalotoolsError(msg % 
                (npts_data1, cond_func, correct_num_weights, npts_weights1, num_weights1))
    
    ### Process the input weights2
    _converted_to_2d_from_1d = False
    # Now convert weights2 into a 2-d ndarray
    if weights2 is None:
        weights2 = np.ones((npts_data2, 1), dtype = np.float64)
    else:
        weights2 = convert_to_ndarray(weights2)
        weights2 = weights2.astype("float64")
        if weights2.ndim == 1:
            _converted_to_2d_from_1d = True
            npts2 = len(weights2)
            weights2 = weights2.reshape((npts2, 1))
        elif weights2.ndim == 2:
            pass
        else:
            ndim2 = weights2.ndim
            msg = ("\n You must either pass in a 1-D or 2-D array \n"
                   "for the input `weights2`. Instead, an array of \n"
                   "dimension %i was received.")
            raise HalotoolsError(msg % ndim2)
    
    npts_weights2 = np.shape(weights2)[0]
    num_weights2 = np.shape(weights2)[1]
    # At this point, weights2 is guaranteed to be a 2-d ndarray
    ### now we check its shape
    if np.shape(weights2) != correct_shape2:
        if _converted_to_2d_from_1d is True:
            msg = ("\n You passed in a 1-D array for `weights2` that \n"
                   "does not have the correct length. The number of \n"
                   "points in `data2` = %i, while the number of points \n"
                   "in your input 1-D `weights2` array = %i")
            raise HalotoolsError(msg % (npts_data2, npts_weights2))
        else:
            msg = ("\n You passed in a 2-D array for `weights2` that \n"
                   "does not have a consistent shape with `data2`. \n"
                   "`data2` has length %i. The input value of `cond_func` = %i \n"
                   "For this value of `cond_func_id`, there should be %i weights \n"
                   "per point. The shape of your input `weights2` is (%i, %i)\n")
            raise HalotoolsError(msg % 
                (npts_data2, cond_func, correct_num_weights, npts_weights2, num_weights2))
    
    return weights1, weights2


def _func_signature_int_from_cond_func(cond_func):
    """
    return the function signiture available conditional functions
    """
    
    if type(cond_func) != int:
        msg = "\n cond_func parameter must be an integer ID of a conditional function."
        raise HalotoolsError(msg)
    
    if cond_func == 1:
        return 1
    elif cond_func == 2:
        return 1
    elif cond_func == 3:
        return 1
    elif cond_func == 4:
        return 1
    elif cond_func == 5:
        return 2
    elif cond_func == 6:
        return 2
    else:
        msg = ("The value ``cond_func`` = %i is not recognized")
        raise HalotoolsError(msg % cond_func)

