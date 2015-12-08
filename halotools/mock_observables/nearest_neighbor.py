# -*- coding: utf-8 -*-

"""
find the nerest neighbor to a point under various definitions
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
####import modules########################################################################
import numpy as np
from .pair_counters.double_tree_pair_matrix import pair_matrix, xy_z_pair_matrix
from warnings import warn
import sys
import time
##########################################################################################


__all__=['nearest_neighbor']
__author__ = ['Duncan Campbell']


np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero in e.g. DD/RR

def nearest_neighbor(sample1, sample2, r_max, period=None, nth_nearest=1,
                     num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    """
    Find the nearest neighbor between two sets of points.
    
    Parameters
    ----------
    sample1 : array_like
        Npts x 3 numpy array containing 3-D positions of points.
    
    sample2 : array_like
        Npts x 3 numpy array containing 3-D positions of points.
    
    r_max : float
        maximum search distance
    
    period : array_like, optional
        length 3 array defining axis-aligned periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*3).
    
    nth_nearest : int
        intger indicating the nth nearest nieghbor with non-zero seperation.
        If less than 0, the absolute nearest neighbor is returned. If there are multiple 
        nth nearest neighbors with the same seperation, the results may not be relaible
        or desired.  If you are looking for the nearest neighbor when `sample1` and 
        `sample2` are the same, but want to exclude the trivial self match, set 
        nth_nearest = 2
    
    num_threads : int, optional
        number of 'threads' to use in the pair counting.  if set to 'max', use all 
        available cores.  num_threads=0 is the default.
    
    approx_cell1_size : array_like, optional 
        Length-3 array serving as a guess for the optimal manner by which 
        the `~halotools.mock_observables.pair_counters.FlatRectanguloidDoubleTree` 
        will apportion the ``sample1`` points into subvolumes of the simulation box. 
        The optimum choice unavoidably depends on the specs of your machine. 
        Default choice is to use *max(rbins)* in each dimension, 
        which will return reasonable result performance for most use-cases. 
        Performance can vary sensitively with this parameter, so it is highly 
        recommended that you experiment with this parameter when carrying out  
        performance-critical calculations. 
    
    approx_cell2_size : array_like, optional 
        Analogous to ``approx_cell1_size``, but for sample2.  See comments for 
        ``approx_cell1_size`` for details.
    
    Returns
    -------
    nearest_nieghbor : numpy.array
        *len(sample1)* array of integers indicating the index of the nearest neighbor in 
        `sample2`.  If no nieghbor was found, set to -1.
    
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
    
    Find the nearest non-self neighbor to each point out to a maximum of seperaiton of 0.1
    
    >>> r_max = 0.05
    >>> matched_inds = nearest_neighbor(coords, coords, r_max, period=period, nth_nearest=1)
    >>> print(matched_inds)
    """
    
    r_max = float(r_max)
    approx_cell1_size = [r_max*4]*3
    approx_cell2_size = [r_max*4]*3
    #print(approx_cell1_size)
    
    start = time.time()
    distance_matrix = pair_matrix(sample1, sample2, r_max, period=period,
                                  approx_cell1_size = approx_cell1_size,
                                  approx_cell2_size = approx_cell1_size,
                                  num_threads = num_threads).tocsc()
    
    distance_matrix.eliminate_zeros()
    inds = sparse_argmin(distance_matrix, axis=0)
    print(inds)
    return inds
    
    """
    print(time.time()-start)
    #create array to store the result
    N1 = len(sample1)
    N2 = len(sample2)
    min_dist_ind = np.zeros(N1)-1
    
    start = time.time()
    i_rows = np.arange(N1)
    for i_row in i_rows:
        min_dist_ind[i_row] = _min_sparse_row(distance_matrix, nth_nearest, i_row)
    print(time.time()-start)
    
    return min_dist_ind
    """

def nth_minimum(m, n, axis=None):
    """
    Find the nth minimum element of a matrix
    
    Parameters
    ----------
    m : matrix
    
    n : integer
    
    """
    
    N0, N1 = m.shape
    
    i,j = m.nonzero()
    d = m[i,j]
    
    if axis=None:
        sort_inds = np.argsort(d)
        sorted_i = i[sort_inds]
        sorted_j = j[sort_inds]
        return sorted_i[0], sorted_j[0]
    elif asix==0:
        sort_inds = np.lexsort((i,d))
        sorted_i = i[sort_inds]
        sorted_j = j[sort_inds]
        sorted_d = d[sort_inds]
        dummy, first_i = np.unique(sorted_i,return_index=True)
        desired_i = first_i+(n-1)
        has_a_match = (sorted_i[desired_i]==sorted_i)
        result[has_a_match] = sorted_j[has_a_match]
    elif asix==1:
        sort_inds = np.lexsort((j,d))
        sorted_i = i[sort_inds]
        sorted_j = j[sort_inds]
        sorted_d = d[sort_inds]
        dummy, first_j = np.unique(sorted_j,return_index=True)
        desired_j = first_j+(n-1)
        has_a_match = (sorted_j[desired_j]==sorted_j)
        result[has_a_match] = sorted_i[has_a_match]
    
    return result
