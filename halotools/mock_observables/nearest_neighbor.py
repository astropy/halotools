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
from ..custom_exceptions import *
from scipy import sparse
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
        intger indicating the nth nearest nieghbor for which to search.
        If the distance between points is 0.0, it is not counted as a match.
        Results are not unique when there are multiple nth nearest neighbors with the 
        same seperation.
    
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
        ``sample2``.  If no nieghbor was found, set to -1.
    
    Notes
    -----
    pairwise distances are calculated using 
    `~halotools.mock_observables.pair_counters.pair_matrix`.  The distance to all pairs
    with seperations less than or equal to ``r_max`` are found and stored in memory.  
    Therefore, if ``r_max`` and/or the number of points in ``sample1`` and ``sample2`` is (are) 
    large, this process can become slow and use copious amounts of memory.
    
    If you need to search many pairs out to large ``r_max``, consider running this code 
    iteratively, finding the nearest neighbor for pairs with smaller seperations first.
    
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
    
    >>> r_max = 0.1
    >>> matched_inds = nearest_neighbor(coords, coords, r_max, period=period, nth_nearest=2)
    """
    
    r_max = float(r_max)
    
    distance_matrix = pair_matrix(sample1, sample2, r_max, period=period,
                                  approx_cell1_size = approx_cell1_size,
                                  approx_cell2_size = approx_cell2_size,
                                  num_threads = num_threads)
    
    result = _nth_matrix_minimum(distance_matrix, nth_nearest, axis=0)
    
    return result


def _nth_matrix_minimum(m, n, axis=None):
    """
    Find the nth minimum element of a sparse matrix.
    
    Parameters
    ----------
    m : scipy.sparse.matrix
        pairwise distance matrix
    
    n : integer
        search for nth smallest value--must be >=1.
    
    axis : integer, optional
        specified axis to preform calculation along.  If 'None' is recevied, the gloabal 
        nth minimum is found.
    
    Notes
    -----
    Only explicitly specified values are searched--by deinfition, most values in a sparse
    matrix are 0, but only values that have explicitly been set to 0.0 are stored, and 
    are therefore searched when caclulating minmimums.  scipy.sparse provides methods
    to search for mimmums including all zeros.
    
    """
    
    #check input
    if not sparse.isspmatrix(m):
        msg = ("\n `m` must be a sparse matrix.")
        raise ValueError(msg)
    if type(n) is not int:
        msg = ("\n `n` must an integer >=1.")
        raise ValueError(msg)
    if n<1:
        msg = ("\n `n` must an integer >=1.")
        raise ValueError(msg)
    if axis not in [0,1,None]:
        msg = ("\n `axis` parameter must 0, 1, or None.")
        raise ValueError(msg)
    
    N0, N1 = m.shape
    
    if not (sparse.isspmatrix_csc(m) or sparse.isspmatrix_csr(m)):
        if axis==0: m = m.tocsr()
        elif axis==1: m = m.tocsc()
    
    i,j = m.nonzero()
    d = np.array(m[i,j]).flatten()
    
    if axis is None:
        sort_inds = np.argsort(d)
        sorted_i = i[sort_inds]
        sorted_j = j[sort_inds]
        return sorted_i[0], sorted_j[0]
    elif axis==0:
        #create array to store result
        result = np.zeros(N0, dtype=int)-1
        #sort by distance and row index
        sort_inds = np.lexsort((d,i))
        sorted_i = i[sort_inds]
        sorted_j = j[sort_inds]
        sorted_d = d[sort_inds]
        #find the index each row index first appears
        unique_is, first_i = np.unique(sorted_i,return_index=True)
        #the first place it appears + n-1 will be the nth place it appears
        nth_i = first_i+(n-1)
        #check to see if adding n-1 takes us out of bounds
        out_of_bounds = (nth_i >= len(sorted_i))
        nth_i[out_of_bounds] = 0
        #check to see if the first place + n-1 is still the same index
        nth_i_matches_first = (sorted_i[nth_i]==sorted_i[first_i])
        nth_i_matches_first[out_of_bounds] = False
        #if so, return the index of the corresponding column j
        result[unique_is[nth_i_matches_first]] = sorted_j[nth_i[nth_i_matches_first]]
    elif axis==1:
        result = np.zeros(N1, dtype=int)-1
        sort_inds = np.lexsort((d,j))
        sorted_i = i[sort_inds]
        sorted_j = j[sort_inds]
        sorted_d = d[sort_inds]
        unique_js, first_j = np.unique(sorted_j,return_index=True)
        nth_j = first_j+(n-1)
        out_of_bounds = (nth_j >= len(sorted_j))
        nth_j[out_of_bounds] = 0
        nth_j_matches_first = (sorted_j[nth_j]==sorted_j[first_j])
        nth_j_matches_first[out_of_bounds] = False
        result[unique_js[nth_j_matches_first]] = sorted_i[nth_j[nth_j_matches_first]]
    else:
        msg = ("\n axis must be None, 0, or 1.")
        raise HalotoolsError(msg)
    
    return result
