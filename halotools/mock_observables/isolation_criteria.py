# -*- coding: utf-8 -*-

"""
detemrine whether a set of points is isolated according to various criteria
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
####import modules########################################################################
import numpy as np
from .pair_counters.double_tree_pair_matrix import pair_matrix, xy_z_pair_matrix
from warnings import warn
##########################################################################################


__all__=['spherical_isolation', 'cylindrical_isolation']
__author__ = ['Duncan Campbell']


np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero in e.g. DD/RR


def spherical_isolation(sample1, r_max, period=None, num_threads=1,
                        approx_cell1_size=None, approx_cell2_size=None):
    """
    detemrine whether a set of points has any neighbor within a spherical volume
    
    Parameters
    ----------
    sample1 : array_like
        Npts x 3 numpy array containing 3-D positions of points.
    
    r_max : float
        size of sphere to search for neighbors
    
    period : array_like, optional
        length 3 array defining axis-aligned periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*3).
    
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
    has_neighbor : numpy.array
        array of booleans indicating if the point as a neighbor.
    
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
    
    >>> r_max = 0.05
    >>> is_isolated = spherical_isolation(coords, r_max, period=period)
    """
    
    
    distance_matrix = pair_matrix(sample1, sample1, r_max, period=period,
                                  approx_cell1_size = approx_cell1_size,
                                  approx_cell2_size = approx_cell1_size)
    
    i ,j = distance_matrix.nonzero()
    
    #self matches should have a distance of zero, but just in case...
    is_not_self_match = (i!=j)
    i = i[is_not_self_match]
    
    #compare list of indices vs. possible indices
    N = len(sample1)
    inds = np.arange(0,N)
    
    is_not_isolated = np.in1d(inds, i)
    is_isolated = (is_not_isolated == False)
    
    return is_isolated


def cylindrical_isolation(sample1, rp_max, pi_max, period=None, num_threads=1,
                          approx_cell1_size=None, approx_cell2_size=None):
    """
    detemrine whether a set of points has any neighbor within a cylinderical volume
    
    Parameters
    ----------
    sample1 : array_like
        Npts x 3 numpy array containing 3-D positions of points.
    
    rp_max : float
        radius of the cylinder to seach for neighbors
    
    pi_max : float
        half the legnth of the cylinder to seach for neighbors
    
    period : array_like, optional
        length 3 array defining axis-aligned periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*3).
    
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
    has_neighbor : numpy.array
        array of booleans indicating if the point as a neighbor.
    
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
    
    >>> rp_max = 0.05
    >>> pi_max = 0.1
    >>> is_isolated = cylindrical_isolation(coords, rp_max, pi_max, period=period)
    """
    
    perp_distance_matrix, para_distance_matrix = \
        xy_z_pair_matrix(sample1, sample1, rp_max, pi_max, period=period, 
                         approx_cell1_size = approx_cell1_size,
                         approx_cell2_size = approx_cell1_size)
    
    distance_matrix = np.sqrt(perp_distance_matrix**2 + para_distance_matrix**2)
    
    i ,j = distance_matrix.nonzero()
    
    #self matches should have a distance of zero, but just in case...
    is_not_self_match = (i!=j)
    i = i[is_not_self_match]
    
    #compare list of indices vs. possible indices
    N = len(sample1)
    inds = np.arange(0,N)
    
    is_not_isolated = np.in1d(inds, i)
    is_isolated = (is_not_isolated == False)
    
    return is_isolated

