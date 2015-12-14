# -*- coding: utf-8 -*-

"""
detemrine whether a set of points is isolated according to various criteria
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
####import modules########################################################################
import numpy as np
from .pair_counters.double_tree_pair_matrix import *
from warnings import warn
##########################################################################################


__all__=['spherical_isolation', 'cylindrical_isolation',\
         'conditional_spherical_isolation','conditional_cylindrical_isolation']
__author__ = ['Duncan Campbell']


np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero in e.g. DD/RR


def spherical_isolation(sample1, sample2, r_max, period=None,
                        num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    """
    Determine whether a set of points, ``sample1``, has a neighbor in ``sample2`` within 
    a spherical volume.
    
    Parameters
    ----------
    sample1 : array_like
        N1pts x 3 numpy array containing 3-D positions of points.
    
    sample2 : array_like
        N2pts x 3 numpy array containing 3-D positions of points.
    
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
    
    Notes
    -----
    Points with zero seperation are considered a self-match, and do no count as neighbors.
    
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
    >>> is_isolated = spherical_isolation(coords, coords, r_max, period=period)
    """
    
    
    distance_matrix = pair_matrix(sample1, sample2, r_max, period=period,
                                  approx_cell1_size = approx_cell1_size,
                                  approx_cell2_size = approx_cell1_size)
    
    i ,j = distance_matrix.nonzero()
    
    #compare list of indices vs. possible indices
    N = len(sample1)
    inds = np.arange(0,N)
    
    is_isolated = np.in1d(inds, i, invert=True)
    
    return is_isolated


def cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=None, num_threads=1,
                          approx_cell1_size=None, approx_cell2_size=None):
    """
    Determine whether a set of points, ``sample1``, has a neighbor in ``sample2`` within 
    a cylinderical volume
    
    Parameters
    ----------
    sample1 : array_like
        N1pts x 3 numpy array containing 3-D positions of points.
    
    sample2 : array_like
        N2pts x 3 numpy array containing 3-D positions of points.
    
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
    
    Notes
    -----
    Points with zero seperation are considered a self-match, and do no count as neighbors.
    
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
    >>> is_isolated = cylindrical_isolation(coords, coords, rp_max, pi_max, period=period)
    """
    
    perp_distance_matrix, para_distance_matrix = \
        xy_z_pair_matrix(sample1, sample1, rp_max, pi_max, period=period, 
                         approx_cell1_size = approx_cell1_size,
                         approx_cell2_size = approx_cell1_size)
    
    distance_matrix = np.sqrt(perp_distance_matrix**2 + para_distance_matrix**2)
    
    i ,j = distance_matrix.nonzero()
    
    #compare list of indices vs. possible indices
    N = len(sample1)
    inds = np.arange(0,N)
    
    is_isolated = np.in1d(inds, i, invert=True)
    
    return is_isolated


def conditional_spherical_isolation(sample1, sample2, r_max,
                        marks1, marks2, cond_func, period=None,
                        num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    """
    Determine whether a set of points, ``sample1``, has a neighbor in ``sample2`` within 
    a spherical volume that satisfies a user specified condition.
    
    Parameters
    ----------
    sample1 : array_like
        N1pts x 3 numpy array containing 3-D positions of points.
    
    sample2 : array_like
        N2pts x 3 numpy array containing 3-D positions of points.
    
    r_max : float
        size of sphere to search for neighbors
    
    marks1 : array_like
        len(sample1) x N_marks array of marks.  The suplied marks array must have the 
        appropiate shape for the chosen ``cond_func`` (see Notes for requirements).  If 
        this parameter is not specified, it is set 
        to numpy.ones((len(sample1), N_marks)).
    
    marks2 : array_like
        len(sample2) x N_marks array of marks.  The suplied marks array must have the 
        appropiate shape for the chosen ``cond_func`` (see Notes for requirements).  If 
        this parameter is not specified, it is set 
        to numpy.ones((len(sample1), N_marks)).
    
    cond_func : int
        Integer ID indicating which conditional function should be used.  See notes for a 
        list of available conditional functions.
    
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
    
    Notes
    -----
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
    
    >>> marks = np.random.random(Npts)
    
    >>> r_max = 0.05
    >>> cond_func = 1
    >>> is_isolated = conditional_spherical_isolation(coords, coords, r_max, marks, marks, cond_func, period=period)
    """
    
    
    distance_matrix = conditional_pair_matrix(sample1, sample2, r_max,
                                              marks1, marks2, cond_func, period=period,
                                              approx_cell1_size = approx_cell1_size,
                                              approx_cell2_size = approx_cell1_size)
    
    i ,j = distance_matrix.nonzero()
    
    #compare list of indices vs. possible indices
    N = len(sample1)
    inds = np.arange(0,N)
    
    is_isolated = np.in1d(inds, i, invert=True)
    
    return is_isolated


def conditional_cylindrical_isolation(sample1, sample2, rp_max, pi_max,
                          marks1, marks2, cond_func, period=None, num_threads=1,
                          approx_cell1_size=None, approx_cell2_size=None):
    """
    Determine whether a set of points, ``sample1``, has a neighbor in ``sample2`` 
    within a cylinderical volume that satisfies a user specified condition.
    
    Parameters
    ----------
    sample1 : array_like
        N1pts x 3 numpy array containing 3-D positions of points.
    
    sample2 : array_like
        N2pts x 3 numpy array containing 3-D positions of points.
    
    rp_max : float
        radius of the cylinder to seach for neighbors
    
    pi_max : float
        half the legnth of the cylinder to seach for neighbors
    
    marks1 : array_like
        len(sample1) x N_marks array of marks.  The suplied marks array must have the 
        appropiate shape for the chosen ``cond_func`` (see Notes for requirements).  If 
        this parameter is not specified, it is set 
        to numpy.ones((len(sample1), N_marks)).
    
    marks2 : array_like
        len(sample2) x N_marks array of marks.  The suplied marks array must have the 
        appropiate shape for the chosen ``cond_func`` (see Notes for requirements).  If 
        this parameter is not specified, it is set 
        to numpy.ones((len(sample1), N_marks)).
    
    cond_func : int
        Integer ID indicating which conditional function should be used.  See notes for a 
        list of available conditional functions.
    
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
    
    Notes
    -----
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
    
    >>> marks = np.random.random(Npts)
    
    >>> rp_max = 0.05
    >>> pi_max = 0.1
    >>> cond_func = 1
    >>> is_isolated = conditional_cylindrical_isolation(coords, coords, rp_max, pi_max, marks, marks, cond_func, period=period)
    """
    
    perp_distance_matrix, para_distance_matrix = \
        conditional_xy_z_pair_matrix(sample1, sample1, rp_max, pi_max,
                         marks1, marks2, cond_func, 
                         period = period,
                         approx_cell1_size = approx_cell1_size,
                         approx_cell2_size = approx_cell1_size)
    
    distance_matrix = np.sqrt(perp_distance_matrix**2 + para_distance_matrix**2)
    
    i ,j = distance_matrix.nonzero()
    
    #compare list of indices vs. possible indices
    N = len(sample1)
    inds = np.arange(0,N)
    
    is_isolated = np.in1d(inds, i, invert=True)
    
    return is_isolated
    
    
