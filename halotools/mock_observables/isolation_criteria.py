# -*- coding: utf-8 -*-
"""
Functions used to determine whether 
a set of points is isolated according to various criteria. 
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from functools import partial 
import multiprocessing 

from .pair_counters.rectangular_mesh import RectangularDoubleMesh
from .pair_counters.double_tree_pair_matrix import pair_matrix, xy_z_pair_matrix
from .pair_counters.double_tree_pair_matrix import conditional_pair_matrix
from .pair_counters.double_tree_pair_matrix import conditional_xy_z_pair_matrix
from .pair_counters.cpairs import spherical_isolation_engine
from .pair_counters.mesh_helpers import (
    _set_approximate_cell_sizes, _cell1_parallelization_indices, _enclose_in_box)

from .pair_counters.marked_npairs_3d import _marked_npairs_process_weights
from .pair_counters.marked_cpairs import marked_spherical_isolation_engine

from ..utils.array_utils import convert_to_ndarray, custom_len

__all__ = ('spherical_isolation', 'cylindrical_isolation',
    'conditional_spherical_isolation','conditional_cylindrical_isolation')
__author__ = ['Duncan Campbell']


np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero in e.g. DD/RR


def spherical_isolation(sample1, sample2, r_max, period=None,
                        num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    """
    Determine whether a set of points, ``sample1``, has a neighbor in ``sample2`` within 
    an input spherical volume centered at each point in ``sample1``.

    See the :ref:`mock_obs_pos_formatting` documentation page for 
    instructions on how to transform your coordinate position arrays into the 
    format accepted by the ``sample1`` and ``sample2`` arguments.   

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

def overhauled_spherical_isolation(sample1, sample2, r_max, period=None,
    num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    """
    Determine whether a set of points, ``sample1``, has a neighbor in ``sample2`` within 
    an input spherical volume centered at each point in ``sample1``.

    See the :ref:`mock_obs_pos_formatting` documentation page for 
    instructions on how to transform your coordinate position arrays into the 
    format accepted by the ``sample1`` and ``sample2`` arguments.   

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
    >>> is_isolated = overhauled_spherical_isolation(coords, coords, r_max, period=period)
    """
    ### Process the inputs with the helper function
    result = _spherical_isolation_process_args(sample1, sample2, r_max, period,
            num_threads, approx_cell1_size, approx_cell2_size)
    x1in, y1in, z1in, x2in, y2in, z2in = result[0:6]
    r_max, period, num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[6:]
    xperiod, yperiod, zperiod = period 

    search_xlength, search_ylength, search_zlength = r_max, r_max, r_max 

    ### Compute the estimates for the cell sizes
    approx_cell1_size, approx_cell2_size = (
        _set_approximate_cell_sizes(approx_cell1_size, approx_cell2_size, period)
        )
    approx_x1cell_size, approx_y1cell_size, approx_z1cell_size = approx_cell1_size
    approx_x2cell_size, approx_y2cell_size, approx_z2cell_size = approx_cell2_size

    # Build the rectangular mesh
    double_mesh = RectangularDoubleMesh(x1in, y1in, z1in, x2in, y2in, z2in,
        approx_x1cell_size, approx_y1cell_size, approx_z1cell_size,
        approx_x2cell_size, approx_y2cell_size, approx_z2cell_size,
        search_xlength, search_ylength, search_zlength, xperiod, yperiod, zperiod, PBCs)

    # Create a function object that has a single argument, for parallelization purposes
    engine = partial(spherical_isolation_engine, 
        double_mesh, sample1[:,0], sample1[:,1], sample1[:,2], 
        sample2[:,0], sample2[:,1], sample2[:,2], r_max)

    # Calculate the cell1 indices that will be looped over by the engine
    num_threads, cell1_tuples = _cell1_parallelization_indices(
        double_mesh.mesh1.ncells, num_threads)

    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        result = pool.map(engine, cell1_tuples)
        counts = np.sum(np.array(result), axis=0)
        pool.close()
    else:
        counts = engine(cell1_tuples[0])

    return np.array(counts, dtype=bool)


def cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=None, num_threads=1,
                          approx_cell1_size=None, approx_cell2_size=None):
    """
    Determine whether a set of points, ``sample1``, has a neighbor in ``sample2`` within 
    a cylindrical volume
    
    Parameters
    ----------
    sample1 : array_like
        N1pts x 3 numpy array containing 3-D positions of points.
    
    sample2 : array_like
        N2pts x 3 numpy array containing 3-D positions of points.
    
    rp_max : float
        radius of the cylinder to seach for neighbors
    
    pi_max : float
        half the length of the cylinder to seach for neighbors
    
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
    Points with zero seperation are considered a self-match, and do not count as neighbors.
    
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
    
    perp_distance_matrix, para_distance_matrix = xy_z_pair_matrix(
        sample1, sample2, rp_max, pi_max, period=period, 
        approx_cell1_size = approx_cell1_size, approx_cell2_size = approx_cell1_size)
    
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
    Determine whether a set of points, ``sample1``, has a neighbor in ``sample2``, 
    where various additional conditions may be applied to judge whether a matching point 
    is considered to be a neighbor. For example, 
    `conditional_spherical_isolation` can be used to identify galaxies as isolated 
    if no other galaxy with a greater stellar mass lies within 500 kpc. 
    Different additional criteria can be built up from different 
    combinations of input ``marks`` and ``cond_func``. 
    See the Examples section for further details.  
    
    Parameters
    ----------
    sample1 : array_like
        N1pts x 3 numpy array containing 3-D positions of points.
    
    sample2 : array_like
        N2pts x 3 numpy array containing 3-D positions of points.
    
    r_max : float
        size of sphere to search for neighbors
    
    marks1 : array_like
        len(sample1) x N_marks array of marks.  The supplied marks array must have the 
        appropiate shape for the chosen ``cond_func`` (see Notes for requirements).  If 
        this parameter is not specified, it is set 
        to numpy.ones((len(sample1), N_marks)).
    
    marks2 : array_like
        len(sample2) x N_marks array of marks.  The supplied marks array must have the 
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
    Points with zero seperation are considered a self-match, and do not count as neighbors.
    
    There are multiple conditional functions available.  In general, each requires a 
    different number of marks per point, N_marks.  The conditional function gets passed 
    two arrays per pair, w1 and w2, of length N_marks and return a float.  
    You can pass in more than one piece of information about each point by choosing a 
    the input ``marks`` arrays to be multi-dimensional of shape (N_points, N_marks). 
    
    One point is considered to be a neighbor of another 
    if it lies within the enclosing sphere *and* 
    if the conditional function ``cond_func`` evaluates as True 
    when operating on the input ``marks`` data for that pair of points. 
    
    The available marking functions, ``cond_func`` and the associated integer 
    ID numbers are:
    
    #. greater than (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] > w_2[0] \\\\
                    False & : w_1[0] \\leq w_2[0] \\\\
                \\end{array}
                \\right.
    
    #. less than (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] < w_2[0] \\\\
                    False & : w_1[0] \\geq w_2[0] \\\\
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
    
    #. tolerance greater than (N_marks = 2)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] > (w_2[0]+w_1[1]) \\\\
                    False & : w_1[0] \\leq (w_2[0]+w_1[1]) \\\\
                \\end{array}
                \\right.
    
    #. tolerance less than (N_marks = 2)
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
    
    
    distance_matrix = conditional_pair_matrix(
        sample1, sample2, r_max, marks1, marks2, cond_func, period=period, 
        approx_cell1_size = approx_cell1_size, approx_cell2_size = approx_cell1_size)
    
    i ,j = distance_matrix.nonzero()
    
    #compare list of indices vs. possible indices
    N = len(sample1)
    inds = np.arange(0,N)
    
    is_isolated = np.in1d(inds, i, invert=True)
    
    return is_isolated

def overhauled_conditional_spherical_isolation(sample1, sample2, r_max,
    marks1, marks2, cond_func, period=None,
    num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    """
    Determine whether a set of points, ``sample1``, has a neighbor in ``sample2``, 
    where various additional conditions may be applied to judge whether a matching point 
    is considered to be a neighbor. For example, 
    `conditional_spherical_isolation` can be used to identify galaxies as isolated 
    if no other galaxy with a greater stellar mass lies within 500 kpc. 
    Different additional criteria can be built up from different 
    combinations of input ``marks`` and ``cond_func``. 
    See the Examples section for further details.  
    
    Parameters
    ----------
    sample1 : array_like
        N1pts x 3 numpy array containing 3-D positions of points.
    
    sample2 : array_like
        N2pts x 3 numpy array containing 3-D positions of points.
    
    r_max : float
        size of sphere to search for neighbors
    
    marks1 : array_like
        len(sample1) x N_marks array of marks.  The supplied marks array must have the 
        appropiate shape for the chosen ``cond_func`` (see Notes for requirements).  If 
        this parameter is not specified, it is set 
        to numpy.ones((len(sample1), N_marks)).
    
    marks2 : array_like
        len(sample2) x N_marks array of marks.  The supplied marks array must have the 
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
    Points with zero seperation are considered a self-match, and do not count as neighbors.
    
    There are multiple conditional functions available.  In general, each requires a 
    different number of marks per point, N_marks.  The conditional function gets passed 
    two arrays per pair, w1 and w2, of length N_marks and return a float.  
    You can pass in more than one piece of information about each point by choosing a 
    the input ``marks`` arrays to be multi-dimensional of shape (N_points, N_marks). 
    
    One point is considered to be a neighbor of another 
    if it lies within the enclosing sphere *and* 
    if the conditional function ``cond_func`` evaluates as True 
    when operating on the input ``marks`` data for that pair of points. 
    
    The available marking functions, ``cond_func`` and the associated integer 
    ID numbers are:
    
    #. greater than (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] > w_2[0] \\\\
                    False & : w_1[0] \\leq w_2[0] \\\\
                \\end{array}
                \\right.
    
    #. less than (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] < w_2[0] \\\\
                    False & : w_1[0] \\geq w_2[0] \\\\
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
    
    #. tolerance greater than (N_marks = 2)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] > (w_2[0]+w_1[1]) \\\\
                    False & : w_1[0] \\leq (w_2[0]+w_1[1]) \\\\
                \\end{array}
                \\right.
    
    #. tolerance less than (N_marks = 2)
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
    >>> is_isolated = overhauled_conditional_spherical_isolation(coords, coords, r_max, marks, marks, cond_func, period=period)
    """
    ### Process the inputs with the helper function
    result = _spherical_isolation_process_args(sample1, sample2, r_max, period,
            num_threads, approx_cell1_size, approx_cell2_size)
    x1in, y1in, z1in, x2in, y2in, z2in = result[0:6]
    r_max, period, num_threads, PBCs, approx_cell1_size, approx_cell2_size = result[6:]
    xperiod, yperiod, zperiod = period 

    search_xlength, search_ylength, search_zlength = r_max, r_max, r_max 

    # Process the input weights and with the helper function
    marks1, marks2 = _marked_npairs_process_weights(sample1, sample2,
            marks1, marks2, cond_func)

    ### Compute the estimates for the cell sizes
    approx_cell1_size, approx_cell2_size = (
        _set_approximate_cell_sizes(approx_cell1_size, approx_cell2_size, period)
        )
    approx_x1cell_size, approx_y1cell_size, approx_z1cell_size = approx_cell1_size
    approx_x2cell_size, approx_y2cell_size, approx_z2cell_size = approx_cell2_size

    # Build the rectangular mesh
    double_mesh = RectangularDoubleMesh(x1in, y1in, z1in, x2in, y2in, z2in,
        approx_x1cell_size, approx_y1cell_size, approx_z1cell_size,
        approx_x2cell_size, approx_y2cell_size, approx_z2cell_size,
        search_xlength, search_ylength, search_zlength, xperiod, yperiod, zperiod, PBCs)

    # Create a function object that has a single argument, for parallelization purposes
    engine = partial(marked_spherical_isolation_engine, 
        double_mesh, sample1[:,0], sample1[:,1], sample1[:,2], 
        sample2[:,0], sample2[:,1], sample2[:,2], marks1, marks2, cond_func, r_max)

    # Calculate the cell1 indices that will be looped over by the engine
    num_threads, cell1_tuples = _cell1_parallelization_indices(
        double_mesh.mesh1.ncells, num_threads)

    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        result = pool.map(engine, cell1_tuples)
        counts = np.sum(np.array(result), axis=0)
        pool.close()
    else:
        counts = engine(cell1_tuples[0])

    return np.array(counts, dtype=bool)


def conditional_cylindrical_isolation(sample1, sample2, rp_max, pi_max,
                          marks1, marks2, cond_func, period=None, num_threads=1,
                          approx_cell1_size=None, approx_cell2_size=None):
    """
    Determine whether a set of points, ``sample1``, has a neighbor in ``sample2`` 
    within a cylindrical volume that satisfies a user specified condition.
    
    Parameters
    ----------
    sample1 : array_like
        N1pts x 3 numpy array containing 3-D positions of points.
    
    sample2 : array_like
        N2pts x 3 numpy array containing 3-D positions of points.
    
    rp_max : float
        radius of the cylinder to seach for neighbors
    
    pi_max : float
        half the length of the cylinder to seach for neighbors
    
    marks1 : array_like
        len(sample1) x N_marks array of marks.  The supplied marks array must have the 
        appropiate shape for the chosen ``cond_func`` (see Notes for requirements).  If 
        this parameter is not specified, it is set 
        to numpy.ones((len(sample1), N_marks)).
    
    marks2 : array_like
        len(sample2) x N_marks array of marks.  The supplied marks array must have the 
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
    Points with zero seperation are considered a self-match, and do not count as neighbors.
    
    There are multiple conditional functions available.  In general, each requires a 
    different number of marks per point, N_marks.  The conditional function gets passed 
    two arrays per pair, w1 and w2, of length N_marks and return a float.  
    You can pass in more than one piece of information about each point by choosing a 
    the input ``marks`` arrays to be multi-dimensional of shape (N_points, N_marks). 
    
    One point is considered to be a neighbor of another 
    if the point lies within the enclosing cylinder *and* 
    if the conditional function ``cond_func`` evaluates as True 
    when operating on the input ``marks`` data for that pair of points. 
    
    The available marking functions, ``cond_func`` and the associated integer 
    ID numbers are:
    
    #. greater than (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] > w_2[0] \\\\
                    False & : w_1[0] \\leq w_2[0] \\\\
                \\end{array}
                \\right.
    
    #. less than (N_marks = 1)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    True & : w_1[0] < w_2[0] \\\\
                    False & : w_1[0] \\geq w_2[0] \\\\
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
    
    perp_distance_matrix, para_distance_matrix = conditional_xy_z_pair_matrix(
        sample1, sample2, rp_max, pi_max, marks1, marks2, cond_func, period = period,
        approx_cell1_size = approx_cell1_size, approx_cell2_size = approx_cell1_size)
    
    distance_matrix = np.sqrt(perp_distance_matrix**2 + para_distance_matrix**2)
    
    i ,j = distance_matrix.nonzero()
    
    #compare list of indices vs. possible indices
    N = len(sample1)
    inds = np.arange(0,N)
    
    is_isolated = np.in1d(inds, i, invert=True)
    
    return is_isolated
    
def _spherical_isolation_process_args(data1, data2, r_max, period, 
    num_threads, approx_cell1_size, approx_cell2_size):
    """
    """
    if num_threads is not 1:
        if num_threads=='max':
            num_threads = multiprocessing.cpu_count()
        if not isinstance(num_threads,int):
            msg = "Input ``num_threads`` argument must be an integer or the string 'max'"
            raise ValueError(msg)
    
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
            _enclose_in_box(x1, y1, z1, x2, y2, z2, 
                min_size=[r_max*3.0,r_max*3.0,r_max*3.0]))
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
            raise ValueError(msg)

    try:
        assert r_max < period[0]/3.
        assert r_max < period[1]/3.
        assert r_max < period[2]/3.
    except AssertionError:
        msg = ("Input ``r_max`` must be less than input period/3 in all dimensions.")
        raise ValueError(msg)

    if approx_cell1_size is None:
        approx_cell1_size = [r_max, r_max, r_max]
    elif custom_len(approx_cell1_size) == 1:
        approx_cell1_size = [approx_cell1_size, approx_cell1_size, approx_cell1_size]
    if approx_cell2_size is None:    
        approx_cell2_size = [r_max, r_max, r_max]
    elif custom_len(approx_cell2_size) == 1:
        approx_cell2_size = [approx_cell2_size, approx_cell2_size, approx_cell2_size]
        
    return (x1, y1, z1, x2, y2, z2, 
        r_max, period, num_threads, PBCs, 
        approx_cell1_size, approx_cell2_size)








