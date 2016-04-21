# -*- coding: utf-8 -*-
"""
This module contains private helper functions used throughout the 
`~halotools.mock_observables.pair_counters` subpackage to perform 
control flow on function arguments, bounds-checking and exception-handling. 
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from copy import copy 
from ...utils.array_utils import convert_to_ndarray

__author__ = ['Duncan Campbell', 'Andrew Hearin']

__all__ = ('_set_approximate_cell_sizes', '_cell1_parallelization_indices')

def _enclose_in_box(x1, y1, z1, x2, y2, z2, min_size=None):
    """
    Build box which encloses all points, shifting the points so that 
    the "leftmost" point is (0,0,0).
    
    Parameters
    ----------
    x1,y1,z1 : array_like
        cartesian positions of points
        
    x2,y2,z2 : array_like
        cartesian positions of points
        
    min_size : array_like
        minimum lengths of a side of the box.  If the minimum box constructed around the 
        points has a side i less than ``min_size[i]``, then the box is padded in order to
        obtain the minimum specified size. 
    
    Returns
    -------
    x1, y1, z1, x2, y2, z2, Lbox
        shifted positions and box size.
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
    
    Lbox = np.array([xyzmax, xyzmax, xyzmax])
    
    if min_size is not None:
        min_size = convert_to_ndarray(min_size)
        if np.any(Lbox<min_size):
            Lbox[(Lbox<min_size)] = min_size[(Lbox<min_size)]
            
    return x1, y1, z1, x2, y2, z2, Lbox

def _set_approximate_cell_sizes(approx_cell1_size, approx_cell2_size, rmax, period):
    """
    process the approximate cell size parameters.  
    If either is set to None, apply default settings.
    """

    #################################################
    ### Set the approximate cell sizes of the trees
    if approx_cell1_size is None:
        #approx_cell1_size = np.array([rmax, rmax, rmax])
        approx_cell1_size = period/10.0
    else:
        approx_cell1_size = convert_to_ndarray(approx_cell1_size)
        try:
            assert len(approx_cell1_size) == 3
            assert type(approx_cell1_size) is np.ndarray
            assert approx_cell1_size.ndim == 1
        except AssertionError:
            msg = ("Input ``approx_cell1_size`` must be a length-3 sequence")
            raise ValueError(msg)

    if approx_cell2_size is None:
        approx_cell2_size = copy(approx_cell1_size)
    else:
        approx_cell2_size = convert_to_ndarray(approx_cell2_size)
        try:
            assert len(approx_cell2_size) == 3
            assert type(approx_cell2_size) is np.ndarray
            assert approx_cell2_size.ndim == 1
        except AssertionError:
            msg = ("Input ``approx_cell2_size`` must be a length-3 sequence")
            raise ValueError(msg)

    return approx_cell1_size, approx_cell2_size

def _cell1_parallelization_indices(ncells, num_threads):
    """
    """
    if num_threads == 1:
        return 1, [(0, ncells)]
    elif num_threads > ncells:
        return ncells, [(a, a+1) for a in np.arange(ncells)]
    else:
        list_with_possibly_empty_arrays = np.array_split(np.arange(ncells), num_threads)
        list_of_nonempty_arrays = [a for a in list_with_possibly_empty_arrays if len(a) > 0]
        list_of_tuples = [(x[0], x[0] + len(x)) for x in list_of_nonempty_arrays]
        return num_threads, list_of_tuples
