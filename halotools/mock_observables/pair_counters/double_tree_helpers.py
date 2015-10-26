# -*- coding: utf-8 -*-

"""
rectangular Cuboid Pair Counter. 
This module contains pair counting functions used to count the number of pairs with 
separations less than or equal to r, optimized for simulation boxes.
This module also contains a 'main' function which runs speed tests.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from time import time
from copy import copy 
import sys
import multiprocessing
from functools import partial

from ...custom_exceptions import *
from ...utils.array_utils import convert_to_ndarray, array_is_monotonic

__all__ = ['_npairs_process_args', '_enclose_in_box', '_set_approximate_cell_sizes']

def _npairs_process_args(data1, data2, rbins, period, 
    verbose, num_threads, approx_cell1_size, approx_cell2_size):
    """
    """
    if num_threads is not 1:
        if num_threads=='max':
            num_threads = multiprocessing.cpu_count()
        if not isinstance(num_threads,int):
            msg = "Input ``num_threads`` argument must be an integer or the string 'max'"
            raise HalotoolsError(msg)
    
    # Passively enforce that we are working with ndarrays
    x1 = data1[:,0]
    y1 = data1[:,1]
    z1 = data1[:,2]
    x2 = data2[:,0]
    y2 = data2[:,1]
    z2 = data2[:,2]
    rbins = convert_to_ndarray(rbins)

    try:
        assert rbins.ndim == 1
        assert len(rbins) > 1
        if len(rbins) > 2:
            assert array_is_monotonic(rbins, strict = True) == 1
    except AssertionError:
        msg = "Input ``rbins`` must be a monotonically increasing 1D array with at least two entries"
        raise HalotoolsError(msg)

    # Set the boolean value for the PBCs variable
    if period is None:
        PBCs = False
        x1, y1, z1, x2, y2, z2, period = (
            _enclose_in_box(x1, y1, z1, x2, y2, z2))
    else:
        PBCs = True
        period = convert_to_ndarray(period)
        if len(period) == 1:
            period = np.array([period[0]]*3)
        try:
            assert np.all(period < np.inf)
            assert np.all(period > 0)
        except AssertionError:
            msg = "Input ``period`` must be a bounded positive number in all dimensions"
            raise HalotoolsError(msg)

    return x1, y1, z1, x2, y2, z2, rbins, period, num_threads, PBCs


def _enclose_in_box(x1, y1, z1, x2, y2, z2):
    """
    build axis aligned box which encloses all points. 
    shift points so cube's origin is at 0,0,0.
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
    
    period = np.array([xyzmax, xyzmax, xyzmax])
    
    return x1, y1, z1, x2, y2, z2, period

def _set_approximate_cell_sizes(approx_cell1_size, approx_cell2_size, rmax):
    """
    """

    #################################################
    ### Set the approximate cell sizes of the trees
    if approx_cell1_size is None:
        approx_cell1_size = np.array([rmax, rmax, rmax])
    else:
        approx_cell1_size = convert_to_ndarray(approx_cell1_size)
        try:
            assert len(approx_cell1_size) == 3
            assert type(approx_cell1_size) is np.ndarray
            assert approx_cell1_size.ndim == 1
        except AssertionError:
            msg = ("Input ``approx_cell1_size`` must be a length-3 sequence")
            raise HalotoolsError(msg)

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
            raise HalotoolsError(msg)

    return approx_cell1_size, approx_cell2_size





    