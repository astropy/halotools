# -*- coding: utf-8 -*-

"""
helper functions for the pairwise velocity statistics module
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from warnings import warn
from multiprocessing import cpu_count 
from ..custom_exceptions import *
from ..utils.array_utils import convert_to_ndarray, array_is_monotonic

__all__ = ['_pairwise_velocity_stats_process_args']
__author__ = ['Duncan Campbell']

def _pairwise_velocity_stats_process_args(sample1, velocities1, rbins, sample2,
                                 velocities2, period, do_auto, do_cross,
                                 num_threads, max_sample_size,
                                 approx_cell1_size, approx_cell2_size):
    """ 
    Private method to do bounds-checking on the arguments passed to 
    `~halotools.mock_observables.pairwise_velocity_stats`. 
    """
    
    sample1 = convert_to_ndarray(sample1)
    velocities1 = convert_to_ndarray(sample1)
    
    if sample2 is not None: 
        sample2 = convert_to_ndarray(sample2)
        if velocities2 is None:
            msg = ("\n If `sample2` is passed as an argument, \n"
                   "`velocities2` must also be specified.")
            raise HalotoolsError(msg)
        else: velocities2 = convert_to_ndarray(velocities2)
        if np.all(sample1==sample2):
            _sample1_is_sample2 = True
            msg = ("\n Warning: `sample1` and `sample2` are exactly the same, \n"
                   "only the auto-function will be returned.\n")
            warn(msg)
            do_cross==False
        else: 
            _sample1_is_sample2 = False
    else: 
        sample2 = sample1
        velocities2 = velocities1
        _sample1_is_sample2 = True
    
    # down sample if sample size exceeds max_sample_size.
    if _sample1_is_sample2 is True:
        if (len(sample1) > max_sample_size):
            inds = np.arange(0,len(sample1))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample1 = sample1[inds]
            velocities1 = velocities1[inds]
            print('\n downsampling `sample1`...')
    else:
        if len(sample1) > max_sample_size:
            inds = np.arange(0,len(sample1))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample1 = sample1[inds]
            velocities1 = velocities1[inds]
            print('\n down sampling `sample1`...')
        if len(sample2) > max_sample_size:
            inds = np.arange(0,len(sample2))
            np.random.shuffle(inds)
            inds = inds[0:max_sample_size]
            sample2 = sample2[inds]
            velocities2 = velocities2[inds]
            print('\n down sampling `sample2`...')
    
    #check the radial bins parameter
    rbins = convert_to_ndarray(rbins)
    rmax = np.max(rbins)
    try:
        assert rbins.ndim == 1
        assert len(rbins) > 1
        if len(rbins) > 2:
            assert array_is_monotonic(rbins, strict = True) == 1
    except AssertionError:
        msg = ("\n Input `rbins` must be a monotonically increasing \n"
               "1-D array with at least two entries")
        raise HalotoolsError(msg)
        
    #Process period entry and check for consistency.
    if period is None:
        PBCs = False
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

    #check for input parameter consistency
    if (period is not None):
        if (rmax >= np.min(period)/3.0):
            msg = ("\n The maximum length over which you search for pairs \n"
                   "of points cannot be larger than Lbox/3 in any dimension. \n"
                   "If you need to count pairs on these length scales, \n"
                   "you should use a larger simulation. \n")
            raise HalotoolsError(msg)
    
    if (sample2 is not None) & (sample1.shape[-1] != sample2.shape[-1]):
        msg = ('\n Sample1 and sample2 must have same dimension.\n')
        raise HalotoolsError(msg)
    
    if (type(do_auto) is not bool) | (type(do_cross) is not bool):
        msg = ('\n  do_auto and do_cross keywords must be of type boolean.')
        raise HalotoolsError(msg)
    
    if num_threads == 'max':
        num_threads = cpu_count()
    
    return sample1, velocities1, rbins, sample2, velocities2, period, do_auto,\
           do_cross, num_threads, _sample1_is_sample2, PBCs